"""The primary measures M1–M4 (study plan, section 4) from the saved predictions of all runs.

    python scripts/measures.py                       # -> results/measures.csv
    python scripts/measures.py --models grud odernn  # only these; other models' rows stay
    python scripts/measures.py --models gru --metrics auroc auprc

Every measure is a difference of AUROCs, averaged over the seeds, with a 95% interval from a
paired patient bootstrap: each draw resamples the patients once and scores every model and
seed on them (``eval.bootstrap_contrast``). The interval reflects the test patients; the
spread over seeds is in ``seed_values``.

Measures (AUROC differences; a positive value means what the name says):

- M1 Δmask: all − values on the clean test set; S (pattern share) next to it when pattern
  runs exist.
- M2: (random − state) for all minus (random − state) for values, per removal fraction p: the
  extra loss under state-dependent removal that the pattern inputs add. With whole busy hours
  (D27) the same measure is ``M2_state_vs_random_hours`` at p 0.2, 0.25, 0.3.
- M3 gain: rule on at strength s minus rule on at strength 1, per input variant (E5).
- M3 loss_off: rule on minus rule off at strength s, per input variant, and all minus values.
  Each design of the rule (column ``planted_q1``: 1.0 for D21, 0.8 for D24) is kept apart.
- M4 thinning: (recorded − thinned) for all minus the same for values, hospital-1 test set.
- M4 transfer: (hospital 1 − hospital 2) for all minus the same for values; the two test sets
  are independent samples, so their bootstrap draws are resampled separately.

RQ2 redesign (D29, proposed). Runs trained on another view (``features: snapshot`` or
``decoded``) are kept apart (column ``view``); for the snapshot view, M1 is the built-in check
(all and values see the same common schedule, so it should be about 0). Across views:

- M1_decoded_gap: pattern model on the true m and delta − pattern model on m and delta rebuilt
  from forward-filled x (leak audit). Near 0 means a values-only model can read the pattern.
- M1_bound_snapshot: forward-fill all − snapshot values, the upper end of the Δmask bound
  (the lower end is M1 with forward fill).
- M2_ref_snapshot, M4_thinning_ref_snapshot: M2 and M4 with the snapshot values model as the
  reference instead of the forward-fill values model.
- M2_channel_pattern / M2_channel_values: (random − state) for the forward-fill all model
  when only its pattern inputs (or only its values) come from the removed data;
  M4_thinning_channel_pattern / _values: recorded − thinned in the same way.

Only runs with the same seeds for all variants of a measure are combined. Rows of all
prediction files of a measure must match (same patients, hours, labels); this is checked.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from missingness.data.views import view_name
from missingness.eval import bootstrap_contrast, information_share
from missingness.eval.metrics import METRICS
from missingness.experiment import eval_subsets
from missingness.paths import OUTPUT_DIR, REPO_ROOT
from missingness.results import write_table

TASK_OF = {"hourly": "sepsis", "stay": "mortality"}


def runs_index(outputs: Path, models: list[str] | None) -> pd.DataFrame:
    rows = []
    for cfg_path in sorted(outputs.glob("*/config.yaml")):
        run = cfg_path.parent
        if run.name.startswith("tune_") or not (run / "metrics.json").exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        if models and cfg["model"]["name"] not in models:
            continue
        pl = cfg.get("planted") or {}
        rows.append(
            {
                "path": run,
                "data_tag": cfg["data"].get("tag", cfg["data"]["name"]),
                "task": cfg["data"]["task"],
                "model": cfg["model"]["name"],
                "inputs": cfg["inputs"],
                "view": view_name(cfg),
                "seed": cfg["seed"],
                "strength": round(pl["q1"] / pl["q0"], 3) if pl else None,
                "q1": pl.get("q1"),
            }
        )
    return pd.DataFrame(rows)


def load_pred(run: Path, test_set: str, tag: str | None = None) -> pd.DataFrame | None:
    """Clean predictions (tag None) or perturbed ones, sorted by patient and hour."""
    folder = run / "predictions" if tag is None else run / "perturbed" / tag
    f = folder / f"{test_set}.parquet"
    if not f.exists():
        return None
    return pd.read_parquet(f).sort_values(["patient_id", "hour"]).reset_index(drop=True)


def contrast(frames, weights, task, metrics, n_boot, seed=0, keep_draws=False):
    """Yield (subset, metric, result, masks) for sum_k w_k metric(frame_k), paired over patients."""
    ref = frames[0]
    for f in frames[1:]:
        same = f[["patient_id", "hour", "label"]].equals(ref[["patient_id", "hour", "label"]])
        assert same, "prediction files of one measure do not have the same rows"
    masks = eval_subsets(ref, task)
    for subset, keep in masks.items():
        y, pid = ref["label"].to_numpy()[keep], ref["patient_id"].to_numpy()[keep]
        scores = [f["score"].to_numpy()[keep] for f in frames]
        for metric in metrics:
            res = bootstrap_contrast(
                y, scores, weights, pid, metric, n_boot, seed=seed, keep_draws=keep_draws
            )
            yield subset, metric, res, masks


def mean_metric(frames, keep, metric) -> list[float]:
    """Metric of each frame on the rows ``keep`` (all frames have the same rows)."""
    fn = METRICS[metric]
    return [fn(f["label"].to_numpy()[keep], f["score"].to_numpy()[keep]) for f in frames]


def seed_values(groups: list[list[pd.DataFrame]], signs, keep, metric) -> list[float]:
    """The measure per seed: groups[k][i] is frame k of seed i."""
    n = len(groups[0])
    vals = [mean_metric([g[i] for g in groups], keep, metric) for i in range(n)]
    return [round(float(np.dot(signs, v)), 4) for v in vals]


def m3_rows(paths: dict, key: dict, ts: str, task: str, add) -> None:
    """M3 (E5) on test set ``ts``; ``paths[(inputs, strength)][seed]`` = run folder."""
    for inp in ("all", "values"):
        if (inp, 1.0) not in paths:
            continue
        for s in sorted(k[1] for k in paths if k[0] == inp and k[1] != 1.0):
            seeds = sorted(set(paths[(inp, s)]) & set(paths[(inp, 1.0)]))
            on = [load_pred(paths[(inp, s)][i], ts) for i in seeds]
            base = [load_pred(paths[(inp, 1.0)][i], ts) for i in seeds]
            add("M3_gain", key | {"inputs": inp}, f"s={s}", ts, [on, base], [1, -1], task)
    for s in sorted({k[1] for k in paths}):
        per = {}
        for inp in ("all", "values"):
            runs = paths.get((inp, s), {})
            on = [load_pred(runs[i], ts) for i in sorted(runs)]
            off = [load_pred(runs[i], f"{ts}_rule_off", "planted_off_seed0") for i in sorted(runs)]
            if not runs or any(f is None for f in on + off):
                continue
            per[inp] = (sorted(runs), on, off)
            add("M3_loss_off", key | {"inputs": inp}, f"s={s}", ts, [on, off], [1, -1], task)
        if {"all", "values"} <= set(per) and per["all"][0] == per["values"][0]:
            groups = [*per["all"][1:], *per["values"][1:]]
            add("M3_loss_off", key | {"inputs": "all-values"}, f"s={s}", ts, groups,
                [1, -1, -1, 1], task)  # fmt: skip


def m2_m4_rows(name: str, key: dict, a: list, v: list, test_sets: list, task: str, add) -> None:
    """M2 at p 0.05-0.15 and M4 thinning with run lists ``a`` (all) and ``v`` (reference)."""
    for p in (0.05, 0.1, 0.15):
        for ts in test_sets:
            tags = [f"random_p{p}_seed0", f"state_p{p}_seed0"]
            groups = [[load_pred(r, ts, t) for r in runs] for runs in (a, v) for t in tags]
            if not any(f is None for grp in groups for f in grp):
                add(f"M2_{name}", key, f"p={p}", ts, groups, [1, -1, -1, 1], task)
    if "test_hospital2" in test_sets:
        thin = [f"thin_seed{t}" for t in (0, 1, 2)]
        groups = []
        for runs in (a, v):
            groups += [
                [load_pred(r, "test_hospital1") for r in runs for _ in thin],
                [load_pred(r, "test_hospital1_thinned", t) for r in runs for t in thin],
            ]
        if not any(f is None for grp in groups for f in grp):
            add(f"M4_thinning_{name}", key, "", "test_hospital1", groups, [1, -1, -1, 1], task)


def paired_runs(g: pd.DataFrame, first: tuple, second: tuple) -> tuple[list, list]:
    """Run folders of two (view, inputs) variants for the seeds both have."""
    sel = [
        g[(g["view"] == view) & (g["inputs"] == inp)].set_index("seed")["path"]
        for view, inp in (first, second)
    ]
    seeds = sorted(set(sel[0].index) & set(sel[1].index))
    return [sel[0][s] for s in seeds], [sel[1][s] for s in seeds]


def cross_view_rows(plain: pd.DataFrame, add) -> None:
    """Comparisons across views (D29): decoded pattern, snapshot bound and reference."""
    for (data_tag, model, task), g in plain.groupby(["data_tag", "model", "task"]):
        key = {"task": TASK_OF[task], "data_tag": data_tag, "model": model, "view": "cross"}
        pairs = [("M1_decoded_gap", ("forward_fill", "pattern"), ("decoded", "pattern"), "")]
        for snap in sorted(v for v in g["view"].unique() if v.startswith("snapshot")):
            suffix = snap.removeprefix("snapshot")  # '' or '-<tag>' for a snapshot variant
            pairs.append(("M1_bound_snapshot", ("forward_fill", "all"), (snap, "values"), suffix))
        for name, first, second, suffix in pairs:
            a, b = paired_runs(g, first, second)
            if not a:
                continue
            test_sets = [f.stem for f in sorted((a[0] / "predictions").glob("*.parquet"))]
            for ts in test_sets:
                groups = [[load_pred(r, ts) for r in a], [load_pred(r, ts) for r in b]]
                add(f"{name}{suffix}", key, "", ts, groups, [1, -1], task)
            if name == "M1_bound_snapshot":
                m2_m4_rows(f"ref_snapshot{suffix}", key, a, b, test_sets, task, add)


def channel_rows(plain: pd.DataFrame, add) -> None:
    """Channel intervention (D29) on the forward-fill all runs."""
    sel = plain[(plain["view"] == "forward_fill") & (plain["inputs"] == "all")]
    for (data_tag, model, task), g in sel.groupby(["data_tag", "model", "task"]):
        key = {"task": TASK_OF[task], "data_tag": data_tag, "model": model, "view": "channel"}
        runs = list(g.sort_values("seed")["path"])
        test_sets = [f.stem for f in sorted((runs[0] / "predictions").glob("*.parquet"))]
        for ch in ("pattern", "values"):
            for p in (0.05, 0.1, 0.15):
                for ts in test_sets:
                    tags = [f"random_p{p}_ch{ch}_seed0", f"state_p{p}_ch{ch}_seed0"]
                    groups = [[load_pred(r, ts, t) for r in runs] for t in tags]
                    if not any(f is None for grp in groups for f in grp):
                        add(f"M2_channel_{ch}", key, f"p={p}", ts, groups, [1, -1], task)
            if "test_hospital2" in test_sets:
                thin = [f"thin_ch{ch}_seed{t}" for t in (0, 1, 2)]
                groups = [
                    [load_pred(r, "test_hospital1") for r in runs for _ in thin],
                    [load_pred(r, "test_hospital1_thinned", t) for r in runs for t in thin],
                ]
                if not any(f is None for grp in groups for f in grp):
                    add(f"M4_thinning_channel_{ch}", key, "", "test_hospital1", groups,
                        [1, -1], task)  # fmt: skip


def measure_rows(idx: pd.DataFrame, metrics: list[str], n_boot: int) -> list[dict]:
    rows = []

    def add(name, key, setting, test_set, groups, signs, task, extra=None):
        n = len(groups[0])
        frames = [f for g in groups for f in g]
        weights = [s / n for s in signs for _ in range(n)]
        for subset, metric, res, masks in contrast(frames, weights, task, metrics, n_boot):
            rows.append(
                {"measure": name, **key, "setting": setting, "test_set": test_set}
                | {"subset": subset, "metric": metric, "n_seeds": n}
                | {k: round(res[k], 4) for k in ("value", "lo", "hi")}
                | {"seed_values": seed_values(groups, signs, masks[subset], metric)}
                | (extra or {})
            )

    plain = idx[idx["strength"].isna()]
    for (data_tag, model, task, view), g in plain.groupby(["data_tag", "model", "task", "view"]):
        key = {"task": TASK_OF[task], "data_tag": data_tag, "model": model, "view": view}
        by = {inp: d.set_index("seed")["path"] for inp, d in g.groupby("inputs")}
        if not {"all", "values"} <= set(by):
            continue
        seeds = sorted(set(by["all"].index) & set(by["values"].index))
        if not seeds:
            continue
        a, v = [by["all"][s] for s in seeds], [by["values"][s] for s in seeds]
        test_sets = [f.stem for f in sorted((a[0] / "predictions").glob("*.parquet"))]
        print(f"{data_tag} {model} {view}: seeds {seeds}")

        for ts in test_sets:  # M1
            ga, gv = [load_pred(r, ts) for r in a], [load_pred(r, ts) for r in v]
            extra = {}
            if "pattern" in by and set(seeds) <= set(by["pattern"].index):
                gp = [load_pred(by["pattern"][s], ts) for s in seeds]
                rows_all = np.ones(len(ga[0]), dtype=bool)
                pa = np.mean(mean_metric(gp, rows_all, "auroc"))
                aa = np.mean(mean_metric(ga, rows_all, "auroc"))
                extra = {"S_all_rows": round(information_share(pa, aa), 3)}
            add("M1_delta_mask", key, "", ts, [ga, gv], [1, -1], task, extra)

        m2 = [("M2_state_vs_random", "", p) for p in (0.05, 0.1, 0.15)]
        m2 += [("M2_state_vs_random_hours", "_hours", p) for p in (0.2, 0.25, 0.3)]
        for name, suffix, p in m2:  # M2
            for ts in test_sets:
                tags = [f"random{suffix}_p{p}_seed0", f"state{suffix}_p{p}_seed0"]
                groups = [[load_pred(r, ts, t) for r in runs] for runs in (a, v) for t in tags]
                if any(f is None for grp in groups for f in grp):
                    continue
                add(name, key, f"p={p}", ts, groups, [1, -1, -1, 1], task)

        if "test_hospital2" in test_sets:  # M4
            thin = [f"thin_seed{t}" for t in (0, 1, 2)]
            groups = []
            for runs in (a, v):
                clean = [load_pred(r, "test_hospital1") for r in runs for _ in thin]
                thinned = [load_pred(r, "test_hospital1_thinned", t) for r in runs for t in thin]
                groups += [clean, thinned]
            if not any(f is None for grp in groups for f in grp):
                add("M4_thinning", key, "", "test_hospital1", groups, [1, -1, -1, 1], task)
            draws = {}
            for ts in ("test_hospital1", "test_hospital2"):
                frames = [load_pred(r, ts) for r in a] + [load_pred(r, ts) for r in v]
                n = len(a)
                w = [1 / n] * n + [-1 / n] * n
                seed = 0 if ts == "test_hospital1" else 1  # independent resamples
                for subset, metric, res, _ in contrast(
                    frames, w, task, metrics, n_boot, seed, True
                ):
                    draws[(ts, subset, metric)] = res
            for (ts, subset, metric), r1 in draws.items():
                if ts != "test_hospital1":
                    continue
                r2 = draws[("test_hospital2", subset, metric)]
                m = min(len(r1["draws"]), len(r2["draws"]))
                d = r1["draws"][:m] - r2["draws"][:m]
                lo, hi = np.quantile(d, [0.025, 0.975])
                rows.append(
                    {"measure": "M4_transfer", **key, "setting": "", "test_set": "h1-h2"}
                    | {"subset": subset, "metric": metric, "n_seeds": len(a)}
                    | {"value": round(r1["value"] - r2["value"], 4)}
                    | {"lo": round(float(lo), 4), "hi": round(float(hi), 4)}
                )

    cross_view_rows(plain, add)
    channel_rows(plain, add)

    planted = idx[idx["strength"].notna()]  # M3, one design (q1) at a time: D21 q1 = 1, D24 0.8
    for (data_tag, model, task, q1), g in planted.groupby(["data_tag", "model", "task", "q1"]):
        key = {"task": TASK_OF[task], "data_tag": data_tag, "model": model, "planted_q1": q1}
        paths = defaultdict(dict)
        for _, r in g.iterrows():
            paths[(r["inputs"], r["strength"])][r["seed"]] = r["path"]
        first = g["path"].iloc[0] / "predictions"
        for ts in [f.stem for f in sorted(first.glob("*.parquet"))]:
            m3_rows(paths, key, ts, task, add)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outputs", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "measures.csv")
    ap.add_argument(
        "--models", nargs="*", help="only these models; the other models' rows in --out stay"
    )
    ap.add_argument("--metrics", nargs="+", default=["auroc"])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    idx = runs_index(args.outputs, args.models)
    if idx.empty:
        raise SystemExit(f"no finished runs in {args.outputs}")
    table = pd.DataFrame(measure_rows(idx, args.metrics, args.n_boot))
    write_table(table, args.out, args.models)
    print(f"{len(table)} rows -> {args.out}")


if __name__ == "__main__":
    main()
