"""SOFA label check for the neural models: hide the pattern of the five SOFA variables.

The sepsis label is computed from SOFA lab values, so part of the pattern signal may be linked
to the label by construction. As for gradient boosting (``run_baselines.py``), we retrain the
`all` model with the measurement pattern (mask and time gaps) of the five SOFA variables
hidden, and compare with hiding the pattern of five randomly chosen other variables (the same
five draws as gradient boosting). The values of the hidden variables stay in the inputs.

    python scripts/sofa_check_neural.py --print-commands | xargs -d '\n' -P 4 -n 1 bash -c  # train
    python scripts/sofa_check_neural.py   # -> results/sofa_check_neural/sofa_check_neural.csv

Loss = AUROC(nothing hidden) - AUROC(pattern hidden), mean over seeds 0-2, 95% interval from
the paired patient bootstrap (every draw scores every model and seed on the same patients).
``sofa_minus_random`` is the SOFA loss minus the mean loss of the five random draws, with the
same paired bootstrap.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from missingness.eval import bootstrap_contrast
from missingness.eval.metrics import auroc
from missingness.experiment import eval_subsets
from missingness.paths import CONFIG_DIR, OUTPUT_DIR, REPO_ROOT

MODELS = ("gru", "grud", "odernn")
SEEDS = (0, 1, 2)
TESTS = ("test_hospital1", "test_hospital2")
SUBSETS = ("pre_onset", "all")


def variants(n_random: int = 5, seed: int = 0) -> list[tuple[str, list[str]]]:
    """('none', []), ('sofa', SOFA variables), ('random<k>', five others): as run_baselines.py."""
    data = yaml.safe_load((CONFIG_DIR / "data" / "physionet2019.yaml").read_text())
    sofa = data["sofa_variables"]
    others = [v for v in data["variables"] if v not in sofa]
    rng = np.random.default_rng(seed)
    out = [("none", []), ("sofa", list(sofa))]
    for k in range(n_random):
        pick = rng.choice(others, size=len(sofa), replace=False)
        out.append((f"random{k}", sorted(str(v) for v in pick)))
    return out


def run_dir(model: str, tag: str, seed: int):
    suffix = "" if tag == "none" else f"_nopattern-{tag}"
    return OUTPUT_DIR / f"physionet2019_{model}_all{suffix}_seed{seed}"


def commands() -> list[str]:
    cmds = []
    for model in MODELS:
        for tag, names in variants():
            for seed in SEEDS:
                extra = ""
                if names:
                    extra = f" 'pattern_drop=[{','.join(names)}]' pattern_drop_tag={tag}"
                cmds.append(
                    f"python -m missingness.train configs/experiment/sepsis_{model}.yaml "
                    f"inputs=all seed={seed}{extra} --skip-done"
                    f" > {OUTPUT_DIR}/log_sofa_{model}_{tag}_{seed}.txt 2>&1"
                )
    return cmds


def load(model: str, tag: str, seed: int, test: str) -> pd.DataFrame:
    f = run_dir(model, tag, seed) / "predictions" / f"{test}.parquet"
    return pd.read_parquet(f).sort_values(["patient_id", "hour"]).reset_index(drop=True)


def table(n_boot: int) -> pd.DataFrame:
    rows = []
    vs = variants()
    for model in MODELS:
        for test in TESTS:
            preds = {(t, s): load(model, t, s, test) for t, _ in vs for s in SEEDS}
            ref = preds[("none", 0)]
            for p in preds.values():
                assert p[["patient_id", "hour", "label"]].equals(
                    ref[["patient_id", "hour", "label"]]
                )
            subsets = eval_subsets(ref, "hourly")
            for subset in SUBSETS:
                keep = subsets[subset]
                y, pid = ref["label"].to_numpy()[keep], ref["patient_id"].to_numpy()[keep]

                def sc(t: str, s: int, keep=keep, preds=preds) -> np.ndarray:
                    return preds[(t, s)]["score"].to_numpy()[keep]

                w = 1 / len(SEEDS)
                for tag, names in vs:
                    base = [sc("none", s) for s in SEEDS]
                    var = [sc(tag, s) for s in SEEDS]
                    d = bootstrap_contrast(
                        y, base + var, [w] * 3 + [-w] * 3, pid, n_boot=n_boot, seed=0
                    )
                    rows.append(
                        {
                            "model": model,
                            "test_set": test,
                            "subset": subset,
                            "variant": tag,
                            "pattern_hidden_for": ",".join(names),
                            "auroc": np.mean([auroc(y, v) for v in var]),
                            "loss": d["value"],
                            "loss_lo": d["lo"],
                            "loss_hi": d["hi"],
                            "seed_losses": " ".join(
                                f"{auroc(y, b) - auroc(y, v):+.4f}"
                                for b, v in zip(base, var, strict=True)
                            ),
                        }
                    )
                # SOFA loss minus the mean loss of the random draws = mean(random) - sofa AUROC.
                rand = [t for t, _ in vs if t.startswith("random")]
                wr = w / len(rand)
                scores = [sc(t, s) for t in rand for s in SEEDS] + [sc("sofa", s) for s in SEEDS]
                weights = [wr] * (len(rand) * 3) + [-w] * 3
                d = bootstrap_contrast(y, scores, weights, pid, n_boot=n_boot, seed=0)
                rows.append(
                    {
                        "model": model,
                        "test_set": test,
                        "subset": subset,
                        "variant": "sofa_minus_random",
                        "pattern_hidden_for": "",
                        "auroc": np.nan,
                        "loss": d["value"],
                        "loss_lo": d["lo"],
                        "loss_hi": d["hi"],
                        "seed_losses": "",
                    }
                )
            print(f"done {model} {test}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--print-commands", action="store_true", help="print the training commands")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    if args.print_commands:
        print("\n".join(commands()))
        return
    out = REPO_ROOT / "results" / "sofa_check_neural" / "sofa_check_neural.csv"
    table(args.n_boot).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
