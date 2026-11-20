"""One figure per primary measure (M1-M4) for one model, from the committed result tables.

    python scripts/plot_findings.py             # the GRU figures of the week-2 handoff
    python scripts/plot_findings.py --model grud

Reads `results/runs.csv` and `results/measures.csv` (no data, no predictions) and writes
`results/findings_<model>_{1_information,2_removal,3_planted,4_transfer}.png`. The same
script for every model keeps the figures comparable side by side.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from missingness.paths import REPO_ROOT  # noqa: E402

RESULTS = REPO_ROOT / "results"
NAMES = {"gru": "GRU", "grud": "GRU-D", "odernn": "ODE-RNN"}

# Colours: one per input variant, the same in every figure.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
VARIANT_COLOR = {"values": "#2a78d6", "pattern": "#eb6834", "all": "#1baf7a"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": INK,
    }
)

CASES = [
    ("physionet2012", "test", "all", "Mortality"),
    ("physionet2019", "test_hospital1", "pre_onset", "Sepsis, hospital 1, pre-onset"),
]


def suptitle(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)


def plain(r: pd.DataFrame) -> pd.DataFrame:
    """Runs without a planted pattern."""
    return r[r.planted_variable.isna()]


def fig_information(r: pd.DataFrame, m: pd.DataFrame, name: str, out: str) -> None:
    """M1: test AUROC of pattern, values and all, on one scale from chance."""
    p = plain(r)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    top = 0.92
    for row, (tag, test, subset, _) in enumerate(CASES):
        q = p[
            (p.data_tag == tag)
            & (p.test_set == test)
            & (p.subset == subset)
            & (p.perturb == "none")
        ]
        means = {}
        for v in ("pattern", "values", "all"):
            vals = q[q.inputs == v]["value"].to_numpy()
            means[v] = vals.mean()
            top = max(top, vals.max() + 0.03)
            ax.scatter(vals, [row] * len(vals), s=14, color=VARIANT_COLOR[v], alpha=0.5, zorder=2)
            ax.scatter(
                [vals.mean()],
                [row],
                s=110,
                color=VARIANT_COLOR[v],
                zorder=3,
                edgecolor=SURFACE,
                lw=2,
                label=v if row == 0 else None,
            )
        ax.plot([0.5, means["all"]], [row, row], color=GRID, lw=6, zorder=1, solid_capstyle="round")
        ax.text(0.5, row - 0.28, "chance", color=INK2, fontsize=9, ha="center")
        mm = m[
            (m.measure == "M1_delta_mask")
            & (m.data_tag == tag)
            & (m.test_set == test)
            & (m.subset == subset)
        ].iloc[0]
        ax.annotate(
            "",
            xy=(means["all"], row + 0.22),
            xytext=(means["values"], row + 0.22),
            arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1},
        )
        ax.text(
            (means["all"] + means["values"]) / 2,
            row + 0.42,
            f"adding the pattern: {mm.value:+.3f}\n[{mm.lo:.3f}, {mm.hi:.3f}]",
            ha="center",
            va="top",
            fontsize=9,
            color=INK,
        )
        share = (means["pattern"] - 0.5) / (means["all"] - 0.5)
        ax.text(
            (0.5 + means["pattern"]) / 2,
            row - 0.28,
            f"pattern alone: {share:.0%} of the way",
            ha="center",
            color=INK2,
            fontsize=9,
        )
    ax.set_yticks(range(len(CASES)), [c[3] for c in CASES])
    ax.set_ylim(len(CASES) - 0.2, -0.6)
    ax.set_xlim(0.48, top)
    ax.set_xlabel("test AUROC (big dot: mean of 3 seeds; small dots: each seed)")
    ax.set_title(f"1. Information (M1), {name}: how far each input gets")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.2))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_removal(r: pd.DataFrame, m: pd.DataFrame, name: str, out: str) -> None:
    """M2: AUROC lost under random and state-dependent removal, and M2 itself."""
    ps = [0.05, 0.1, 0.15]
    p = plain(r)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (tag, test, subset, title) in zip(axes[:2], CASES, strict=True):
        q = p[(p.data_tag == tag) & (p.test_set == test) & (p.subset == subset)]
        for v in ("all", "values"):
            base = q[(q.inputs == v) & (q.perturb == "none")].set_index("seed")["value"]
            for method, ls in (("random", "-"), ("state", "--")):
                drops = []
                for pp in ps:
                    pert = q[(q.inputs == v) & (q.perturb == f"{method}_p{pp:g}_seed0")]
                    drops.append((base - pert.set_index("seed")["value"]).mean())
                label = "state-dependent" if method == "state" else "random"
                ax.plot(
                    ps,
                    drops,
                    color=VARIANT_COLOR[v],
                    ls=ls,
                    lw=2,
                    marker="o",
                    markersize=6,
                    label=f"{v}, {label}",
                )
        ax.set_title(title)
        ax.set_xticks(ps, ["5%", "10%", "15%"])
        ax.set_xlabel("share of values removed (p)")
        ax.set_ylabel("AUROC lost")
        ax.axhline(0, color=INK2, lw=1)
    axes[0].legend(loc="upper left", fontsize=8.5)
    ax = axes[2]
    for i, (tag, test, subset, title) in enumerate(CASES):
        q = m[
            (m.measure == "M2_state_vs_random")
            & (m.data_tag == tag)
            & (m.test_set == test)
            & (m.subset == subset)
        ].copy()
        q["p"] = q.setting.str.replace("p=", "").astype(float)
        q = q.sort_values("p")
        x = np.array(ps) + (i - 0.5) * 0.006
        ax.errorbar(
            x,
            q.value,
            yerr=[q.value - q.lo, q.hi - q.value],
            fmt="os"[i],
            color=INK,
            ms=6,
            lw=1.6,
            capsize=0,
            mfc=SURFACE if i else INK,
            mew=1.6,
            label=title.split(",")[0],
        )
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(ps, ["5%", "10%", "15%"])
    ax.set_xlabel("share of values removed (p)")
    ax.set_ylabel("M2 with 95% interval")
    ax.set_title("M2 (0 = no extra sensitivity)")
    ax.legend(loc="upper left", fontsize=8.5)
    suptitle(fig, f"2. Removal (M2), {name}: random against state-dependent removal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_planted(r: pd.DataFrame, name: str, out: str) -> None:
    """M3: planted pattern, AUROC with the rule on and switched off, per strength."""
    mort = r[(r.data_tag == "physionet2012") & (r.subset == "all")]
    control = plain(mort)[(plain(mort).perturb == "none") & (plain(mort).test_set == "test")]
    designs = [
        (1.0, [1.5, 2.0, 4.0], "First design (D21): q1 = 1"),
        (0.8, [1.05, 1.1, 1.2], "Redesign (D24): q1 = 0.8"),
    ]
    designs = [d for d in designs if (mort.planted_q1 == d[0]).any()]
    fig, axes = plt.subplots(1, len(designs), figsize=(6 * len(designs), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for (q1, strengths, title), ax in zip(designs, axes, strict=True):
        xs = [1.0] + strengths
        for v in ("all", "values"):
            on, off = [], []
            for s in xs:
                if q1 == 1.0 and s == 1.0:
                    c = control[control.inputs == v]["value"].mean()
                    on.append(c)
                    off.append(c)
                    continue
                pr = mort[
                    (mort.planted_q1 == q1)
                    & np.isclose(mort.planted_strength, s)
                    & (mort.inputs == v)
                ]
                on.append(pr[(pr.perturb == "none") & (pr.test_set == "test")]["value"].mean())
                off.append(pr[pr.test_set == "test_rule_off"]["value"].mean())
            xpos = range(len(xs))
            ax.plot(xpos, on, color=VARIANT_COLOR[v], lw=2, marker="o", ms=7, label=f"{v}, rule on")
            ax.plot(
                xpos,
                off,
                color=VARIANT_COLOR[v],
                lw=2,
                ls="--",
                marker="o",
                ms=7,
                markerfacecolor=SURFACE,
                label=f"{v}, rule switched off",
            )
            if v == "all":
                for y in (on[-1], off[-1]):
                    ax.text(len(xs) - 1 + 0.08, y, f"{y:.3f}", va="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(xs)), [f"{s:g}" for s in xs])
        ax.set_xlim(-0.3, len(xs) - 0.4)
        ax.set_xlabel("strength = q1 / q0 (1 = no difference between the groups)")
        ax.set_title(title)
    axes[0].set_ylabel("test AUROC, mortality (mean of 3 seeds)")
    axes[-1].legend(loc="upper left", fontsize=8.5)
    suptitle(fig, f"3. Planted heart-rate pattern (M3), {name}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_transfer(r: pd.DataFrame, m: pd.DataFrame, name: str, out: str) -> None:
    """M4: AUROC on hospital 1, thinned hospital 1 and hospital 2, and M4 itself."""
    p = plain(r)
    q = p[(p.data_tag == "physionet2019") & (p.subset == "pre_onset")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    xs = ["hospital 1\n(recorded)", "hospital 1, thinned\nto hospital-2 rates", "hospital 2"]
    lo, hi = 1.0, 0.0
    for v in ("all", "values"):
        qq = q[q.inputs == v]
        ys = [
            qq[(qq.test_set == "test_hospital1") & (qq.perturb == "none")]["value"].mean(),
            qq[qq.test_set == "test_hospital1_thinned"]["value"].mean(),
            qq[(qq.test_set == "test_hospital2") & (qq.perturb == "none")]["value"].mean(),
        ]
        lo, hi = min(lo, *ys), max(hi, *ys)
        ax.plot(range(3), ys, color=VARIANT_COLOR[v], lw=2, marker="o", ms=8, label=v)
        for x, y in enumerate(ys):
            ax.text(
                x + 0.07,
                y + (0.002 if v == "all" else -0.002),
                f"{y:.3f}",
                fontsize=9,
                color=INK,
                va="bottom" if v == "all" else "top",
            )
    ax.set_xticks(range(3), xs)
    ax.set_xlim(-0.2, 2.4)
    ax.set_ylim(lo - 0.012, hi + 0.008)
    ax.set_ylabel("test AUROC, pre-onset (mean of 3 seeds)")
    ax.set_title(f"Sepsis {name} trained on hospital 1")
    ax.legend(loc="upper right")
    ax = axes[1]
    rows = [
        ("M4_thinning", "physionet2019", "thinning, 34 variables"),
        ("M4_thinning", "physionet2019_comparable", "thinning, 25 variables"),
        ("M4_transfer", "physionet2019", "hospital 2, 34 variables"),
        ("M4_transfer", "physionet2019_comparable", "hospital 2, 25 variables"),
    ]
    for i, (meas, tag, _) in enumerate(rows):
        mm = m[(m.measure == meas) & (m.data_tag == tag) & (m.subset == "pre_onset")].iloc[0]
        ax.plot([mm.lo, mm.hi], [i, i], color=INK2, lw=2, solid_capstyle="round")
        ax.scatter([mm.value], [i], s=60, color=INK, zorder=3, edgecolor=SURFACE)
        ax.text(mm.hi + 0.003, i, f"{mm.value:+.3f}", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)), [x[2] for x in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel("extra AUROC lost by all over values, 95% interval")
    ax.set_title("M4 = loss of all − loss of values")
    ax.grid(axis="y", visible=False)
    suptitle(fig, f"4. Transfer (M4), {name}: losses where the measurements change")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gru", choices=sorted(NAMES))
    args = parser.parse_args()
    runs = pd.read_csv(RESULTS / "runs.csv")
    measures = pd.read_csv(RESULTS / "measures.csv")
    r = runs[(runs.model == args.model) & (runs.metric == "auroc") & ~runs.tuning.astype(bool)]
    m = measures[measures.model == args.model]
    if r.empty or m.empty:
        raise SystemExit(f"no rows for {args.model} in results/runs.csv or results/measures.csv")
    name = NAMES[args.model]
    stem = RESULTS / f"findings_{args.model}"
    fig_information(r, m, name, f"{stem}_1_information.png")
    fig_removal(r, m, name, f"{stem}_2_removal.png")
    fig_planted(r, name, f"{stem}_3_planted.png")
    fig_transfer(r, m, name, f"{stem}_4_transfer.png")
    print(f"4 figures: {stem}_*.png")


if __name__ == "__main__":
    main()
