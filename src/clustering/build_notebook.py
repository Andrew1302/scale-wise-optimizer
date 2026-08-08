"""Generate ``clustering_kfold.ipynb``.

The notebook is written from here rather than by hand so it stays reviewable in
diffs and reproducible: run this module to regenerate it.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).with_name("clustering_kfold.ipynb")

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Per-cluster resolution policies — k-fold evaluation

Does routing each question to its own image resolution beat simply running every
question at one well-chosen fixed resolution?

**The bar is `lift`**: accuracy minus that of a *fixed* budget costing the same.
Beating "always full resolution" while spending less is not evidence — a cheaper
fixed budget usually does that too. Every number below is measured on documents
the clustering never saw: KMeans and the per-cluster budgets are fit on the
training folds and applied to the held-out fold.

Two benchmarks are evaluated side by side, using the sweeps under `benchmarks/`.""",
    ),
    (
        "code",
        """from pathlib import Path

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "pyproject.toml").exists():
    if PROJECT_ROOT == PROJECT_ROOT.parent:
        raise RuntimeError("could not locate the project root")
    PROJECT_ROOT = PROJECT_ROOT.parent

# ── Sweeps to evaluate: label -> results directory ────────────────────────────
RUNS = {
    "seedbench_2_plus": PROJECT_ROOT / "benchmarks/seedbench_2_plus/Qwen-Qwen3.5-4B",
    "mmmu_pro_standard": PROJECT_ROOT / "benchmarks/mmmu_pro_standard/Qwen-Qwen3.5-4B-tok32k",
}

# ── Cross-validation ──────────────────────────────────────────────────────────
K_VALUES = [2, 3, 5, 8, 12, 20, 30, 50]
N_FOLDS = 5
RANDOM_STATE = 42

# K used for the single-K strategy comparison chart (section 5); must be in K_VALUES.
SUMMARY_K = 30

# ── Cost basis. input_tokens is what the model actually pays for; sent_px is
#    what the pipeline controls. Both are recorded per document per budget. ─────
COST = "input_tokens"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RESULTS_DIR = PROJECT_ROOT / "src/clustering/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)""",
    ),
    (
        "code",
        """import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer

from clustering.data import load_sweep, question_text
from clustering.kfold import cross_validate, fixed_frontier, lift

sns.set_style("whitegrid")
%matplotlib inline

encoder = SentenceTransformer(EMBEDDING_MODEL)

sweeps, embeddings = {}, {}
for label, run_dir in RUNS.items():
    sweep = load_sweep(run_dir, cost=COST)
    sweeps[label] = sweep
    embeddings[label] = encoder.encode(
        question_text(sweep.task, sweep.doc_ids), show_progress_bar=False, convert_to_numpy=True
    )
    print(f"{label}: {sweep.n_documents} documents x {len(sweep.budgets)} budgets")""",
    ),
    (
        "markdown",
        """---
## 1. The fixed-budget frontier

This is what routing has to beat. Each point is one budget applied to every
document, averaged over the held-out folds.""",
    ),
    (
        "code",
        """folds = {label: cross_validate(sweeps[label], embeddings[label], K_VALUES, N_FOLDS, RANDOM_STATE)
         for label in RUNS}

for label in RUNS:
    curve = fixed_frontier(folds[label])
    curve.insert(0, "benchmark", label)
    print(curve.to_string(index=False))
    print()""",
    ),
    (
        "markdown",
        """---
## 2. Lift — the result that matters

`lift > 0` means routing beat a fixed budget of the same cost. `NaN` marks
policies cheaper than the cheapest fixed budget, where no comparison exists
rather than an extrapolated one.""",
    ),
    (
        "code",
        """lifts = {}
for label in RUNS:
    table = lift(folds[label])
    table.insert(0, "benchmark", label)
    lifts[label] = table
    best = table.dropna(subset=["lift"]).sort_values("lift", ascending=False)
    print(f"── {label} ── best lift by strategy")
    print(best.groupby("strategy").head(1)[
        ["strategy", "k", "accuracy", "fixed_at_same_cost", "lift", "cost_share"]
    ].round(4).to_string(index=False))
    print(f"   positive lift in {(table['lift'] > 0).sum()} of {table['lift'].notna().sum()} comparable settings")
    print()

all_lift = pd.concat(lifts.values(), ignore_index=True)
all_lift.to_csv(RESULTS_DIR / "kfold_lift.csv", index=False)
pd.concat(folds.values(), keys=folds, names=["benchmark", None]).reset_index(level=0).to_csv(
    RESULTS_DIR / "kfold_folds.csv", index=False
)""",
    ),
    (
        "markdown",
        """---
## 3. Lift against K

If routing worked, lift would be reliably positive across K rather than
flickering around zero.""",
    ),
    (
        "code",
        """# Shared styling for every strategy chart: colours, markers, display names and
# legend order (Fixed budget, BPC, E99, E95) stay identical across sections 3-5.
STYLE = {
    "best": ("*", "#2ecc71", 280),
    "efficient_99": ("s", "#e74c3c", 180),
    "efficient_95": ("D", "#e67e22", 180),
}
NAMES = {"best": "BPC", "efficient_99": "E99", "efficient_95": "E95"}
SHORT_RES = {"2000": "2k", "12500": "12.5k", "25000": "25k", "50000": "50k",
             "100000": "100k", "150000": "150k", "250000": "250k",
             "400000": "400k", "600000": "600k", "800000": "800k"}
COST_LABEL = COST.replace("_", " ")

fig, axes = plt.subplots(1, len(RUNS), figsize=(6 * len(RUNS), 4), squeeze=False)
for ax, label in zip(axes[0], RUNS, strict=True):
    table = lifts[label]
    for name, (marker, colour, _) in STYLE.items():
        group = table[table["strategy"] == name]
        ax.plot(group["k"], group["lift"], marker=marker, ms=7, color=colour, label=NAMES[name])
    ax.axhline(0, color="black", lw=1)
    ax.set_title(label)
    ax.set_xlabel("clusters (K)")
    ax.set_ylabel(f"Lift vs fixed budget at equal {COST_LABEL}")
    ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "kfold_lift_vs_k.pdf", bbox_inches="tight")
plt.show()""",
    ),
    (
        "markdown",
        """---
## 4. Accuracy vs cost

The routed strategies are plotted against the fixed frontier. A strategy is only
interesting if it sits **above** the line, not merely to the left of the most
expensive point.""",
    ),
    (
        "code",
        """# Axis limits per (benchmark, cost basis), shared with the section 5 charts so
# figures of the same benchmark and axis stay directly comparable.
SHARED_LIMITS = {}

for label in RUNS:
    curve = fixed_frontier(folds[label])
    full_cost = curve["cost"].max()
    table = lifts[label]
    for basis in ("tokens", "relative"):
        if basis == "tokens":
            curve_x = curve["cost"]
            xlabel = f"Mean {COST_LABEL} per document"
        else:
            curve_x = curve["cost"] / full_cost * 100
            xlabel = f"Relative computational cost (% of full-resolution {COST_LABEL})"
        routed_x = table["cost"] if basis == "tokens" else table["cost_share"] * 100
        xs = pd.concat([curve_x, routed_x])
        ys = pd.concat([curve["accuracy"], table["accuracy"]])
        x_pad, y_pad = 0.05 * (xs.max() - xs.min()), 0.14 * (ys.max() - ys.min())
        xlim = (xs.min() - x_pad, xs.max() + x_pad)
        ylim = (ys.min() - y_pad, ys.max() + y_pad)
        SHARED_LIMITS[(label, basis)] = (xlim, ylim)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.plot(curve_x, curve["accuracy"], "o-", color="#3498db", label="Fixed budget", zorder=3)
        for _, row in curve.iterrows():
            x = row["cost"] if basis == "tokens" else row["cost"] / full_cost * 100
            ax.annotate(SHORT_RES[str(int(row["budget"]))], (x, row["accuracy"]),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=9, color="#2c3e50")
        for name, (marker, colour, _) in STYLE.items():
            group = table[table["strategy"] == name]
            group_x = group["cost"] if basis == "tokens" else group["cost_share"] * 100
            ax.scatter(group_x, group["accuracy"], marker=marker, c=colour, s=70,
                       alpha=0.8, edgecolors="black", linewidth=0.5, label=NAMES[name], zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Out-of-fold accuracy")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        suffix = "_tokens" if basis == "tokens" else ""
        fig.savefig(RESULTS_DIR / f"kfold_accuracy_vs_cost_all_k_{label}{suffix}.pdf", bbox_inches="tight")
        plt.show()""",
    ),
    (
        "markdown",
        """---
## 5. Strategy comparison at a single K

The chart the original notebook produced, rebuilt on k-fold: accuracy against
relative cost, with error bars showing the spread across folds on **both** axes.
The routed strategies vary in cost between folds (different clusters get
different budgets), which the original's single split could not show.""",
    ),
    (
        "code",
        """from clustering.kfold import strategy_summary

summaries = {label: strategy_summary(folds[label], SUMMARY_K) for label in RUNS}
for label, table in summaries.items():
    table.insert(0, "benchmark", label)
    print(f"── {label} ── {N_FOLDS}-fold, K={SUMMARY_K} (mean ± std across folds)")
    print(table[["strategy", "accuracy", "accuracy_sd", "cost_pct", "cost_pct_sd"]].round(4).to_string(index=False))
    print()

pd.concat(summaries.values(), ignore_index=True).to_csv(RESULTS_DIR / "kfold_strategy_comparison.csv", index=False)""",
    ),
    (
        "code",
        """# One frame per distinct point (coincident strategies share a frame), placed
# just below its point; the horizontal split keeps near-coincident points apart
FRAME_OFFSETS = [(-70, -46), (70, -46), (0, -100)]

for label in RUNS:
    table = summaries[label]
    for basis in ("tokens", "relative"):
        xcol, xsd = ("cost", "cost_sd") if basis == "tokens" else ("cost_pct", "cost_pct_sd")
        fig, ax = plt.subplots(figsize=(12, 7))

        # Same axis limits as the section 4 chart of this benchmark and basis
        xlim, ylim = SHARED_LIMITS[(label, basis)]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        y0, y1 = ylim

        fixed = table[table["is_fixed"]]
        ax.errorbar(fixed[xcol], fixed["accuracy"], yerr=fixed["accuracy_sd"],
                    fmt="o-", color="#3498db", lw=2, ms=8, capsize=3,
                    label="Fixed budget", zorder=3)

        # Labels sit just past the error bar so label and bar never collide
        ax_h_pts = fig.get_size_inches()[1] * 72 * 0.78
        for _, row in fixed.iterrows():
            err_pts = row["accuracy_sd"] / (y1 - y0) * ax_h_pts
            ax.annotate(SHORT_RES[row["strategy"].replace("fixed_", "")],
                        (row[xcol], row["accuracy"]),
                        textcoords="offset points", xytext=(0, 6 + err_pts), ha="center",
                        fontsize=9, color="#2c3e50")

        groups = {}
        for name, (marker, colour, size) in STYLE.items():
            row = table[table["strategy"] == name].iloc[0]
            ax.errorbar(row[xcol], row["accuracy"],
                        xerr=row[xsd], yerr=row["accuracy_sd"],
                        fmt="none", ecolor=colour, capsize=3, zorder=4)
            ax.scatter(row[xcol], row["accuracy"], marker=marker, c=colour, s=size,
                       zorder=5, edgecolors="black", linewidth=0.8, label=NAMES[name])
            key = (round(row[xcol], 3), round(row["accuracy"], 3))
            groups.setdefault(key, ([], row["accuracy_sd"]))[0].append(name)

        for offset, ((cost, acc), (names, acc_sd)) in zip(FRAME_OFFSETS, sorted(groups.items())):
            colour = STYLE[names[0]][1]
            cost_text = f"{cost:.1f}% cost" if basis == "relative" else f"{cost:.0f} {COST_LABEL}"
            ax.annotate(
                " = ".join(NAMES[n] for n in names) + f"\\n({cost_text}, {acc:.4f} ± {acc_sd:.4f})",
                xy=(cost, acc), xycoords="data",
                xytext=offset, textcoords="offset points",
                ha="center", va="top", fontsize=9, fontweight="bold", color=colour,
                arrowprops=dict(arrowstyle="->", color=colour, lw=1.3, shrinkB=10),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=colour, alpha=0.95))

        xlabel = (f"Relative computational cost (% of full-resolution {COST_LABEL})"
                  if basis == "relative" else f"Mean {COST_LABEL} per document")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Out-of-fold accuracy", fontsize=11)
        ax.legend(loc="lower right", fontsize=11, markerscale=0.6, labelspacing=0.7, borderpad=0.8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        suffix = "_tokens" if basis == "tokens" else ""
        fig.savefig(RESULTS_DIR / f"kfold_accuracy_vs_cost_{label}{suffix}.pdf", bbox_inches="tight")
        plt.show()""",
    ),
    (
        "markdown",
        """---
## 6. Findings

Read the lift column, not the accuracy column. Fill this in from the run:

- is lift positive, and at how many of the K values tested?
- is it larger than the fold-to-fold standard deviation (`accuracy_sd`)?
- does either benchmark behave differently from the other?

A routing policy that only matches the fixed frontier is telling you the useful
signal is *how much resolution the benchmark needs on average*, not *which
question needs what* — in which case the simplest correct action is to pick one
cheaper fixed budget.""",
    ),
]


def build() -> Path:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
        for kind, source in CELLS
    ]
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.validator.normalize(notebook)
    notebook.nbformat_minor = 5
    nbformat.write(notebook, NOTEBOOK)
    return NOTEBOOK


if __name__ == "__main__":
    print(f"wrote {build()}")
