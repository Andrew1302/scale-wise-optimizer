# README.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

**scale-wise-optimizer** is a research project that clusters VQA (Visual Question Answering) questions by their textual characteristics and determines the optimal image resolution for each cluster. The goal is adaptive resolution selection: use high resolutions only when necessary, reducing computational cost while maintaining accuracy.

**Dataset**: 2,277 questions from SeedBench 2+ benchmark (Chart / Map / Web), evaluated across 10 resolutions (2,000 to 800,000 pixels).

**Train/test split**: 80/20, stratified by benchmark subject, `random_state=42` — 1,821 train / 456 test. The training notebook owns the split and writes it to `data_split.csv`; the inference notebook reads that file and evaluates on the held-out rows only. Both the KMeans fit and the per-cluster resolution selection see training rows exclusively, so the inference numbers are genuinely out-of-sample.

## Environment & Package Management

- **Package manager**: `uv` — use `uv add <package>` to install (never `pip install`)
- **Python**: >=3.10, <3.13
- **Activate venv**: `source .venv/Scripts/activate` (Windows/Git Bash)
- **Run notebooks**: Use Jupyter or `jupyter nbconvert --to notebook --execute <notebook.ipynb>`
- **Windows encoding**: Always use `sys.stdout.reconfigure(encoding='utf-8')` in standalone Python scripts (Windows cp1252 causes UnicodeEncodeError)

## Architecture

The project is organized as a three-stage pipeline. Each stage is a Jupyter notebook:

### 1. EDA (`clustering/eda/clustering_eda.ipynb`)
Exploratory analysis only. Loads data, extracts 18 text features, produces distribution plots. No model training.

### 2. Training (`clustering/training/clustering_production.ipynb`)
The core pipeline. Loads data, holds out a stratified test set, then — **on the training half only** — generates sentence embeddings (`all-MiniLM-L6-v2`, 384 dims), runs KMeans with silhouette score optimization (K=5..30), performs Friedman statistical tests per cluster, and determines resolution recommendations at three thresholds (best, 95%, 99%). Clustering uses the embeddings alone; no hand-crafted text features are involved.

The accuracy-vs-cost section at the end of this notebook is in-sample by construction and is labelled as such — treat the inference notebook as the source of truth for reported numbers.

**Outputs** (all three consumed by the inference notebook):
- `clustering/training/artifacts/clustering_model.pkl` — pickled dict with keys `kmeans`, `embedding_model`, `optimal_k`
- `clustering/training/results/cluster_resolution_mapping.csv` — columns: `cluster, n_questions, best_resolution, best_accuracy, efficient_res_95pct, acc_at_95pct, efficient_res_99pct, acc_at_99pct` (`n_questions` counts training rows only)
- `clustering/training/results/data_split.csv` — columns: `doc_id, split` where `split ∈ {train, test}`

**Section 7 — K-selection diagnostics (reporting only, changes nothing).** Documents how
well K=30 holds up. Results are cached in three committed CSVs so the section renders
instantly; set `RECOMPUTE = True` in the diagnostics config cell to regenerate (~18 min
on CPU). The K grids are written out explicitly so a recompute reproduces the committed
numbers. Diagnostics sort training rows by `doc_id` first, so they do not depend on the
split's shuffle order.
- `k_diag_internal.csv` — silhouette / Calinski-Harabasz / Davies-Bouldin / inertia / cluster sizes for K=2..200
- `k_diag_cv.csv` — out-of-fold accuracy, cost and **lift** per K (5-fold CV inside the training split)
- `k_diag_stability.csv` — the same lift re-measured over 8 independent CV partitions

"Lift" is strategy accuracy minus what a *fixed* resolution delivers at the same cost.
It is the bar that matters: beating "always 800000" while spending less is trivially
achievable by just picking a cheaper fixed resolution.

Headline findings: silhouette keeps rising past `MAX_K` (peak ≈ K=165), the three internal
criteria disagree completely, silhouette never exceeds 0.075 at any K (≈ no cluster
structure), and out-of-fold lift is positive at only 2 of 19 K values — at K=30 it is
about −0.015, i.e. *below* the fixed-resolution curve. **K=30 is a convention retained for
continuity, not an optimum.** See the notebook's findings cell for the full write-up.

### 3. Inference (`clustering/inference/clustering_inference.ipynb`)
Evaluation notebook. Loads the trained model + mapping CSV + split CSV, filters the benchmark down to the held-out rows, classifies them into clusters via `kmeans.predict` (no re-fitting), assigns resolutions, and produces out-of-sample accuracy vs. computational cost analysis.

**Output**: `clustering/inference/results/test_predictions.csv` — columns: `doc_id, cluster, resolution`

Note: a small cluster can end up with zero held-out questions, so the per-cluster test tables may cover fewer than `optimal_k` clusters.

### Data flow
```
benchmarks/seedbench_2_plus/test_questions.jsonl  ──┐
benchmarks/seedbench_2_plus/...resolution_analysis.xlsx ──┤
                                                          ├──> Training (train split) ──> .pkl + mapping.csv + data_split.csv
                                                          │                                            │
                                                          │                                            ▼
                                                          │                              Inference (test split)
                                                          └──> EDA (standalone, full dataset)
```

## Key Shared Patterns

All notebooks use this pattern to find the project root:
```python
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
```

All notebooks share these constants (defined independently in each):
- `RESOLUTIONS = ['2000', '12500', '25000', '50000', '100000', '150000', '250000', '400000', '600000', '800000']` — must stay ascending; the 95%/99% recommendation loops take the first resolution over threshold
- `clean_input()` function that strips MCQ options and instruction text from questions

EDA-only (the training pipeline does not use them):
- `QUESTION_TYPES` dict with 8 keyword-based categories
- `extract_features()` and its 18 text features

## Notebook Editing Notes

- When restructuring notebooks with many cells, write a Python script that manipulates the `.ipynb` JSON directly rather than using many individual cell edits. Use a `keep(cells, idx)` helper to reuse existing cells (strip outputs, preserve source).
- Always add `os.makedirs(os.path.dirname(path), exist_ok=True)` before writing artifacts.
- `.pkl` files are gitignored — the training notebook must be re-executed to generate artifacts before inference can run. Re-running training also rewrites `data_split.csv`; the split is deterministic under `RANDOM_STATE`, but changing `TEST_SIZE`, `STRATIFY_BY` or the seed invalidates any previously reported test numbers.
- Always re-run inference after re-running training — the mapping, the model and the split must come from the same run.
- Notebooks written by script must be normalized before use (`nbformat.validator.normalize`) so every cell carries an `id`; set `nbformat_minor` to 5.
- Ordering convention: all tables and graphs must be ordered by cluster ID.
- Executing `clustering_production.ipynb` takes several minutes. Run it in the foreground with a generous timeout — do **not** wrap `nbconvert` in a shell `timeout`, and note that piping to `tail` masks the real exit code (`$?` becomes `tail`'s). Long-running background jobs get killed in this environment.
- Long sweeps should append results to CSV incrementally and skip K values already on disk, so an interrupted run loses nothing and can be resumed.
