# README.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

**scale-wise-optimizer** is a research project that clusters VQA (Visual Question Answering) questions by their textual characteristics and determines the optimal image resolution for each cluster. The goal is adaptive resolution selection: use high resolutions only when necessary, reducing computational cost while maintaining accuracy.

**Dataset**: 2,277 questions from SeedBench 2+ benchmark, evaluated across 10 resolutions (2,000 to 800,000 pixels).

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
The core pipeline. Loads data, generates sentence embeddings (`all-MiniLM-L6-v2`, 384 dims), runs KMeans with silhouette score optimization (K=5..30), performs Friedman statistical tests per cluster, and determines resolution recommendations at three thresholds (best, 95%, 99%).

**Outputs** (both consumed by the inference notebook):
- `clustering/training/artifacts/clustering_model.pkl` — pickled dict with keys `kmeans`, `embedding_model`, `optimal_k`
- `clustering/training/results/cluster_resolution_mapping.csv` — columns: `cluster, n_questions, best_resolution, best_accuracy, efficient_res_95pct, acc_at_95pct, efficient_res_99pct, acc_at_99pct`

### 3. Inference (`clustering/inference/clustering_inference.ipynb`)
Oracle/evaluation notebook. Loads the trained model + mapping CSV, classifies test questions into clusters, assigns resolutions, and produces accuracy vs. computational cost analysis.

**Output**: `clustering/inference/results/test_predictions.csv` — columns: `doc_id, cluster, resolution`

### Data flow
```
benchmarks/seedbench_2_plus/test_questions.jsonl  ──┐
benchmarks/seedbench_2_plus/...resolution_analysis.xlsx ──┤
                                                          ├──> Training ──> .pkl + .csv ──> Inference
                                                          └──> EDA (standalone)
```

## Key Shared Patterns

All notebooks use this pattern to find the project root:
```python
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
```

All notebooks share these constants (defined independently in each):
- `RESOLUTIONS = ['2000', '12500', '25000', '50000', '100000', '150000', '250000', '400000', '600000', '800000']`
- `QUESTION_TYPES` dict with 8 keyword-based categories
- `clean_input()` function that strips MCQ options and instruction text from questions

## Notebook Editing Notes

- When restructuring notebooks with many cells, write a Python script that manipulates the `.ipynb` JSON directly rather than using many individual cell edits. Use a `keep(cells, idx)` helper to reuse existing cells (strip outputs, preserve source).
- Always add `os.makedirs(os.path.dirname(path), exist_ok=True)` before writing artifacts.
- `.pkl` files are gitignored — the training notebook must be re-executed to generate artifacts before inference can run.
- Ordering convention: all tables and graphs must be ordered by cluster ID.
