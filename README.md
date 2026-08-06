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
- **lmms-eval** is pinned to a git commit, not PyPI: the published `0.7.2` wheel omits the
  `_default_template_yaml` task files (no extension, and that release predates the
  `tasks/**/*` package-data glob), so every task fails to load. See `[tool.uv.sources]`.
- **Windows encoding**: Always use `sys.stdout.reconfigure(encoding='utf-8')` in standalone Python scripts (Windows cp1252 causes UnicodeEncodeError)

## Benchmarking (`src/swo/`) — generating the resolution sweep

The clustering notebooks below consume a table of "was this question answered correctly at
budget X". `src/swo/` is the code that produces that table. It is a Python package, tested
and linted; the notebooks are not.

**`swo.model.BudgetedModel`** is a custom lmms-eval model that downscales every image to a
pixel budget and then delegates to a real VLM. It wraps *any* chat-capable lmms-eval model
rather than subclassing one, so the backend is swappable. Only pixels change — the prompt
reaching the backend is byte-identical to an unbudgeted run.

It registers through lmms-eval's entry-point plugin group (`[project.entry-points."lmms_eval.models"]`
in `pyproject.toml`), so nothing in the lmms-eval installation is patched:

```bash
lmms-eval --model resolution_budget --tasks seedbench_2_plus \
  --model_args 'backend=vllm,resolution_budget=100000,backend_args={"model":"Qwen/Qwen3-VL-8B-Instruct"}'
```

**`python -m swo.sweep`** runs the whole sweep and is the normal entry point:

```bash
python -m swo.sweep --task seedbench_2_plus \
  --budgets 2000,12500,25000,50000,100000,150000,250000,400000,600000 \
  --chunk-size 500 --backend qwen3_5 \
  --backend-args '{"pretrained":"Qwen/Qwen3.5-4B","min_pixels":256,"max_pixels":16777216,
                   "enable_thinking":false,
                   "system_prompt":"You are answering a multiple-choice question. Reply with exactly one character: A, B, C, or D. Do not explain."}'
```

Every one of those backend args is load-bearing — see "Answer-format compliance"
and "Pixel budgets vs. the model's own floor" below. Note the budget ladder stops
at 600,000: SeedBench-2-Plus images are uniformly 800×800 = 640,000 px, so any
budget at or above that is a no-op identical to the native image.

It writes `benchmarks/<task>/<model>/samples.csv` (one row per document per budget, carrying
`original_px`, `sent_px`, the response and the task's per-sample metrics) and `summary.csv`
(one row per lmms-eval invocation, with that invocation's own aggregates — a budget-level
metric is the `n_docs`-weighted mean of its rows). `samples.csv` is what the clustering stage
should consume — it supersedes `seedbench_2_plus_resolution_analysis.xlsx`, which records no
model identity and whose `Aggregated_Scores` sheet stops at 250000.

The backend is loaded once and reused across budgets, and work already in `samples.csv` is
skipped, so an interrupted sweep resumes. Pass `--chunk-size N` on long runs to make that
resolution finer than a whole budget.

### Running on the lab GPU VMs

`python -m swo.vm` deploys this package to a lab machine, runs the sweep there in
`tmux`, waits, and fetches results. One command, no shell scripts, no job configs,
and **no WSL** — it uses `ssh` + `tar`, both of which Git Bash has.

```bash
python -m swo.vm check                      # GPUs, sessions, disk — always first
python -m swo.vm --vm vm03 run -- \
  --task seedbench_2_plus --budgets 2000,12500,100000,800000 --chunk-size 500 \
  --backend vllm --backend-args '{"model":"Qwen/Qwen3-VL-8B-Instruct"}'
python -m swo.vm status | logs | fetch | stop
```

Targets are `vm03` (default), `vm02` and `c2d`; `--vm` goes before the subcommand.
Everything after `--` is passed verbatim to `swo.sweep`, so this module never has
to track sweep's flags. Remote paths mirror the lab's other tooling
(`/media/<user>/ssd1T/andrew/...`) so the HuggingFace and uv caches are shared;
`/home` is small and is never written to.

**Recovering from a failure is re-running the same command.** With `--chunk-size`,
each budget is evaluated in document slices and every finished slice is on disk
before the next starts; the rows already in `samples.csv` are the resume cursor.
A crashed slice wrote nothing, so there is no partial state to clean up — no
sentinels, no merge step. `fetch` is safe mid-run for the same reason.

These are shared machines and **c2d is borrowed from another lab** — `run` refuses
to start there if its pinned GPU is busy. See `.claude/skills/vm-runs/SKILL.md`
for the etiquette and the failure catalog.

### Two backends: in-process HF, or a vLLM server

| backend | `system_prompt` | `input_tokens` | throughput |
|---|:-:|:-:|---|
| `qwen3_5` (in-process HF) | yes | no — output only | one document at a time |
| `vllm` (in-process) | **no** | **no counts at all** | batched |
| `openai` | **no** | yes | 32 concurrent |
| **`async_openai`** (vLLM server) | yes | **yes** (`usage.prompt_tokens`) | adaptive, 1→128 |

`async_openai` is the only backend with both, which is why it is the one to use
against a served model. `openai` looks equivalent but has no `system_prompt`, and
losing that costs ~50 points of answer-format compliance.

```bash
# start once; the server outlives any number of sweeps
python -m swo.vm --vm c2d serve --model Qwen/Qwen3.5-4B --gpu-memory-utilization 0.9

python -m swo.vm --vm c2d run -- --task seedbench_2_plus \
  --budgets 2000,12500,25000,50000,100000,150000,250000,400000,600000 \
  --chunk-size 500 --backend async_openai \
  --backend-args '{"model_version":"Qwen/Qwen3.5-4B","base_url":"http://127.0.0.1:8000/v1",
                   "api_key":"EMPTY","system_prompt":"You are answering a multiple-choice question. Reply with exactly one character: A, B, C, or D. Do not explain."}'

python -m swo.vm --vm c2d serve-stop
```

Measured on c2d (one RTX 5090, Qwen3.5-4B, 200 documents at native resolution):
**0.06 s/doc vs 0.28–0.33 s/doc** in-process — about **5× faster**, with 100%
answer compliance and real `input_tokens`. The gain needs a few hundred documents
to show; at 10 documents the adaptive concurrency never ramps and the two paths
look equivalent.

`serve` installs the `vllm` extra, waits for `/health`, and prints the exact
backend args. Three things move when you switch:

- **Thinking is disabled server-side.** `async_openai` has no per-request
  `chat_template_kwargs`, so the switch lives on the server:
  `--default-chat-template-kwargs '{"enable_thinking": false}'`, which `serve`
  passes by default. Without it a reasoning model spends its whole token budget
  thinking and **every answer scores wrong** — measured 0% compliance.

- **`min_pixels` moves server-side.** The image processor now lives in vLLM, so
  the floor is `--min-pixels` on `serve`, not a backend arg. Leaving it at the
  model default reinstates Qwen3.5's 65,536-px floor and silently upscales every
  low budget. The default here is 256; `--max-pixels` defaults above the largest
  budget.
- **vLLM pre-allocates GPU memory** for the server's whole life
  (`--gpu-memory-utilization`, default 0.9). On a shared box, set it to what you
  actually need. `--tensor-parallel-size` and `--gpus` shard across cards.

`async_openai` does *not* resize images: its `max_pixels`/`min_pixels` feed only
`video_kwargs`, so the budget still binds exactly as with the HF backend.

### Reading the results

```bash
python -m swo.report benchmarks/seedbench_2_plus/<model> [--markdown]
```

One row per budget: `n`, `accuracy`, `mean_sent_px`, the token counts lmms-eval
reported, and `elapsed_s` / `s_per_doc`. **The token columns are the audit**: if
they don't move with the budget, the backend re-resized the images and the sweep
measured nothing.

Which token counts exist depends on the backend — local HF backends report
`output_tokens` only, while API-style ones (`openai`, `async_openai`, `gemini`,
`async_hf_model`) also report `input_tokens`, which is where vision tokens live.

### Pixel budgets vs. the model's own floor

A budget only binds if the backend's `min_pixels` is below it. Worked example for
Qwen3.5 (`patch_size=16`, `spatial_merge_size=2`):

- one visual token covers 32×32 = **1,024 px** — the hard floor
- its `preprocessor_config.json` declares `shortest_edge: 65536`, i.e. a *default*
  `min_pixels` of 65,536 px — 64 tokens
- so without `min_pixels` overridden, every budget below 65,536 is **upscaled**,
  and the four lowest steps of the 2,000–800,000 sweep measure nothing

Pass `min_pixels` low (and `max_pixels` high, or budgets above its default get
clamped too). Note that going far under a model's default also introduces
distribution shift, not just information loss — worth separating when reading
results.

### Answer-format compliance is a confound, not a nuisance

`seedbench_2_plus` scores `pred[0]` — the **first character** of the response
(`utils.seed_process_result`) — and caps generation at `max_new_tokens: 16`. A
model that leads with prose is scored wrong no matter how good the answer is, and
raising `max_new_tokens` alone does not help.

Measured on Qwen3.5-4B, 10 docs × 10 budgets (`enable_thinking=false`):

- responses split almost perfectly: 50 answered with a letter and stopped, 49
  started explaining and hit the 16-token cap; only 1 case was mixed
- accuracy over **compliant** answers was **0.80**, versus **0.40** overall
- compliance **fell with resolution**: 80% at 2,000 px → 40% at ≥100,000 px

That last line is the dangerous one: more visual detail makes the model
elaborate, which then gets truncated and scored wrong. The formatting effect runs
*opposite* to the resolution effect, so leaving it uncorrected systematically
understates the value of high resolution — exactly the quantity this project is
trying to measure. Always report compliance rate alongside accuracy, and treat any
resolution curve without it as unreliable.

**The fix is a directive `system_prompt`.** The task's own `post_prompt` is not
enough for Qwen; the instruction has to arrive at the system level:

```json
"system_prompt": "You are answering a multiple-choice question. Reply with exactly one character: A, B, C, or D. Do not explain."
```

Measured on Qwen3.5-4B at budgets 2,000 and 800,000 (10 docs each):

| config | compliance | output tokens | s/doc |
|---|---:|---:|---:|
| `enable_thinking=true`, 16 tokens (lmms-eval default) | 0% | 16.0 | 0.40 |
| `enable_thinking=false`, 16 tokens | 51% | 6–10 | 0.33 |
| `enable_thinking=true`, 512 tokens | 50% | 408 | 5.90 |
| `enable_thinking=false` **+ system_prompt** | **100%** | **2.0** | **0.28** |

It is also the cheapest option — one-letter answers cut generation to 2 tokens.

Two things this table settles. **Thinking mode is a trap for this task**: at 512
tokens half the responses came back *empty*, because `_strip_thinking` returns
whatever follows `</think>` and the block never closed — 18× the latency for no
gain. And **`enable_thinking` defaults to `True`** in
`lmms_eval/models/simple/qwen3_5.py`, so the out-of-the-box configuration scores
0.00 at every budget.

This is calibrated on the Qwen3/Qwen3.5 family, which is where it was measured,
but the failure mode is generic — any chatty instruction-tuned VLM on a task
scored by `pred[0]` can do this. **Check the compliance rate on 10 samples before
starting any sweep with a new model**, and adjust the system prompt if it is not
~100%. Note that `system_prompt` is a per-backend argument: the qwen/HF backends
accept it, others may not.

### Three things that silently corrupt a sweep

1. **Backend `min_pixels` undoes the downscaling.** Qwen processors re-resize server-side and
   default to `min_pixels=200704`, which *upscales* a 2,000-px image back to ~200,000 px. Pass
   a pixel floor below your smallest budget in `--backend-args`, and check the
   `resolution_budget=... mean X -> Y px/image` line each run logs.
2. **Never enable lmms-eval's response cache.** It keys on prompt text only
   (`lmms_eval/caching/response_cache.py`), so it would replay one budget's answers for every
   other budget. `run_sweep` passes `use_cache=None` explicitly.
3. **Don't hand-edit `samples.csv`.** Appends are column-aligned against the existing header
   and a new column raises rather than shifting every row.

`--dry-run` swaps in a backend that loads no weights but still builds every prompt: it
validates budgets and collects native image sizes without a GPU.

**Development**: `pytest` and `ruff check src tests` / `ruff format src tests`.

## Architecture

The clustering half of the project is organized as a three-stage pipeline. Each stage is a
Jupyter notebook:

### 1. EDA (`clustering/eda/clustering_eda.ipynb`)
Exploratory analysis only. Loads data, extracts 18 text features, produces distribution plots. No model training.

### 2. Training (`clustering/training/clustering_training.ipynb`)
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
python -m swo.sweep ──> benchmarks/<task>/<model>/samples.csv ──┐
                                                               ├──> Training (train split) ──> .pkl + mapping.csv + data_split.csv
benchmarks/seedbench_2_plus/test_questions.jsonl  ─────────────┤                                            │
benchmarks/seedbench_2_plus/...resolution_analysis.xlsx (legacy)┤                                           ▼
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
- Executing `clustering_training.ipynb` takes several minutes. Run it in the foreground with a generous timeout — do **not** wrap `nbconvert` in a shell `timeout`, and note that piping to `tail` masks the real exit code (`$?` becomes `tail`'s). Long-running background jobs get killed in this environment.
- Long sweeps should append results to CSV incrementally and skip K values already on disk, so an interrupted run loses nothing and can be resumed.
