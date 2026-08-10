# Do You Need to See It All?

### On the Impact of Image Resolution in Multimodal LLMs

Andrew Carvalho de Sá · Erick Diogo de Almeida Sousa — PCS5022, Neural Networks and Deep Learning

Multimodal LLMs encode images at full resolution because nobody knows in advance how much
visual detail a question needs, so every request pays for detail it may never use. We measure
that waste across ten pixel budgets (2,000 → 800,000) on SEED-Bench-2-Plus and MMMU-Pro,
with Qwen3.5 as the model.

## The gap

An *ideal* policy — one that grants each question the cheapest budget that answers it
correctly — is unattainable, since it reads the ground truth. But it bounds what any
per-question selector could win:

| benchmark | largest budget | best fixed budget | ideal policy |
|---|---|---|---|
| SEED-Bench-2-Plus | 71.2% @ 100% cost | 71.2% @ 100% | **75.8% @ 4.5%** |
| MMMU-Pro | 43.8% @ 100% cost | 44.6% @ 39.4% | **54.9% @ 4.8%** |

Higher accuracy than any fixed budget, at under 5% of the pixels. The largest budget rarely
earns its cost — on MMMU-Pro a budget at 39% of the pixels scores *better* than full
resolution.

## The pipeline

```
question ──> selector ──┐
                        r (pixel budget)
image ──────────────> downscale to r ──> MLLM ──> answer
```

The selector fixes the budget **before** the encoder runs, and the image is downscaled
before any tokenization happens. Neither the weights nor the training of the MLLM are
touched, so the selector and the model are both interchangeable.

## The selector we tried

A clusterizer over question embeddings: embed the question text, k-means into K = 30
clusters, and give each cluster the cheapest budget within a tolerance τ of its best
accuracy (τ = 1, 0.99, 0.95). The hypothesis was that the wording of a question indicates
how much visual detail its answer needs.

It does not close the gap. Under five-fold cross-validation every operating point sits
below the fixed-budget curve, dominated by a cheaper fixed budget on both axes. The
per-cluster preferences are real on the training folds — 28 of the 30 MMMU-Pro clusters
peak below the largest budget — but they do not transfer to unseen questions.

Whether the prompt alone carries enough signal is still open. The margin the clusterizer
leaves is what the next selector has to close.

## Layout

| path | what it is |
|---|---|
| `src/swo/` | the measurement pipeline — downscale to a budget, run the benchmark, one row per question per budget |
| `src/clustering/` | the selector and its cross-validated evaluation |
| `clustering/` | earlier notebook pipeline (EDA → training → inference) |
| `benchmarks/` | sweep results, one directory per benchmark and model |

## Running it

Needs [uv](https://docs.astral.sh/uv/), and a GPU for the sweeps.

```bash
uv sync

python -m swo.sweep --task seedbench_2_plus --budgets 2000,100000,800000 \
  --backend vllm --backend-args '{"model": "Qwen/Qwen3.5-4B"}'

python -m swo.report benchmarks/seedbench_2_plus/<model>
```

**One caveat about `benchmarks/`:** the per-question sweep tables (`samples.csv`,
`summary.csv`) are not in the repository — `.gitignore` excludes all CSVs, and the largest
is 128 MB, past what GitHub accepts. The notebooks under `clustering/` read the committed
spreadsheets and are unaffected; anything reading `samples.csv` fails with a
`FileNotFoundError` until you copy the tables over or re-run the sweep.
