"""Load a sweep into the matrices the cross-validation works on."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from swo.report import correctness, metric_names, overall_metric
from swo.sweep import SAMPLES_FILE

#: Cost columns a sweep records, cheapest-to-interpret first. ``sent_px`` is what
#: the pipeline controls; ``input_tokens`` is what the model actually pays for.
COST_COLUMNS = ("input_tokens", "sent_px")


@dataclasses.dataclass(frozen=True)
class Sweep:
    """One benchmark evaluated at every budget, aligned into matrices.

    ``scores`` and ``costs`` are ``(n_documents, n_budgets)`` and share row and
    column order with ``doc_ids`` and ``budgets``, so a policy expressed as a
    per-document budget *index* indexes both.
    """

    task: str
    doc_ids: np.ndarray
    budgets: np.ndarray
    scores: np.ndarray
    costs: np.ndarray
    subjects: np.ndarray
    cost_name: str
    native_px: np.ndarray | None = None

    def __post_init__(self) -> None:
        expected = (len(self.doc_ids), len(self.budgets))
        if self.scores.shape != expected or self.costs.shape != expected:
            raise ValueError(f"scores/costs must be {expected}, got {self.scores.shape} and {self.costs.shape}")

    @property
    def n_documents(self) -> int:
        return len(self.doc_ids)

    def effective_budget(self, budget: int) -> int:
        """What ``budget`` actually caps at, given the images this sweep ran on.

        A budget above every image's native size is a no-op — the pipeline never
        upscales — so naming such a rung after its nominal budget overstates it.
        seedbench is uniformly 800x800, which makes its 800,000 rung identical to
        640,000. Falls back to the nominal budget when no native sizes are known.
        """
        if self.native_px is None:
            return int(budget)
        return min(int(budget), int(self.native_px.max()))

    def accuracy_at(self, budget_index: np.ndarray) -> float:
        """Mean accuracy when each document is run at its own budget index."""
        return float(self.scores[np.arange(self.n_documents), budget_index].mean())

    def cost_at(self, budget_index: np.ndarray) -> float:
        return float(self.costs[np.arange(self.n_documents), budget_index].mean())


def load_sweep(run_dir: Path, cost: str = "input_tokens") -> Sweep:
    """Read ``samples.csv`` into aligned score/cost matrices.

    Documents missing at any budget are dropped, so every row is comparable
    across the whole ladder.
    """
    if cost not in COST_COLUMNS:
        raise ValueError(f"cost must be one of {COST_COLUMNS}, got {cost!r}")

    samples = pd.read_csv(Path(run_dir) / SAMPLES_FILE)
    metric = overall_metric(samples)
    if metric is None:
        raise ValueError(f"{run_dir} has no scorable metric")
    samples = samples.assign(correct=correctness(samples, metric).fillna(False).astype(float))

    budgets = np.sort(samples["budget"].unique())
    scores = samples.pivot_table(index="doc_id", columns="budget", values="correct", aggfunc="first")
    costs = samples.pivot_table(index="doc_id", columns="budget", values=cost, aggfunc="first")

    complete = scores.notna().all(axis=1) & costs.notna().all(axis=1)
    scores, costs = scores[complete].reindex(columns=budgets), costs[complete].reindex(columns=budgets)

    # Constant across budgets — the source image never changes — so any of a
    # document's rows will do. Absent from sweeps predating the pixel columns.
    native = None
    if "original_px" in samples.columns:
        per_doc = samples.drop_duplicates("doc_id").set_index("doc_id")["original_px"]
        native = per_doc.reindex(scores.index).to_numpy(dtype=float)

    return Sweep(
        task=str(samples["task"].iloc[0]),
        doc_ids=scores.index.to_numpy(),
        budgets=budgets,
        scores=scores.to_numpy(dtype=float),
        costs=costs.to_numpy(dtype=float),
        subjects=_subjects(samples, metric, scores.index),
        cost_name=cost,
        native_px=native,
    )


def _subjects(samples: pd.DataFrame, metric: str, doc_ids: pd.Index) -> np.ndarray:
    """A per-document group label, used only to stratify the folds.

    mmmu_pro states the subject outright; seedbench_2_plus encodes it in *which*
    per-subject metric column is populated. Falls back to a single group.
    """
    per_doc = samples.drop_duplicates("doc_id").set_index("doc_id")

    if "mmmu_acc.subject" in per_doc.columns:
        return per_doc.loc[doc_ids, "mmmu_acc.subject"].fillna("unknown").to_numpy()

    task_metrics = [name for name in metric_names(samples) if name != metric]
    if not task_metrics:
        return np.full(len(doc_ids), "all")

    answered = per_doc.loc[doc_ids, [f"{name}.answer" for name in task_metrics]].notna()
    names = [name.rsplit("_", 1)[-1] for name in task_metrics]
    return np.where(answered.any(axis=1), np.array(names)[answered.to_numpy().argmax(axis=1)], "unknown")


def question_text(task: str, doc_ids: np.ndarray) -> list[str]:
    """The question stem for each document, for embedding.

    Pulled from the source dataset rather than a sweep, because a sweep records
    pixels and answers — not prompts. ``doc_id`` is the row index of the split
    lmms-eval evaluated, so it indexes the dataset directly.
    """
    import datasets

    name, config, split = _DATASETS[task]
    data = datasets.load_dataset(name, config, split=split)
    return [_stem(data[int(index)]) for index in doc_ids]


#: task -> (hf dataset, config, split). Mirrors each task's yaml.
_DATASETS = {
    "seedbench_2_plus": ("doolayer/SEED-Bench-2-Plus", None, "test"),
    "mmmu_pro_standard": ("MMMU/MMMU_Pro", "standard (10 options)", "test"),
}


def _stem(doc: dict) -> str:
    """The question without its answer options — options are near-identical noise."""
    return str(doc.get("question", "")).strip()
