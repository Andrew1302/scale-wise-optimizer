import numpy as np
import pandas as pd
import pytest

from clustering.data import Sweep, load_sweep
from clustering.kfold import cross_validate, fixed_frontier, lift, strategy_summary
from clustering.policy import derive_policy

TASK = "seedbench_2_plus"
BUDGETS = np.array([2_000, 100_000, 800_000])


def sweep(scores: np.ndarray, costs: np.ndarray | None = None, subjects=None) -> Sweep:
    n = len(scores)
    return Sweep(
        task=TASK,
        doc_ids=np.arange(n),
        budgets=BUDGETS,
        scores=np.asarray(scores, dtype=float),
        costs=np.asarray(costs if costs is not None else np.tile(BUDGETS, (n, 1)), dtype=float),
        subjects=np.array(subjects if subjects is not None else ["a"] * n),
        cost_name="sent_px",
    )


# -- policy ----------------------------------------------------------------


def test_best_strategy_takes_the_peak():
    scores = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    assert derive_policy(scores, np.zeros(2, int), 1, "best") == [1]


def test_efficient_takes_the_cheapest_budget_within_tolerance():
    """0.96 of the peak clears the 95% bar, so the cheaper budget wins."""
    scores = np.array([[0.96, 1.0, 1.0]] * 100)

    assert derive_policy(scores, np.zeros(100, int), 1, "efficient_95") == [0]
    assert derive_policy(scores, np.zeros(100, int), 1, "efficient_99") == [1]


def test_empty_clusters_fall_back_to_the_most_expensive_budget():
    """A cluster with no training rows is not one to economise on."""
    scores = np.array([[1.0, 0.0, 0.0]])

    assert list(derive_policy(scores, np.zeros(1, int), 3, "best")) == [0, 2, 2]


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        derive_policy(np.zeros((1, 3)), np.zeros(1, int), 1, "cheapest")


# -- sweep -----------------------------------------------------------------


def test_accuracy_and_cost_follow_the_chosen_budget():
    s = sweep(np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]))

    assert s.accuracy_at(np.array([1, 0])) == 1.0
    assert s.cost_at(np.array([0, 0])) == 2_000


def test_misshapen_matrices_are_rejected():
    with pytest.raises(ValueError, match="scores/costs"):
        Sweep(TASK, np.arange(2), BUDGETS, np.zeros((2, 2)), np.zeros((2, 3)), np.array(["a", "a"]), "sent_px")


def test_documents_missing_at_some_budget_are_dropped(tmp_path):
    """Every row must be comparable across the whole ladder."""
    rows = [
        {
            "task": TASK,
            "budget": b,
            "doc_id": d,
            f"{TASK}_all.pred": "A",
            f"{TASK}_all.answer": "A",
            "input_tokens": 10,
            "sent_px": b,
        }
        for d in (0, 1)
        for b in (2_000, 100_000)
    ]
    rows = [r for r in rows if not (r["doc_id"] == 1 and r["budget"] == 100_000)]
    pd.DataFrame(rows).to_csv(tmp_path / "samples.csv", index=False)

    assert load_sweep(tmp_path).doc_ids.tolist() == [0]


# -- cross-validation ------------------------------------------------------


@pytest.fixture
def folds():
    rng = np.random.default_rng(0)
    n = 120
    scores = rng.integers(0, 2, size=(n, 3)).astype(float)
    subjects = np.array(["a", "b"] * (n // 2))
    embeddings = rng.normal(size=(n, 8))
    return cross_validate(sweep(scores, subjects=subjects), embeddings, k_values=[2, 3], n_folds=3)


def test_every_fold_reports_each_strategy_and_baseline(folds):
    assert set(folds["fold"]) == {0, 1, 2}
    assert set(folds[folds.k.notna()]["strategy"]) == {"best", "efficient_95", "efficient_99"}
    assert len(folds[folds["strategy"].str.startswith("fixed_")]) == 3 * len(BUDGETS)


def test_baselines_are_k_independent(folds):
    assert folds[folds["strategy"].str.startswith("fixed_")]["k"].isna().all()


def test_frontier_is_sorted_by_cost(folds):
    curve = fixed_frontier(folds)

    assert curve["cost"].is_monotonic_increasing
    assert len(curve) == len(BUDGETS)


def test_lift_compares_against_a_fixed_budget_of_equal_cost(folds):
    table = lift(folds)

    assert set(table.columns) >= {"k", "strategy", "accuracy", "cost", "fixed_at_same_cost", "lift"}
    scored = table.dropna(subset=["lift"])
    assert np.allclose(scored["lift"], scored["accuracy"] - scored["fixed_at_same_cost"])


def test_summary_reports_fold_spread_for_error_bars(folds):
    table = strategy_summary(folds, k=2)

    assert set(table.columns) >= {"accuracy", "accuracy_sd", "cost", "cost_sd", "cost_pct", "cost_pct_sd"}
    assert (table["accuracy_sd"] >= 0).all(), "std across folds is what the error bars draw"


def test_summary_keeps_only_the_requested_k(folds):
    table = strategy_summary(folds, k=2)

    assert set(table[~table["is_fixed"]]["strategy"]) == {"best", "efficient_95", "efficient_99"}
    assert len(table) == len(BUDGETS) + 3


def test_summary_orders_fixed_budgets_by_cost_then_routed(folds):
    table = strategy_summary(folds, k=2)

    assert table["is_fixed"].tolist() == [True] * len(BUDGETS) + [False] * 3
    assert table[table["is_fixed"]]["cost"].is_monotonic_increasing


def test_cost_is_a_percentage_of_the_dearest_fixed_budget(folds):
    table = strategy_summary(folds, k=2)

    assert table[table["is_fixed"]]["cost_pct"].max() == pytest.approx(100.0)


def test_summary_rejects_a_k_that_was_never_run(folds):
    with pytest.raises(ValueError, match="k=99"):
        strategy_summary(folds, k=99)


def test_lift_is_not_extrapolated_below_the_cheapest_fixed_budget():
    """A policy cheaper than every fixed budget has no baseline to beat."""
    rows = pd.DataFrame(
        [
            {"fold": 0, "k": np.nan, "strategy": "fixed_2000", "budget": 2_000, "accuracy": 0.3, "cost": 2_000},
            {"fold": 0, "k": np.nan, "strategy": "fixed_800000", "budget": 800_000, "accuracy": 0.7, "cost": 800_000},
            {"fold": 0, "k": 5, "strategy": "best", "budget": np.nan, "accuracy": 0.5, "cost": 100},
        ]
    )

    assert lift(rows)["lift"].isna().all()
