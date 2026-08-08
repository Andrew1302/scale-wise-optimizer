"""Out-of-fold evaluation of per-cluster resolution policies.

Every number here is produced on documents the clustering never saw: KMeans is
fit on the training folds, the per-cluster budgets are derived from the training
folds, and both are then applied to the held-out fold. Fitting the policy on the
same rows it is scored on inflates the result badly, which is the whole reason
this module exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

from clustering.data import Sweep
from clustering.policy import STRATEGIES, derive_policy


def cross_validate(
    sweep: Sweep,
    embeddings: np.ndarray,
    k_values: list[int],
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """One row per (fold, K, strategy), plus the fixed-budget baselines.

    Fixed baselines are emitted once per fold — they do not depend on K — with
    ``k`` set to NaN, so they can be joined against any K.
    """
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    rows = []

    for fold, (train, test) in enumerate(folds.split(embeddings, sweep.subjects)):
        for index, budget in enumerate(sweep.budgets):
            held = np.full(len(test), index)
            rows.append(
                {
                    "fold": fold,
                    "k": np.nan,
                    "strategy": f"fixed_{budget}",
                    "budget": budget,
                    "accuracy": float(sweep.scores[test, index].mean()),
                    "cost": float(sweep.costs[test, index].mean()),
                    "n_test": len(held),
                }
            )

        for k in k_values:
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(embeddings[train])
            assigned = model.predict(embeddings[test])
            for strategy in STRATEGIES:
                policy = derive_policy(sweep.scores[train], model.labels_, k, strategy)
                chosen = policy[assigned]
                rows.append(
                    {
                        "fold": fold,
                        "k": k,
                        "strategy": strategy,
                        "budget": np.nan,
                        "accuracy": float(sweep.scores[test, chosen].mean()),
                        "cost": float(sweep.costs[test, chosen].mean()),
                        "n_test": len(test),
                    }
                )

    return pd.DataFrame(rows)


def fixed_frontier(folds: pd.DataFrame) -> pd.DataFrame:
    """The fixed-budget accuracy/cost curve, averaged over folds and sorted by cost."""
    fixed = folds[folds["strategy"].str.startswith("fixed_")]
    curve = fixed.groupby("budget")[["cost", "accuracy"]].mean().reset_index()
    return curve.sort_values("cost").reset_index(drop=True)


def strategy_summary(folds: pd.DataFrame, k: int) -> pd.DataFrame:
    """Mean ± fold standard deviation for every fixed budget and the routed
    strategies at one ``k``, ordered fixed-then-routed by cost.

    ``cost_pct`` rescales cost against the most expensive *fixed* budget, so the
    axis reads as "share of what running everything at full resolution costs".
    """
    # Checked against the routed rows specifically: the fixed baselines match every
    # k, so an unrun k would otherwise yield a chart with no strategies on it.
    if not (folds["k"] == k).any():
        raise ValueError(f"no folds recorded for k={k}")
    selected = folds[folds["strategy"].str.startswith("fixed_") | (folds["k"] == k)]

    summary = (
        selected.groupby("strategy")
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            cost=("cost", "mean"),
            cost_sd=("cost", "std"),
        )
        .reset_index()
    )
    summary["is_fixed"] = summary["strategy"].str.startswith("fixed_")
    full_cost = summary.loc[summary["is_fixed"], "cost"].max()
    summary["cost_pct"] = summary["cost"] / full_cost * 100
    summary["cost_pct_sd"] = summary["cost_sd"] / full_cost * 100
    return summary.sort_values(["is_fixed", "cost"], ascending=[False, True]).reset_index(drop=True)


def lift(folds: pd.DataFrame) -> pd.DataFrame:
    """Accuracy gained over a *fixed* budget of the same cost, per (k, strategy).

    Positive lift is the only result that supports routing. Beating the most
    expensive fixed budget while spending less is not evidence — a cheaper fixed
    budget usually does that too, which is exactly what the frontier encodes.
    Strategies costing less than the cheapest fixed budget cannot be compared and
    are returned with NaN lift rather than an extrapolated number.
    """
    curve = fixed_frontier(folds)
    routed = folds[~folds["strategy"].str.startswith("fixed_")]

    summary = (
        routed.groupby(["k", "strategy"])
        .agg(accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"), cost=("cost", "mean"))
        .reset_index()
    )
    inside = summary["cost"].between(curve["cost"].min(), curve["cost"].max())
    baseline = np.interp(summary["cost"], curve["cost"], curve["accuracy"])
    summary["fixed_at_same_cost"] = np.where(inside, baseline, np.nan)
    summary["lift"] = summary["accuracy"] - summary["fixed_at_same_cost"]
    summary["cost_share"] = summary["cost"] / curve["cost"].max()
    return summary.sort_values(["strategy", "k"]).reset_index(drop=True)
