"""Turning per-cluster accuracy curves into a budget for each cluster."""

from __future__ import annotations

import numpy as np

#: Strategy name -> the fraction of a cluster's best accuracy it must retain.
#: ``best`` takes the peak outright; the others take the *cheapest* budget within
#: a tolerance of it, which is where the efficiency gain is supposed to come from.
STRATEGIES: dict[str, float | None] = {"best": None, "efficient_95": 0.95, "efficient_99": 0.99}


def derive_policy(scores: np.ndarray, labels: np.ndarray, n_clusters: int, strategy: str) -> np.ndarray:
    """Choose a budget index for every cluster, from training rows only.

    Returns an array of length ``n_clusters``. Clusters with no training rows
    fall back to the most expensive budget — the safe choice, since a cluster we
    know nothing about is not one to economise on.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")

    tolerance = STRATEGIES[strategy]
    most_expensive = scores.shape[1] - 1
    policy = np.full(n_clusters, most_expensive, dtype=int)

    for cluster in range(n_clusters):
        rows = labels == cluster
        if not rows.any():
            continue
        accuracy = scores[rows].mean(axis=0)
        best = int(accuracy.argmax())
        if tolerance is None:
            policy[cluster] = best
            continue
        # Budgets are ascending, so the first index clearing the bar is the cheapest.
        within = np.flatnonzero(accuracy >= accuracy[best] * tolerance)
        policy[cluster] = int(within[0]) if within.size else best

    return policy
