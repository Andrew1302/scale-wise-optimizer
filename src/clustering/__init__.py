"""Cross-validated evaluation of per-cluster resolution policies.

The question this package answers is *not* "does routing beat always running at
full resolution" — that is trivially winnable by picking any cheaper fixed
resolution. It is "does routing beat the best **fixed** resolution at the same
cost", measured out-of-fold. That quantity is :func:`~clustering.kfold.lift`.
"""

from clustering.data import Sweep, load_sweep
from clustering.kfold import cross_validate, lift, strategy_summary
from clustering.policy import STRATEGIES, derive_policy

__all__ = ["STRATEGIES", "Sweep", "cross_validate", "derive_policy", "lift", "load_sweep", "strategy_summary"]
