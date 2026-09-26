import numpy as np
import scores_services as ss

# Single source of truth: each formula lives in its own module.
# Re-exported here so `from scores_services.metric_service import ochiai`
# keeps working.
from .trantula import Tarontula
from .ochiai import ochiai
from .jaccard import jaccard
from .gp13 import gp13
from .op2 import op2
from .kulczynski2 import kulczynski2
from .zoltar import zoltar
from .ample import ample

__all__ = [
    'Tarontula', 'ochiai', 'jaccard', 'gp13',
    'op2', 'kulczynski2', 'zoltar', 'ample',
    'calculate_all_metrics',
]

_BASE_FUNCS = {
    'tar': Tarontula,
    'och': ochiai,
    'jac': jaccard,
    'gp': gp13,
    'op2': op2,
    'kul2': kulczynski2,
    'zol': zoltar,
    'amp': ample,
}

def calculate_all_metrics(p: np.ndarray, n: np.ndarray, metrics=None, decimals: int = 2) -> tuple:
    """Computes the selected metrics (default: all) with the given score precision.

    Aggregators (mj/apv) are computed over the *selected* base subset only.
    Returns (score_matrix, keys) where keys are in canonical order.
    """
    from scores_services import get_all_metric_keys
    if metrics is None:
        metrics = get_all_metric_keys()
    keys = [k for k in get_all_metric_keys() if k in metrics]
    num_instr = p.shape[1]

    score_matrix = np.zeros((num_instr, len(keys)))
    key_to_idx = {key: i for i, key in enumerate(keys)}

    # 1. Standard Metrics (only the selected ones are computed)
    for key, func in _BASE_FUNCS.items():
        if key in key_to_idx:
            score_matrix[:, key_to_idx[key]] = func(p, n, decimals=decimals)

    # 2. Aggregators over the selected base subset
    active_indices = [key_to_idx[k] for k in _BASE_FUNCS if k in key_to_idx]

    if 'apv' in key_to_idx:
        score_matrix[:, key_to_idx['apv']] = ss.calculate_approval_voting_score(score_matrix, active_indices)
    if 'mj' in key_to_idx:
        score_matrix[:, key_to_idx['mj']] = ss.calculate_majority_judgment_score(score_matrix, active_indices)

    return score_matrix, keys
