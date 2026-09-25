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

def calculate_all_metrics(p: np.ndarray, n: np.ndarray) -> tuple:
    """Computes all metrics in a single optimized pass while preserving exact rounding."""
    all_keys = ss.get_all_metric_keys()
    num_instr = p.shape[1]
    
    score_matrix = np.zeros((num_instr, len(all_keys)))
    key_to_idx = {key: i for i, key in enumerate(all_keys)}

    # 1. Standard Metrics
    score_matrix[:, key_to_idx['tar']] = Tarontula(p, n)
    score_matrix[:, key_to_idx['och']] = ochiai(p, n)
    score_matrix[:, key_to_idx['jac']] = jaccard(p, n)
    score_matrix[:, key_to_idx['gp']] = gp13(p, n)
    score_matrix[:, key_to_idx['op2']] = op2(p, n)
    score_matrix[:, key_to_idx['kul2']] = kulczynski2(p, n)
    score_matrix[:, key_to_idx['zol']] = zoltar(p, n)
    score_matrix[:, key_to_idx['amp']] = ample(p, n)

    # 2. Aggregators
    active_indices = [key_to_idx[k] for k in ['tar', 'och', 'jac', 'gp', 'op2', 'kul2', 'zol', 'amp']]
    
    if 'apv' in key_to_idx:
        score_matrix[:, key_to_idx['apv']] = ss.calculate_approval_voting_score(score_matrix, active_indices)
    if 'mj' in key_to_idx:
        score_matrix[:, key_to_idx['mj']] = ss.calculate_majority_judgment_score(score_matrix, active_indices)
        
    return score_matrix, all_keys
