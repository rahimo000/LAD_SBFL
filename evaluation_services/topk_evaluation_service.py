from typing import List, Dict
import numpy as np
from evaluation_services.shared import calculate_rank

def calculate_topk_score(score_matrix: np.ndarray, fault_index: int, k: int, case: str, metric_keys: List[str]) -> Dict[str, int]:
    """Calculates Top-K for all metrics in matrix."""
    results = {}
    for idx, key in enumerate(metric_keys):
        rank = calculate_rank(score_matrix, fault_index, idx, case=case)
        results[key] = 1 if rank <= k else 0
    return results

def get_version_topk_results(score_matrix: np.ndarray, fault_index: int, metric_keys: List[str], evals=None) -> Dict[str, Dict[str, int]]:
    """Computes Top-K variations for a specific version (only selected evals)."""
    topk_results = {}
    for k in [1, 3, 5]:
        for prefix, case in (('l', 'lexical'), ('r', 'reverse_lexical')):
            key = f'{prefix}-Top{k}'
            if evals is not None and key not in evals:
                continue
            topk_results[key] = calculate_topk_score(score_matrix, fault_index, k, case, metric_keys)
    return topk_results
