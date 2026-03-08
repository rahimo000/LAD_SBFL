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

def get_version_topk_results(score_matrix: np.ndarray, fault_index: int, metric_keys: List[str]) -> Dict[str, Dict[str, int]]:
    """Computes Top-K variations for a specific version."""
    topk_results = {}
    for k in [1, 3, 5]:
        topk_results[f'l-Top{k}'] = calculate_topk_score(score_matrix, fault_index, k, 'lexical', metric_keys)
        topk_results[f'r-Top{k}'] = calculate_topk_score(score_matrix, fault_index, k, 'reverse_lexical', metric_keys)
    return topk_results
