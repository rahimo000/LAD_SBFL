import numpy as np
from typing import List, Dict, Union

def calculate_rank(score_matrix: np.ndarray, fault_index: int, metric_idx: int, case: str = 'best') -> int:
    """
    Highly optimized rank calculation using NumPy vectorization.
    
    Args:
        score_matrix: 2D NumPy array [instructions x metrics]
        fault_index: Row index of the fault.
        metric_idx: Column index of the metric.
        case: 'best', 'worst', 'lexical', 'reverse_lexical'.
    """
    scores = score_matrix[:, metric_idx]
    fault_score = scores[fault_index]
    
    # Vectorized comparisons are much faster than Python loops
    greater_than = np.count_nonzero(scores > fault_score)
    
    if case == 'best':
        return greater_than + 1
    
    elif case == 'lexical':
        # Count instructions with same score but smaller index
        equal_scores = (scores == fault_score)
        smaller_index = np.arange(len(scores)) <= fault_index
        equal_and_smaller = np.count_nonzero(equal_scores & smaller_index)
        return greater_than + equal_and_smaller
        
    elif case == 'reverse_lexical':
        # Count instructions with same score but larger index
        equal_scores = (scores == fault_score)
        larger_index = np.arange(len(scores)) >= fault_index
        equal_and_larger = np.count_nonzero(equal_scores & larger_index)
        return greater_than + equal_and_larger
        
    else: # worst
        equal_to = np.count_nonzero(scores == fault_score)
        return greater_than + equal_to

def detect_metrics(score_matrix_or_list) -> List[int]:
    """Helper to get metric indices if using matrix."""
    if isinstance(score_matrix_or_list, np.ndarray):
        return list(range(score_matrix_or_list.shape[1]))
    return []
