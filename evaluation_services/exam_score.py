from typing import List, Dict, Any
import numpy as np
from evaluation_services.shared import calculate_rank

def _calc_exam_all_metrics(score_matrix: np.ndarray, fault_index: int, metric_keys: List[str], case: str) -> Dict[str, float]:
    """Helper to calculate EXAM for all metrics in the matrix."""
    num_instructions = score_matrix.shape[0]
    results = {}
    for idx, key in enumerate(metric_keys):
        rank = calculate_rank(score_matrix, fault_index, idx, case=case)
        results[key] = rank / num_instructions
    return results

def best_case_exam_score(score_matrix: np.ndarray, fault_index: int, metric_keys: List[str]) -> Dict[str, float]:
    return _calc_exam_all_metrics(score_matrix, fault_index, metric_keys, 'best')

def worst_case_exam_score(score_matrix: np.ndarray, fault_index: int, metric_keys: List[str]) -> Dict[str, float]:
    return _calc_exam_all_metrics(score_matrix, fault_index, metric_keys, 'worst')

def lexical_exam_score(score_matrix: np.ndarray, fault_index: int, metric_keys: List[str]) -> Dict[str, float]:
    return _calc_exam_all_metrics(score_matrix, fault_index, metric_keys, 'lexical')

def reverse_lexical_exam_score(score_matrix: np.ndarray, fault_index: int, metric_keys: List[str]) -> Dict[str, float]:
    return _calc_exam_all_metrics(score_matrix, fault_index, metric_keys, 'reverse_lexical')

def exam_score_for_eval(score_matrix: np.ndarray, fault_index: int, metric_keys: List[str], eval_key: str) -> Dict[str, float]:
    """Computes EXAM for a single selected evaluation (oexam/pexam/lex-exam/rev-exam)."""
    from scores_services.selection import EXAM_CASES
    return _calc_exam_all_metrics(score_matrix, fault_index, metric_keys, EXAM_CASES[eval_key])
