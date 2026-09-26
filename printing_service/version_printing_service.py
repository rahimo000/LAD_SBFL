import scores_services as ss
from typing import Optional
from scores_services.selection import Selection, default_selection, EVAL_RESULT_KEYS

EXAM_COLUMNS = {
    'oexam': 'oexam (%)',
    'pexam': 'pexam (%)',
    'lex-exam': 'lex-ex (%)',
    'rev-exam': 'rev-ex (%)',
}
DELTA_COLUMN = 'delta (%)'

def print_exam_scores(version_name: str, results: dict, selection: Optional[Selection] = None, decimals: int = 2):
    """
    Prints the EXAM scores (%) and Top-K results for a specific version
    (only selected metrics/evaluations, version-phase decimals).
    """
    if selection is None:
        selection = default_selection()
    if not results: return

    print(f"\n--- EXAM Scores (%) and Top-K for {version_name} (Fault Index: {results['fault_index']}) ---")

    metrics = selection.metrics

    header = f"{'Metric':<10}"
    for eval_key in selection.exam_evals:
        header += f" | {EXAM_COLUMNS[eval_key]:<12}"
    if selection.show_delta:
        header += f" | {DELTA_COLUMN:<10}"
    for key in selection.topk_evals:
        header += f" | {key.replace('-Top', '-T'):<4}"
    print(header)
    print("-" * len(header))

    exam_data = {EVAL_RESULT_KEYS[e]: results.get(EVAL_RESULT_KEYS[e], {}) for e in selection.exam_evals}
    topk = results.get('topk', {})

    for metric in metrics:
        vals = {rkey: data.get(metric, 0.0) * 100 for rkey, data in exam_data.items()}
        line = f"{metric:<10}"
        for eval_key in selection.exam_evals:
            line += f" | {vals[EVAL_RESULT_KEYS[eval_key]]:<12.{decimals}f}"
        if selection.show_delta:
            delta = vals.get('worst', 0.0) - vals.get('best', 0.0)
            line += f" | {delta:<10.{decimals}f}"
        for key in selection.topk_evals:
            line += f" | {str(topk.get(key, {}).get(metric, 0)):<4}"
        print(line)

def print_instruction_scores_table(version_name: str, results: dict, decimals: int = 2):
    """Prints detailed scores for each instruction from the score_matrix."""
    score_matrix = results.get('score_matrix')
    metric_keys = results.get('metric_keys')
    fault_index = results.get('fault_index')
    
    if score_matrix is None: return

    print(f"\n--- Detailed Instruction Scores for {version_name} ---")
    header = f"{'Index':<8} | " + " | ".join([f"{m:<8}" for m in metric_keys])
    print(header)
    print("-" * len(header))

    for i in range(score_matrix.shape[0]):
        is_fault = (i == fault_index)
        scores = score_matrix[i, :]
        line = f"{i:<8} | " + " | ".join([f"{s:<8.{decimals}f}" for s in scores])
        if is_fault: print(f"{line}  <-- FAULTY")
        else: print(line)
