import scores_services as ss
from typing import Dict, Optional, Tuple
from scores_services.selection import Selection, default_selection, EVAL_RESULT_KEYS, EVAL_AVG_KEYS

EXAM_LABELS = {
    'oexam': 'oexam',
    'pexam': 'pexam',
    'lex-exam': 'lex-exam',
    'rev-exam': 'rev-exam',
}
DELTA_LABEL = 'deltaexam'


def _resolve(selection: Optional[Selection]) -> Selection:
    return selection if selection is not None else default_selection()


def print_project_exam_summary(project_name: str, all_results: Dict[str, dict], selection: Optional[Selection] = None):
    """
    Prints a large consolidated EXAM and Top-K scores table for the project
    (only selected metrics/evaluations).
    """
    selection = _resolve(selection)
    if not all_results: return
    print(f"\n--- Project EXAM Summary Table (%) and Top-K : {project_name} ---")

    metrics = selection.metrics
    v_width, c_width, m_width = 10, 10, 10

    header = f"{'Version':<{v_width}} | {'Type':<{c_width}} | " + " | ".join([f"{m:<{m_width}}" for m in metrics])
    print(header)
    print("-" * len(header))

    sorted_versions = sorted(all_results.keys(), key=lambda x: [int(t) if t.isdigit() else t.lower() for t in __import__('re').split('([0-9]+)', x)])

    for version in sorted_versions:
        res = all_results[version]
        topk = res.get('topk', {})

        first = True
        for eval_key in selection.exam_evals:
            rkey = EVAL_RESULT_KEYS[eval_key]
            vlabel = version if first else ""
            first = False
            print(f"{vlabel:<{v_width}} | {EXAM_LABELS[eval_key]:<{c_width}} | " + " | ".join([f"{res.get(rkey, {}).get(m, 0.0)*100:<{m_width}.4f}" for m in metrics]))
        if selection.show_delta:
            print(f"{'':<{v_width}} | {DELTA_LABEL:<{c_width}} | " + " | ".join([f"{(res.get('worst', {}).get(m, 0.0)-res.get('best', {}).get(m, 0.0))*100:<{m_width}.4f}" for m in metrics]))

        for key in selection.topk_evals:
            print(f"{'':<{v_width}} | {key:<{c_width}} | " + " | ".join([f"{topk.get(key, {}).get(m, 0):<{m_width}}" for m in metrics]))
        print("-" * len(header))

def calculate_overall_averages(all_results: Dict[str, dict], selection: Optional[Selection] = None) -> Tuple[Dict[str, Dict[str, float]], int]:
    """Calculates average scores across all versions of a project (selected evaluations only)."""
    selection = _resolve(selection)
    metrics = selection.metrics
    count = len(all_results)

    exam_keys = [EVAL_AVG_KEYS[e] for e in selection.exam_evals]
    topk_keys = selection.topk_evals
    types = exam_keys + topk_keys
    totals = {m: {t: 0.0 for t in types} for m in metrics}

    for res in all_results.values():
        topk = res.get('topk', {})
        for m in metrics:
            for eval_key, avg_key in zip(selection.exam_evals, exam_keys):
                totals[m][avg_key] += res.get(EVAL_RESULT_KEYS[eval_key], {}).get(m, 0.0)
            for key in topk_keys:
                totals[m][key] += topk.get(key, {}).get(m, 0)

    # Create average map
    avg_map = {m: {t: (totals[m][t] / count) for t in types} for m in metrics}
    return avg_map, count

def print_overall_exam_scores(project_name: str, all_results: Dict[str, dict], selection: Optional[Selection] = None):
    """Prints the overall average EXAM and Top-K scores (%) across all versions."""
    selection = _resolve(selection)
    if not all_results: return
    metrics = selection.metrics
    avg_map, count = calculate_overall_averages(all_results, selection=selection)

    print(f"\n--- Overall Average EXAM and Top-K Scores (%) : {project_name} ({count} versions) ---")
    m_width, c_width = 10, 10
    header = f"{'Type':<{c_width}} | " + " | ".join([f"{m:<{m_width}}" for m in metrics])
    print(header)
    print("-" * len(header))

    def pr(label, key, is_delta=False):
        if is_delta:
            row = [f"{(avg_map[m]['pexam'] - avg_map[m]['oexam']) * 100:<{m_width}.4f}" for m in metrics]
        else:
            row = [f"{avg_map[m][key] * 100:<{m_width}.4f}" for m in metrics]
        print(f"{label:<{c_width}} | " + " | ".join(row))

    for eval_key in selection.exam_evals:
        pr(EXAM_LABELS[eval_key], EVAL_AVG_KEYS[eval_key])
    if selection.show_delta:
        pr(DELTA_LABEL, '', True)
    for key in selection.topk_evals:
        pr(key, key)
    print("-" * len(header))
