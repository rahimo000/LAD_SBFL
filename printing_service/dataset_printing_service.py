import scores_services as ss
from typing import Dict, List, Any, Optional
from scores_services.rounding import round_value
from scores_services.selection import Selection, default_selection, EVAL_AVG_KEYS, DELTA_LABEL


def selection_rows(selection: Selection):
    """Ordered (label, avg_map_key) rows for overall tables (console + Excel)."""
    rows = [(e, EVAL_AVG_KEYS[e]) for e in selection.exam_evals]
    if selection.show_delta:
        rows.append((DELTA_LABEL, 'delta'))
    rows += [(k, k) for k in selection.topk_evals]
    return rows

def row_values(avg_map, metrics, key):
    """Row values in percentages (same as the console table)."""
    if key == "delta":
        return [(avg_map[m]['pexam'] - avg_map[m]['oexam']) * 100 for m in metrics]
    return [avg_map[m][key] * 100 for m in metrics]

def print_dataset_overall_summary(dataset_name: str, project_averages: Dict[str, Dict[str, Dict[str, float]]], selection: Optional[Selection] = None, decimals: int = 2):
    """
    Prints a consolidated table of overall average scores for all projects in a dataset
    (only selected metrics/evaluations, overall-phase decimals).

    Args:
        dataset_name: Name of the dataset (e.g., 'issta13')
        project_averages: Dict mapping project_name -> { metric -> { type -> avg_score } }
        selection: metric/eval subset (default: all).
        decimals: decimals shown (and used for the grand TOTAL calculation).
    """
    if selection is None:
        selection = default_selection()
    if not project_averages:
        print(f"No results found for dataset {dataset_name}.")
        return

    print(f"\n--- Dataset Overall Average EXAM and Top-K Summary (%) : {dataset_name} ---")

    metrics = selection.metrics
    p_width, t_width, m_width = 15, 12, 10

    header = f"{'Project':<{p_width}} | {'Type':<{t_width}} | " + " | ".join([f"{m:<{m_width}}" for m in metrics])
    print(header)
    print("-" * len(header))

    sorted_projects = sorted(project_averages.keys())

    all_types = selection_rows(selection)

    def get_row_data(avg_map, label, key):
        return row_values(avg_map, metrics, key)

    for project in sorted_projects:
        avg_map = project_averages[project]
        for idx, (label, key) in enumerate(all_types):
            p_label = project if idx == 0 else ""
            row_data = get_row_data(avg_map, label, key)
            row_str = " | ".join([f"{val:<{m_width}.{decimals}f}" for val in row_data])
            print(f"{p_label:<{p_width}} | {label:<{t_width}} | {row_str}")
        print("-" * len(header))

    # Grand Total (Average of project averages, overall-phase precision)
    grand_avg = calculate_grand_average(project_averages, all_types, decimals=decimals)

    # Print Grand Total Row
    for idx, (label, key) in enumerate(all_types):
        p_label = "TOTAL (AVG)" if idx == 0 else ""
        row_data = [grand_avg[m][key] * 100 for m in metrics]
        row_str = " | ".join([f"{val:<{m_width}.{decimals}f}" for val in row_data])
        print(f"{p_label:<{p_width}} | {label:<{t_width}} | {row_str}")
    print("-" * len(header))


def calculate_grand_average(project_averages: Dict[str, Dict[str, Dict[str, float]]], all_types: List[tuple], decimals: int = 2) -> Dict[str, Dict[str, float]]:
    """Averages project averages into a grand TOTAL map (rounded to the overall-phase precision)."""
    metrics = [m for m in ss.get_all_metric_keys()
               if any(m in avg for avg in project_averages.values())]
    num_projects = len(project_averages)
    grand_avg = {m: {key: 0.0 for _, key in all_types} for m in metrics}

    for avg_map in project_averages.values():
        for m in metrics:
            for _, key in all_types:
                if key == "delta":
                    val = avg_map[m]['pexam'] - avg_map[m]['oexam']
                else:
                    val = avg_map[m][key]
                grand_avg[m][key] += val

    for m in metrics:
        for _, key in all_types:
            grand_avg[m][key] = round_value(grand_avg[m][key] / num_projects, decimals)
    return grand_avg
