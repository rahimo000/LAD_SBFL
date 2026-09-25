import scores_services as ss
from typing import Dict, List, Any

def print_dataset_overall_summary(dataset_name: str, project_averages: Dict[str, Dict[str, Dict[str, float]]], include_topk: bool = True):
    """
    Prints a consolidated table of overall average scores for all projects in a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'issta13')
        project_averages: Dict mapping project_name -> { metric -> { type -> avg_score } }
        include_topk: When False, Top-K rows are omitted (EXAM only).
    """
    if not project_averages:
        print(f"No results found for dataset {dataset_name}.")
        return

    print(f"\n--- Dataset Overall Average EXAM and Top-K Summary (%) : {dataset_name} ---")
    
    metrics = ss.get_all_metric_keys()
    p_width, t_width, m_width = 15, 12, 10
    
    header = f"{'Project':<{p_width}} | {'Type':<{t_width}} | " + " | ".join([f"{m:<{m_width}}" for m in metrics])
    print(header)
    print("-" * len(header))
    
    sorted_projects = sorted(project_averages.keys())
    
    exam_types = [
        ("oexam", "oexam"),
        ("pexam", "pexam"),
        ("lex-exam", "lexical"),
        ("rev-exam", "reverse"),
        ("deltaexam", "delta")
    ]
    
    topk_keys = ([f"l-Top{k}" for k in [1, 3, 5]] + [f"r-Top{k}" for k in [1, 3, 5]]) if include_topk else []
    all_types = exam_types + [(k, k) for k in topk_keys]

    def get_row_data(avg_map, label, key):
        if key == "delta":
            return [ (avg_map[m]['pexam'] - avg_map[m]['oexam']) * 100 for m in metrics]
        else:
            return [avg_map[m][key] * 100 for m in metrics]

    for project in sorted_projects:
        avg_map = project_averages[project]
        for idx, (label, key) in enumerate(all_types):
            p_label = project if idx == 0 else ""
            row_data = get_row_data(avg_map, label, key)
            row_str = " | ".join([f"{val:<{m_width}.4f}" for val in row_data])
            print(f"{p_label:<{p_width}} | {label:<{t_width}} | {row_str}")
        print("-" * len(header))

    # Grand Total (Average of project averages)
    grand_avg = calculate_grand_average(project_averages, all_types)

    # Print Grand Total Row
    for idx, (label, key) in enumerate(all_types):
        p_label = "TOTAL (AVG)" if idx == 0 else ""
        row_data = [grand_avg[m][key] * 100 for m in metrics]
        row_str = " | ".join([f"{val:<{m_width}.4f}" for val in row_data])
        print(f"{p_label:<{p_width}} | {label:<{t_width}} | {row_str}")
    print("-" * len(header))


def calculate_grand_average(project_averages: Dict[str, Dict[str, Dict[str, float]]], all_types: List[tuple]) -> Dict[str, Dict[str, float]]:
    """Averages project averages into a grand TOTAL map (average of averages)."""
    metrics = ss.get_all_metric_keys()
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
            grand_avg[m][key] /= num_projects
    return grand_avg
