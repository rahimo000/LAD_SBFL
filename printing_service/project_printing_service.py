import scores_services as ss
from typing import Dict, Tuple

def print_project_exam_summary(project_name: str, all_results: Dict[str, dict], include_topk: bool = True):
    """
    Prints a large consolidated EXAM and Top-K scores table for the project.
    """
    if not all_results: return
    print(f"\n--- Project EXAM Summary Table (%) and Top-K : {project_name} ---")
    
    metrics = ss.get_all_metric_keys()
    v_width, c_width, m_width = 10, 10, 10
    
    header = f"{'Version':<{v_width}} | {'Type':<{c_width}} | " + " | ".join([f"{m:<{m_width}}" for m in metrics])
    print(header)
    print("-" * len(header))
    
    sorted_versions = sorted(all_results.keys(), key=lambda x: [int(t) if t.isdigit() else t.lower() for t in __import__('re').split('([0-9]+)', x)])

    for version in sorted_versions:
        res = all_results[version]
        topk = res.get('topk', {})
        
        # oexam, pexam, lex, rev, delta
        print(f"{version:<{v_width}} | {'oexam':<{c_width}} | " + " | ".join([f"{res['best'].get(m, 0.0)*100:<{m_width}.4f}" for m in metrics]))
        print(f"{'':<{v_width}} | {'pexam':<{c_width}} | " + " | ".join([f"{res['worst'].get(m, 0.0)*100:<{m_width}.4f}" for m in metrics]))
        print(f"{'':<{v_width}} | {'lex-exam':<{c_width}} | " + " | ".join([f"{res.get('lexical', {}).get(m, 0.0)*100:<{m_width}.4f}" for m in metrics]))
        print(f"{'':<{v_width}} | {'rev-exam':<{c_width}} | " + " | ".join([f"{res.get('reverse', {}).get(m, 0.0)*100:<{m_width}.4f}" for m in metrics]))
        print(f"{'':<{v_width}} | {'deltaexam':<{c_width}} | " + " | ".join([f"{(res['worst'].get(m, 0.0)-res['best'].get(m, 0.0))*100:<{m_width}.4f}" for m in metrics]))
        
        # Top-K
        if include_topk:
            topk_keys = [f"l-Top{k}" for k in [1, 3, 5]] + [f"r-Top{k}" for k in [1, 3, 5]]
            for key in topk_keys:
                print(f"{'':<{v_width}} | {key:<{c_width}} | " + " | ".join([f"{topk.get(key, {}).get(m, 0):<{m_width}}" for m in metrics]))
        print("-" * len(header))

def calculate_overall_averages(all_results: Dict[str, dict], include_topk: bool = True) -> Tuple[Dict[str, Dict[str, float]], int]:
    """Calculates average scores across all versions of a project."""
    metrics = ss.get_all_metric_keys()
    count = len(all_results)

    types = ["oexam", "pexam", "lexical", "reverse"]
    topk_keys = ([f"l-Top{k}" for k in [1, 3, 5]] + [f"r-Top{k}" for k in [1, 3, 5]]) if include_topk else []
    totals = {m: {t: 0.0 for t in (types + topk_keys)} for m in metrics}
    
    for res in all_results.values():
        topk = res.get('topk', {})
        for m in metrics:
            totals[m]["oexam"] += res['best'].get(m, 0.0)
            totals[m]["pexam"] += res['worst'].get(m, 0.0)
            totals[m]["lexical"] += res.get('lexical', {}).get(m, 0.0)
            totals[m]["reverse"] += res.get('reverse', {}).get(m, 0.0)
            for key in topk_keys: 
                totals[m][key] += topk.get(key, {}).get(m, 0)
                
    # Create average map
    avg_map = {m: {t: (totals[m][t] / count) for t in (types + topk_keys)} for m in metrics}
    return avg_map, count

def print_overall_exam_scores(project_name: str, all_results: Dict[str, dict], include_topk: bool = True):
    """Prints the overall average EXAM and Top-K scores (%) across all versions."""
    if not all_results: return
    metrics = ss.get_all_metric_keys()
    avg_map, count = calculate_overall_averages(all_results, include_topk=include_topk)
    
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

    pr('oexam', 'oexam'); pr('pexam', 'pexam'); pr('lex-exam', 'lexical'); pr('rev-exam', 'reverse'); pr('deltaexam', '', True)
    if include_topk:
        topk_keys = [f"l-Top{k}" for k in [1, 3, 5]] + [f"r-Top{k}" for k in [1, 3, 5]]
        for key in topk_keys: pr(key, key)
    print("-" * len(header))
