import scores_services as ss

def print_exam_scores(version_name: str, results: dict):
    """
    Prints the EXAM scores (%) and Top-K results for a specific version.
    """
    if not results: return

    print(f"\n--- EXAM Scores (%) and Top-K for {version_name} (Fault Index: {results['fault_index']}) ---")
    
    metrics = ss.get_all_metric_keys()
    
    header = f"{'Metric':<10} | {'oexam (%)':<12} | {'pexam (%)':<12} | {'lex-ex (%)':<12} | {'rev-ex (%)':<12} | {'delta (%)':<10} | " + \
             " | ".join([f"l-T{k}" for k in [1, 3, 5]]) + " | " + \
             " | ".join([f"r-T{k}" for k in [1, 3, 5]])
    print(header)
    print("-" * len(header))

    o_exam = results['best']
    p_exam = results['worst']
    lex_exam = results.get('lexical', {})
    rev_exam = results.get('reverse', {})
    topk = results.get('topk', {})

    for metric in metrics:
        o_val = o_exam.get(metric, 0.0) * 100
        p_val = p_exam.get(metric, 0.0) * 100
        l_val = lex_exam.get(metric, 0.0) * 100
        r_val = rev_exam.get(metric, 0.0) * 100
        delta = p_val - o_val
        
        l_topks = [str(topk.get(f'l-Top{k}', {}).get(metric, 0)) for k in [1, 3, 5]]
        r_topks = [str(topk.get(f'r-Top{k}', {}).get(metric, 0)) for k in [1, 3, 5]]
        
        line = f"{metric:<10} | {o_val:<12.4f} | {p_val:<12.4f} | {l_val:<12.4f} | {r_val:<12.4f} | {delta:<10.4f} | " + \
               " | ".join([f"{v:<4}" for v in l_topks]) + " | " + \
               " | ".join([f"{v:<4}" for v in r_topks])
        print(line)

def print_instruction_scores_table(version_name: str, results: dict):
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
        line = f"{i:<8} | " + " | ".join([f"{s:<8.4f}" for s in scores])
        if is_fault: print(f"{line}  <-- FAULTY")
        else: print(line)
