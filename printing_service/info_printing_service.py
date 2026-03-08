from typing import Dict, Any, List

def print_version_info(version_name: str, results: Dict[str, Any]):
    """Prints metadata info for a specific version."""
    meta = results.get("metadata", {})
    fault_idx = results.get("fault_index")
    
    print(f"\n--- Version Information: {version_name} ---")
    width = 30
    print(f"{'Property':<{width}} | {'Value'}")
    print("-" * (width + 15))
    print(f"{'Number of Instructions':<{width}} | {meta.get('num_instructions')}")
    print(f"{'Fault Index (err index)':<{width}} | {fault_idx}")
    print(f"{'Number of B+ (Passing)':<{width}} | {meta.get('num_b_plus')}")
    print(f"{'Number of B- (Failing)':<{width}} | {meta.get('num_b_moins')}")
    print(f"{'Test Pass Rate (%)':<{width}} | {meta.get('pass_rate', 0)*100:.2f}%")
    print(f"{'Matrix Density (%)':<{width}} | {meta.get('density', 0)*100:.2f}%")
    print(f"{'Fault Covered by Failing Test':<{width}} | {meta.get('fault_covered')}")
    print("-" * (width + 15))

def print_project_info(project_name: str, all_results: Dict[str, Dict[str, Any]]):
    """Prints aggregated metadata info for a project."""
    num_versions = len(all_results)
    if num_versions == 0: return

    total_b_plus = sum(r['metadata']['num_b_plus'] for r in all_results.values())
    total_b_moins = sum(r['metadata']['num_b_moins'] for r in all_results.values())
    total_instr = sum(r['metadata']['num_instructions'] for r in all_results.values())
    
    # Range (choice info)
    min_instr = min(r['metadata']['num_instructions'] for r in all_results.values())
    max_instr = max(r['metadata']['num_instructions'] for r in all_results.values())

    print(f"\n--- Project Information: {project_name} ---")
    width = 30
    print(f"{'Property':<{width}} | {'Total':<12} | {'Average (per version)'}")
    print("-" * (width + 40))
    print(f"{'Number of Versions':<{width}} | {num_versions:<12} | N/A")
    print(f"{'Total B+ (Passing)':<{width}} | {total_b_plus:<12} | {total_b_plus/num_versions:.2f}")
    print(f"{'Total B- (Failing)':<{width}} | {total_b_moins:<12} | {total_b_moins/num_versions:.2f}")
    print(f"{'Total Instructions':<{width}} | {total_instr:<12} | {total_instr/num_versions:.2f}")
    print(f"{'Instruction Count Range':<{width}} | {f'{min_instr}-{max_instr}':<12} | N/A")
    print("-" * (width + 40))

def print_dataset_info(dataset_name: str, dataset_results: Dict[str, Dict[str, Dict[str, Any]]]):
    """Prints aggregated metadata info for a dataset."""
    print(f"\n--- Dataset Information: {dataset_name} ---")
    
    p_width, v_width, bp_width, bm_width, in_width = 15, 10, 10, 10, 15
    header = f"{'Project':<{p_width}} | {'Versions':<{v_width}} | {'Total B+':<{bp_width}} | {'Total B-':<{bm_width}} | {'Total Instr':<{in_width}}"
    print(header)
    print("-" * len(header))
    
    grand_versions = 0
    grand_b_plus = 0
    grand_b_moins = 0
    grand_instr = 0
    
    for project, all_results in sorted(dataset_results.items()):
        num_v = len(all_results)
        t_bp = sum(r['metadata']['num_b_plus'] for r in all_results.values())
        t_bm = sum(r['metadata']['num_b_moins'] for r in all_results.values())
        t_in = sum(r['metadata']['num_instructions'] for r in all_results.values())
        
        print(f"{project:<{p_width}} | {num_v:<{v_width}} | {t_bp:<{bp_width}} | {t_bm:<{bm_width}} | {t_in:<{in_width}}")
        
        grand_versions += num_v
        grand_b_plus += t_bp
        grand_b_moins += t_bm
        grand_instr += t_in
        
    print("-" * len(header))
    print(f"{'TOTAL DATASET':<{p_width}} | {grand_versions:<{v_width}} | {grand_b_plus:<{bp_width}} | {grand_b_moins:<{bm_width}} | {grand_instr:<{in_width}}")
    print("-" * len(header))
