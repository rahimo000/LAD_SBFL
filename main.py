import os
import argparse
import numpy as np
import fileManagment as fm
import scores_services as ss
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor
from scores_services.metric_service import calculate_all_metrics
from evaluation_services.exam_score import best_case_exam_score, worst_case_exam_score, lexical_exam_score, reverse_lexical_exam_score
from evaluation_services.topk_evaluation_service import get_version_topk_results
from printing_service.version_printing_service import print_exam_scores, print_instruction_scores_table
from printing_service.project_printing_service import print_project_exam_summary, print_overall_exam_scores, calculate_overall_averages
from printing_service.dataset_printing_service import print_dataset_overall_summary
from printing_service.info_printing_service import print_version_info, print_project_info, print_dataset_info
from printing_service.csv_export_service import capture_to_csv
from logging_config import setup_logging, get_logger

logger = get_logger(__name__)

def natural_sort_key(s: str) -> List[Any]:
    """Sort strings with numbers naturally."""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def process_version_wrapper(args: Tuple[str, str]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Wrapper for multiprocessing support."""
    version_path, version_name = args
    res = process_version(version_path)
    return version_name, res

def process_version(version_path: str) -> Optional[Dict[str, Any]]:
    """Processes a single version folder using optimized NumPy operations."""
    base_dir = os.path.join(version_path, "Base")
    if not os.path.exists(base_dir):
        logger.debug(f"Base directory not found: {base_dir}")
        return None

    b_plus_path = os.path.join(base_dir, "Bplus")
    b_moins_path = os.path.join(base_dir, "Bmoins")
    
    p = fm.readBFile(b_plus_path)
    n = fm.readBFile(b_moins_path)
    if p is None or n is None:
        logger.error(f"Failed to read coverage files in {base_dir}")
        return None

    # Determine fault index
    fault_index = -1
    try:
        for filename in os.listdir(version_path):
            if filename.isdigit():
                fault_index = int(filename) - 1
                break
    except Exception as e:
        logger.error(f"Error finding fault index in {version_path}: {e}")
        return None
    if fault_index == -1:
        logger.warning(f"No fault index file found in {version_path}")
        return None

    # Vectorized Metric Calculation
    try:
        score_matrix, all_keys = calculate_all_metrics(p, n)
    except Exception as e:
        logger.error(f"Error calculating metrics for {version_path}: {e}")
        return None

    # Evaluation
    best_exam = best_case_exam_score(score_matrix, fault_index, all_keys)
    worst_exam = worst_case_exam_score(score_matrix, fault_index, all_keys)
    lex_exam = lexical_exam_score(score_matrix, fault_index, all_keys)
    rev_exam = reverse_lexical_exam_score(score_matrix, fault_index, all_keys)
    topk_results = get_version_topk_results(score_matrix, fault_index, all_keys)
    
    # Metadata for -info
    num_instructions = p.shape[1]
    num_b_plus = p.shape[0]
    num_b_moins = n.shape[0]
    
    # Choice infos: Pass rate, Density, and Fault coverage status
    pass_rate = num_b_plus / (num_b_plus + num_b_moins) if (num_b_plus + num_b_moins) > 0 else 0
    density = (np.sum(p) + np.sum(n)) / ((num_b_plus + num_b_moins) * num_instructions) if num_instructions > 0 else 0
    is_fault_covered = (np.sum(n[:, fault_index]) > 0)

    return {
        "best": best_exam,
        "worst": worst_exam,
        "lexical": lex_exam,
        "reverse": rev_exam,
        "topk": topk_results,
        "fault_index": fault_index,
        "score_matrix": score_matrix,
        "metric_keys": all_keys,
        "metadata": {
            "num_instructions": num_instructions,
            "num_b_plus": num_b_plus,
            "num_b_moins": num_b_moins,
            "pass_rate": pass_rate,
            "density": density,
            "fault_covered": is_fault_covered
        }
    }

def process_project(dataset_name: str, project_name: str) -> Optional[Dict[str, Any]]:
    """Helper to process all versions of a project and return results."""
    dataset_root = fm.get_project_path(dataset_name, project_name)
    if not dataset_root:
        return None
    
    versions = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))], key=natural_sort_key)
    
    # Using ProcessPoolExecutor to parallelize version processing
    version_args = [(os.path.join(dataset_root, v), v) for v in versions]
    results = {}
    
    with ProcessPoolExecutor() as executor:
        for v_name, res in executor.map(process_version_wrapper, version_args):
            if res:
                results[v_name] = res
                
    return results

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="SBFL Optimized CLI")
    subparsers = parser.add_subparsers(dest="service")
    
    # Version service
    version_parser = subparsers.add_parser("version")
    version_parser.add_argument("-examscore", action="store_true")
    version_parser.add_argument("-scores", action="store_true")
    version_parser.add_argument("-info", action="store_true")
    version_parser.add_argument("-o", "--output", help="Export output to a CSV file")
    version_parser.add_argument("dataset")
    version_parser.add_argument("project")
    version_parser.add_argument("version")
    
    # Project service
    project_parser = subparsers.add_parser("project")
    project_parser.add_argument("-examscore", action="store_true")
    project_parser.add_argument("-overall", action="store_true")
    project_parser.add_argument("-info", action="store_true")
    project_parser.add_argument("-o", "--output", help="Export output to a CSV file")
    project_parser.add_argument("dataset")
    project_parser.add_argument("project")

    # Dataset service
    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument("-overall", action="store_true")
    dataset_parser.add_argument("-info", action="store_true")
    dataset_parser.add_argument("-o", "--output", help="Export output to a CSV file")
    dataset_parser.add_argument("dataset")

    args = parser.parse_args()
    if not args.service:
        parser.print_help()
        return

    if args.service == "version":
        dataset_root = fm.get_project_path(args.dataset, args.project)
        if not dataset_root:
            logger.error(f"Project path not found for {args.dataset}/{args.project}")
            return
        version_path = os.path.join(dataset_root, args.version)
        results = process_version(version_path)
        if results:
            with capture_to_csv(args.output):
                if args.info: print_version_info(args.version, results)
                if args.examscore: print_exam_scores(args.version, results)
                if args.scores: print_instruction_scores_table(args.version, results)

    elif args.service == "project":
        results = process_project(args.dataset, args.project)
        if not results:
            logger.error(f"Project {args.project} not found or has no versions.")
            return

        with capture_to_csv(args.output):
            if args.info: print_project_info(args.project, results)
            if args.examscore: print_project_exam_summary(args.project, results)
            if args.overall: print_overall_exam_scores(args.project, results)

    elif args.service == "dataset":
        dataset_name = args.dataset.lower()
        if dataset_name not in fm.CONFIG:
            logger.error(f"Dataset {dataset_name} not found in config.")
            return
        
        projects = fm.CONFIG[dataset_name]["projects"]
        project_averages = {}
        dataset_results = {}
        
        for project in projects:
            logger.info(f"Processing project {project}...")
            results = process_project(dataset_name, project)
            if results:
                if args.overall:
                    avg_map, _ = calculate_overall_averages(results)
                    project_averages[project] = avg_map
                if args.info:
                    dataset_results[project] = results
                
        with capture_to_csv(args.output):
            if args.info:
                print_dataset_info(dataset_name, dataset_results)
            if args.overall:
                print_dataset_overall_summary(dataset_name, project_averages)

if __name__ == "__main__":
    main()
