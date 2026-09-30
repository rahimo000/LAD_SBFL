import os
import argparse
import numpy as np
import fileManagment as fm
import scores_services as ss
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor
from scores_services.metric_service import calculate_all_metrics
from scores_services.selection import Selection, default_selection, EVAL_RESULT_KEYS
from evaluation_services.exam_score import exam_score_for_eval
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

def process_version_wrapper(args: Tuple[str, str, Selection]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Wrapper for multiprocessing support."""
    version_path, version_name, selection = args
    res = process_version(version_path, selection=selection)
    return version_name, res

def process_version(version_path: str, selection: Optional[Selection] = None) -> Optional[Dict[str, Any]]:
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

    # Vectorized Metric Calculation (only selected metrics)
    if selection is None:
        selection = default_selection()
    try:
        score_matrix, all_keys = calculate_all_metrics(p, n, metrics=selection.metrics)
    except Exception as e:
        logger.error(f"Error calculating metrics for {version_path}: {e}")
        return None

    # Evaluation (only selected evaluations)
    exam_data = {}
    for eval_key in selection.exam_evals:
        exam_data[EVAL_RESULT_KEYS[eval_key]] = exam_score_for_eval(score_matrix, fault_index, all_keys, eval_key)
    topk_results = get_version_topk_results(score_matrix, fault_index, all_keys, evals=selection.topk_evals)
    
    # Metadata for -info
    num_instructions = p.shape[1]
    num_b_plus = p.shape[0]
    num_b_moins = n.shape[0]
    
    # Choice infos: Pass rate, Density, and Fault coverage status
    pass_rate = num_b_plus / (num_b_plus + num_b_moins) if (num_b_plus + num_b_moins) > 0 else 0
    density = (np.sum(p) + np.sum(n)) / ((num_b_plus + num_b_moins) * num_instructions) if num_instructions > 0 else 0
    is_fault_covered = (np.sum(n[:, fault_index]) > 0)

    return {
        "best": exam_data.get('best', {}),
        "worst": exam_data.get('worst', {}),
        "lexical": exam_data.get('lexical', {}),
        "reverse": exam_data.get('reverse', {}),
        "topk": topk_results,
        "selection": {"metrics": selection.metrics, "evals": selection.evals},
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

def process_project(dataset_name: str, project_name: str, selection: Optional[Selection] = None) -> Optional[Dict[str, Any]]:
    """Helper to process all versions of a project and return results."""
    if selection is None:
        selection = default_selection()
    dataset_root = fm.get_project_path(dataset_name, project_name)
    if not dataset_root:
        return None

    versions = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))], key=natural_sort_key)

    # Using ProcessPoolExecutor to parallelize version processing
    version_args = [(os.path.join(dataset_root, v), v, selection) for v in versions]
    results = {}
    
    with ProcessPoolExecutor() as executor:
        for v_name, res in executor.map(process_version_wrapper, version_args):
            if res:
                results[v_name] = res
                
    return results

def run_version_cmd(dataset: str, project: str, version: str,
                   examscore: bool = False, scores: bool = False,
                   info: bool = False, output: Optional[str] = None,
                   selection: Optional[Selection] = None) -> bool:
    """Runs the version service. Shared by the CLI and the interactive TUI."""
    if selection is None:
        selection = default_selection()
    dataset_root = fm.get_project_path(dataset, project)
    if not dataset_root:
        logger.error(f"Project path not found for {dataset}/{project}")
        return False
    version_path = os.path.join(dataset_root, version)
    results = process_version(version_path, selection=selection)
    if results is None:
        logger.error(f"Could not process version {version} of {dataset}/{project}")
        return False
    if output and output.lower().endswith((".xlsx", ".xls")):
        logger.error("Excel export (.xlsx) is only supported for: dataset -overall")
        return False
    with capture_to_csv(output):
        if info: print_version_info(version, results)
        if examscore: print_exam_scores(version, results, selection=selection)
        if scores: print_instruction_scores_table(version, results)
    return True

def run_project_cmd(dataset: str, project: str,
                    examscore: bool = False, overall: bool = False,
                    info: bool = False, output: Optional[str] = None,
                    selection: Optional[Selection] = None) -> bool:
    """Runs the project service. Shared by the CLI and the interactive TUI."""
    if selection is None:
        selection = default_selection()
    results = process_project(dataset, project, selection=selection)
    if not results:
        logger.error(f"Project {project} not found or has no versions.")
        return False
    if output and output.lower().endswith((".xlsx", ".xls")):
        logger.error("Excel export (.xlsx) is only supported for: dataset -overall")
        return False
    with capture_to_csv(output):
        if info: print_project_info(project, results)
        if examscore: print_project_exam_summary(project, results, selection=selection)
        if overall: print_overall_exam_scores(project, results, selection=selection)
    return True

def run_dataset_cmd(dataset: str,
                    overall: bool = False, info: bool = False,
                    output: Optional[str] = None,
                    selection: Optional[Selection] = None) -> bool:
    """Runs the dataset service. Shared by the CLI and the interactive TUI."""
    if selection is None:
        selection = default_selection()
    dataset_name = dataset.lower()
    if dataset_name not in fm.CONFIG:
        logger.error(f"Dataset {dataset_name} not found in config.")
        return False

    want_excel = bool(output and output.lower().endswith((".xlsx", ".xls")))
    if want_excel and not overall:
        logger.error("Excel export (.xlsx) is only supported for: dataset -overall")
        return False

    projects = fm.CONFIG[dataset_name]["projects"]
    project_averages = {}
    dataset_results = {}

    for project in projects:
        logger.info(f"Processing project {project}...")
        results = process_project(dataset_name, project, selection=selection)
        if results:
            if overall:
                avg_map, _ = calculate_overall_averages(results, selection=selection)
                project_averages[project] = avg_map
            if info:
                dataset_results[project] = results

    if want_excel:
        from printing_service.excel_export_service import export_dataset_overall
        if info:
            print_dataset_info(dataset_name, dataset_results)
        if overall:
            print_dataset_overall_summary(dataset_name, project_averages, selection=selection)
            try:
                real_path = export_dataset_overall(output, dataset_name, project_averages, selection=selection)
                print(f"Exported results to {real_path}")
            except Exception as e:
                logger.error(f"Error exporting to Excel: {e}")
                return False
        return True

    with capture_to_csv(output):
        if info:
            print_dataset_info(dataset_name, dataset_results)
        if overall:
            print_dataset_overall_summary(dataset_name, project_averages, selection=selection)
    return True

def _selection_from_args(parser, args) -> Selection:
    """Builds a Selection from --metrics/--evals (exits with usage error if invalid)."""
    from scores_services.selection import parse_metrics, parse_evals
    try:
        return Selection(metrics=parse_metrics(args.metrics),
                         evals=parse_evals(args.evals))
    except ValueError as e:
        parser.error(str(e))

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="SBFL Optimized CLI")
    subparsers = parser.add_subparsers(dest="service")
    
    # Version service
    version_parser = subparsers.add_parser("version")
    version_parser.add_argument("-examscore", action="store_true")
    version_parser.add_argument("-scores", action="store_true")
    version_parser.add_argument("-info", action="store_true")
    version_parser.add_argument("--metrics",
                                help="Comma-separated score metrics "
                                     "(default: all). E.g. --metrics tar,och,jac")
    version_parser.add_argument("--evals",
                                help="Comma-separated evaluations "
                                     "(default: all). E.g. --evals oexam,pexam")
    version_parser.add_argument("-o", "--output", help="Export file (.csv), always saved under output/")
    version_parser.add_argument("dataset")
    version_parser.add_argument("project")
    version_parser.add_argument("version")
    
    # Project service
    project_parser = subparsers.add_parser("project")
    project_parser.add_argument("-examscore", action="store_true")
    project_parser.add_argument("-overall", action="store_true")
    project_parser.add_argument("-info", action="store_true")
    project_parser.add_argument("--metrics",
                                help="Comma-separated score metrics "
                                     "(default: all). E.g. --metrics tar,och,jac")
    project_parser.add_argument("--evals",
                                help="Comma-separated evaluations "
                                     "(default: all). E.g. --evals oexam,pexam")
    project_parser.add_argument("-o", "--output", help="Export file (.csv), always saved under output/")
    project_parser.add_argument("dataset")
    project_parser.add_argument("project")

    # Dataset service
    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument("-overall", action="store_true")
    dataset_parser.add_argument("-info", action="store_true")
    dataset_parser.add_argument("--metrics",
                                help="Comma-separated score metrics "
                                     "(default: all). E.g. --metrics tar,och,jac")
    dataset_parser.add_argument("--evals",
                                help="Comma-separated evaluations "
                                     "(default: all). E.g. --evals oexam,pexam")
    dataset_parser.add_argument("-o", "--output",
                                help="Export path, always saved under output/ "
                                     "(.csv for console tables, .xlsx for the formatted "
                                     "dataset overall workbook)")
    dataset_parser.add_argument("dataset")

    args = parser.parse_args()
    if not args.service:
        # No subcommand: launch the interactive TUI (unless non-interactive).
        import sys
        try:
            import questionary  # noqa: F401
        except ImportError:
            parser.print_help()
            print("\nTip: install the TUI dependencies with: pip install questionary rich")
            return
        if not sys.stdin.isatty():
            parser.print_help()
            return
        try:
            from tui import run_tui
        except ImportError as e:
            logger.error(f"Could not load the TUI: {e}")
            return
        try:
            run_tui()
        except KeyboardInterrupt:
            print("\nBye!")
        return

    if args.service == "version":
        run_version_cmd(args.dataset, args.project, args.version,
                        examscore=args.examscore, scores=args.scores,
                        info=args.info, output=args.output,
                        selection=_selection_from_args(parser, args))

    elif args.service == "project":
        run_project_cmd(args.dataset, args.project,
                        examscore=args.examscore, overall=args.overall,
                        info=args.info, output=args.output,
                        selection=_selection_from_args(parser, args))

    elif args.service == "dataset":
        run_dataset_cmd(args.dataset, overall=args.overall,
                        info=args.info, output=args.output,
                        selection=_selection_from_args(parser, args))

if __name__ == "__main__":
    main()
