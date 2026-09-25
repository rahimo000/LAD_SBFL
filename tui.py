"""Interactive TUI for the SBFL tool (questionary + rich).

Launched automatically by `python main.py` with no subcommand.
All menus are data-driven: datasets/projects come from config.json,
versions are listed from the dataset directory on disk.
Execution reuses the run_*_cmd helpers from main.py, so the TUI
always stays in parity with the CLI.
"""
import os
from typing import Dict, List, Optional

import questionary
from rich.console import Console

import fileManagment as fm
from main import natural_sort_key, run_dataset_cmd, run_project_cmd, run_version_cmd

console = Console()

BACK = "< Back"
QUIT = "< Quit>"


def _ask_single(message: str, choices: List[str]) -> Optional[str]:
    """Single-choice menu. Returns None on abort (Ctrl+C) or Quit."""
    answer = questionary.select(message, choices=choices, use_arrow_keys=True).ask()
    if answer is None or answer == QUIT:
        return None
    if answer == BACK:
        return BACK
    return answer


def _ask_flags(message: str, options: List[str]) -> Optional[Dict[str, bool]]:
    """Checkbox menu for report flags. Returns None if nothing selected."""
    selected = questionary.checkbox(message, choices=options).ask()
    if selected is None:  # aborted with Ctrl+C
        raise KeyboardInterrupt
    if not selected:
        console.print("[yellow]Pick at least one report option.[/yellow]")
        return None
    return {opt: (opt in selected) for opt in options}


def _ask_output(excel_hint: bool = False) -> Optional[str]:
    """Optional export path. None means console only."""
    export = questionary.confirm("Export the output?", default=False).ask()
    if export is None:  # aborted with Ctrl+C
        raise KeyboardInterrupt
    if not export:
        return None
    hint = ".csv, or .xlsx for a formatted dataset-overall workbook" if excel_hint else ".csv"
    path = questionary.text(f"Export file path ({hint}, empty = console only):").ask()
    if path is None:  # aborted with Ctrl+C
        raise KeyboardInterrupt
    if not path.strip():
        return None
    return path.strip()


def _ask_no_topk() -> bool:
    """Whether to skip Top-K evaluation (EXAM scores only)."""
    answer = questionary.confirm("Skip Top-K evaluation (EXAM scores only)?", default=False).ask()
    if answer is None:  # aborted with Ctrl+C
        raise KeyboardInterrupt
    return bool(answer)


def build_command(service: str, flags: Dict[str, bool], dataset: str,
                  project: Optional[str] = None, version: Optional[str] = None,
                  output: Optional[str] = None, no_topk: bool = False) -> str:
    """Builds the equivalent CLI command string for preview (pure function)."""
    parts = ["python main.py", service]
    parts += [f"-{name}" for name, enabled in flags.items() if enabled]
    if no_topk:
        parts.append("--no-topk")
    parts.append(dataset)
    if project:
        parts.append(project)
    if version:
        parts.append(version)
    if output:
        parts += ["-o", output]
    return " ".join(parts)


def _confirm_and_run(command: str, runner) -> bool:
    """Shows the equivalent CLI command, confirms, then runs it."""
    console.print(f"\n[bold cyan]Equivalent command:[/bold cyan] [green]{command}[/green]")
    if not questionary.confirm("Run it?", default=True).ask():
        return False
    with console.status("[bold green]Processing...[/bold green]"):
        ok = runner()
    if not ok:
        console.print("[red]The run failed — see the log messages above.[/red]")
    return ok


def _list_versions(dataset: str, project: str) -> List[str]:
    """Lists available versions for a project from disk (natural sort)."""
    root = fm.get_project_path(dataset, project)
    if not root or not os.path.isdir(root):
        return []
    return sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
        key=natural_sort_key,
    )


def _version_flow(dataset: str, project: str) -> None:
    versions = _list_versions(dataset, project)
    if not versions:
        console.print(f"[red]No versions found for {dataset}/{project}.[/red]")
        return
    while True:
        version = _ask_single(f"Version of {project}?", versions + [BACK])
        if version is None:
            raise KeyboardInterrupt
        if version == BACK:
            return
        while True:
            flags = _ask_flags("Version reports?", ["examscore", "scores", "info"])
            if flags is not None:
                break
        no_topk = _ask_no_topk() if flags["examscore"] else False
        output = _ask_output()
        cmd = build_command("version", flags, dataset, project, version, output, no_topk)
        _confirm_and_run(cmd, lambda: run_version_cmd(
            dataset, project, version, examscore=flags["examscore"],
            scores=flags["scores"], info=flags["info"], output=output,
            no_topk=no_topk))


def _project_flow(dataset: str) -> None:
    projects = fm.CONFIG[dataset]["projects"]
    while True:
        project = _ask_single(f"Project in {dataset}?", projects + [BACK])
        if project is None:
            raise KeyboardInterrupt
        if project == BACK:
            return
        while True:
            action = _ask_single(f"What for {project}?",
                                 ["Single version", "Whole project (all versions)", BACK])
            if action is None:
                raise KeyboardInterrupt
            if action == BACK:
                break
            if action == "Single version":
                _version_flow(dataset, project)
            else:
                while True:
                    flags = _ask_flags("Project reports?", ["examscore", "overall", "info"])
                    if flags is not None:
                        break
                no_topk = _ask_no_topk() if (flags["examscore"] or flags["overall"]) else False
                output = _ask_output()
                cmd = build_command("project", flags, dataset, project, None, output, no_topk)
                _confirm_and_run(cmd, lambda: run_project_cmd(
                    dataset, project, examscore=flags["examscore"],
                    overall=flags["overall"], info=flags["info"], output=output,
                    no_topk=no_topk))


def _dataset_flow() -> None:
    datasets = sorted(fm.CONFIG.keys())
    while True:
        dataset = _ask_single("Dataset?", datasets + [BACK])
        if dataset is None:
            raise KeyboardInterrupt
        if dataset == BACK:
            return
        while True:
            scope = _ask_single(f"What for {dataset}?",
                                ["Single project...", "Whole dataset (all projects)", BACK])
            if scope is None:
                raise KeyboardInterrupt
            if scope == BACK:
                break
            if scope == "Single project...":
                _project_flow(dataset)
            else:
                console.print("[yellow]Whole-dataset runs process every project "
                              "and may take a while.[/yellow]")
                while True:
                    flags = _ask_flags("Dataset reports?", ["overall", "info"])
                    if flags is not None:
                        break
                no_topk = _ask_no_topk() if flags["overall"] else False
                output = _ask_output(excel_hint=flags["overall"])
                cmd = build_command("dataset", flags, dataset, None, None, output, no_topk)
                _confirm_and_run(cmd, lambda: run_dataset_cmd(
                    dataset, overall=flags["overall"],
                    info=flags["info"], output=output, no_topk=no_topk))


def run_tui() -> None:
    """Main TUI loop: pick a scope, drill down, run, repeat until quit."""
    console.print("[bold]SBFL Explorer[/bold] — pick what to analyze "
                  "(equivalent CLI command is always shown).")
    try:
        _dataset_flow()
    except KeyboardInterrupt:
        raise
    console.print("Bye!")
