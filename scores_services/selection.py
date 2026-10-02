"""Selectable score metrics and evaluation metrics (single source of truth).

Score metrics: 8 base keys (tar och jac gp op2 kul2 zol amp) plus the
aggregators mj/apv (computed over the *selected* base subset).
Evaluations: oexam pexam lex-exam rev-exam plus l/r-Top1/3/5.
`deltaexam` is derived (pexam - oexam) and auto-shown whenever oexam
and pexam are both selected.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

BASE_METRICS = ['tar', 'och', 'jac', 'gp', 'op2', 'kul2', 'zol', 'amp', 'dst', 'tks']
AGGREGATORS = ['mj', 'apv']

MIN_COMBO_SIZE = 3

EXAM_EVALS = ['oexam', 'pexam', 'lex-exam', 'rev-exam']
TOPK_EVALS = ['l-Top1', 'l-Top3', 'l-Top5', 'r-Top1', 'r-Top3', 'r-Top5']

# eval key -> exam case used by the evaluation services
EXAM_CASES = {
    'oexam': 'best',
    'pexam': 'worst',
    'lex-exam': 'lexical',
    'rev-exam': 'reverse_lexical',
}

# eval key -> key in the per-version results dict
EVAL_RESULT_KEYS = {
    'oexam': 'best',
    'pexam': 'worst',
    'lex-exam': 'lexical',
    'rev-exam': 'reverse',
}

# eval key -> key in the project-average maps
EVAL_AVG_KEYS = {
    'oexam': 'oexam',
    'pexam': 'pexam',
    'lex-exam': 'lexical',
    'rev-exam': 'reverse',
}

DELTA_LABEL = 'deltaexam'


def all_score_metrics() -> List[str]:
    from scores_services import get_all_metric_keys
    return get_all_metric_keys()


def all_evals() -> List[str]:
    return EXAM_EVALS + TOPK_EVALS


def _parse_list(spec: Optional[str], valid: List[str], what: str) -> List[str]:
    if spec is None or not spec.strip():
        return list(valid)
    chosen = [s.strip() for s in spec.split(',') if s.strip()]
    unknown = [c for c in chosen if c not in valid]
    if unknown:
        raise ValueError(f"Unknown {what}: {', '.join(unknown)}. Valid: {', '.join(valid)}")
    if not chosen:
        return list(valid)
    # Canonical order, duplicates removed.
    return [v for v in valid if v in chosen]


def parse_metrics(spec: Optional[str]) -> List[str]:
    return _parse_list(spec, all_score_metrics(), "score metrics")


def parse_evals(spec: Optional[str]) -> List[str]:
    return _parse_list(spec, all_evals(), "evaluations")


@dataclass
class Selection:
    """Ordered metric + eval subset driving calculation, tables and charts."""
    metrics: List[str] = field(default_factory=all_score_metrics)
    evals: List[str] = field(default_factory=all_evals)

    def __post_init__(self):
        if not self.metrics:
            raise ValueError("Select at least one score metric.")
        base_selected = [m for m in self.metrics if m in BASE_METRICS]
        for agg in AGGREGATORS:
            if agg in self.metrics and not base_selected:
                raise ValueError(f"'{agg}' needs at least one selected base metric.")
        if not self.evals:
            raise ValueError("Select at least one evaluation.")

    @property
    def base_metrics(self) -> List[str]:
        return [m for m in self.metrics if m in BASE_METRICS]

    @property
    def exam_evals(self) -> List[str]:
        return [e for e in self.evals if e in EXAM_EVALS]

    @property
    def topk_evals(self) -> List[str]:
        return [e for e in self.evals if e in TOPK_EVALS]

    @property
    def show_delta(self) -> bool:
        return 'oexam' in self.evals and 'pexam' in self.evals

    @property
    def needs_topk(self) -> bool:
        return bool(self.topk_evals)

    def cli_flags(self) -> List[str]:
        """Equivalent --metrics/--evals CLI flags (omitted when all selected)."""
        parts = []
        if self.metrics != all_score_metrics():
            parts += ["--metrics", ",".join(self.metrics)]
        if self.evals != all_evals():
            parts += ["--evals", ",".join(self.evals)]
        return parts


def default_selection() -> Selection:
    return Selection()


def parse_combo_metrics(spec: Optional[str]) -> List[str]:
    """Parses the base-metric pool for combination search.

    Aggregators (mj/apv) are rejected: they are auto-added to every combo.
    At least MIN_COMBO_SIZE base metrics are required.
    """
    chosen = _parse_list(spec, BASE_METRICS, "combo metrics")
    if len(chosen) < MIN_COMBO_SIZE:
        raise ValueError(
            f"Select at least {MIN_COMBO_SIZE} base metrics to combine "
            f"(got {len(chosen)}). Aggregators mj/apv are added automatically."
        )
    return chosen


def generate_combinations(base_metrics: List[str], min_size: int = MIN_COMBO_SIZE) -> List[tuple]:
    """All metric combinations of length >= min_size (canonical order kept)."""
    import itertools
    pool = [m for m in BASE_METRICS if m in base_metrics]
    combos = []
    for size in range(max(min_size, 1), len(pool) + 1):
        combos.extend(itertools.combinations(pool, size))
    return combos


def combo_label(combo) -> str:
    """Human-readable label: 'tar, och, jac'."""
    return ", ".join(combo)


def combo_slug(combo) -> str:
    """Filename-safe slug: 'tar_och_jac'."""
    return "_".join(combo)


def rank_columns(scores: Dict[str, float], higher_is_better: bool = False) -> Dict[str, int]:
    """Competition ranking (1, 2, 2, 4): best value gets rank 1, ties share it.

    Lower wins for EXAM-style scores, higher wins for Top-K counts.
    """
    ranks: Dict[str, int] = {}
    for metric, value in scores.items():
        if higher_is_better:
            better = sum(1 for v in scores.values() if v > value)
        else:
            better = sum(1 for v in scores.values() if v < value)
        ranks[metric] = better + 1
    return ranks
