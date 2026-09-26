"""Canonical score rounding for all SBFL metrics (single source of truth)."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_PRECISION = 2
MIN_PRECISION = 0
MAX_PRECISION = 10


def format_scores(scores, decimals=2):
    """Rounds to match the original float("{:.2f}".format(x)) behavior exactly."""
    fmt = f"{{:.{decimals}f}}"
    return np.array([float(fmt.format(x)) for x in scores])


def round_value(x: float, decimals: int) -> float:
    """Rounds a single scalar with the same half-even behavior as format_scores."""
    return float(f"{x:.{decimals}f}")


def validate_precision(value, name: str) -> int:
    """Validates a user-supplied precision (integer within range)."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer between {MIN_PRECISION} and {MAX_PRECISION}.")
    if not (MIN_PRECISION <= n <= MAX_PRECISION):
        raise ValueError(f"{name} must be an integer between {MIN_PRECISION} and {MAX_PRECISION}.")
    return n


@dataclass
class Precision:
    """Decimal precision per calculation phase.

    scores: instruction suspiciousness scores.
    version: per-version EXAM values (rank fractions).
    program: project averages (means over versions).
    overall: dataset grand averages (means over projects).
    Display in each table/workbook follows its phase precision.
    """
    scores: int = DEFAULT_PRECISION
    version: int = DEFAULT_PRECISION
    program: int = DEFAULT_PRECISION
    overall: int = DEFAULT_PRECISION

    def cli_flags(self) -> list:
        """Non-default values as CLI flags (for TUI previews)."""
        parts = []
        if self.scores == self.version == self.program == self.overall:
            if self.scores != DEFAULT_PRECISION:
                parts += ["--precision", str(self.scores)]
            return parts
        parts += ["--score-precision", str(self.scores),
                  "--version-precision", str(self.version),
                  "--program-precision", str(self.program),
                  "--overall-precision", str(self.overall)]
        return parts


def resolve_precision(global_prec: Optional[int] = None,
                      scores: Optional[int] = None,
                      version: Optional[int] = None,
                      program: Optional[int] = None,
                      overall: Optional[int] = None) -> Precision:
    """Builds a Precision from a global value with per-phase overrides."""
    base = DEFAULT_PRECISION if global_prec is None else validate_precision(global_prec, "--precision")
    return Precision(
        scores=validate_precision(scores, "--score-precision") if scores is not None else base,
        version=validate_precision(version, "--version-precision") if version is not None else base,
        program=validate_precision(program, "--program-precision") if program is not None else base,
        overall=validate_precision(overall, "--overall-precision") if overall is not None else base,
    )
