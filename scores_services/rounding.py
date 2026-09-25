"""Canonical score rounding for all SBFL metrics (single source of truth)."""
import numpy as np


def format_scores(scores, decimals=2):
    """Rounds to match the original float("{:.2f}".format(x)) behavior exactly."""
    fmt = f"{{:.{decimals}f}}"
    return np.array([float(fmt.format(x)) for x in scores])
