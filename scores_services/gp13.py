import numpy as np

from .rounding import format_scores


def gp13(p, n, decimals=2):
    """Vectorized GP13 with exact rounding (configurable decimals)."""
    n_e = np.sum(n, axis=0)
    p_e = np.sum(p, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        raw_score = n_e * (1 + (1 / ((2 * p_e) + n_e)))
        scores = raw_score / (n_e + 1)

    return format_scores(np.nan_to_num(scores), decimals)
