import numpy as np

from .rounding import format_scores


def jaccard(p, n, decimals=2):
    """Vectorized Jaccard with exact rounding (configurable decimals)."""
    n_e = np.sum(n, axis=0)
    p_e = np.sum(p, axis=0)
    total_f = n.shape[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = n_e / (p_e + total_f)

    return format_scores(np.nan_to_num(scores), decimals)
