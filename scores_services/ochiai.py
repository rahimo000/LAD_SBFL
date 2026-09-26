import numpy as np

from .rounding import format_scores


def ochiai(p, n, decimals=2):
    """Vectorized Ochiai with exact rounding (configurable decimals)."""
    n_e = np.sum(n, axis=0)
    p_e = np.sum(p, axis=0)
    total_f = n.shape[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = n_e / np.sqrt(total_f * (n_e + p_e))

    return format_scores(np.nan_to_num(scores), decimals)
