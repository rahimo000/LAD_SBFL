import numpy as np

from .rounding import format_scores


def topkscore(p, n):
    """Vectorized TopKScore with exact rounding (2 decimals).

    raw = ef - ep / (total_p + 1) + 1, normalized as raw / (total_f + 1).
    Already bounded in (0, 1].
    """
    n_e = np.sum(n, axis=0).astype(float)
    p_e = np.sum(p, axis=0).astype(float)
    total_f = n.shape[0]
    total_p = p.shape[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        raw = n_e - p_e / (total_p + 1) + 1
        scores = raw / (total_f + 1)

    return format_scores(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 2)
