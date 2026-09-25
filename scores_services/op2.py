import numpy as np


def op2(p, n):
    """Vectorized normalized Op2 with exact rounding (2 decimals).

    Raw Op2 = ef - ep / (ep + np_ + 1) ranges in (-1, total_f], so it is
    normalized into [0, 1] as: (raw + 1) / (total_f + 1).
    """
    n_e = np.sum(n, axis=0).astype(float)
    p_e = np.sum(p, axis=0).astype(float)
    total_f = n.shape[0]
    total_p = p.shape[0]
    np_ = total_p - p_e

    with np.errstate(divide='ignore', invalid='ignore'):
        raw = n_e - p_e / (p_e + np_ + 1)
        scores = (raw + 1) / (total_f + 1)

    return np.round(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 2)
