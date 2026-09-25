import numpy as np


def zoltar(p, n):
    """Vectorized Zoltar with exact rounding (2 decimals).

    Zoltar = ef / (ef + nf + ep + (10000 * nf * ep) / ef).
    Already bounded in [0, 1]; ef == 0 scores 0 by convention
    (avoids division by ef).
    """
    n_e = np.sum(n, axis=0).astype(float)
    p_e = np.sum(p, axis=0).astype(float)
    total_f = n.shape[0]
    nf = total_f - n_e

    with np.errstate(divide='ignore', invalid='ignore'):
        penalty = np.where(n_e == 0, 0.0, (10000.0 * nf * p_e) / np.where(n_e == 0, 1.0, n_e))
        denom = n_e + nf + p_e + penalty
        scores = np.where(n_e == 0, 0.0, n_e / denom)

    return np.round(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), 2)
