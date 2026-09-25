import numpy as np


def kulczynski2(p, n):
    """Vectorized Kulczynski2 with exact rounding (2 decimals).

    Kulczynski2 = (1/2) * [ ef / (ef + nf) + ef / (ef + ep) ].
    Already bounded in [0, 1]; 0/0 terms are treated as 0.
    """
    n_e = np.sum(n, axis=0).astype(float)
    p_e = np.sum(p, axis=0).astype(float)
    total_f = n.shape[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        recall_f = n_e / total_f
        precision = n_e / (n_e + p_e)
        scores = 0.5 * (recall_f + precision)

    return np.round(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), 2)
