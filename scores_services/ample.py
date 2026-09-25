import numpy as np

from .rounding import format_scores


def ample(p, n):
    """Vectorized Ample with exact rounding (2 decimals).

    Ample = | ef / (ef + nf) - ep / (ep + np_) |.
    Already bounded in [0, 1]; 0/0 terms are treated as 0.
    """
    n_e = np.sum(n, axis=0).astype(float)
    p_e = np.sum(p, axis=0).astype(float)
    total_f = n.shape[0]
    total_p = p.shape[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        fail_rate = n_e / total_f
        pass_rate = p_e / total_p
        scores = np.abs(fail_rate - pass_rate)

    return format_scores(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 2)
