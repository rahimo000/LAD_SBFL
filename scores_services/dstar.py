import numpy as np

from .rounding import format_scores

STAR_EXPONENT = 2


def dstar(p, n):
    """Vectorized D-Star with exact rounding (2 decimals).

    raw = ef**STAR_EXPONENT / (ep + nf), normalized as raw / (raw + 1).
    A zero denominator scores 1.0 when ef > 0, else 0.0.
    Already bounded in [0, 1].
    """
    n_e = np.sum(n, axis=0).astype(float)
    p_e = np.sum(p, axis=0).astype(float)
    total_f = n.shape[0]
    nf = total_f - n_e
    denom = p_e + nf

    with np.errstate(divide='ignore', invalid='ignore'):
        safe_denom = np.where(denom == 0, 1.0, denom)
        raw = np.where(denom == 0,
                       np.where(n_e > 0, np.inf, 0.0),
                       (n_e ** STAR_EXPONENT) / safe_denom)
        scores = np.where(np.isinf(raw), 1.0, raw / (raw + 1))

    return format_scores(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 2)
