import numpy as np

def gp13(p, n):
    """Vectorized GP13 with exact rounding (3 decimals)."""
    n_e = np.sum(n, axis=0)
    p_e = np.sum(p, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_score = n_e * (1 + (1 / ((2 * p_e) + n_e)))
        scores = raw_score / (n_e + 1)
        
    return np.round(np.nan_to_num(scores), 3)
