import numpy as np

def Tarontula(p, n):
    """Vectorized Tarantula with exact rounding."""
    n_e = np.sum(n, axis=0)
    p_e = np.sum(p, axis=0)
    total_f, total_p = n.shape[0], p.shape[0]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f_ratio = n_e / total_f
        p_ratio = p_e / total_p
        scores = f_ratio / (p_ratio + f_ratio)
    
    return np.round(np.nan_to_num(scores), 2)
