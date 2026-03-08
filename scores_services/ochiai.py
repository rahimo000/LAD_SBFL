import numpy as np

def ochiai(p, n):
    """Vectorized Ochiai with exact rounding."""
    n_e = np.sum(n, axis=0)
    p_e = np.sum(p, axis=0)
    total_f = n.shape[0]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = n_e / np.sqrt(total_f * (n_e + p_e))
        
    return np.round(np.nan_to_num(scores), 2)
