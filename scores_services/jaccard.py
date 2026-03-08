import numpy as np

def jaccard(p, n):
    """Vectorized Jaccard with exact rounding."""
    n_e = np.sum(n, axis=0)
    p_e = np.sum(p, axis=0)
    total_f = n.shape[0]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = n_e / (p_e + total_f)
        
    return np.round(np.nan_to_num(scores), 2)
