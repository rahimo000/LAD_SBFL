import numpy as np
import scores_services as ss

def _format_scores(scores, decimals=2):
    """Utility to match the original float("{:.2f}".format(x)) rounding exactly."""
    fmt = f"{{:.{decimals}f}}"
    return np.array([float(fmt.format(x)) for x in scores])

def Tarontula(p, n):
    n_e = np.sum(n, axis=0)
    p_e = np.sum(p, axis=0)
    total_f, total_p = n.shape[0], p.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        f_ratio = n_e / total_f
        p_ratio = p_e / total_p
        scores = f_ratio / (p_ratio + f_ratio)
    return _format_scores(np.nan_to_num(scores), 2)

def ochiai(p, n):
    n_e = np.sum(n, axis=0)
    p_e = np.sum(p, axis=0)
    total_f = n.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = n_e / np.sqrt(total_f * (n_e + p_e))
    return _format_scores(np.nan_to_num(scores), 2)

def jaccard(p, n):
    n_e = np.sum(n, axis=0)
    p_e = np.sum(p, axis=0)
    total_f = n.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = n_e / (p_e + total_f)
    return _format_scores(np.nan_to_num(scores), 2)

def gp13(p, n):
    n_e = np.sum(n, axis=0)
    p_e = np.sum(p, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_score = n_e * (1 + (1 / ((2 * p_e) + n_e)))
        scores = raw_score / (n_e + 1)
    return np.array([round(x, 3) for x in np.nan_to_num(scores)])

def op2(p, n):
    n_e = np.sum(n, axis=0).astype(float)
    p_e = np.sum(p, axis=0).astype(float)
    total_f = n.shape[0]
    total_p = p.shape[0]
    np_ = total_p - p_e
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = n_e - p_e / (p_e + np_ + 1)
        scores = (raw + 1) / (total_f + 1)
    return _format_scores(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 2)

def kulczynski2(p, n):
    n_e = np.sum(n, axis=0).astype(float)
    p_e = np.sum(p, axis=0).astype(float)
    total_f = n.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = 0.5 * ((n_e / total_f) + (n_e / (n_e + p_e)))
    return _format_scores(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), 2)

def zoltar(p, n):
    n_e = np.sum(n, axis=0).astype(float)
    p_e = np.sum(p, axis=0).astype(float)
    total_f = n.shape[0]
    nf = total_f - n_e
    with np.errstate(divide='ignore', invalid='ignore'):
        penalty = np.where(n_e == 0, 0.0, (10000.0 * nf * p_e) / np.where(n_e == 0, 1.0, n_e))
        denom = n_e + nf + p_e + penalty
        scores = np.where(n_e == 0, 0.0, n_e / denom)
    return _format_scores(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), 2)

def ample(p, n):
    n_e = np.sum(n, axis=0).astype(float)
    p_e = np.sum(p, axis=0).astype(float)
    total_f = n.shape[0]
    total_p = p.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.abs((n_e / total_f) - (p_e / total_p))
    return _format_scores(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 2)

def calculate_all_metrics(p: np.ndarray, n: np.ndarray) -> tuple:
    """Computes all metrics in a single optimized pass while preserving exact rounding."""
    all_keys = ss.get_all_metric_keys()
    num_instr = p.shape[1]
    
    score_matrix = np.zeros((num_instr, len(all_keys)))
    key_to_idx = {key: i for i, key in enumerate(all_keys)}

    # 1. Standard Metrics
    score_matrix[:, key_to_idx['tar']] = Tarontula(p, n)
    score_matrix[:, key_to_idx['och']] = ochiai(p, n)
    score_matrix[:, key_to_idx['jac']] = jaccard(p, n)
    score_matrix[:, key_to_idx['gp']] = gp13(p, n)
    score_matrix[:, key_to_idx['op2']] = op2(p, n)
    score_matrix[:, key_to_idx['kul2']] = kulczynski2(p, n)
    score_matrix[:, key_to_idx['zol']] = zoltar(p, n)
    score_matrix[:, key_to_idx['amp']] = ample(p, n)

    # 2. Aggregators
    active_indices = [key_to_idx[k] for k in ['tar', 'och', 'jac', 'gp', 'op2', 'kul2', 'zol', 'amp']]
    
    if 'apv' in key_to_idx:
        score_matrix[:, key_to_idx['apv']] = ss.calculate_approval_voting_score(score_matrix, active_indices)
    if 'mj' in key_to_idx:
        score_matrix[:, key_to_idx['mj']] = ss.calculate_majority_judgment_score(score_matrix, active_indices)
        
    return score_matrix, all_keys
