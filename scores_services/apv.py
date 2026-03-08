import numpy as np

def calculate_approval_voting_score(score_matrix: np.ndarray, metric_indices: list, threshold: float = 0.5) -> np.ndarray:
    """Vectorized APV: Count metrics >= threshold for each instruction."""
    subset = score_matrix[:, metric_indices]
    approvals = np.sum(subset >= threshold, axis=1)
    return approvals.astype(float)
