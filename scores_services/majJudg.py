import numpy as np

def calculate_majority_judgment_score(score_matrix: np.ndarray, metric_indices: list) -> np.ndarray:
    """Vectorized MJ: Pick lower-middle element (n//2 - 1) from sorted row scores."""
    subset = score_matrix[:, metric_indices]
    num_metrics = subset.shape[1]
    target_idx = (num_metrics // 2) - 1
    
    # Sort each row independently to pick the correct index
    sorted_subset = np.sort(subset, axis=1)
    return sorted_subset[:, target_idx]
