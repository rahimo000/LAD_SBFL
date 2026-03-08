from .apv import calculate_approval_voting_score
from .majJudg import calculate_majority_judgment_score

# Central registry of active SBFL metrics
# Format: (key, function, display_name)
ACTIVE_METRICS = [
    ('tar', None, 'Tarantula'),
    ('och', None, 'Ochiai'),
    ('jac', None, 'Jaccard'),
    ('gp', None, 'GP13'),
]

def get_all_metric_keys():
    """Returns a list of all active and aggregate metric keys in the preferred order."""
    return ['tar', 'och', 'jac', 'gp', 'mj', 'apv']
