from .apv import calculate_approval_voting_score
from .majJudg import calculate_majority_judgment_score
from .trantula import Tarontula
from .ochiai import ochiai
from .jaccard import jaccard
from .gp13 import gp13
from .op2 import op2
from .kulczynski2 import kulczynski2
from .zoltar import zoltar
from .ample import ample

# Central registry of active SBFL metrics
# Format: (key, function, display_name)
ACTIVE_METRICS = [
    ('tar', Tarontula, 'Tarantula'),
    ('och', ochiai, 'Ochiai'),
    ('jac', jaccard, 'Jaccard'),
    ('gp', gp13, 'GP13'),
    ('op2', op2, 'Op2'),
    ('kul2', kulczynski2, 'Kulczynski2'),
    ('zol', zoltar, 'Zoltar'),
    ('amp', ample, 'Ample'),
]

_METRIC_BY_KEY = {key: func for key, func, _ in ACTIVE_METRICS}

def get_all_metric_keys():
    """Returns a list of all active and aggregate metric keys in the preferred order."""
    return ['tar', 'och', 'jac', 'gp', 'op2', 'kul2', 'zol', 'amp', 'mj', 'apv']

def get_metric_function(key):
    """Returns the scoring function for a base metric key (None for aggregators)."""
    return _METRIC_BY_KEY.get(key)
