import os
import json
import numpy as np
from typing import Optional, List, Dict

def load_config() -> Dict:
    """Loads dataset configuration from config.json."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not os.path.exists(config_path):
        # Fallback or default if config not found
        return {}
    with open(config_path, 'r') as f:
        return json.load(f)

CONFIG = load_config()

def get_project_path(dataset_name: str, project_name: str) -> Optional[str]:
    """Returns the full path to a project based on config.json."""
    ds_key = dataset_name.lower()
    if ds_key not in CONFIG:
        return None
    
    ds_info = CONFIG[ds_key]
    base_dir = ds_info['base_dir']
    
    # Check if base_dir exists locally, otherwise fallback to current directory
    if not os.path.exists(base_dir):
        base_dir = os.getcwd()
        
    return os.path.join(base_dir, project_name, ds_info['sub_dir'])

def readBFile(fpath: str) -> Optional[np.ndarray]:
    """Loads a binary coverage matrix efficiently using NumPy (dtype=int8, 2D)."""
    try:
        if not os.path.exists(fpath):
            return None
        # ndmin=2 ensures even 1-line files are treated as matrices
        return np.loadtxt(fpath, dtype=np.int8, ndmin=2)
    except Exception:
        return None
