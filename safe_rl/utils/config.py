import os
import yaml

def load_config(config_path):
    """
    Load a YAML config file, given a path relative to the project root.

    Args:
        config_path (str): Relative path to the config file from the project root,
                           e.g., 'configs/expected/ppo/ppo.yaml'

    Returns:
        dict: parsed configuration dictionary
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  
    root_dir = os.path.normpath(os.path.join(current_dir, '../'))
    cfg_path = os.path.join(root_dir, config_path)

    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    return config