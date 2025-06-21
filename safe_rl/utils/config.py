import os
import copy
import yaml
import torch
import torch.nn.functional as F

activations = {
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    'softplus': F.softplus,
    'tanh': F.tanh
}

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
        original_config = copy.deepcopy(config)

        if 'ac_kwargs' in config and 'activation' in config['ac_kwargs']:
            cfg_activation = config['ac_kwargs']['activation']
            if cfg_activation in activations:
                config['ac_kwargs']['activation'] = activations[cfg_activation]
            else:
                raise ValueError(f"Unknown activation: {cfg_activation}")
            
        if 'lagrange_kwargs' in config and 'activation' in config['lagrange_kwargs']:
            cfg_activation = config['lagrange_kwargs']['activation']
            if cfg_activation in activations:
                config['lagrange_kwargs']['activation'] = activations[cfg_activation]
            else:
                raise ValueError(f"Unknown activation: {cfg_activation}")

    return config, original_config

def get_device(device_type=None):
    try:
        if device_type is not None:
            device = torch.device(device_type)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available.")
            if device.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available.")
        else:
            raise RuntimeError("No device specified.")
    except Exception:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cpu")
        )
    return device