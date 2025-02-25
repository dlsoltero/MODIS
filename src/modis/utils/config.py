import sys
from itertools import product

import torch
from omegaconf import OmegaConf, DictConfig

from modis.utils.utils import fix_data_type

def validate_config(config: DictConfig) -> None:
    """Process and verifies a configuration"""
    error = None
    if config.training_mode not in ['semisupervised', 'supervised']:
        error = f'Invalid training mode: {config.training_mode}'
    elif config.device not in ['auto', 'cuda', 'cpu']:
        error = f'Invalid device: {config.device}'

    if error is not None:
        print(f'[-] Configuration error: {error}')
        sys.exit(1)

    if config.device == 'auto':
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(config_file: str) -> DictConfig:
    """Load and return configuration from config file"""
    config = OmegaConf.load(config_file)
    validate_config(config)
    return config

def read_config(config_file: str) -> DictConfig:
    """Decorator to read the config file and adjust it using input from the command-line"""

    def decorator(func):
        def wrapper():
            config = {}

            # Parse command line arguments
            for arg in sys.argv[1:]:
                if '=' not in arg or arg.count('=') != 1:
                    raise Exception(f'invalid argument: {arg}')
                param, param_value = arg.split('=')
                if param in config:
                    raise Exception(f'duplicated config param: {param}')
                config[param] = []
                if ',' in param_value:    
                    config[param].extend(param_value.split(','))
                else:
                    config[param].append(param_value)

            # Fix params data type
            config = {k:[fix_data_type(p) for p in v] for k,v in config.items()}

            if config:
                # Temporately update the configuration,
                # if there are multiple parameter configurations generate combinations and run each
                params = list(config.keys())
                param_values = list(product(*config.values()))
                for i, values in enumerate(param_values):
                    run_config = dict(zip(params, values))
                    orig_config = load_config(config_file)
                    
                    # Used to create saving folder structure
                    if len(param_values) > 1:
                        run_config['run_id'] = f"{i+1:03d}"

                    # Make config adjustment
                    for param in run_config:
                        if param != 'run_id' and OmegaConf.select(orig_config, param) is None:
                            raise Exception(f'invalid config parameter: {param}')
                        OmegaConf.update(orig_config, param, run_config[param])
                    
                    if len(param_values) > 1:
                        print(f'==> Multi-run {i+1}/{len(param_values)}')
                    print(f"Adjusted configuration: {run_config}\n")
                    func(orig_config)
            else:
                orig_config = load_config(config_file)
                func(orig_config)
        return wrapper
    return decorator