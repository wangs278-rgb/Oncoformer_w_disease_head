from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from typing import List, Dict, Any

def load_config(config_dir: str = '../config',config_name: str = 'config_pretraining_paired', overrides: List[str] = []) -> Dict[str, Any]:
    with initialize(version_base=None, config_path=config_dir):
        config = compose(config_name=config_name, overrides=overrides)
    config = OmegaConf.to_container(config)
    return config
