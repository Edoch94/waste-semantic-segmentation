"""
This is an example that uses hydra
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='../configs/project', config_name='main', version_base=None)
def function(cfg: DictConfig):
    """A function"""
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    function()
