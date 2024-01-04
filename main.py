"""This is the main script"""

import logging

import hydra
import torch
from icecream import ic
from omegaconf import DictConfig, OmegaConf

from src.loading_data import loading_data

# Disabled icecream
ic.disable()
log = logging.getLogger(__name__)


@hydra.main(config_path='./configs/project', config_name='main', version_base=None)
def main(cfg: DictConfig) -> None:
    """This is the main function"""
    ic(OmegaConf.to_yaml(cfg))
    log.info(
        f'Loading configuration file:\t{cfg.data.dataset.name}\t{cfg.model.name}\t{cfg.model.version}'  # noqa: E501
    )

    # Set GPU or CPU
    if cfg.gpu:
        torch.cuda.set_device(cfg.gpu[0])
        log.info(f'Using GPU: {cfg.gpu}')
    else:
        log.info('Using CPU')

    # Set benchmark to True to make the training reproducible
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, restore_transform = loading_data(
        normalization_mean=cfg.data.normalization_mean,
        normalization_std=cfg.data.normalization_std,
        scaling_factor=cfg.data.scaling_factor,
        padding=cfg.data.padding,
        img_size=cfg.data.img_size,
        ignore_label=cfg.data.ignore_label,
        num_classes=cfg.data.num_classes,
        train_batch_size=cfg.model.train.input.batch_size,
        val_batch_size=cfg.model.validation.input.batch_size,
        num_workers_train=cfg.model.train.num_workers,
        num_workers_val=cfg.model.validation.num_workers,
    )

    # Train and validate the model
    if cfg.model.name == 'ENet':
        log.info('Training and validating ENet')
        from src.model.ENet.train_val import ENet_main

        ENet_main(
            cfg=cfg,
            restore_transform=restore_transform,
            train_loader=train_loader,
            val_loader=val_loader,
        )


if __name__ == '__main__':
    main()
