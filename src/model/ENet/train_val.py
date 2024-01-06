import logging
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import optim
from torch.nn.modules.loss import BCEWithLogitsLoss, _Loss
from torch.nn.modules.module import Module
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.transforms import Compose
from tqdm.auto import trange

from ...utils import Timer, calculate_mean_iu
from .model import ENet

log = logging.getLogger(__name__)


def ENet_main(
    cfg: DictConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    restore_transform: standard_transforms.Compose,
) -> None:
    checkpoint_dir = Path(HydraConfig.get().runtime.output_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(HydraConfig.get().runtime.output_dir) / 'tensorboard'
    writer = SummaryWriter(tensorboard_dir)

    # Choose the training strategy: train complete ENet or only encoder
    if cfg.model.train.stage.encoder.totrain:
        net = ENet(cfg=cfg, only_encode=False)
        if cfg.model.train.stage.encoder.pretrained != '':
            encoder_weight = torch.load(cfg.model.train.stage.encoder.pretrained)
            del encoder_weight['classifier.bias']
            del encoder_weight['classifier.weight']
            # pdb.set_trace()
            net.encoder.load_state_dict(encoder_weight)
        log.info('Chosen training strategy: train complete ENet')
    else:
        net = ENet(cfg=cfg, only_encode=True)
        log.info('Chosen training strategy: train only encoder')

    # Model parallelization
    if len(cfg.gpu) > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            module=net, device_ids=cfg.gpu
        ).cuda()
        log.info('Using DistributedDataParallel')
    else:
        net = net.cuda()
        log.info('Using GPU')

    # Setting mode to train
    net.train()
    log.info('Setting mode to train')

    # Setting loss function
    criterion = BCEWithLogitsLoss().cuda()  # Binary Classification
    log.info(f'Setting loss function: {criterion.__class__.__name__}')

    # Setting optimizer
    optimizer = optim.Adam(
        net.parameters(),
        lr=cfg.model.train.learning_rate.lr,
        weight_decay=cfg.model.train.learning_rate.weight_decay,
    )
    log.info(f'Setting optimizer: {optimizer.__class__.__name__}')

    # Setting scheduler
    scheduler = StepLR(
        optimizer,
        step_size=cfg.model.train.learning_rate.num_epoch_lr_decay,
        gamma=cfg.model.train.learning_rate.lr_decay,
    )
    log.info(f'Setting scheduler: {scheduler.__class__.__name__}')

    # Instantiating timer
    _t = {'train time': Timer(), 'val time': Timer()}

    # Validate the model before training
    mean_iu, mean_loss = validate(
        val_loader=val_loader,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        epoch=-1,
        restore=restore_transform,
    )
    log.info(
        f'Validating model before training\t Mean IU: {mean_iu:.4f}\t Mean Loss: {mean_loss:.4f}'  # noqa: E501
    )
    torch.save(
        net.state_dict(),
        f'{checkpoint_dir}/{cfg.model.name}_{cfg.model.version}__initial.pth',
    )

    writer.add_scalar('Validation/Mean IU', mean_iu, -1)
    writer.add_scalar('Validation/Mean Loss', mean_loss, -1)

    # Training-validation loop
    log.info('Starting training-validation loop')
    for epoch in trange(cfg.model.train.max_epoch):
        current_epoch = epoch + 1

        _t['train time'].tic()
        train(
            train_loader=train_loader,
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )
        _t['train time'].toc(average=False)
        training_time = _t['train time'].diff

        _t['val time'].tic()
        mean_iu, mean_loss = validate(
            val_loader=val_loader,
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            restore=restore_transform,
        )
        _t['val time'].toc(average=False)
        validation_time = _t['val time'].diff

        if (current_epoch) % 10 == 0:
            torch.save(
                net.state_dict(),
                f'{checkpoint_dir}/{cfg.model.name}_{cfg.model.version}__epoch_{current_epoch}.pth',
            )
            # mlflow.log_artifact(f'{HydraConfig.get().runtime.output_dir}_epoch_{epoch}.pth')  # noqa: E501

        log.debug(
            f'Epoch: {epoch}\t Training Time [s]: {training_time:.2f}\t Validation Time [s]: {validation_time:.2f}\t Mean IU: {mean_iu:.4f}\t Mean Loss: {mean_loss:.4f}'  # noqa: E501
        )
        writer.add_scalar('Validation/Mean IU', mean_iu, epoch)
        writer.add_scalar('Validation/Mean Loss', mean_loss, epoch)
    log.info('Training-validation loop finished')

    # Validate the model after training
    mean_iu, mean_loss = validate(
        val_loader=val_loader,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        epoch=cfg.model.train.max_epoch,
        restore=restore_transform,
    )
    torch.save(
        net.state_dict(),
        f'{checkpoint_dir}/{cfg.model.name}_{cfg.model.version}__final.pth',
    )

    log.info(
        f'Validating model after training \t Mean IU: {mean_iu:.4f}\t Mean Loss: {mean_loss:.4f}'  # noqa: E501
    )
    writer.add_scalar('Validation/Mean IU', mean_iu, cfg.model.train.max_epoch)
    writer.add_scalar('Validation/Mean Loss', mean_loss, cfg.model.train.max_epoch)

    writer.flush()
    writer.close()


def train(
    train_loader: DataLoader,
    net: Module,
    criterion: _Loss,
    optimizer: Optimizer,
    epoch: int,
) -> None:
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()


def validate(
    val_loader: DataLoader,
    net: Module,
    criterion: _Loss,
    optimizer: Optimizer,
    epoch: int,
    restore: Compose,
) -> tuple[float, float]:
    with net.eval() and criterion.cpu() and torch.no_grad():
        iou_ = 0.0
        loss_ = 0.0
        for inputs, labels in val_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            # for binary classification
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            # for multi-classification ???

            iou_ += calculate_mean_iu(
                predictions=[outputs.squeeze_(1).data.cpu().numpy()],
                gts=[labels.data.cpu().numpy()],
                num_classes=2,
            )
            loss_ += criterion(outputs, labels.float())

        mean_iu = iou_ / len(val_loader)
        mean_loss = loss_ / len(val_loader)

    return mean_iu, mean_loss
