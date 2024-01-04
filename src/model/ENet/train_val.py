import logging

import torch
import torchvision.transforms as standard_transforms
from omegaconf import DictConfig
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import trange

# from tensorboardX import SummaryWriter
from ...utils import Timer, calculate_mean_iu
from .model import ENet

# from utils import *

# exp_name = cfg.TRAIN.EXP_NAME
# log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
# writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

log = logging.getLogger(__name__)


def ENet_main(
    cfg: DictConfig,
    restore_transform: standard_transforms.Compose,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> None:
    # Instantiating ENet
    log.info('Instantiating ENet')
    net = []

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
    criterion = torch.nn.BCEWithLogitsLoss().cuda()  # Binary Classification
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
    mean_iu = validate(
        val_loader=val_loader,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        epoch=-1,
        restore=restore_transform,
    )
    log.info(f'Validating model before training\tMean IU: {mean_iu:.4f}')

    # Training-validation loop
    log.info('Starting training-validation loop')
    for epoch in trange(cfg.model.train.max_epoch):
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
        mean_iu = validate(
            val_loader=val_loader,
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            restore=restore_transform,
        )
        _t['val time'].toc(average=False)
        validation_time = _t['val time'].diff

        log.debug(
            f'Epoch: {epoch}\t Training Time [s]: {training_time:.2f}\t Validation Time [s]: {validation_time:.2f}\t Mean IU: {mean_iu:.4f}'  # noqa: E501
        )
    log.info('Training-validation loop finished')

    # Validate the model after training
    mean_iu = validate(
        val_loader=val_loader,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        epoch=cfg.model.train.max_epoch,
        restore=restore_transform,
    )
    log.info(f'Validating model after training \tMean IU: {mean_iu:.4f}')


def train(train_loader, net, criterion, optimizer, epoch):
    for _i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    iou_ = 0.0
    for _vi, data in enumerate(val_loader, 0):
        inputs, labels = data

        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
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

    mean_iu = iou_ / len(val_loader)

    net.train()
    criterion.cuda()

    # print('[mean iu %.4f]' % (mean_iu))
    return mean_iu
