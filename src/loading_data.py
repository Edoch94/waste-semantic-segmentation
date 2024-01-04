import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

from .custom_transforms import (
    CenterCropCustom,
    ChangeLabel,
    ComposeCustom,
    DeNormalize,
    MaskToTensor,
    RandomCropCustom,
    RandomHorizontalFlipCustom,
    Scale,
)
from .resortit import resortit


def loading_data(
    normalization_mean: list[float],
    normalization_std: list[float],
    scaling_factor: float,
    padding: int,
    img_size: list[int],
    ignore_label: int,
    num_classes: int,
    train_batch_size: int,
    val_batch_size: int,
    num_workers_train: int,
    num_workers_val: int,
) -> tuple[DataLoader, DataLoader, standard_transforms.Compose]:
    train_simul_transform = ComposeCustom(
        [
            Scale(int(img_size[0] / scaling_factor)),
            RandomCropCustom(img_size, padding=padding),
            RandomHorizontalFlipCustom(),
        ]
    )
    val_simul_transform = ComposeCustom(
        [
            Scale(int(img_size[0] / scaling_factor)),
            CenterCropCustom(img_size),
        ]
    )
    img_transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=normalization_mean, std=normalization_std
            ),
        ]
    )
    target_transform = standard_transforms.Compose(
        [
            MaskToTensor(),
            ChangeLabel(
                ori_label=ignore_label,
                new_label=num_classes - 1,
            ),
        ]
    )
    restore_transform = standard_transforms.Compose(
        [
            DeNormalize(
                mean=normalization_mean,
                std=normalization_std,
            ),
            standard_transforms.ToPILImage(),
        ]
    )

    train_set = resortit(
        'train',
        simul_transform=train_simul_transform,
        transform=img_transform,
        target_transform=target_transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        num_workers=num_workers_train,
        shuffle=True,
    )

    val_set = resortit(
        'val',
        simul_transform=val_simul_transform,
        transform=img_transform,
        target_transform=target_transform,
    )
    val_loader = DataLoader(
        val_set, batch_size=val_batch_size, num_workers=num_workers_val, shuffle=False
    )

    return train_loader, val_loader, restore_transform
