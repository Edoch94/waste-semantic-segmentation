import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip

# ===============================img tranforms============================


class ComposeCustom(Compose):
    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCropCustom(RandomCrop):
    def __call__(self, img, mask):
        if self.padding and self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize(
                (tw, th), Image.NEAREST
            )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop(
            (x1, y1, x1 + tw, y1 + th)
        )


class CenterCropCustom(CenterCrop):
    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop(
            (x1, y1, x1 + tw, y1 + th)
        )


class RandomHorizontalFlipCustom(RandomHorizontalFlip):
    def __call__(self, img, mask):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(
                Image.FLIP_LEFT_RIGHT
            )
        return img, mask


class FreeScale:
    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size  # (h, w)
        self.interpolation = interpolation

    def __call__(self, img, mask):
        return img.resize(
            (self.size[1], self.size[0]), self.interpolation
        ), mask.resize(self.size, self.interpolation)


class Scale:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize(
                (ow, oh), Image.NEAREST
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize(
                (ow, oh), Image.NEAREST
            )


# ===============================label tranforms============================


class DeNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor:
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class ChangeLabel:
    def __init__(self, ori_label, new_label):
        self.ori_label = ori_label
        self.new_label = new_label

    def __call__(self, mask):
        mask[mask == self.ori_label] = self.new_label
        return mask
