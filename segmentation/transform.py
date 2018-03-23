import collections
import math
import numbers
import random

import numpy as np
import torch
from PIL import Image


class Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


class ToParallel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        yield img
        for t in self.transforms:
            yield t(img)


class ToLabel(object):
    def __call__(self, inputs):
        # tensors = []
        # for i in inputs:
        # tensors.append(torch.from_numpy(np.array(i)).long())
        tensors = torch.from_numpy(np.array(inputs)).long()
        return tensors


class ToLabel_P(object):
    def __call__(self, inputs):
        tensors = []
        for i in inputs:
            tensors.append(torch.from_numpy(np.array(i)).long())
            # tensors = torch.from_numpy(np.array(inputs)).long()
        return tensors


class ReLabel(object):
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        # assert isinstance(input, torch.LongTensor), 'tensor needs to be LongTensor'
        for i in inputs:
            i[i == self.olabel] = self.nlabel
        return inputs


class ToSP(object):
    def __init__(self, size):
        self.scale2 = Scale(size / 2, Image.NEAREST)
        self.scale4 = Scale(size / 4, Image.NEAREST)
        self.scale8 = Scale(size / 8, Image.NEAREST)
        self.scale16 = Scale(size / 16, Image.NEAREST)
        self.scale32 = Scale(size / 32, Image.NEAREST)
        self.scale64 = Scale(size / 64, Image.NEAREST)

    def __call__(self, input):
        # input2 = self.scale2(input)
        # input4 = self.scale4(input)
        # input8 = self.scale8(input)
        # input16 = self.scale16(input)
        # input32 = self.scale32(input)
        input64 = self.scale64(input)
        inputs = input  # [input, input64]
        # inputs =input

        return inputs


class HorizontalFlip(object):
    """Horizontally flips the given PIL.Image with a probability of 0.5."""

    def __call__(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)


class VerticalFlip(object):
    def __call__(self, img):
        return img.transpose(Image.FLIP_TOP_BOTTOM)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


def pallet():
    pallet = [[128, 64, 128],
              [244, 35, 232],
              [70, 70, 70],
              [102, 102, 156],
              [190, 153, 153],
              [153, 153, 153],
              [250, 170, 30],
              [220, 220, 0],
              [107, 142, 35],
              [152, 251, 152],
              [70, 130, 180],
              [220, 20, 60],
              [255, 0, 0],
              [0, 0, 142],
              [0, 0, 70],
              [0, 60, 100],
              [0, 80, 100],
              [0, 0, 230],
              [119, 11, 32],
              [0, 0, 0]]
    pallet = np.array(pallet)
    return pallet


def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
            g = g + (1 << (7 - j)) * ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
            b = b + (1 << (7 - j)) * ((i & (1 << (3 * j + 2))) >> (3 * j + 2))

        cmap[i, :] = np.array([r, g, b])

    return cmap


class Colorize(object):
    def __init__(self, n=20):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


class Colorize2(object):
    def __init__(self, n=20):
        self.cmap = pallet()
        self.cmap = torch.from_numpy(self.cmap)

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


class RandomSizedCrop:
    """This is random sized cropping."""

    def __init__(self, size=None, interpolation=Image.BILINEAR):
        """Set output size and type of interpolation."""
        self.size = size
        self.img_interpolation = interpolation
        self.target_interpolation = Image.NEAREST

    def __call__(self, img):
        """Random sized cropp -> resize into 'self.size'."""

        # default size
        if self.size is None:
            self.size = img.size

        # try 10times
        for attempt in range(10):
            area = img.size[0] * img.size[1]

            # decide w, h
            cropped_area = random.uniform(0.5, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)
            w = int(round(math.sqrt(cropped_area * aspect_ratio)))
            h = int(round(math.sqrt(cropped_area / aspect_ratio)))

            # which is larger (prob: 0.5)
            if random.random() < 0.5:
                w, h = h, w

            # random crop, if possible
            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize(self.size, self.img_interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.img_interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip:
    """
    Random horizontal flip.

    prob = 0.5
    """

    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomVerticalFlip:
    """
    Random vertical flip.

    prob = 0.5
    """

    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotation:
    """
    Random roatation.

    -max_deg ~ deg
    """

    def __call__(self, img, max_deg=10):
        deg = np.random.randint(-max_deg, max_deg, 1)[0]
        return img.rotate(deg)
