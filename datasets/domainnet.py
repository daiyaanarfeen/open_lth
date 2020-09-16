# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import concurrent
import numpy as np
import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform


class Dataset(base.ImageDataset):
    """DomainNet"""

    def __init__(self, loc: str, image_transforms, train=True):
        # Load the data.
        splits = os.listdir(os.path.join(loc, 'splits'))
        splits = [f for f in splits if ("train" in f) == train]
        examples = []
        for f in splits:
            examples += open(os.path.join(loc, 'splits', f), 'r').readlines()
        examples = [ex.replace('\n', '').split(' ') for ex in examples]
        labels = [int(ex[1]) for ex in examples]
        examples = [os.path.join(loc, 'images', ex[0]) for ex in examples]

        super(Dataset, self).__init__(
            np.array(examples), np.array(labels), image_transforms,
            [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @staticmethod
    def num_train_examples(): return {'clipart': 34019, 'infograph': 37087, 'painting': 52867, 'quickdraw': 120750, 'real': 122563, 'sketch': 49115}

    @staticmethod
    def num_test_examples(): return {'clipart': 14814, 'infograph': 16114, 'painting': 22892, 'quickdraw': 51750, 'real': 52764, 'sketch': 21271}

    @staticmethod
    def num_classes(): return 345

    @staticmethod
    def _augment_transforms():
        return [
            torchvision.transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(0.8, 1.25)),
            torchvision.transforms.RandomHorizontalFlip()
        ]

    @staticmethod
    def _transforms():
        return [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224)]

    @staticmethod
    def get_train_set(use_augmentation):
        transforms = Dataset._augment_transforms() if use_augmentation else Dataset._transforms()
        return Dataset(get_platform().domainnet_root, transforms)

    @staticmethod
    def get_test_set():
        return Dataset(get_platform().domainnet_root, Dataset._transforms(), train=False)

    @staticmethod
    def example_to_image(example):
        with get_platform().open(example, 'rb') as fp:
            return Image.open(fp).convert('RGB')

    def domains(domains):
        domains = split(domains, ',')
        indices = [i for (i, ex) in enumerate(self._examples) if any([d in ex for d in domains])]
        self._examples = [self._examples[i] for i in indices]
        self._labels = [self._labels[i] for i in indices]


DataLoader = base.DataLoader
