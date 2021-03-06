# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from datasets import base, cifar10, mnist, imagenet, domainnet
from foundations.hparams import DatasetHparams
from platforms.platform import get_platform

registered_datasets = {'cifar10': cifar10, 'mnist': mnist, 'imagenet': imagenet, 'domainnet': domainnet}


def get(dataset_hparams: DatasetHparams, train: bool = True):
    """Get the train or test set corresponding to the hyperparameters."""

    seed = dataset_hparams.transformation_seed or 0

    # Get the dataset itself.
    if dataset_hparams.dataset_name in registered_datasets:
        use_augmentation = train and not dataset_hparams.do_not_augment
        if train:
            dataset = registered_datasets[dataset_hparams.dataset_name].Dataset.get_train_set(use_augmentation)
        else:
            dataset = registered_datasets[dataset_hparams.dataset_name].Dataset.get_test_set()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    # Transform the dataset.
    if train and dataset_hparams.random_labels_fraction is not None:
        dataset.randomize_labels(seed=seed, fraction=dataset_hparams.random_labels_fraction)

    if train and dataset_hparams.subsample_fraction is not None:
        dataset.subsample(seed=seed, fraction=dataset_hparams.subsample_fraction, inverse=dataset_hparams.subsample_inverse)

    if not train and dataset_hparams.subsample_fraction_test is not None:
        dataset.subsample(seed=seed, fraction=dataset_hparams.subsample_fraction_test, inverse=dataset_hparams.subsample_inverse)

    if dataset_hparams.domains is not None:
        dataset.domains(dataset_hparams.domains.split(','))

    if train and dataset_hparams.blur_factor is not None:
        if not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can blur images.')
        else:
            if dataset_hparams.random_blur:
                dataset.random_blur(blur_factor=dataset_hparams.blur_factor, subsample_blur=dataset_hparams.subsample_blur)
            else:
                dataset.blur(blur_factor=dataset_hparams.blur_factor, subsample_blur=dataset_hparams.subsample_blur)

    if train and dataset_hparams.rotate_array is not None:
        if not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can blur images.')
        else:
            if dataset_hparams.random_rotate:
                dataset.random_rotate([float(r) for r in dataset_hparams.rotate_array.split(',')])
            else:
                dataset.rotate([float(r) for r in dataset_hparams.rotate_array.split(',')])

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        elif not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can only do unsupervised rotation to images.')
        else:
            dataset.unsupervised_rotation(seed=seed)

    # Create the loader.
    if train or dataset_hparams.batch_size_test is None:
        batch_size = dataset_hparams.batch_size
    else:
        batch_size = dataset_hparams.batch_size_test
    return registered_datasets[dataset_hparams.dataset_name].DataLoader(
        dataset, batch_size=batch_size, num_workers=get_platform().num_workers)


def iterations_per_epoch(dataset_hparams: DatasetHparams):
    """Get the number of iterations per training epoch."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_train_examples = registered_datasets[dataset_hparams.dataset_name].Dataset.num_train_examples()
        if dataset_hparams.dataset_name == "domainnet":
            if dataset_hparams.domains is not None:
                num_train_examples = sum([num_train_examples[k] for k in dataset_hparams.domains.split(',')])
            else:
                num_train_examples = sum(list(num_train_examples.values()))
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.subsample_fraction is not None:
        num_train_examples *= dataset_hparams.subsample_fraction

    return np.ceil(num_train_examples / dataset_hparams.batch_size).astype(int)


def num_classes(dataset_hparams: DatasetHparams):
    """Get the number of classes."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_classes = registered_datasets[dataset_hparams.dataset_name].Dataset.num_classes()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        else:
            return 4

    return num_classes
