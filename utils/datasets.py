import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
from utils.helpers import *


def unormalize_CIFAR10_image(image):
    return image*torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)

def plot_image(input, unormalize=True):
    if len(input.shape) > 3:
        print("Use plot_images function instead!")
        raise NotImplementedError
    npimg = input.numpy()
    if unormalize:
        npimg = npimg * np.array([0.2023, 0.1994, 0.2010]).reshape(3,1,1) + np.array([0.4914, 0.4822, 0.4465]).reshape(3,1,1)
    npimg = np.transpose(npimg, (1, 2, 0))
    if npimg.shape[-1] != 3:
        npimg = npimg[:, :, 0]
    #print(npimg.shape)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.imshow(npimg, cmap='gray')
    plt.show()
    return fig

def plot_images(batch, padding=2, unormalize=True):
    if len(batch.shape) == 3:
        plot_image(batch, unormalize=unormalize)
    elif len(batch.shape) == 4:
        n_images = batch.shape[0]
        if n_images == 1:
            plot_image(batch[0], unormalize=unormalize)
        else:
            grid_img = torchvision.utils.make_grid(batch, nrow=int(np.ceil(np.sqrt(n_images))), padding=padding)
            plot_image(grid_img, unormalize=unormalize)

class Cutout(object):
    def __init__(self, length, prob=1.0):
      self.length = length
      self.prob = prob
      assert prob<=1, f"Cutout prob given ({prob}) must be <=1"

    def __call__(self, img):
      if np.random.binomial(1, self.prob):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
      return img

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

def get_loaders(datasets_path,
                dataset,
                train_batch_size=128,
                val_batch_size=128,
                val_source='train',
                val_train_fraction=0.1,
                val_train_overlap=False,
                workers=0,
                train_infinite=False,
                val_infinite=False,
                cutout=False,
                cutout_length=16,
                cutout_prob=1):
    """
    NB: val_train_fraction and val_train_overlap only used if val_source='train'
    Note that infinite=True changes the seed/order of the batches
    Validation is never augmented since validation stochasticity comes
    from sampling different validation images anyways
    """
    assert val_source in ['test', 'train']
    TrainLoader = InfiniteDataLoader if train_infinite else DataLoader
    ValLoader = InfiniteDataLoader if val_infinite else DataLoader

    ## Select relevant dataset
    if dataset in ['MNIST', 'FashionMNIST']:
        mean, std = (0.1307,), (0.3081,)
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        if cutout: transform_train.transforms.append(Cutout(length=cutout_length, prob=cutout_prob))
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        if dataset == 'MNIST':
            train_dataset = datasets.MNIST(datasets_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.MNIST(datasets_path, train=False, download=True, transform=transform_test)
            val_dataset = test_dataset if val_source=='test' else datasets.MNIST(datasets_path, train=True, download=True, transform=transform_test)
        elif dataset == 'FashionMNIST':
            train_dataset = datasets.FashionMNIST(datasets_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.FashionMNIST(datasets_path, train=False, download=True, transform=transform_test)
            val_dataset = test_dataset if val_source=='test' else datasets.FashionMNIST(datasets_path, train=True, download=True, transform=transform_test)

    elif dataset == 'SVHN':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        dataset_path = os.path.join(datasets_path, 'SVHN') #Pytorch is inconsistent in folder structure
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        if cutout: transform_train.transforms.append(Cutout(length=cutout_length, prob=cutout_prob))
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        train_dataset = datasets.SVHN(dataset_path, split='train', download=True, transform=transform_train)
        test_dataset = datasets.SVHN(dataset_path, split='test', download=True, transform=transform_test)
        val_dataset = test_dataset if val_source=='test' else datasets.SVHN(dataset_path, split='train', download=True, transform=transform_test)

        #print(len(train_dataset))

    elif dataset in ['CIFAR10', 'CIFAR100']:
        # official CIFAR10 std seems to be wrong (actual is [0.2470, 0.2435, 0.2616])
        mean = (0.4914, 0.4822, 0.4465) if dataset == 'CIFAR10' else (0.5071, 0.4867, 0.4408)
        std = (0.2023, 0.1994, 0.2010) if dataset == 'CIFAR10' else (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        if cutout: transform_train.transforms.append(Cutout(length=cutout_length, prob=cutout_prob))
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        if dataset == 'CIFAR10':
            dataset_path = os.path.join(datasets_path, 'CIFAR10') #Pytorch is inconsistent in folder structure
            train_dataset = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform_test)
            val_dataset = test_dataset if val_source=='test' else datasets.CIFAR10(datasets_path, train=True, download=True, transform=transform_test)
        elif dataset == 'CIFAR100':
            dataset_path = os.path.join(datasets_path, 'CIFAR100')
            train_dataset = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transform_test)
            val_dataset = test_dataset if val_source=='test' else datasets.CIFAR10(datasets_path, train=True, download=True, transform=transform_test)

    else:
        print(f'{dataset} is not implemented')
        raise NotImplementedError

    ## Create dataloaders
    n_train_images = len(train_dataset)
    #print(train_dataset)
    pin_memory = True if dataset == 'ImageNet' else False

    if val_source == 'test':
        train_loader = TrainLoader(
            dataset=train_dataset, batch_size=train_batch_size,
            shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin_memory)

        val_loader = ValLoader(
            dataset=val_dataset, batch_size=val_batch_size,
            shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin_memory)

    elif val_source == 'train':
        all_indices = list(range(n_train_images))
        val_indices = np.random.choice(all_indices, size=int(val_train_fraction * n_train_images), replace=False)

        val_loader = ValLoader(
            dataset=val_dataset, batch_size=val_batch_size,
            sampler=SubsetRandomSampler(val_indices), drop_last=True,
            num_workers=workers, pin_memory=pin_memory)

        if val_train_overlap:
            train_loader = TrainLoader(
                dataset=train_dataset, batch_size=train_batch_size,
                shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin_memory)
        else:
            train_indices = list(set(all_indices) - set(val_indices))
            train_loader = TrainLoader(
                dataset=train_dataset, batch_size=train_batch_size,
                sampler=SubsetRandomSampler(train_indices), drop_last=True,
                num_workers=workers, pin_memory=pin_memory)

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=val_batch_size,
        shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin_memory) # test loader never infinite


    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_loaders('~/Datasets/Pytorch/',
                                                        'MNIST',
                                                        train_batch_size=500,
                                                        val_batch_size=500,
                                                        val_source='train',
                                                        val_train_fraction=0.05,
                                                        val_train_overlap=False,
                                                        workers=0,
                                                        train_infinite=False,
                                                        val_infinite=False,
                                                        cutout=True,
                                                        cutout_length=16,
                                                        cutout_prob=1)

    print(len(train_loader)*500)
    print(len(val_loader)*500)
    for x_val, y_val in val_loader:
        print(x_val.shape)
    for x_train, y_train in train_loader:
        break

    #plot_images(x_val[:100])
    plot_images(x_train[:100])

















