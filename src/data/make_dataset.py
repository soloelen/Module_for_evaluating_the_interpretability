import torch
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from ..config.config import BATCH_SIZE, PATH_TO_DATA
from .data_transform import transform


def get_dataloaders(get_original_test_data=False):
    """
    Сборка dataloader-ов.
    """

    train_data = datasets.CIFAR10(root=PATH_TO_DATA, train=True, download=False, transform=transform['train'])

    test_data = datasets.CIFAR10(root=PATH_TO_DATA, train=False, download=False, transform=transform['test'])
    original_test_data = datasets.CIFAR10(root=PATH_TO_DATA, train=False, download=False, transform=None)

    generator = torch.Generator().manual_seed(42)
    test_size = len(test_data)
    orig_test_size = len(original_test_data)
    val_data, test_data = random_split(test_data, [test_size // 2, test_size - test_size // 2], generator)
    temp, original_test_data = random_split(original_test_data, [orig_test_size // 2,
                                                                 orig_test_size - orig_test_size // 2], generator)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    if get_original_test_data:
        return train_loader, val_loader, original_test_data
    return train_loader, val_loader
