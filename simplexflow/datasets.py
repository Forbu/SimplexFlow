"""
Module for discretizing the MNIST dataset.
"""

import torch
import torchvision
import torchvision.transforms as transforms

# import dataset class from torch
from torch.utils.data import Dataset

import numpy as np


class DiscretizeMnist(Dataset):
    """
    Simple dataset that discretizes the MNIST dataset.
    """

    def __init__(self, num_bins=10):
        """
        Args:
            data (torch.Tensor): data to be discretized
            labels (torch.Tensor): labels for the data
            num_bins (int): number of bins to discretize the data into
        """
        # get the data and labels
        self.num_bins = num_bins

        # import classic mnist dataset
        self.mnist = torchvision.datasets.MNIST(
            root="./data", train=True, download=True
        )

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        """
        Returns a tuple of (discretized data, label)
        """
        # get the data and label
        data, label = self.mnist[idx]

        # transform the data
        data = self.transform(data)

        # discretize the data
        data_onehot, data = self.discretize(data)

        return data_onehot, data, label

    def discretize(self, data):
        """
        Discretizes the data into self.num_bins bins.
        """
        # get the min and max of the data
        min_val = torch.min(data)
        max_val = torch.max(data)

        # get the bin size
        bin_size = (max_val - min_val) / self.num_bins

        # discretize the data
        data = torch.floor((data - min_val) / bin_size)

        # max being num_bins - 1
        data = torch.clamp(data, 0, self.num_bins - 1)

        # data is of (1, 28, 28)
        # we want to convert it to (1, 28, 28, self.num_bins)
        # with a one hot encoding
        data_onehot = torch.nn.functional.one_hot(data.long(), num_classes=self.num_bins)

        # data is now of (1, 28, 28, self.num_bins)

        return data_onehot.squeeze(0), data.squeeze(0).long()
