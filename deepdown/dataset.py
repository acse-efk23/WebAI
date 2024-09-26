import os
import h5py
from torch.utils.data import Dataset

from .transforms import *


class Hdf5Dataset(Dataset):
    """
    A PyTorch Dataset class for loading data from an HDF5 file.
    """

    def __init__(self):
        self.file_path = None
        self.transform = None

    def initialise(self, file_path, transform=None):
        """
        Initializes the dataset by loading the file and setting up transformations.

        Args:
            file_path (str): The path to the dataset file.
            transform (callable, optional): A function/transform to apply to the 
                                            data. Defaults to None.
        """
        self.transform = transform

        # Check if entered data file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        self.file_path = file_path

        with h5py.File(self.file_path, "r") as file:
            self.length = file["input"].shape[0]

    def load_data(self, index):
        """
        Loads data from an HDF5 file at the specified index.

        Args:
            index (int): The index of the data to load.

        Returns:
            tuple: A tuple containing the input data and 
                   output data at the specified index.
        """
        with h5py.File(self.file_path, "r") as file:
            return file["input"][index], file["output"][index]

    def __getitem__(self, index):
        """
        Retrieve the data sample and its corresponding label at the specified index.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing the data sample and its corresponding label.
                   If a transform is specified, both the data sample and label are
                   transformed before being returned.
        """
        x, y = self.load_data(index)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return self.length
