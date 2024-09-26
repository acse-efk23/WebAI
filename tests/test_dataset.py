import os
import h5py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from torch.utils.data import Dataset
from deepdown.dataset import Hdf5Dataset

@pytest.fixture
def hdf5_file(tmp_path):
    # Create a temporary HDF5 file for testing
    file_path = tmp_path / "test.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("input", data=np.random.rand(10, 3, 32, 32))
        f.create_dataset("output", data=np.random.randint(0, 10, size=(10,)))
    return file_path

def test_initialise_file_not_found():
    dataset = Hdf5Dataset()
    with pytest.raises(FileNotFoundError):
        dataset.initialise("non_existent_file.h5")

def test_initialise(hdf5_file):
    dataset = Hdf5Dataset()
    dataset.initialise(hdf5_file)
    assert dataset.file_path == hdf5_file
    assert dataset.length == 10

def test_load_data(hdf5_file):
    dataset = Hdf5Dataset()
    dataset.initialise(hdf5_file)
    x, y = dataset.load_data(0)
    assert x.shape == (3, 32, 32)
    assert isinstance(y, np.integer)

def test_getitem(hdf5_file):
    dataset = Hdf5Dataset()
    dataset.initialise(hdf5_file)
    x, y = dataset[0]
    assert x.shape == (3, 32, 32)
    assert isinstance(y, np.integer)

def test_len(hdf5_file):
    dataset = Hdf5Dataset()
    dataset.initialise(hdf5_file)
    assert len(dataset) == 10