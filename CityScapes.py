import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from torchvision import transforms

from PIL import Image


class CityScapesInterface(data.Dataset):

    # constructor
    def __init__(self):
        pass

    # get length
    def __len__(self):
        pass

    # get an item
    def __getitem__(self, idx):
        pass