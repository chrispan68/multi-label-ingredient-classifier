import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data
from data_utils import read_inputs_1M

class IngredientDataset(data.dataset.Dataset):
    def __init__(self, images_file, labels_file, transforms, data_dir = "data"):
        self.images, self.labels = read_inputs_1M(images_file, labels_file, data_dir)
        self.data_dir = data_dir
        self.transforms = transforms

    def __getitem__(self, index):
        im = Image.open("{}/{}".format(self.data_dir, self.images[index]))
        if im.getbands()[0] == "L" or im.mode == "CMYK":
            im = im.convert("RGB")
        if self.transforms:
            im = self.transforms(im)
        
        return im, torch.Tensor(self.labels[self.images[index][1:]])

    def __len__(self):
        return len(self.images)

