import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data

def read_inputs(images_file, labels_file, data_dir):
    """
    Reads the images and labels from files
    Input: 
        images_file: the file containing image filenames
        labels_file: the file that the labels are read from
    Output:
        images: list of image filenames. 
        labels: returns a dictionary from filename to numpy labels
    """
    images = []
    with open("{}/SplitAndIngreLabel/{}".format(data_dir, images_file)) as f:
        for line in f:
            images.append(line.split()[0])
    
    labels = {}
    with open("{}/SplitAndIngreLabel/{}".format(data_dir, labels_file)) as f:
        for line in f:
            words = line.split()
            filename = words[0]
            #Converts the labels from -1, 1 to 0, 1
            labels[filename] = (np.asarray([int(i) for i in words[1:]]) + 1) * 0.5

    return images, labels

class IngredientDataset(data.dataset.Dataset):
    def __init__(self, images_file, labels_file, transforms, data_dir = "data"):
        self.images, self.labels = read_inputs(images_file, labels_file, data_dir)
        self.data_dir = data_dir
        self.transforms = transforms

    def __getitem__(self, index):
        im = Image.open("{}/ready_chinese_food{}".format(self.data_dir, self.images[index]))

        if im.getbands()[0] == "L" or im.mode == "CMYK":
            im = im.convert("RGB")
        if self.transforms:
            im = self.transforms(im)
        
        return im, torch.Tensor(self.labels[self.images[index]])

    def __len__(self):
        return len(self.images)

