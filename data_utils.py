import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import os
import sys
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def get_ingredients_list(data_dir):
    '''
    Returns the list of ingredients. 
    '''
    filename = data_dir + "/SplitAndIngreLabel/IngredientList.txt"
    ingredients = []
    with open(filename) as f:
        for line in f:
            ingredients.append(line.rstrip("\n"))
    return ingredients
