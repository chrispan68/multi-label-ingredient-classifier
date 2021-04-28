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
from nearest_neighbor_query import ImageNearestNeighbors, encode
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

def evaluate(model, dataloader, num_labels, ingredients, device, mode="baseline", search_tree=None):
    '''
    Returns the classification report for the model
    '''
    # Toggle flag
    model.eval()
    y_pred = []
    y_scores = []
    y_truth = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            images = images.to(device)
            outputs = torch.squeeze(model(images)).cpu().numpy()
            pred_labels = (outputs  > 0.5).astype(int)
            if(mode == "neighborhood_search"):
                neighborhood_concensus = search_tree.query(outputs)
                pred_labels = np.max(pred_labels, neighborhood_concensus)
            labels = torch.squeeze(labels.to(device)).cpu().numpy()
            y_pred.append(pred_labels)
            y_scores.append(outputs)
            y_truth.append(labels)

    y_pred = np.asarray(y_pred)
    y_scores = np.asarray(y_scores)
    y_truth = np.asarray(y_truth)

    # Compute precision recall curve
    tpr, fpr = get_roc_curve(y_truth, y_scores, num_labels)
    area = {}
    area['micro'] = auc(fpr['micro'], tpr['micro'])
    area['macro'] = auc(fpr['macro'], tpr['macro'])
    return classification_report(y_truth, y_pred, target_names=ingredients), tpr, fpr, area

def get_roc_curve(y_truth, y_scores, num_labels):
    fpr = {}
    tpr = {}
    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(y_truth[:, i], y_scores[:, i])
    
    fpr['micro'], tpr['micro'], _ = roc_curve(y_truth.ravel(), y_scores.ravel())
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_labels)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_labels):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_labels
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    return tpr, fpr
