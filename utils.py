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

def evaluate(model, dataloader, num_labels, ingredients, device, mode="baseline", search_tree=None):
    '''
    Returns the classification report for the model
    '''
    thresh_prediction = 0.5 
    thresh_concensus = 0.8 #80% of neighbors have to agree for class to be positive. 
    # Toggle flag
    model.eval()
    y_pred = []
    y_scores = []
    y_truth = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            images = images.to(device)
            if(mode == "baseline"):
                outputs = model(images).cpu().numpy()
                pred_labels = (outputs  > thresh_prediction).astype(int)
            if(mode == "neighborhood_search"):
                model_outputs = model(images).cpu().numpy()
                outputs = search_tree.batch_query(model_outputs)
                pred_labels = outputs > thresh_concensus
            elif(mode == "both"):
                model_outputs = model(images).cpu().numpy()
                neighbor_outputs = search_tree.batch_query(model_outputs)
                outputs = model_outputs
                pred_labels = np.maximum((model_outputs > thresh_prediction), (neighbor_outputs > thresh_concensus))
            labels = labels.to(device).cpu().numpy()
            y_pred.extend(list(pred_labels))
            y_scores.extend(list(outputs))
            y_truth.extend(list(labels))

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
