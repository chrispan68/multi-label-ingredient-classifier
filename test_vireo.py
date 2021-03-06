import argparse
import numpy as np
from IngredientDataset import IngredientDataset
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import os
import sys
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from utils import get_roc_curve, evaluate
from data_utils import get_ingredients_list
from model import Resnet50, Resnet50_baseline
from nearest_neighbor_query import ImageNearestNeighbors
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def test(model_filename, data_dir, mode, output_dir, batch_size, use_resnet):
    '''
    Evaluates the model on the test set. 
    '''
    num_labels = 353
    if use_resnet:
        num_labels = 1000
    test_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 1}
    test_transforms = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    print("Loading Dataset...")
    sys.stdout.flush()
    test_dataset = IngredientDataset("TE.txt", "IngreLabel.txt", test_transforms, f'{data_dir}')
    test_loader = data.DataLoader(test_dataset, **test_params)
    ingredients = get_ingredients_list(data_dir)

    print("Loading model...")
    sys.stdout.flush()
    model = Resnet50(num_labels, False).to(device)
    if not use_resnet:
        model.load_state_dict(torch.load("/n/fs/pvl-mvs/sahanp_dev/datasets/multi-label-ingredient-classifier/checkpoint/{}".format(model_filename), map_location=device))
    else:
        model = Resnet50_baseline(True).to(device)
    
    if torch.cuda.device_count() > 1: 
        print("Using", torch.cuda.device_count(), "GPUs.")
        sys.stdout.flush()
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    print("Initializing Tests...")
    search_tree = None
    if mode == "neighborhood_search" or mode == "both":
        dataset = IngredientDataset("TR.txt", "IngreLabel.txt", transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), f'{data_dir}')
        params = {"batch_size": batch_size, "shuffle": False, "num_workers": 1}
        loader = data.DataLoader(dataset, **params)
        search_tree = ImageNearestNeighbors(model=model, device=device, dataloader=loader,num_ingredients=353, input_size=num_labels)
    sys.stdout.flush()
    print("Evaluating Model...")
    results, tpr, fpr, area = evaluate(model, test_loader, 353, ingredients, device, mode, search_tree)
    print("Testing Results:")
    print(results)

    #Plot precision recall curve
    plt.title('ROC-Curves')
    plt.plot(fpr['micro'], tpr['micro'], 'b', label = 'Micro-AUC = %0.4f' % area['micro'])
    plt.plot(fpr['macro'], tpr['macro'], 'g', label = 'Macro-AUC = %0.4f' % area['macro'])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('{}/roc-curves.jpg'.format(output_dir))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["baseline", "neighborhood_search", "both"], default="neighborhood_search")
    parser.add_argument("--model_filename", type=str, default="model.bin")
    parser.add_argument("--data_dir", type=str, default="/n/fs/pvl-mvs/sahanp_dev/datasets/food/vireo")
    parser.add_argument("--output_dir", type=str, default="analysis_1M_neighborhood_search")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_resnet", type=bool, default=False)
    args = parser.parse_args()
    args = vars(args)

    mode = args["mode"]
    model_filename = args["model_filename"]
    data_dir = args["data_dir"]
    output_dir = args["output_dir"]
    batch_size = args["batch_size"]
    use_resnet = args["use_resnet"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(
        model_filename,
        data_dir, 
        mode,
        output_dir, 
        batch_size,
        use_resnet
    )