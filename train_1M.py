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
from model import Resnet50

def train(
    num_epochs,
    eval_interval,
    save_interval,
    learning_rate,
    batch_size,
    initialization,
    model_filename,
    optimizer_filename,
    data_dir,
):
    '''
    Trains the model on the training set and validation set. 
    '''
    num_labels = 1000
    train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 1}
    test_params = {"batch_size": 1, "shuffle": True, "num_workers": 1}
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.Resize((224, 224)),  # ImageNet standard
            transforms.ToTensor(),
        ]
    )
    test_transforms = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    print("Loading Dataset...")
    sys.stdout.flush()
    train_dataset = IngredientDataset("TR.txt", "IngreLabel.txt", train_transforms, f'/{data_dir}/train')
    test_dataset = IngredientDataset("VAL.txt", "IngreLabel.txt", test_transforms, f'/{data_dir}/val')
    train_loader = data.DataLoader(train_dataset, **train_params)
    test_loader = data.DataLoader(test_dataset, **test_params)
    total_steps = len(train_loader)
    ingredients = get_ingredients_list_1M(data_dir)

    print("Loading model...")
    sys.stdout.flush()
    criterion = nn.BCELoss()
    if initialization == 'random':
        model = Resnet50(num_labels, False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif initialization == 'pretrained':
        model = Resnet50(num_labels, True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif initialization == 'from_file':
        model = Resnet50(num_labels, False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.load_state_dict(torch.load("checkpoint/{}".format(model_filename)))
        optimizer.load_state_dict(torch.load("checkpoint/{}".format(optimizer_filename)))
    
    if torch.cuda.device_count() > 1: 
        print("Using", torch.cuda.device_count(), "GPUs.")
        sys.stdout.flush()
        model = nn.DataParallel(model)
    model = model.to(device)

    print("Beginning Training...")
    sys.stdout.flush()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Batchnorm1D can't handle batch size of 1
            if images.shape[0] < 2:
                break
            images = images.to(device)
            labels = labels.to(device).float()
            # Toggle training flag
            model.train()

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                curr_iter = epoch * len(train_loader) + i
                print(
                    "Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_steps, loss.item()
                    )
                )
                sys.stdout.flush()
        
        if (epoch + 1) % save_interval == 0:
            print("Saving model...")
            sys.stdout.flush()
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(model.state_dict(), "checkpoint/{}".format(model_filename))
            torch.save(optimizer.state_dict(), "checkpoint/{}".format(optimizer_filename))
        
        if (epoch + 1) % eval_interval == 0:
            print("Evaluating:")
            sys.stdout.flush()
            results, _, _, area = evaluate(model, test_loader, num_labels, ingredients, device)
            print("Epoch [{}/{}] Validation Results".format(epoch+1, num_epochs))
            print(results)
            print("ROC Curve AUC. Micro: {} Macro: {}.".format(area['micro'], area['macro']))
            sys.stdout.flush()

    # Save final version of the model
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    torch.save(model.state_dict(), "checkpoint/{}".format(model_filename))
    torch.save(optimizer.state_dict(), "checkpoint/{}".format(optimizer_filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--initialization", type=str, choices=["random", "pretrained", "from_file"], default="pretrained")
    parser.add_argument("--model_filename", type=str, default="model.bin")
    parser.add_argument("--optimizer_filename", type=str, default="optimizer.bin")
    parser.add_argument("--data_dir", type=str, default="/n/fs/pvl-mvs/sahanp_dev/datasets/food/1M_data")
    args = parser.parse_args()
    args = vars(args)

    num_epochs = args["num_epochs"]
    eval_interval = args["eval_interval"]
    save_interval = args["save_interval"]
    learning_rate = args["learning_rate"]
    batch_size = args["batch_size"]
    initialization = args["initialization"]
    model_filename = args["model_filename"]
    optimizer_filename = args["optimizer_filename"]
    data_dir = args["data_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(
        num_epochs,
        eval_interval,
        save_interval,
        learning_rate,
        batch_size,
        initialization,
        model_filename,
        optimizer_filename,
        data_dir
    )
