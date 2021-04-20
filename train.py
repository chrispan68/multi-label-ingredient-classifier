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

class Resnet50(nn.Module):
    '''
    Resnet architecture for multi-label ingredient classification
    '''
    def __init__(self, n_classes, is_pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=is_pretrained).to(device)
        resnet.fc = nn.Sequential(
            nn.BatchNorm1d(resnet.fc.in_features),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(resnet.fc.in_features, n_classes),
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))

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
    num_labels = 353
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
    train_dataset = IngredientDataset("TR.txt", "IngreLabel.txt", train_transforms, data_dir)
    test_dataset = IngredientDataset("VAL.txt", "IngreLabel.txt", test_transforms, data_dir)
    train_loader = data.DataLoader(train_dataset, **train_params)
    test_loader = data.DataLoader(test_dataset, **test_params)
    total_steps = len(train_loader)
    ingredients = get_ingredients_list(data_dir)

    print("Loading model...")
    sys.stdout.flush()
    criterion = nn.BCELoss()
    if initialization == 'random':
        model = Resnet50(num_labels, False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif initialization == 'pretrained':
        model = Resnet50(num_labels, True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif initialization == 'from_file':
        model = Resnet50(num_labels, False)
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
            results, _, _, area = evaluate(model, test_loader, num_labels, ingredients)
            print("Epoch [{}/{}] Validation Results".format(epoch+1, num_epochs))
            print(results)
            print("ROC Curve AUC. Micro: {} Macro: {}.".format(area['micro'], area['macro']))
            sys.stdout.flush()

    # Save final version of the model
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    torch.save(model.state_dict(), "checkpoint/{}".format(model_filename))
    torch.save(optimizer.state_dict(), "checkpoint/{}".format(optimizer_filename))

def test(model_filename, data_dir):
    '''
    Evaluates the model on the test set. 
    '''
    num_labels = 353
    test_params = {"batch_size": 1, "shuffle": True, "num_workers": 1}
    test_transforms = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    print("Loading Dataset...")
    sys.stdout.flush()
    test_dataset = IngredientDataset("TE.txt", "IngreLabel.txt", test_transforms, data_dir)
    test_loader = data.DataLoader(test_dataset, **test_params)
    ingredients = get_ingredients_list(data_dir)

    print("Loading model...")
    sys.stdout.flush()
    model = Resnet50(num_labels, False)
    model.load_state_dict(torch.load("checkpoint/{}".format(model_filename)))
    
    if torch.cuda.device_count() > 1: 
        print("Using", torch.cuda.device_count(), "GPUs.")
        sys.stdout.flush()
        model = nn.DataParallel(model)
    model = model.to(device)

    print("Beginning Testing...")
    sys.stdout.flush()
    results, tpr, fpr, area = evaluate(model, test_loader, num_labels, ingredients)
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
    plt.savefig('analysis/roc-curves.jpg')
    
def evaluate(model, dataloader, num_labels, ingredients):
    '''
    Returns the classification report for the model
    '''
    # Toggle flag
    model.eval()
    y_pred = []
    y_scores = []
    y_truth = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = torch.squeeze(model(images))
            labels = torch.squeeze(labels.to(device))
            pred_labels = outputs  > 0.5
            y_pred.append(pred_labels.cpu().numpy().astype(int))
            y_scores.append(outputs.cpu().numpy())
            y_truth.append(labels.cpu().numpy())

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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--initialization", type=str, choices=["random", "pretrained", "from_file"], default="pretrained")
    parser.add_argument("--model_filename", type=str, default="model.bin")
    parser.add_argument("--optimizer_filename", type=str, default="optimizer.bin")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    args = vars(args)

    mode = args["mode"]
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
    if mode == "train":
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
    elif mode == "test":
        test(
            model_filename,
            data_dir
        )