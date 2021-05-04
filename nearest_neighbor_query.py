import numpy as np
import sys
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from data_utils import read_inputs
from PIL import Image
from IngredientDataset import IngredientDataset

def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    print("\tLoading Data...")
    with torch.no_grad():
        for images, label in tqdm(dataloader, total=len(dataloader)):
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            embeddings.extend(list(outputs))
            labels.extend(list(label.cpu().numpy()))
    return np.asarray(embeddings), np.asarray(labels)

class ImageNearestNeighbors:
    def __init__(self, model, device, dataloader, neighborhood_size = 100, input_size=353, dimension_size=50, num_ingredients=353):
        torch.multiprocessing.set_sharing_strategy('file_system')
        print("Initializing Model...")
        self.pca = PCA(n_components=dimension_size)
        self.input_size = input_size
        self.dimension_size = dimension_size
        self.num_ingredients = num_ingredients
        self.neighborhood_size = neighborhood_size

        arr, self.labels = get_embeddings(model=model, dataloader=dataloader, device=device)
        arr = self.pca.fit_transform(arr)

        print("\tComputing KD Tree...")
        sys.stdout.flush()
        self.tree = cKDTree(arr)
    
    def query(self, encoding):
        """
        Given an (K) length numpy array representing a batch of encodings,
        return a (num_ingredients) length numpy array representing the ingredients that have reached a neighborhood concensus.
        
        Input:
            encodings: (K) length numpy array of encodings
            neighborhood_size: number of neighbors to check
            p: fraction of neighbors with positive ingredient needed to reach concensus
        
        Output:
            neighborhood_concensus: A (num_ingredients) length array, where a 1 at index j represents that
                what percentage of neighbors had ingredient j. 
        """
        distances, indices = self.tree.query(self.pca.transform(encoding), k=self.neighborhood_size)
        neighborhood_concensus = np.zeros(self.num_ingredients)
        for index in indices: 
            neighborhood_concensus += self.labels[index]
        scores
        return neighborhood_concensus / self.neighborhood_size

    def batch_query(self, encodings):
        """
        Given an (N x K) numpy array representing a batch of encodings,
        return a (N x num_ingredients) numpy array representing the ingredients that have reached a neighborhood concensus.
        (Batched version of query)

        Input:
            encodings: (N x K) numpy array of encodings
            neighborhood_size: number of neighbors to check
            p: fraction of neighbors with positive ingredient needed to reach concensus
        
        Output:
            neighborhood_concensus: A (N x num_ingredients) array, where a 1 at index (i, j) represents that for the ith input,
                what percentage of neighbors had ingredient j. 
        """
        distances, indices = self.tree.query(self.pca.transform(encodings), k=self.neighborhood_size)
        neighborhood_concensus = []
        for row in indices:
            cumulative_sum = np.zeros(self.num_ingredients)
            for index in row:
                cumulative_sum += self.labels[index]
            neighborhood_concensus.append((cumulative_sum / self.neighborhood_size))
        return np.array(neighborhood_concensus, dtype=float)