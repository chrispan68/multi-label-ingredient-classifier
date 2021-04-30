import numpy as np
import sys
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from data_utils import read_inputs
from PIL import Image

class ImageNearestNeighbors:
    def __init__(self, images_file="TR.txt", labels_file="IngreLabel.txt", data_dir = "data", neighborhood_size = 100, input_size=353, dimension_size=50, num_ingredients=353):
        self.images, self.labels = read_inputs(images_file, labels_file, data_dir)
        self.data_dir = data_dir
        self.pca = PCA(n_components=dimension_size)
        self.input_size = input_size
        self.dimension_size = dimension_size
        self.num_ingredients = num_ingredients
        self.neighborhood_size = neighborhood_size
        arr = []

        print("Encoding Training Examples...")
        sys.stdout.flush()
        for img in tqdm(self.images, total=len(self.images)):
            label = self.labels[img]
            im = Image.open("{}/ready_chinese_food{}".format(self.data_dir, img))
            if im.getbands()[0] == "L" or im.mode == "CMYK":
                im = im.convert("RGB")
            arr.append(encode(im, label))

        print("Computing PCA...")
        sys.stdout.flush()
        arr = np.asarray(arr)
        arr = self.pca.fit_transform(arr)

        print("Computing KD Tree...")
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
            neighborhood_concensus += self.labels[self.images[index]]
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
                cumulative_sum += self.labels[self.images[index]]
            neighborhood_concensus.append((cumulative_sum / self.neighborhood_size))
        return np.array(neighborhood_concensus, dtype=float)

def encode(img, label, mode="label"):
    if mode == "label":
        return label
    else:
        raise Exception("the mode entered was not a valid choice.")