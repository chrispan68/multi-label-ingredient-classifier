import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree

class ImageNearestNeighbors:
    def __init__(self, images_file="TR.txt", labels_file="IngreLabel.txt", data_dir = "data", num_ingredients=353):
        self.images, self.labels = read_inputs(images_file, labels_file, data_dir)
        self.data_dir = data_dir
        self.k = num_ingredients
        arr = []

        print("Initializing KD-Tree")
        sys.stdout.flush()
        for img, label in tqdm(zip(self.images, self.labels), total=len(self.images)):
            im = Image.open("{}/ready_chinese_food{}".format(self.data_dir, self.images[index]))
            if im.getbands()[0] == "L" or im.mode == "CMYK":
                im = im.convert("RGB")
            arr.append(encode(img, label))
        
        self.tree = cKDTree(np.asarray(arr))
    
    def query(self, encoding, neighborhood_size=10, p=0.7):
        """
        Given an (K) length numpy array representing a batch of encodings,
        return a (num_ingredients) length numpy array representing the ingredients that have reached a neighborhood concensus.
        
        Input:
            encodings: (K) length numpy array of encodings
            neighborhood_size: number of neighbors to check
            p: fraction of neighbors with positive ingredient needed to reach concensus
        
        Output:
            neighborhood_concensus: A (num_ingredients) length array, where a 1 at index j represents that
                a large number of neighbors had ingredient j. 
        """
        distances, indices = self.tree.query(encoding)
        neighborhood_concensus = np.zeros(self.k)
        for index in indices: 
            neighborhood_concensus += self.labels[index]
        
        return (neighborhood_concensus / neighborhood_size) >= p

    def batch_query(self, encodings, neighborhood_size=10, p=0.7):
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
                a large number of neighbors had ingredient j. 
        """
        distances, indices = self.tree.query(encodings)
        neighborhood_concensus = []
        for row in indices:
            cumulative_sum = np.zeros(self.k)
            for index in row:
                cumulative_sum += self.labels[index]
            neighborhood_concensus.append((cumulative_sum / neighborhood_size) >= p)
        return np.array(neighborhood_concensus)

def encode(img, label, mode="label"):
    if mode == "label":
        return label
    else:
        raise Exception("the mode entered was not a valid choice.")