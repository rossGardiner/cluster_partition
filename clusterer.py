import os
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
# === CONFIGURATION ===
PROJ_ROOT = '../image-classifier-project/'
SOURCE_ROOT = PROJ_ROOT + 'data/expert_labelled'  # <-- CHANGE THIS
CSV_PATH = PROJ_ROOT + 'src/tailled_expert.csv'   # class_id,count
TRAIN_DIR = 'data/expert_labelled_train_2_not_shuffled'
TEST_DIR = 'data/expert_labelled_test_2_not_shuffled'
MIN_IMAGES = 110
NUM_TEST_IMAGES = 10
NUM_TRAIN_IMAGES = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SHUFFLE = False  # Set to True if you want to shuffle the clusters


class Model:
    """This class does two things:
        1) create a pretrained model instance, resnet50 by default, and the transforms to go with it. 
        2) using this model, compute embeddings from a given image from the image path, if this isnt possible for any reason, return null
    """
    def __init__(self, model_name='resnet50'):
        self.model = models.__dict__[model_name](pretrained=True)
        self.model.fc = torch.nn.Identity()  # Remove classifier
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.model = self.model.to(DEVICE)
        self.model.eval()

    def get_embedding(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = self.model(tensor)
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

class Data:
    """This class does the following, given a directory of images, where the subdirs are class names and the images inside are instances of that class:
        1) Find all eligible classes based on the minimum number of images required for training and testing.
        2) For each eligible class, extract embeddings from the images using a model provided externally.
        3) Cluster the embeddings using Agglomerative Clustering.
        4) Select a specified number of test images from the clusters, ensuring that the test set is balanced. These can be selected either by sorting the clusters based on size, so the outliers are in the test set, or by randomly shuffling the clusters 
        5) Select a specified number of training images from the remaining images after selecting the test set.
        6) Copy the selected images to the respective train and test directories.
        
        The input source root should be in the following format:
        DIRNAME
        ├── CLASS1
        │   └── IMG1.jpg
        ├── CLASS2
        │   |── IMG1.jpg
        │   ├── IMG2.jpg
        │   ├── IMG3.jpg
        │   ├── IMG4.jpg
        │   |...
        ├── CLASS3
        |   |...
    """
    def __init__(self, source_root, min_train_images, min_test_images):
        self.source_root = source_root
        self.min_train_images = min_train_images
        self.min_test_images = min_test_images
        self.eligibles =  self.find_eligibles()
    
    def find_eligibles(self):
        #First, find the eligible classes. Use glob to query the root directory. 
        dirs = glob(self.source_root + '/*')  #get each directory in the source root
        qualified_classes = []
        for dir_path in dirs:
            if not os.path.isdir(dir_path):
                continue
            class_id = os.path.basename(dir_path)
            all_images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(all_images) >= self.min_train_images + self.min_test_images:
                qualified_classes.append(class_id)
        return qualified_classes
    
    def main_process(self, model, shuffle, min_clusters=5):
        for eligible in tqdm(self.eligibles, desc="Processing classes"):
            class_path = os.path.join(self.source_root, eligible)
            if not os.path.isdir(class_path):
                continue
            all_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(all_images) < (self.min_train_images + self.min_test_images):
                continue

            image_paths = [os.path.join(class_path, f) for f in all_images]

            # Extract embeddings
            embeddings = []
            valid_paths = []
            for path in image_paths:
                emb = model.get_embedding(path)
                if emb is not None:
                    embeddings.append(emb)
                    valid_paths.append(path)

            if len(embeddings) < (self.min_train_images + self.min_test_images):
                print(f"Skipping {eligible}, not enough usable images.")
                continue

            embeddings = np.vstack(embeddings)

            # Cluster embeddings
            #number of clusters chosen heuristically, either 5 (default) or sqrt the nr embeddings you have, which is apparently a common approximation.
            n_clusters = max(min_clusters, int(np.sqrt(len(embeddings))))
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
            cluster_labels = clustering.fit_predict(embeddings)

            # Group images by cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[label].append(idx)

            # Select test set from clusters
            cluster_keys = list(clusters.keys())
            if shuffle:
                random.shuffle(cluster_keys)
            else:
                cluster_keys = sorted(cluster_keys, key=lambda k: len(clusters[k]))
                
            # Finally, select the train and test images per class
            selected_test_indices = []
            for key in cluster_keys:
                if len(selected_test_indices) >= self.min_test_images:
                    break
                cluster = clusters[key]
                space_left = self.min_test_images - len(selected_test_indices)
                take = min(space_left, len(cluster))
                selected_test_indices.extend(cluster[:take])
                
            if len(selected_test_indices) < NUM_TEST_IMAGES:
                print(f"Class {eligible}: could not get 10 test images from clusters.")
                continue

            # === Select train set from remaining
            remaining_indices = list(set(range(len(valid_paths))) - set(selected_test_indices))
            if len(remaining_indices) < NUM_TRAIN_IMAGES:
                print(f"Class {eligible}: not enough remaining for train set.")
                continue

            train_indices = random.sample(remaining_indices, NUM_TRAIN_IMAGES)

            # === Create output folders
            test_out_dir = os.path.join(TEST_DIR, eligible)
            train_out_dir = os.path.join(TRAIN_DIR, eligible)
            os.makedirs(test_out_dir, exist_ok=True)
            os.makedirs(train_out_dir, exist_ok=True)

            # === Copy files
            for idx in selected_test_indices:
                src = valid_paths[idx]
                dst = os.path.join(test_out_dir, os.path.basename(src))
                shutil.copy(src, dst)

            for idx in train_indices:
                src = valid_paths[idx]
                dst = os.path.join(train_out_dir, os.path.basename(src))
                shutil.copy(src, dst)
            

def main(
    source_root,
    train_dir,
    test_dir,
    min_images,
    num_train_images,
    num_test_images,
    shuffle
):
    # Initialize model and data handler
    model = Model()
    data_handler = Data(source_root, num_train_images, num_test_images)

    # Update global output directories for this run
    global TRAIN_DIR, TEST_DIR, MIN_IMAGES, NUM_TRAIN_IMAGES, NUM_TEST_IMAGES, SHUFFLE
    TRAIN_DIR = train_dir
    TEST_DIR = test_dir
    MIN_IMAGES = min_images
    NUM_TRAIN_IMAGES = num_train_images
    NUM_TEST_IMAGES = num_test_images
    SHUFFLE = shuffle

    # Process the data
    data_handler.main_process(model, shuffle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/test split for image classification.")
    parser.add_argument('--source_root', type=str, default=SOURCE_ROOT, help='Root directory of source images. This can be from a long-tailed/uneven distribution of classes, the programme will try and balance the final dataset.')
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR, help='Directory to save training images.')
    parser.add_argument('--test_dir', type=str, default=TEST_DIR, help='Directory to save testing images.')
    parser.add_argument('--min_images', type=int, default=MIN_IMAGES, help='Minimum number of images per class.')
    parser.add_argument('--num_train_images', type=int, default=NUM_TRAIN_IMAGES, help='Number of training images per class.')
    parser.add_argument('--num_test_images', type=int, default=NUM_TEST_IMAGES, help='Number of testing images per class.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle clusters before selecting test images. This reduces the chance of the test set containing only outliers.')

    args = parser.parse_args()

    main(
        args.source_root,
        args.train_dir,
        args.test_dir,
        args.min_images,
        args.num_train_images,
        args.num_test_images,
        args.shuffle
    )