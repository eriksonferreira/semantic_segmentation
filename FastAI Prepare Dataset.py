# Databricks notebook source
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
import os
import shutil
import glob
from tqdm.notebook import tqdm
from dotenv import load_dotenv

# COMMAND ----------

# Initialize a SegmentsDataset from the release file
load_dotenv()
SEGMENTS_KEY = os.getenv("SEGMENTS_KEY")
client = SegmentsClient(SEGMENTS_KEY)
release = client.get_release('eferreira/ltq_v3', 'v1')  # Example: release = 'flowers-v1.0.json'
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled'])

# Export to COCO panoptic format
export_dataset(dataset, export_format='semantic', export_folder="/dbfs/mnt/cobble/cleansed/images/segmentation/new/labels")

# COMMAND ----------

# Define directories
source_dir = '/dbfs/mnt/cobble/cleansed/images/segmentation/new/labels'
search_dir = '/dbfs/mnt/cobble/cleansed/images/segmentation/new/seg'
destination_dir = '/dbfs/mnt/cobble/cleansed/images/segmentation/new/imgs'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Create a set with the names of the files already present in the destination directory
destination_files = set(os.listdir(destination_dir))
search_files = set(os.listdir(search_dir))
source_files = os.listdir(source_dir)

# Adjust source filenames to match search filenames
source_files = [x.replace('_label_ground-truth_semantic.png', ".jpg") for x in source_files]
source_files = set(source_files)

# Identify files that need to be copied
need_copy = source_files - destination_files
print(need_copy)

for img in need_copy:
    search_filename = img
    source_file = os.path.join(search_dir, search_filename)
    destination_file = os.path.join(destination_dir, search_filename)

    # Check if the file already exists in the destination directory
    if search_filename not in destination_files:
        # Copy the file to the destination directory
        shutil.copy(source_file, destination_file)
        print(f'Copied: {search_filename}')

# COMMAND ----------

def create_train_val_test_split(dataset_path, train_split, val_split, split_by='stand'):
    """
    Create a train, validation, and test split from a dataset.

    Args:
    dataset_path (str): Path to the dataset.
    train_split (float): Proportion of the dataset to allocate to the training set.
    val_split (float): Proportion of the dataset to allocate to the validation set.
    split_by (str): Criterion to split the dataset ('stand' is implemented).

    """
    imgs_path = os.path.join(dataset_path, 'imgs')
    labels_path = os.path.join(dataset_path, 'labels')
    label_suffix = '_label_ground-truth_semantic.png'

    # Define paths for train, validation, and test splits
    train_path = os.path.join(dataset_path, 'train')
    train_label_path = os.path.join(dataset_path, 'train_label')

    val_path = os.path.join(dataset_path, 'val')
    val_label_path = os.path.join(dataset_path, 'val_label')
    
    test_path = os.path.join(dataset_path, 'test')
    test_label_path = os.path.join(dataset_path, 'test_label')

    # Define stands (specific categories for splitting)
    stands = ["F"+str(x) for x in range(1, 6)]

    # Clear existing files in the split directories
    print("Clearing directories")
    for dir_path in [train_path, train_label_path, val_path, val_label_path, test_path, test_label_path]:
        files = glob.glob(os.path.join(dir_path, '*'))
        for f in files:
            os.remove(f)
    print("Directories cleared")

    print("Creating split")
    train_split_files = []
    val_split_files = []
    test_split_files = []

    if split_by == 'stand':
        for stand in tqdm(stands, desc="Splitting by Stand"): 
            imgs = glob.glob(os.path.join(imgs_path, f'{stand}_*'))
            train_idx = int(len(imgs) * train_split)
            val_idx = int(len(imgs) * (train_split + val_split))

            train = imgs[:train_idx]
            val = imgs[train_idx:val_idx]
            test = imgs[val_idx:]

            train_split_files.extend(train)
            val_split_files.extend(val)
            test_split_files.extend(test)
    else: 
        print("Split criterion not implemented!")
        return

    # Copy files to train split
    for img in tqdm(train_split_files, desc="Processing train split"):
        if img not in train_path:
            file_name = os.path.basename(img)
            train_dest = os.path.join(train_path, file_name)
            shutil.copy(img, train_dest)
            label_name = file_name.replace('.jpg', label_suffix)
            shutil.copy(os.path.join(labels_path, label_name), os.path.join(train_label_path, label_name))

    # Copy files to validation split
    for img in tqdm(val_split_files, desc="Processing validation split"):
        if img not in val_path:
            file_name = os.path.basename(img)
            val_dest = os.path.join(val_path, file_name)
            shutil.copy(img, val_dest)
            label_name = file_name.replace('.jpg', label_suffix)
            shutil.copy(os.path.join(labels_path, label_name), os.path.join(val_label_path, label_name))

    # Copy files to test split
    for img in tqdm(test_split_files, desc="Processing test split"):
        if img not in test_path:
            file_name = os.path.basename(img)
            test_dest = os.path.join(test_path, file_name)
            shutil.copy(img, test_dest)
            label_name = file_name.replace('.jpg', label_suffix)
            shutil.copy(os.path.join(labels_path, label_name), os.path.join(test_label_path, label_name))

    print("Done!")

# COMMAND ----------

dataset_path = '/home/eferreira/master/mestrado_cnn/projeto_final/semantic_segmentation/dataset'
create_train_val_test_split(dataset_path, train_split=0.7, val_split=0.2)