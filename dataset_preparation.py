import argparse
import os
import random
import shutil
from typing import Tuple, List
from tqdm import tqdm

class DatasetPreparation:
   
    def __init__(self, data_path: str, train_path: str, valid_path: str, split_ratio: float = 0.8):
        
        self.data_path = data_path
        self.train_path = train_path
        self.valid_path = valid_path
        self.split_ratio = split_ratio

    @staticmethod
    def create_dir(path: str):
      
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def copy_data(files: List[str], source_dir: str, target_dir: str, file_type: str):
        
        for file in files:
            source = os.path.join(source_dir, f'{file}{file_type}')
            target = os.path.join(target_dir, f'{file}{file_type}')
            if os.path.exists(source):
                shutil.copy(source, target)
            else:
                print(f"Warning: {source} does not exist. Check if the file is missing or the file naming is incorrect.")

    def prepare(self) -> Tuple[List[str], List[str]]:
       
        # Check if data_path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path '{self.data_path}' does not exist.")

        # Check if images directory exists within data_path
        images_dir = os.path.join(self.data_path, "dataset_mentah/resize")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory '{images_dir}' not found.")

        # Create necessary directories for training and validation sets
        self.create_dir(os.path.join(self.train_path, 'images'))
        self.create_dir(os.path.join(self.train_path, 'labels'))
        self.create_dir(os.path.join(self.valid_path, 'images'))
        self.create_dir(os.path.join(self.valid_path, 'labels'))

        # List all files and shuffle them
        files = [os.path.splitext(file)[0] for file in os.listdir(images_dir)]
        random.shuffle(files)
        split_point = int(len(files) * self.split_ratio)

        # Split files into training and validation sets
        train_files = files[:split_point]
        valid_files = files[split_point:]

        # Copy image files to the corresponding directories with progress bar
        image_source_dir = images_dir
        label_source_dir = os.path.join(self.data_path, "dataset_mentah/labels")

        for file_set, storage_path, file_type in [
            (train_files, os.path.join(self.train_path, "images"), '.jpg'),
            (train_files, os.path.join(self.train_path, "labels"), '.txt'),
            (valid_files, os.path.join(self.valid_path, "images"), '.jpg'),
            (valid_files, os.path.join(self.valid_path, "labels"), '.txt')
        ]:
            for file in tqdm(file_set, desc=f"Copying files to {storage_path}"):
                source_dir = image_source_dir if file_type == '.jpg' else label_source_dir
                self.copy_data([file], source_dir, storage_path, file_type)

        return train_files, valid_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset by creating a training and validation split.')
    
    parser.add_argument('--data_path', type=str, default='./', help='Path to the data directory')
    parser.add_argument('--train_path', type=str, default='dataset/train', help='Path to the training directory')
    parser.add_argument('--valid_path', type=str, default='dataset/val', help='Path to the validation directory')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='Ratio to split the training and validation data')

    args = parser.parse_args()

    # Instantiate the class and prepare the dataset
    dataset_preparation = DatasetPreparation(args.data_path, args.train_path, args.valid_path, args.split_ratio)
    train_files, valid_files = dataset_preparation.prepare()

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(valid_files)}")
