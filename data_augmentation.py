import glob
import os
from typing import List, Tuple
import  imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
from imgaug.augmenters import contrast as iaa_contrast
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataAugmentation:    
    def __init__(self, train_path: str, num_augmentations: int = 1):
        self.train_patha = train_path
        self.num_augmentations = num_augmentations
        self.seq = self._get_augmentation_sequence()

    def _get_augmentation_sequence(self):
        augmentation_sequence = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-15, 15)),
            iaa.GaussianBlur(sigma=(0, 0.5)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
        ]) 
        
        return augmentation_sequence

    def augment_image(self, image_path: str, num_augmentations: int, seq: iaa.Sequential, file_extension: str):
 
        image = imageio.imread(image_path)
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
        image_shape = image.shape
        bbs = BoundingBoxesOnImage(self.read_label_file(label_path, image_shape), shape=image_shape)

        for i in range(num_augmentations):
            if image.shape[2] == 4:
                image = image[:, :, :3]

            image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            aug_image_filename = f"{base_filename}_aug_{i}{file_extension}"
            aug_label_filename = f"{base_filename}_aug_{i}.txt"
            
            image_aug_path = os.path.join(self.train_path, 'images', aug_image_filename)
            label_aug_path = os.path.join(self.train_path, 'labels', aug_label_filename)

            imageio.imwrite(image_aug_path, image_aug)
            self.write_label_file(bbs_aug.remove_out_of_image().clip_out_of_image(), label_aug_path, image_shape[1], image_shape[0])

    def augment_data(self):
        
        # Support file types
        file_types = ['*.png', '*.jpg', '*.jpeg']
        
        # Fetch all the file paths of matching files
        image_paths = []
        for file_type in file_types:
            image_paths.extend(glob.glob(os.path.join(self.train_path, 'images', file_type)))

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Conduct correct parameters to augment_image
            futures = [executor.submit(self.augment_image, image_path, self.num_augmentations, self.seq, os.path.splitext(image_path)[1]) for image_path in image_paths]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()

    @staticmethod
    def read_label_file(label_path: str, image_shape: Tuple[int, int, int]) -> List[BoundingBox]:
        
        bounding_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    x1 = (x_center - width / 2) * image_shape[1]
                    y1 = (y_center - height / 2) * image_shape[0]
                    x2 = (x_center + width / 2) * image_shape[1]
                    y2 = (y_center + height / 2) * image_shape[0]
                    bounding_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=int(class_id)))
        return bounding_boxes

    @staticmethod
    def write_label_file(bounding_boxes: List[BoundingBox], label_path: str, image_width: int, image_height: int):
        
        with open(label_path, 'w') as f:
            for bb in bounding_boxes:
                x_center = ((bb.x1 + bb.x2) / 2) / image_width
                y_center = ((bb.y1 + bb.y2) / 2) / image_height
                width = (bb.x2 - bb.x1) / image_width
                height = (bb.y2 - bb.y1) / image_height
                x_center, y_center, width, height = [max(0, min(1, val)) for val in [x_center, y_center, width, height]]
                class_index = bb.label
                f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform data augmentation on image datasets.')
    parser.add_argument('--train_path', type=str, default='dataset/train', help='Path to the training data')
    parser.add_argument('--num_augmentations', type=int, default=5, help='Number of augmentations per image')
    args = parser.parse_args()
    augmenter = DataAugmentation(args.train_path, args.num_augmentations)
    augmenter.augment_data()