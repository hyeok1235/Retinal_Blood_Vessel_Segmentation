import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(image_filenames, mask_filenames, image_dir, mask_dir, image_size=(256, 256)):
    image_list, mask_list = [], []
    for img_file, mask_file in zip(image_filenames, mask_filenames):
        if img_file.endswith(".png") and mask_file.endswith(".png"):
            image = cv2.imread(os.path.join(image_dir, img_file))
            image = cv2.resize(image, image_size)
            image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            image_list.append(image)

            mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, image_size)
            mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            mask_list.append(mask)
    return np.array(image_list), np.array(mask_list)

def prepare_data(image_path):
    # Define directories
    train_image_dir = os.path.join(image_path, 'train/Original/')
    train_mask_dir = os.path.join(image_path, 'train/Ground truth/')
    test_image_dir = os.path.join(image_path, 'test/Original/')
    test_mask_dir = os.path.join(image_path, 'test/Ground truth/')

    # Get and sort filenames
    train_image_filenames = sorted(os.listdir(train_image_dir))
    train_mask_filenames = sorted(os.listdir(train_mask_dir))
    test_image_filenames = sorted(os.listdir(test_image_dir))
    test_mask_filenames = sorted(os.listdir(test_mask_dir))

    # Load data
    train_images, train_masks = load_data(train_image_filenames, train_mask_filenames, train_image_dir, train_mask_dir)
    test_images, test_masks = load_data(test_image_filenames, test_mask_filenames, test_image_dir, test_mask_dir)

    # Split test into validation and test sets
    images_val, images_test, masks_val, masks_test = train_test_split(test_images, test_masks, test_size=0.5, random_state=42)
    
    return train_images, train_masks, images_val, masks_val, images_test, masks_test
