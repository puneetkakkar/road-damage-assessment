import os
import cv2
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from utils.utils import load_config, get_config_key
from sklearn.model_selection import train_test_split
from .transforms import get_dino_v2_transforms, get_segformer_transforms
from transformers import AutoImageProcessor


# Custom dataset class for DiNOv2 model to perform semantic segmentation
class DINOv2SemanticSegmentationDataset(Dataset):

    # Initialze the dataset class with image paths, mask paths, and
    # a transform function for both images, and masks
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Fetch the image and mask paths for the given index
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Read image and convert to RGB
        original_image = cv2.imread(image_path)
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Read mask as grayscale and binarize
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (original_mask > 0).astype(np.uint8)

        # Apply transformations (if any) to the image and mask
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert image to PyTorch tensor with channels first (C, H, W)
        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()

        return image, mask, original_image, original_mask


# Custom dataset class for SegFormer model to perform semantic segmentation
class SegformerSemanticSegmentationDataset(Dataset):

    # Initialize the dataset class with image and mask paths, transformations,
    # and image processor for SegFormer
    def __init__(self, image_paths, mask_paths, transform=None, image_processor=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_processor = image_processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        # print(image_path)

        # Read image and convert to RGB
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Read mask in grayscale
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure binary mask (damage = 1, background = 0)
        original_mask = (original_mask > 0).astype(np.uint8)

        # Apply transformations if any
        if self.transform:
            transformed = self.transform(image=original_image, mask=original_mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Prepare inputs for the model
        inputs = self.image_processor(image, mask, return_tensors="pt")

        # Squeeze the input tensors to remove unnecessary dimensions
        for k, v in inputs.items():
            inputs[k] = inputs[k].squeeze_()

        return inputs, original_image, original_mask


# It is a helper function to get image and mask paths from specified directories
def custom_dataset(image_folder_path, mask_folder_path):

    # Get all image filenames with ".jpg" extension
    filenames = [f for f in os.listdir(image_folder_path) if f.endswith(".jpg")]

    # Create full paths for images and masks
    image_paths = [os.path.join(image_folder_path, f) for f in filenames]
    mask_paths = [os.path.join(mask_folder_path, f) for f in filenames]

    return image_paths, mask_paths


# Collate function for DINOv2 dataset, used to stack images, and masks in batches
# while ignoring the original images, and original masks
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    return images, masks


# Collate function for SegFormer dataset, used to stack data in batches
def segformer_collate_fn(batch):
    pixel_values = torch.stack([item[0]["pixel_values"] for item in batch])
    labels = torch.stack([item[0]["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


# This function helps in preparing the datasets for training, validation, and testing
def get_dataset(config, model_name):

    # Initialzes transformations for training and validation
    train_transform = A.Compose([], is_check_shapes=False)
    val_transform = A.Compose([], is_check_shapes=False)

    if model_name == "dino_v2":
        train_transform, val_transform = get_dino_v2_transforms()
    elif model_name == "segformer":
        train_transform, val_transform = get_segformer_transforms()

    processed_data_dir = get_config_key(config, "data.processed_data_dir")
    image_dir = f"{processed_data_dir}/img"
    mask_dir = f"{processed_data_dir}/mask"

    image_paths, mask_paths = custom_dataset(image_dir, mask_dir)

    # Split the data into train/validation/test sets
    train_val_image_paths, test_image_paths, train_val_mask_paths, test_mask_paths = (
        train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
    )

    # Furthermore, split the training set into train and validation
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = (
        train_test_split(
            train_val_image_paths, train_val_mask_paths, test_size=0.25, random_state=42
        )
    )

    # Prepare datasets for DINOv2 model
    if model_name == "dino_v2":
        train_dataset = DINOv2SemanticSegmentationDataset(
            train_image_paths, train_mask_paths, transform=train_transform
        )
        val_dataset = DINOv2SemanticSegmentationDataset(
            val_image_paths, val_mask_paths, transform=val_transform
        )
        test_dataset = DINOv2SemanticSegmentationDataset(
            test_image_paths, test_mask_paths, transform=val_transform
        )

    # Prepare datasets for SegFormer model
    elif model_name == "segformer":

        base_model_name = get_config_key(config, "training.segformer.base_model")

        # Load the image processor for SegFormer from HuggingFace
        image_processor = AutoImageProcessor.from_pretrained(base_model_name)

        train_dataset = SegformerSemanticSegmentationDataset(
            train_image_paths,
            train_mask_paths,
            transform=train_transform,
            image_processor=image_processor,
        )
        val_dataset = SegformerSemanticSegmentationDataset(
            val_image_paths,
            val_mask_paths,
            transform=val_transform,
            image_processor=image_processor,
        )
        test_dataset = SegformerSemanticSegmentationDataset(
            test_image_paths,
            test_mask_paths,
            transform=val_transform,
            image_processor=image_processor,
        )

    return train_dataset, val_dataset, test_dataset


# This function gets data loaders for training, validation, and testing based on
# selection of the models.
def get_data_loaders(config, model_name):

    train_dataset, val_dataset, test_dataset = get_dataset(config, model_name)

    batch_size = get_config_key(config, f"training.{model_name}.batch_size")

    if model_name == "dino_v2":
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

    if model_name == "segformer":
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=segformer_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=segformer_collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=segformer_collate_fn,
        )

    return train_loader, val_loader, test_loader
