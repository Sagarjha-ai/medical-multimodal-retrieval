"""
Data loading utilities for Medical Multimodal Retrieval System
"""

import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ============================================================
# CLAHE Enhancement
# ============================================================

def apply_clahe(pil_image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to enhance X-ray image contrast
    
    Args:
        pil_image: PIL Image
        
    Returns:
        Enhanced PIL Image
    """
    image = np.array(pil_image)
    
    # Convert to grayscale for CLAHE
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    image = clahe.apply(image)
    
    # Convert back to RGB
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(image)

# ============================================================
# Dataset Class
# ============================================================

class ChestXrayDataset(Dataset):
    """
    Dataset class for Chest X-ray images and reports
    """
    
    def __init__(self, dataframe, transforms=None, image_size=224):
        """
        Initialize dataset
        
        Args:
            dataframe: DataFrame with image_path and caption columns
            transforms: Image transformations
            image_size: Target image size
        """
        self.dataframe = dataframe
        self.transforms = transforms
        self.image_size = image_size
        
    def __len__(self):
        """Return dataset size"""
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Get sample by index
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image and text data
        """
        row = self.dataframe.iloc[idx]
        
        # Load image
        image_path = row["image_path"]
        caption = row["structured_caption"]
        
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        
        return {
            "image": image,
            "caption": caption,
            "image_path": image_path
        }

# ============================================================
# Transform Functions
# ============================================================

def get_train_transforms(image_size=224):
    """Get training transforms with augmentation"""
    return transforms.Compose([
        transforms.Lambda(apply_clahe),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms(image_size=224):
    """Get validation transforms without augmentation"""
    return transforms.Compose([
        transforms.Lambda(apply_clahe),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_inference_transforms(image_size=224):
    """Get inference transforms"""
    return transforms.Compose([
        transforms.Lambda(apply_clahe),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ============================================================
# Data Loading Utilities
# ============================================================

def load_dataset(csv_path, transform_type="train", image_size=224):
    """
    Load dataset from CSV file
    
    Args:
        csv_path: Path to CSV file
        transform_type: Type of transforms ("train", "val", "inference")
        image_size: Target image size
        
    Returns:
        ChestXrayDataset instance
    """
    df = pd.read_csv(csv_path)
    
    if transform_type == "train":
        transforms = get_train_transforms(image_size)
    elif transform_type == "val":
        transforms = get_val_transforms(image_size)
    else:
        transforms = get_inference_transforms(image_size)
    
    return ChestXrayDataset(df, transforms, image_size)

def validate_dataset(dataset, num_samples=5):
    """
    Validate dataset by printing sample information
    
    Args:
        dataset: Dataset instance
        num_samples: Number of samples to validate
    """
    print(f"\nDataset validation - Total samples: {len(dataset)}")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Caption: {sample['caption'][:100]}...")
        print(f"  Image path: {sample['image_path']}")
