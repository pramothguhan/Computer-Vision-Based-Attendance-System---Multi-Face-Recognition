import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from tqdm import tqdm

class CelebrityDataAugmentation:
    def __init__(self, base_dir, target_images_per_celebrity=100):
        """
        Initialize the data augmentation pipeline for celebrity images.
        
        Args:
            base_dir: Base directory containing celebrity subfolders
            target_images_per_celebrity: Target number of images per celebrity
        """
        self.base_dir = Path(base_dir)
        self.target_images = target_images_per_celebrity
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.Rotate(limit=15, p=1.0),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=1.0),
            ], p=0.8),
            
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
                A.CLAHE(clip_limit=2.0, p=1.0),
            ], p=0.7),
            
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.5),
            
            A.OneOf([
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            ], p=0.5),
            
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
        ])
    
    def augment_celebrity_folder(self, celebrity_folder):
        """Augment images in a single celebrity folder."""
        folder_path = Path(celebrity_folder)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        original_images = [
            f for f in folder_path.iterdir() 
            if f.suffix.lower() in image_extensions and not f.stem.startswith('aug_')
        ]
        
        num_original = len(original_images)
        if num_original == 0:
            print(f"No images found in {folder_path.name}")
            return
        
        num_to_generate = self.target_images - num_original
        
        if num_to_generate <= 0:
            print(f"{folder_path.name}: Already has {num_original} images")
            return
        
        print(f"{folder_path.name}: Generating {num_to_generate} augmented images")
        
        aug_count = 0
        while aug_count < num_to_generate:
            source_img_path = np.random.choice(original_images)
            
            image = cv2.imread(str(source_img_path))
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=image_rgb)
            augmented_image = augmented['image']
            augmented_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            
            aug_filename = f"aug_{aug_count:04d}_{source_img_path.stem}{source_img_path.suffix}"
            aug_path = folder_path / aug_filename
            
            cv2.imwrite(str(aug_path), augmented_bgr)
            aug_count += 1
        
        print(f"{folder_path.name}: Completed! Total: {num_original + num_to_generate}")
    
    def augment_all_celebrities(self):
        """Augment images for all celebrity subfolders."""
        celebrity_folders = [f for f in self.base_dir.iterdir() if f.is_dir()]
        
        if not celebrity_folders:
            print(f"No subdirectories found in {self.base_dir}")
            return
        
        print(f"Found {len(celebrity_folders)} celebrity folders")
        print(f"Target: {self.target_images} images per celebrity\n")
        
        for celeb_folder in tqdm(celebrity_folders, desc="Processing celebrities"):
            self.augment_celebrity_folder(celeb_folder)
        
        print("\nData augmentation completed!")