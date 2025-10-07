import os
import cv2
import numpy as np
import random
import yaml
from pathlib import Path
from tqdm import tqdm

class CelebrityYOLODataset:
    def __init__(self, base_dir, output_dir, grid_size=(6, 6), img_size=640):
        """
        Initialize the celebrity detection dataset generator.
        
        Args:
            base_dir: Directory containing celebrity subfolders
            output_dir: Output directory for the dataset
            grid_size: Grid layout (rows, cols) for concatenation
            img_size: Size of each concatenated image
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.grid_size = grid_size
        self.img_size = img_size
        self.cell_size = img_size // grid_size[0]
        
        self.create_directories()
        
        self.celebrity_folders = sorted([f for f in self.base_dir.iterdir() if f.is_dir()])
        self.num_celebrities = len(self.celebrity_folders)
        self.celebrity_to_id = {folder.name: idx for idx, folder in enumerate(self.celebrity_folders)}
        
        print(f"Found {self.num_celebrities} celebrities")
        print(f"Grid size: {grid_size[0]}x{grid_size[1]} = {grid_size[0]*grid_size[1]} images")
        print(f"Cell size: {self.cell_size}x{self.cell_size} pixels")
    
    def create_directories(self):
        """Create directory structure for YOLO dataset."""
        dirs = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'images' / 'test',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val',
            self.output_dir / 'labels' / 'test',
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def load_celebrity_images(self):
        """Load all celebrity images organized by celebrity ID."""
        celebrity_images = {}
        
        for celeb_folder in self.celebrity_folders:
            celeb_name = celeb_folder.name
            celeb_id = self.celebrity_to_id[celeb_name]
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            images = [
                str(f) for f in celeb_folder.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            
            celebrity_images[celeb_id] = {
                'name': celeb_name,
                'images': images
            }
            
            print(f"Celebrity {celeb_id} ({celeb_name}): {len(images)} images")
        
        return celebrity_images
    
    def create_concatenated_image(self, celebrity_images):
        """
        Create a single concatenated image with grid of celebrities.
        
        Returns:
            concat_img: The grid image
            annotations: YOLO format annotations
        """
        rows, cols = self.grid_size
        concat_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        annotations = []
        
        for row in range(rows):
            for col in range(cols):
                celeb_id = random.randint(0, self.num_celebrities - 1)
                img_path = random.choice(celebrity_images[celeb_id]['images'])
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img_resized = cv2.resize(img, (self.cell_size, self.cell_size))
                
                y_start = row * self.cell_size
                y_end = y_start + self.cell_size
                x_start = col * self.cell_size
                x_end = x_start + self.cell_size
                
                concat_img[y_start:y_end, x_start:x_end] = img_resized
                
                # YOLO format: normalized coordinates
                x_center = (x_start + self.cell_size / 2) / self.img_size
                y_center = (y_start + self.cell_size / 2) / self.img_size
                width = self.cell_size / self.img_size
                height = self.cell_size / self.img_size
                
                annotations.append([celeb_id, x_center, y_center, width, height])
        
        return concat_img, annotations
    
    def generate_dataset(self, num_train=800, num_val=150, num_test=50):
        """Generate complete dataset with train/val/test splits."""
        celebrity_images = self.load_celebrity_images()
        
        splits = {
            'train': num_train,
            'val': num_val,
            'test': num_test
        }
        
        for split_name, num_images in splits.items():
            print(f"\nGenerating {split_name} split: {num_images} images")
            
            for i in tqdm(range(num_images), desc=f"Creating {split_name}"):
                concat_img, annotations = self.create_concatenated_image(celebrity_images)
                
                img_filename = f"{split_name}_{i:05d}.jpg"
                img_path = self.output_dir / 'images' / split_name / img_filename
                cv2.imwrite(str(img_path), concat_img)
                
                label_filename = f"{split_name}_{i:05d}.txt"
                label_path = self.output_dir / 'labels' / split_name / label_filename
                
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
        
        self.create_yaml_config()
        
        print("\n" + "="*60)
        print("Dataset generation completed!")
        print(f"Total images: {sum(splits.values())}")
        print(f"Output directory: {self.output_dir}")
    
    def create_yaml_config(self):
        """Create YAML configuration file for YOLOv8."""
        class_names = [self.celebrity_folders[i].name for i in range(self.num_celebrities)]
        
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.num_celebrities,
            'names': class_names
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"\nYAML config saved to: {yaml_path}")
