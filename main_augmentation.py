import sys
from pathlib import Path
sys.path.append('src')

from dataset.augmentation import CelebrityDataAugmentation

def main():
    """Run data augmentation pipeline."""
    
    # Configuration
    BASE_DIR = "data/Celebrity_Image_Subsets"
    TARGET_IMAGES = 100
    
    print("="*60)
    print("Celebrity Image Data Augmentation")
    print("="*60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Target Images per Celebrity: {TARGET_IMAGES}")
    print("="*60 + "\n")
    
    # Initialize augmentor
    augmentor = CelebrityDataAugmentation(
        base_dir=BASE_DIR,
        target_images_per_celebrity=TARGET_IMAGES
    )
    
    # Run augmentation
    augmentor.augment_all_celebrities()
    
    print("\n" + "="*60)
    print("Augmentation Complete!")
    print("="*60)

if __name__ == "__main__":
    main()