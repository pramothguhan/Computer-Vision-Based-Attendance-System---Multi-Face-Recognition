import sys
from pathlib import Path
sys.path.append('src')

from dataset.dataset_generator import CelebrityYOLODataset

def main():
    """Generate celebrity detection dataset."""
    
    # Configuration
    BASE_DIR = "data/Celebrity_Image_Subsets"
    OUTPUT_DIR = "data/celebrity_detection_dataset"
    GRID_SIZE = (6, 6)
    IMG_SIZE = 640
    
    # Dataset splits
    NUM_TRAIN = 800
    NUM_VAL = 150
    NUM_TEST = 50
    
    print("="*60)
    print("Celebrity Detection Dataset Generation")
    print("="*60)
    print(f"Input Directory: {BASE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Grid Size: {GRID_SIZE[0]}x{GRID_SIZE[1]}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Train/Val/Test: {NUM_TRAIN}/{NUM_VAL}/{NUM_TEST}")
    print("="*60 + "\n")
    
    # Initialize generator
    generator = CelebrityYOLODataset(
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        grid_size=GRID_SIZE,
        img_size=IMG_SIZE
    )
    
    # Generate dataset
    generator.generate_dataset(
        num_train=NUM_TRAIN,
        num_val=NUM_VAL,
        num_test=NUM_TEST
    )
    
    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print(f"Dataset saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
