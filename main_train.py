import sys
from pathlib import Path
sys.path.append('src')

from training.train import CelebrityDetectionTrainer

def main():
    """Train celebrity detection model."""
    
    # Configuration
    DATA_YAML = "data/celebrity_detection_dataset/data.yaml"
    MODEL_SIZE = 'n'  # Options: 'n', 's', 'm', 'l', 'x'
    DEVICE = '0'  # '0' for GPU, 'cpu' for CPU
    
    # Training hyperparameters
    EPOCHS = 60
    IMG_SIZE = 640
    BATCH_SIZE = 16
    PROJECT_NAME = 'celebrity_detector'
    PATIENCE = 50
    
    print("="*60)
    print("Celebrity Detection Model Training")
    print("="*60)
    print(f"Data Config: {DATA_YAML}")
    print(f"Model: YOLOv8{MODEL_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = CelebrityDetectionTrainer(
        data_yaml_path=DATA_YAML,
        model_size=MODEL_SIZE,
        device=DEVICE
    )
    
    # Train model
    results = trainer.train(
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=PROJECT_NAME,
        patience=PATIENCE
    )
    
    # Validate model
    print("\n" + "="*60)
    print("Running Validation...")
    print("="*60)
    trainer.validate()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best model: runs/detect/{PROJECT_NAME}/weights/best.pt")
    print("="*60)

if __name__ == "__main__":
    main()
