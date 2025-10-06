import yaml
from pathlib import Path
from ultralytics import YOLO

class CelebrityDetectionTrainer:
    def __init__(self, data_yaml_path, model_size='n', device='0'):
        """
        Initialize trainer for celebrity detection.
        
        Args:
            data_yaml_path: Path to data.yaml
            model_size: YOLOv8 model size
            device: Device for training ('0' for GPU, 'cpu' for CPU)
        """
        self.data_yaml = Path(data_yaml_path)
        self.model_size = model_size
        self.device = device
        self.model = None
        
        # Verify data.yaml exists
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Data config not found: {self.data_yaml}")
        
        print(f"Training Configuration:")
        print(f"  Data config: {self.data_yaml}")
        print(f"  Model: YOLOv8{model_size}")
        print(f"  Device: {device}")
    
    def train(self, epochs=100, imgsz=640, batch=16, name='celebrity_detector', 
              patience=50, save_period=10, optimizer='auto', lr0=0.01, momentum=0.937):
        """
        Train YOLOv8 model.
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            name: Project name
            patience: Early stopping patience
            save_period: Save checkpoint every n epochs
            optimizer: Optimizer type
            lr0: Initial learning rate
            momentum: Momentum
        """
        print("\n" + "="*60)
        print(f"Starting Training: {name}")
        print("="*60)
        
        # Initialize model
        self.model = YOLO(f'yolov8{self.model_size}.pt')
        
        # Train
        results = self.model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            patience=patience,
            save=True,
            save_period=save_period,
            device=self.device,
            plots=True,
            optimizer=optimizer,
            lr0=lr0,
            momentum=momentum,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("Training Completed!")
        print(f"Best weights: runs/detect/{name}/weights/best.pt")
        print(f"Last weights: runs/detect/{name}/weights/last.pt")
        print("="*60)
        
        return results
    
    def validate(self, weights_path=None):
        """
        Validate model on validation set.
        
        Args:
            weights_path: Path to weights (optional)
        """
        if weights_path:
            self.model = YOLO(weights_path)
        
        if self.model is None:
            raise ValueError("No model loaded. Train first or provide weights_path")
        
        print("\nValidating model...")
        metrics = self.model.val()
        
        print("\nValidation Metrics:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        
        return metrics
