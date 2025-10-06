from ultralytics import YOLO
from pathlib import Path

class CelebrityYOLOModel:
    def __init__(self, model_size='n', pretrained=True):
        """
        Initialize YOLOv8 model for celebrity detection.
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            pretrained: Whether to use pretrained weights
        """
        self.model_size = model_size
        self.pretrained = pretrained
        self.model = None
        
        self.model_configs = {
            'n': {'params': '3.2M', 'description': 'Nano - Fastest'},
            's': {'params': '11.2M', 'description': 'Small - Balanced'},
            'm': {'params': '25.9M', 'description': 'Medium - Accurate'},
            'l': {'params': '43.7M', 'description': 'Large - Very Accurate'},
            'x': {'params': '68.2M', 'description': 'Extra Large - Most Accurate'}
        }
        
        print(f"Model: YOLOv8{model_size}")
        print(f"Config: {self.model_configs[model_size]['description']}")
        print(f"Parameters: {self.model_configs[model_size]['params']}")
    
    def load_model(self, weights_path=None):
        """
        Load YOLOv8 model.
        
        Args:
            weights_path: Path to custom weights, None for pretrained
        """
        if weights_path:
            print(f"Loading custom weights from: {weights_path}")
            self.model = YOLO(weights_path)
        else:
            model_name = f'yolov8{self.model_size}.pt'
            print(f"Loading pretrained model: {model_name}")
            self.model = YOLO(model_name)
        
        return self.model
    
    def get_model_info(self):
        """Get model architecture information."""
        if self.model is None:
            return "Model not loaded"
        
        return self.model.info()