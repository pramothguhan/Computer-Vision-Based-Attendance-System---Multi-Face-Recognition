import cv2
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class CelebrityDetector:
    def __init__(self, model_path, data_yaml_path=None, conf_threshold=0.25):
        """
        Initialize celebrity detector.
        
        Args:
            model_path: Path to trained model weights
            data_yaml_path: Path to data.yaml (optional, for class names)
            conf_threshold: Confidence threshold
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load model
        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Load class names
        if data_yaml_path:
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            self.class_names = data_config['names']
        else:
            self.class_names = self.model.names
        
        print(f"Model loaded! Classes: {len(self.class_names)}")
    
    def detect_image(self, image_path, save_path=None, show_conf=True):
        """
        Detect celebrities in a single image.
        
        Args:
            image_path: Path to input image
            save_path: Path to save annotated image
            show_conf: Whether to show confidence scores
        
        Returns:
            detections: List of detected celebrities
        """
        print(f"\nProcessing: {image_path}")
        
        # Run inference
        results = self.model(image_path, conf=self.conf_threshold)[0]
        
        # Parse detections
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            detection = {
                'celebrity_id': class_id,
                'celebrity_name': self.class_names[class_id],
                'confidence': confidence,
                'bbox': bbox.tolist(),
                'bbox_center': box.xywh[0].cpu().numpy().tolist()
            }
            detections.append(detection)
        
        # Print results
        print(f"\nDetected {len(detections)} celebrities:")
        print("-" * 80)
        for i, det in enumerate(detections, 1):
            print(f"{i:2d}. {det['celebrity_name']:30s} "
                  f"(ID: {det['celebrity_id']:3d}) "
                  f"Conf: {det['confidence']:.3f}")
            if show_conf:
                x1, y1, x2, y2 = det['bbox']
                print(f"    Location: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
        print("-" * 80)
        
        # Save annotated image
        if save_path:
            annotated_img = results.plot()
            cv2.imwrite(save_path, annotated_img)
            print(f"\nAnnotated image saved: {save_path}")
        
        return detections, results
    
    def detect_batch(self, image_dir, output_dir=None, save_annotations=True):
        """
        Detect celebrities in multiple images.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            save_annotations: Whether to save annotated images
        
        Returns:
            all_detections: Dictionary of detections per image
        """
        image_dir = Path(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if output_dir and save_annotations:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        all_detections = {}
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        for img_path in image_paths:
            save_path = None
            if output_dir and save_annotations:
                save_path = output_dir / f"detected_{img_path.name}"
            
            detections, _ = self.detect_image(img_path, save_path, show_conf=False)
            all_detections[img_path.name] = detections
        
        print(f"\nBatch processing completed!")
        print(f"Total images: {len(image_paths)}")
        print(f"Total detections: {sum(len(d) for d in all_detections.values())}")
        
        return all_detections
    
    def evaluate_test_set(self, data_yaml_path):
        """
        Evaluate model on test set.
        
        Args:
            data_yaml_path: Path to data.yaml configuration
        """
        print("\nEvaluating on test set...")
        
        results = self.model.val(
            data=data_yaml_path,
            split='test',
            verbose=True
        )
        
        print("\nTest Set Metrics:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        print(f"  Precision: {results.box.mp:.4f}")
        print(f"  Recall: {results.box.mr:.4f}")
        
        return results


# ------------------- NEW FUNCTION -------------------
def map_detections_to_grid(detections, img_size=640, grid_rows=6, grid_cols=6):
    """
    Map YOLO detections to grid coordinates (row, col).

    Args:
        detections: list of detection dicts from detect_image()
        img_size: image dimension (assumed square)
        grid_rows, grid_cols: number of rows/cols in grid

    Returns:
        list of (row, col, celebrity_name)
    """
    cell_size = img_size / grid_rows
    mapped = []

    for det in detections:
        x_center, y_center, _, _ = det["bbox_center"]
        col = int(x_center // cell_size) + 1
        row = int(y_center // cell_size) + 1
        mapped.append((row, col, det["celebrity_name"]))

    mapped.sort(key=lambda x: (x[0], x[1]))
    return mapped