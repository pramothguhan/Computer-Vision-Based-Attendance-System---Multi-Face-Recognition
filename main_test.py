import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from test.test import CelebrityDetector

def main():
    """Test celebrity detection model."""
    
    # Configuration
    MODEL_PATH = "runs/detect/celebrity_detector/weights/best.pt"
    DATA_YAML = "data/celebrity_detection_dataset/data.yaml"
    TEST_IMAGE = "data/celebrity_detection_dataset/images/test/test_00000.jpg"
    OUTPUT_PATH = "test_results/detection_result.jpg"
    CONF_THRESHOLD = 0.25
    
    print("="*60)
    print("Celebrity Detection Testing")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Test Image: {TEST_IMAGE}")
    print(f"Confidence Threshold: {CONF_THRESHOLD}")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = CelebrityDetector(
        model_path=MODEL_PATH,
        data_yaml_path=DATA_YAML,
        conf_threshold=CONF_THRESHOLD
    )
    
    # Test single image
    print("\n--- Single Image Detection ---")
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    detections, results = detector.detect_image(
        image_path=TEST_IMAGE,
        save_path=OUTPUT_PATH
    )
    
    # Test batch of images
    print("\n\n--- Batch Detection ---")
    TEST_DIR = "data/celebrity_detection_dataset/images/test"
    OUTPUT_DIR = "test_results/batch_detection"
    
    all_detections = detector.detect_batch(
        image_dir=TEST_DIR,
        output_dir=OUTPUT_DIR,
        save_annotations=True
    )
    
    # Evaluate on test set
    print("\n\n--- Test Set Evaluation ---")
    detector.evaluate_test_set(DATA_YAML)
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print(f"Results saved to: test_results/")
    print("="*60)

if __name__ == "__main__":
    main()