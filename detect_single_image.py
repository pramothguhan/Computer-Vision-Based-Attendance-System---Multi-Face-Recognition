import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from test.test import CelebrityDetector, map_detections_to_grid


def main():
    """Detect multiple celebrities in a single custom image."""
    
    # --- Configuration ---
    MODEL_PATH = "runs/detect/celebrity_detector/weights/best.pt"
    DATA_YAML = "data/celebrity_detection_dataset/data.yaml"
    INPUT_IMAGE = "samples/multiple_celebrities.jpg"  
    OUTPUT_IMAGE = "results/multiple_celebrities_detected.jpg"
    CONF_THRESHOLD = 0.25
    GRID_ROWS, GRID_COLS = 6, 6   # adjust if your collage uses different layout
    IMG_SIZE = 640

    # --- Initialize Detector ---
    detector = CelebrityDetector(
        model_path=MODEL_PATH,
        data_yaml_path=DATA_YAML,
        conf_threshold=CONF_THRESHOLD
    )

    # --- Run Detection ---
    detections, _ = detector.detect_image(
        image_path=INPUT_IMAGE,
        save_path=OUTPUT_IMAGE
    )

    # --- Print Detailed Results ---
    print("\nFinal Detection Summary:")
    for i, d in enumerate(detections, 1):
        print(f"[{i}] {d['celebrity_name']} (ID: {d['celebrity_id']})")
        print(f"    Confidence: {d['confidence']:.3f}")
        x1, y1, x2, y2 = d['bbox']
        print(f"    Location: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")

    # --- Map to Grid Layout ---
    grid_mapped = map_detections_to_grid(detections, img_size=IMG_SIZE,
                                         grid_rows=GRID_ROWS, grid_cols=GRID_COLS)
    
    print("\nCelebrities detected in grid layout:")
    print("=" * 60)
    for r, c, name in grid_mapped:
        print(f"Row {r}, Col {c}  →  {name}")
    print("=" * 60)

    print(f"\nAnnotated image saved to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()