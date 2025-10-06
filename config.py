from pathlib import Path

class Config:
    """Project configuration."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    CELEBRITY_DIR = DATA_DIR / "Celebrity_Image_Subsets"
    DATASET_DIR = DATA_DIR / "celebrity_detection_dataset"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Data Augmentation
    TARGET_IMAGES_PER_CELEBRITY = 100
    
    # Dataset Generation
    GRID_SIZE = (6, 6)  # 6x6 grid = 36 celebrities per image
    IMG_SIZE = 640
    NUM_TRAIN = 800
    NUM_VAL = 150
    NUM_TEST = 50
    
    # Model
    MODEL_SIZE = 'n'  # Options: 'n', 's', 'm', 'l', 'x'
    MODEL_PATH = None  # Will be set after training
    
    # Training
    EPOCHS = 100
    BATCH_SIZE = 16
    DEVICE = '0'  # '0' for GPU, 'cpu' for CPU
    PATIENCE = 50
    LEARNING_RATE = 0.01
    MOMENTUM = 0.937
    
    # Testing
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    
    @classmethod
    def get_data_yaml_path(cls):
        """Get path to data.yaml file."""
        return cls.DATASET_DIR / "data.yaml"
    
    @classmethod
    def get_best_model_path(cls, project_name='celebrity_detector'):
        """Get path to best trained model."""
        return cls.PROJECT_ROOT / "runs" / "detect" / project_name / "weights" / "best.pt"
