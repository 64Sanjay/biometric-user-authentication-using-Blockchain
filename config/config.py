# config/config.py

import os

class Config:
    """
    Configuration settings for the Biometric Authentication System
    Based on the paper's specifications
    Updated for actual dataset paths
    """
    
    # ============================================
    # PATH CONFIGURATIONS
    # ============================================
    
    # Base directory (project root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data directories
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # ACTUAL DATASET PATHS (Updated for your structure)
    FACE_DATA_DIR = os.path.join(DATA_DIR, 'face_images', '48229375_CASIA-FaceV5')
    HAND_DATA_DIR = os.path.join(DATA_DIR, 'hand_images', 'Hands')
    
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    PROCESSED_FACE_DIR = os.path.join(PROCESSED_DIR, 'face')
    PROCESSED_HAND_DIR = os.path.join(PROCESSED_DIR, 'hand')
    
    # Model directory
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # Logs directory
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    # ============================================
    # IMAGE SETTINGS
    # ============================================
    
    # Face image size (as per CASIA-FaceV5 dataset - 640x480)
    FACE_IMAGE_SIZE = (640, 480)
    
    # Hand image size (standardized for processing)
    HAND_IMAGE_SIZE = (224, 224)
    
    # Median filter size for preprocessing (Section 4.1)
    MEDIAN_FILTER_SIZE = 3
    
    # Supported image formats
    FACE_IMAGE_FORMATS = ('.bmp', '.jpg', '.jpeg', '.png')
    HAND_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # ============================================
    # FUZZY VAULT SETTINGS (Section 4.4)
    # ============================================
    
    # Number of features (128 as per paper - Section 4.3)
    N_FEATURES = 128
    
    # Fuzzy vault tolerance
    FV_TOLERANCE = 0.1
    
    # Grid columns (3 as per paper)
    GRID_COLUMNS = 3
    
    # Authentication threshold (0.35 as per paper - Section 5.3)
    AUTH_THRESHOLD = 0.35
    
    # ============================================
    # BLOCKCHAIN SETTINGS
    # ============================================
    
    # Local blockchain URL (Ganache)
    BLOCKCHAIN_URL = "http://127.0.0.1:8545"
    
    # IPFS URL
    IPFS_URL = "/ip4/127.0.0.1/tcp/5001"
    
    # ============================================
    # GLCM SETTINGS (Section 4.2.2)
    # ============================================
    
    # GLCM distances
    GLCM_DISTANCES = [1, 2, 3]
    
    # GLCM angles (0, 45, 90, 135 degrees in radians)
    GLCM_ANGLES = [0, 0.785398, 1.5708, 2.35619]
    
    # ============================================
    # REGION GROWING SETTINGS (Section 4.2.1)
    # ============================================
    
    # MRG threshold
    MRG_THRESHOLD = 10
    
    # MRG connectivity
    MRG_CONNECTIVITY = 8
    
    # ============================================
    # DATASET SETTINGS
    # ============================================
    
    # Number of users to use for testing (can be adjusted)
    NUM_USERS = 100  # Use first 100 users for testing
    
    # Images per user
    IMAGES_PER_USER = 5  # CASIA-FaceV5 has 5 images per user
    
    # ============================================
    # PERFORMANCE SETTINGS
    # ============================================
    
    # Number of iterations for testing
    TEST_ITERATIONS = [50, 75, 100]
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    @classmethod
    def print_config(cls):
        """Print all configuration settings"""
        print("=" * 60)
        print("CONFIGURATION SETTINGS")
        print("=" * 60)
        print(f"\nBase Directory: {cls.BASE_DIR}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"\nDataset Paths:")
        print(f"  Face Images: {cls.FACE_DATA_DIR}")
        print(f"  Hand Images: {cls.HAND_DATA_DIR}")
        print(f"  Processed: {cls.PROCESSED_DIR}")
        print(f"\nImage Settings:")
        print(f"  Face Size: {cls.FACE_IMAGE_SIZE}")
        print(f"  Hand Size: {cls.HAND_IMAGE_SIZE}")
        print(f"  Median Filter: {cls.MEDIAN_FILTER_SIZE}")
        print(f"\nFuzzy Vault Settings:")
        print(f"  N Features: {cls.N_FEATURES}")
        print(f"  Tolerance: {cls.FV_TOLERANCE}")
        print(f"  Threshold: {cls.AUTH_THRESHOLD}")
        print(f"\nDataset Settings:")
        print(f"  Num Users: {cls.NUM_USERS}")
        print(f"  Images/User: {cls.IMAGES_PER_USER}")
        print("=" * 60)
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.PROCESSED_DIR,
            cls.PROCESSED_FACE_DIR,
            cls.PROCESSED_HAND_DIR,
            cls.MODEL_DIR,
            cls.LOG_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Directory ready: {directory}")
    
    @classmethod
    def verify_datasets(cls):
        """Verify that datasets exist and count images"""
        print("\n" + "=" * 60)
        print("DATASET VERIFICATION")
        print("=" * 60)
        
        # Check face dataset
        face_count = 0
        face_users = 0
        if os.path.exists(cls.FACE_DATA_DIR):
            print(f"\n✓ Face dataset found: {cls.FACE_DATA_DIR}")
            for folder in os.listdir(cls.FACE_DATA_DIR):
                folder_path = os.path.join(cls.FACE_DATA_DIR, folder)
                if os.path.isdir(folder_path):
                    face_users += 1
                    images = [f for f in os.listdir(folder_path) 
                              if f.lower().endswith(cls.FACE_IMAGE_FORMATS)]
                    face_count += len(images)
            print(f"  Users: {face_users}")
            print(f"  Total Images: {face_count}")
        else:
            print(f"\n✗ Face dataset NOT found: {cls.FACE_DATA_DIR}")
        
        # Check hand dataset
        hand_count = 0
        if os.path.exists(cls.HAND_DATA_DIR):
            print(f"\n✓ Hand dataset found: {cls.HAND_DATA_DIR}")
            images = [f for f in os.listdir(cls.HAND_DATA_DIR) 
                      if f.lower().endswith(cls.HAND_IMAGE_FORMATS)]
            hand_count = len(images)
            print(f"  Total Images: {hand_count}")
        else:
            print(f"\n✗ Hand dataset NOT found: {cls.HAND_DATA_DIR}")
        
        print("\n" + "=" * 60)
        
        return face_count, hand_count


# Test configuration
if __name__ == "__main__":
    Config.print_config()
    print("\nCreating directories...")
    Config.create_directories()
    Config.verify_datasets()
    print("\nConfiguration complete!")