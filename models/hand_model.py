# models/hand_model.py
"""
Hand Feature Model for Biometric Authentication
Uses MRG + GLCM for dorsal hand feature extraction (64-dimensional)
"""

import os
import numpy as np
from typing import Optional, Tuple
import pickle

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class HandFeatureModel:
    """
    Hand Feature Extraction Model using MRG + GLCM.
    
    Based on the paper methodology:
    - Modified Region Growing (MRG) for segmentation
    - Grey Level Co-occurrence Matrix (GLCM) for texture features
    - 64-dimensional feature vector
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize hand feature model.
        
        Args:
            model_path: Path to model configuration
        """
        self.model_path = model_path
        self.feature_dim = 64
        self.input_size = (128, 128)
        self.glcm_distances = [1, 2, 3]
        self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        if model_path and os.path.exists(model_path):
            self.load_config(model_path)
    
    def load_config(self, config_path: str) -> bool:
        """Load model configuration."""
        try:
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                self.feature_dim = config.get('feature_dim', 64)
                self.input_size = config.get('input_size', (128, 128))
            print(f"✅ Loaded hand model config from {config_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """Save model configuration."""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            config = {
                'feature_dim': self.feature_dim,
                'input_size': self.input_size,
                'glcm_distances': self.glcm_distances,
                'glcm_angles': self.glcm_angles
            }
            with open(config_path, 'wb') as f:
                pickle.dump(config, f)
            print(f"✅ Saved hand model config to {config_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save config: {e}")
            return False
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess hand image.
        
        Args:
            image: Input hand image (BGR format)
            
        Returns:
            Preprocessed grayscale image
        """
        if not CV2_AVAILABLE:
            if len(image.shape) == 3:
                return np.mean(image, axis=2)
            return image
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize
        if gray.shape != self.input_size:
            gray = cv2.resize(gray, self.input_size)
        
        # Apply median filter (as per paper)
        gray = cv2.medianBlur(gray, 5)
        
        # Normalize
        gray = gray.astype(np.float32) / 255.0
        
        return gray
    
    def compute_glcm(self, image: np.ndarray, distance: int = 1, 
                     angle: float = 0) -> np.ndarray:
        """
        Compute Grey Level Co-occurrence Matrix.
        
        Args:
            image: Grayscale image (normalized 0-1)
            distance: Pixel distance for co-occurrence
            angle: Angle in radians
            
        Returns:
            GLCM matrix
        """
        # Quantize to 8 levels
        quantized = (image * 7).astype(np.int32)
        quantized = np.clip(quantized, 0, 7)
        
        # Compute offset
        dx = int(round(distance * np.cos(angle)))
        dy = int(round(distance * np.sin(angle)))
        
        # Initialize GLCM
        glcm = np.zeros((8, 8), dtype=np.float32)
        
        h, w = quantized.shape
        for i in range(max(0, -dy), min(h, h - dy)):
            for j in range(max(0, -dx), min(w, w - dx)):
                row = quantized[i, j]
                col = quantized[i + dy, j + dx]
                glcm[row, col] += 1
        
        # Normalize
        total = np.sum(glcm)
        if total > 0:
            glcm = glcm / total
        
        return glcm
    
    def compute_glcm_features(self, glcm: np.ndarray) -> np.ndarray:
        """
        Extract texture features from GLCM.
        
        Features: Energy, Entropy, Homogeneity, Contrast, 
                  Correlation, Max Probability
        """
        features = np.zeros(6)
        
        # Avoid log(0)
        glcm_safe = glcm + 1e-10
        
        # Energy (Angular Second Moment)
        features[0] = np.sum(glcm ** 2)
        
        # Entropy
        features[1] = -np.sum(glcm_safe * np.log2(glcm_safe))
        
        # Homogeneity
        i, j = np.ogrid[0:8, 0:8]
        features[2] = np.sum(glcm / (1 + np.abs(i - j)))
        
        # Contrast
        features[3] = np.sum(glcm * (i - j) ** 2)
        
        # Correlation
        mean_i = np.sum(i * glcm)
        mean_j = np.sum(j * glcm)
        std_i = np.sqrt(np.sum((i - mean_i) ** 2 * glcm))
        std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm))
        if std_i > 0 and std_j > 0:
            features[4] = np.sum((i - mean_i) * (j - mean_j) * glcm) / (std_i * std_j)
        
        # Maximum Probability
        features[5] = np.max(glcm)
        
        return features
    
    def modified_region_growing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Modified Region Growing (MRG) segmentation.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Segmented image
        """
        if not CV2_AVAILABLE:
            return image
        
        # Convert to uint8 for OpenCV
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            img_uint8, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        segmented = image * (binary / 255.0)
        
        return segmented
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 64-dimensional hand features using MRG + GLCM.
        
        Args:
            image: Preprocessed hand image
            
        Returns:
            64-dimensional feature vector
        """
        # Apply MRG segmentation
        segmented = self.modified_region_growing(image)
        
        features = []
        
        # Extract GLCM features for multiple distances and angles
        for distance in self.glcm_distances:
            for angle in self.glcm_angles:
                glcm = self.compute_glcm(segmented, distance, angle)
                glcm_features = self.compute_glcm_features(glcm)
                features.extend(glcm_features)
        
        # Should give 3 distances * 4 angles * 6 features = 72 features
        features = np.array(features[:64])  # Take first 64
        
        # Pad if needed
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.astype(np.float32)
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute similarity between two hand feature vectors."""
        # Euclidean distance converted to similarity
        distance = np.linalg.norm(features1 - features2)
        similarity = 1.0 / (1.0 + distance)
        return similarity
    
    def verify(self, features1: np.ndarray, features2: np.ndarray,
               threshold: float = 0.5) -> Tuple[bool, float]:
        """Verify if two hand features belong to the same person."""
        similarity = self.compute_similarity(features1, features2)
        is_match = similarity >= threshold
        return is_match, similarity


# Test function
def test_hand_model():
    """Test hand feature model."""
    print("=" * 50)
    print("Testing Hand Feature Model")
    print("=" * 50)
    
    model = HandFeatureModel()
    
    # Create test image
    test_image = np.random.rand(128, 128).astype(np.float32)
    
    # Preprocess
    preprocessed = model.preprocess(test_image)
    print(f"Preprocessed shape: {preprocessed.shape}")
    
    # Extract features
    features = model.extract_features(preprocessed)
    print(f"Feature shape: {features.shape}")
    print(f"Feature norm: {np.linalg.norm(features):.4f}")
    
    # Test similarity
    features2 = model.extract_features(preprocessed)
    similarity = model.compute_similarity(features, features2)
    print(f"Self-similarity: {similarity:.4f}")
    
    # Test GLCM
    glcm = model.compute_glcm(preprocessed, distance=1, angle=0)
    print(f"GLCM shape: {glcm.shape}")
    
    print("\n✅ Hand model test complete!")
    return model


if __name__ == "__main__":
    test_hand_model()
