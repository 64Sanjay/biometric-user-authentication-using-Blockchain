# models/face_model.py
"""
Face Recognition Model for Biometric Authentication
Uses Deep Learning for feature extraction (128-dimensional embeddings)
"""

import os
import numpy as np
from typing import Optional, List, Tuple
import pickle

# Try importing deep learning libraries
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class FaceRecognitionModel:
    """
    Face Recognition Model for extracting 128-dimensional face embeddings.
    
    Based on the paper methodology:
    - HOG encoding for face detection
    - Neural network for feature extraction
    - 128 measurements per face
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize face recognition model.
        
        Args:
            model_path: Path to pre-trained model weights
        """
        self.model_path = model_path
        self.model = None
        self.feature_dim = 128
        self.input_size = (160, 160)
        self.is_loaded = False
        
        # Try to load model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load pre-trained model weights."""
        try:
            if TORCH_AVAILABLE and model_path.endswith('.pth'):
                self.model = torch.load(model_path, map_location='cpu')
                self.is_loaded = True
                print(f"✅ Loaded face model from {model_path}")
                return True
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_loaded = True
                print(f"✅ Loaded face model from {model_path}")
                return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
        return False
    
    def save_model(self, model_path: str) -> bool:
        """Save model weights."""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if TORCH_AVAILABLE and self.model is not None:
                torch.save(self.model, model_path)
                print(f"✅ Saved face model to {model_path}")
                return True
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
        return False
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for model input.
        
        Args:
            image: Input face image (BGR format)
            
        Returns:
            Preprocessed image tensor
        """
        if not CV2_AVAILABLE:
            return image
        
        # Resize to model input size
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Standardize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        return image
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 128-dimensional face embedding.
        
        Args:
            image: Preprocessed face image
            
        Returns:
            128-dimensional feature vector
        """
        if self.model is not None and TORCH_AVAILABLE:
            # Use actual model
            with torch.no_grad():
                tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                features = self.model(tensor)
                return features.numpy().flatten()
        else:
            # Generate pseudo-features based on image statistics
            # This is a fallback when no model is loaded
            features = self._generate_pseudo_features(image)
            return features
    
    def _generate_pseudo_features(self, image: np.ndarray) -> np.ndarray:
        """
        Generate pseudo-features from image statistics.
        Used when no trained model is available.
        """
        np.random.seed(int(np.sum(image) * 1000) % (2**31))
        
        # Extract statistical features
        if len(image.shape) == 3:
            # Color statistics
            mean_per_channel = np.mean(image, axis=(0, 1))
            std_per_channel = np.std(image, axis=(0, 1))
            
            # Gradient features
            gray = np.mean(image, axis=2)
        else:
            gray = image
            mean_per_channel = np.array([np.mean(gray)])
            std_per_channel = np.array([np.std(gray)])
        
        # HOG-like features (simplified)
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Create feature vector
        features = np.zeros(self.feature_dim)
        
        # Fill with various statistics
        features[0:3] = mean_per_channel[:3] if len(mean_per_channel) >= 3 else [mean_per_channel[0]]*3
        features[3:6] = std_per_channel[:3] if len(std_per_channel) >= 3 else [std_per_channel[0]]*3
        features[6] = np.mean(magnitude)
        features[7] = np.std(magnitude)
        
        # Histogram features
        hist, _ = np.histogram(gray.flatten(), bins=32, range=(0, 1))
        features[8:40] = hist / np.sum(hist)
        
        # Region-based features
        h, w = gray.shape[:2]
        regions = [
            gray[:h//2, :w//2],  # Top-left
            gray[:h//2, w//2:],  # Top-right
            gray[h//2:, :w//2],  # Bottom-left
            gray[h//2:, w//2:]   # Bottom-right
        ]
        
        for i, region in enumerate(regions):
            features[40 + i*4:40 + i*4 + 4] = [
                np.mean(region),
                np.std(region),
                np.min(region),
                np.max(region)
            ]
        
        # Fill remaining with random but deterministic values
        remaining = self.feature_dim - 56
        features[56:] = np.random.randn(remaining) * 0.1
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.astype(np.float32)
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute similarity between two face embeddings.
        
        Args:
            features1: First face embedding
            features2: Second face embedding
            
        Returns:
            Similarity score (0 to 1, higher is more similar)
        """
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Convert to range [0, 1]
        return (similarity + 1) / 2
    
    def verify(self, features1: np.ndarray, features2: np.ndarray, 
               threshold: float = 0.6) -> Tuple[bool, float]:
        """
        Verify if two face embeddings belong to the same person.
        
        Args:
            features1: First face embedding
            features2: Second face embedding
            threshold: Similarity threshold for match
            
        Returns:
            Tuple of (is_match, similarity_score)
        """
        similarity = self.compute_similarity(features1, features2)
        is_match = similarity >= threshold
        return is_match, similarity


# Test function
def test_face_model():
    """Test face recognition model."""
    print("=" * 50)
    print("Testing Face Recognition Model")
    print("=" * 50)
    
    model = FaceRecognitionModel()
    
    # Create test image
    test_image = np.random.rand(160, 160, 3).astype(np.float32)
    
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
    
    # Test with different image
    test_image2 = np.random.rand(160, 160, 3).astype(np.float32)
    preprocessed2 = model.preprocess(test_image2)
    features3 = model.extract_features(preprocessed2)
    similarity2 = model.compute_similarity(features, features3)
    print(f"Different image similarity: {similarity2:.4f}")
    
    print("\n✅ Face model test complete!")
    return model


if __name__ == "__main__":
    test_face_model()
