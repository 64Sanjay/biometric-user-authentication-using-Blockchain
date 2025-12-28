# src/feature_extraction/glcm_features.py

"""
Grey Level Co-occurrence Matrix (GLCM) Feature Extraction
Based on Section 4.2.2 of the paper

GLCM measures texture by computing the spatial relationship
between pairs of pixel values.

Features extracted:
- Energy (Equation 1)
- Entropy (Equation 2)
- Homogeneity (Equation 3)
- Maximum Probability (Equation 4)
- Contrast
- Correlation
"""

import os
import sys
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config


class GLCMFeatureExtractor:
    """
    GLCM-based texture feature extraction
    As described in Section 4.2.2 of the paper
    """
    
    def __init__(self):
        # GLCM parameters from config
        self.distances = Config.GLCM_DISTANCES  # [1, 2, 3]
        self.angles = Config.GLCM_ANGLES  # [0, π/4, π/2, 3π/4]
        
        # Properties to extract
        self.properties = ['energy', 'homogeneity', 'contrast', 'correlation']
    
    def _ensure_grayscale_uint8(self, image):
        """
        Convert image to grayscale uint8 format for GLCM
        """
        # Handle float images [0, 1]
        if image.dtype in [np.float32, np.float64]:
            image = (image * 255).astype(np.uint8)
        
        # Handle color images
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image
    
    def compute_glcm(self, image):
        """
        Compute Grey Level Co-occurrence Matrix
        
        Args:
            image: Input image (grayscale or color)
        
        Returns:
            GLCM matrix
        """
        # Ensure proper format
        image = self._ensure_grayscale_uint8(image)
        
        # Compute GLCM
        glcm = graycomatrix(
            image,
            distances=self.distances,
            angles=self.angles,
            levels=256,
            symmetric=True,
            normed=True
        )
        
        return glcm
    
    def compute_energy(self, glcm):
        """
        Energy = Σ P(i,j)²
        Equation (1) in the paper
        
        Energy measures textural uniformity.
        High energy = uniform texture
        """
        return graycoprops(glcm, 'energy').flatten()
    
    def compute_entropy(self, glcm):
        """
        Entropy = -Σ P(i,j) * log2(P(i,j))
        Equation (2) in the paper
        
        Entropy measures randomness/complexity.
        High entropy = complex texture
        """
        # Normalize GLCM
        eps = 1e-10
        glcm_sum = glcm.sum(axis=(0, 1), keepdims=True)
        glcm_normalized = glcm / (glcm_sum + eps)
        
        # Compute entropy for each distance-angle combination
        entropy_values = []
        for d in range(glcm.shape[2]):
            for a in range(glcm.shape[3]):
                glcm_slice = glcm_normalized[:, :, d, a]
                entropy = -np.sum(glcm_slice * np.log2(glcm_slice + eps))
                entropy_values.append(entropy)
        
        return np.array(entropy_values)
    
    def compute_homogeneity(self, glcm):
        """
        Homogeneity = Σ P(i,j) / (1 + |i-j|)
        Equation (3) in the paper
        
        Homogeneity measures closeness of distribution to diagonal.
        High homogeneity = uniform local texture
        """
        return graycoprops(glcm, 'homogeneity').flatten()
    
    def compute_max_probability(self, glcm):
        """
        Maximum Probability = max(P(i,j))
        Equation (4) in the paper
        
        Maximum probability of any co-occurrence.
        """
        max_probs = []
        for d in range(glcm.shape[2]):
            for a in range(glcm.shape[3]):
                max_probs.append(np.max(glcm[:, :, d, a]))
        
        return np.array(max_probs)
    
    def compute_contrast(self, glcm):
        """
        Contrast measures local intensity variation.
        High contrast = high local variation
        """
        return graycoprops(glcm, 'contrast').flatten()
    
    def compute_correlation(self, glcm):
        """
        Correlation measures linear dependency of grey levels.
        """
        return graycoprops(glcm, 'correlation').flatten()
    
    def extract_features(self, image):
        """
        Extract all GLCM features from image
        
        Args:
            image: Input image
        
        Returns:
            Feature vector containing all GLCM features
        """
        # Compute GLCM
        glcm = self.compute_glcm(image)
        
        # Extract all features
        energy = self.compute_energy(glcm)
        entropy = self.compute_entropy(glcm)
        homogeneity = self.compute_homogeneity(glcm)
        max_prob = self.compute_max_probability(glcm)
        contrast = self.compute_contrast(glcm)
        correlation = self.compute_correlation(glcm)
        
        # Concatenate all features
        features = np.concatenate([
            energy,
            entropy,
            homogeneity,
            max_prob,
            contrast,
            correlation
        ])
        
        return features
    
    def get_feature_names(self):
        """Get names of all features"""
        
        names = []
        n_combinations = len(self.distances) * len(self.angles)
        
        for feature_type in ['energy', 'entropy', 'homogeneity', 
                             'max_prob', 'contrast', 'correlation']:
            for i in range(n_combinations):
                names.append(f'{feature_type}_{i}')
        
        return names


def test_glcm_features():
    """Test GLCM feature extraction"""
    
    print("=" * 60)
    print("TESTING GLCM FEATURE EXTRACTION")
    print("=" * 60)
    
    extractor = GLCMFeatureExtractor()
    
    # Load a sample hand image
    hand_dir = Config.HAND_DATA_DIR
    images = sorted([f for f in os.listdir(hand_dir) 
                     if f.lower().endswith(Config.HAND_IMAGE_FORMATS)])
    
    if not images:
        print("No hand images found!")
        return
    
    sample_path = os.path.join(hand_dir, images[0])
    print(f"\nTest image: {sample_path}")
    
    # Load and preprocess image
    image = cv2.imread(sample_path)
    image = cv2.resize(image, Config.HAND_IMAGE_SIZE)
    print(f"Image shape: {image.shape}")
    
    # Extract features
    features = extractor.extract_features(image)
    
    print(f"\nExtracted Features:")
    print(f"  Total features: {len(features)}")
    print(f"  Feature range: [{features.min():.6f}, {features.max():.6f}]")
    print(f"  Feature mean: {features.mean():.6f}")
    print(f"  Feature std: {features.std():.6f}")
    
    # Print feature breakdown
    n_combinations = len(Config.GLCM_DISTANCES) * len(Config.GLCM_ANGLES)
    print(f"\nFeature Breakdown ({n_combinations} combinations per type):")
    
    feature_types = ['Energy', 'Entropy', 'Homogeneity', 
                     'Max Prob', 'Contrast', 'Correlation']
    
    start = 0
    for ftype in feature_types:
        end = start + n_combinations
        fvalues = features[start:end]
        print(f"  {ftype:12s}: mean={fvalues.mean():.6f}, "
              f"min={fvalues.min():.6f}, max={fvalues.max():.6f}")
        start = end
    
    print("\n" + "=" * 60)
    print("GLCM FEATURE EXTRACTION TEST COMPLETE")
    print("=" * 60)
    
    return features


if __name__ == "__main__":
    test_glcm_features()