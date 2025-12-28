# src/feature_extraction/hand_features.py

"""
Combined Hand Feature Extraction Module
Based on Section 4.2 of the paper

Combines:
- Modified Region Growing (MRG) for segmentation
- Grey Level Co-occurrence Matrix (GLCM) for texture features

This generates C2 features as described in the paper.
"""

import os
import sys
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.feature_extraction.glcm_features import GLCMFeatureExtractor
from src.feature_extraction.mrg_segmentation import ModifiedRegionGrowing
from src.preprocessing.hand_preprocessing import HandPreprocessor


class HandFeatureExtractor:
    """
    Combined MRG + GLCM feature extraction for dorsal hand images
    As described in Section 4.2 of the paper
    
    This generates C2 (biometric 1 features) used for encrypting the fuzzy vault.
    """
    
    def __init__(self, n_features=Config.N_FEATURES):
        self.n_features = n_features
        
        # Initialize components
        self.preprocessor = HandPreprocessor()
        self.mrg = ModifiedRegionGrowing()
        self.glcm = GLCMFeatureExtractor()
    
    def extract_features(self, image_path):
        """
        Extract features from dorsal hand image
        
        Steps:
        1. Preprocess image (median filter, resize)
        2. Segment using MRG
        3. Extract GLCM features from segmented region
        4. Normalize and adjust to n_features length
        
        Args:
            image_path: Path to hand image
        
        Returns:
            C2: Feature vector of length n_features
        """
        # Step 1: Preprocess
        preprocessed, _ = self.preprocessor.preprocess(
            image_path, 
            return_color=True
        )
        
        # Convert to uint8 for further processing
        image_uint8 = (preprocessed * 255).astype(np.uint8)
        
        # Step 2: Segment using MRG
        segmented, mask, _ = self.mrg.apply_segmentation(image_uint8)
        
        # Step 3: Extract GLCM features from segmented image
        glcm_features = self.glcm.extract_features(segmented)
        
        # Step 4: Adjust feature length to n_features
        features = self._adjust_feature_length(glcm_features)
        
        # Step 5: Normalize features to [0, 1]
        features = self._normalize_features(features)
        
        return features
    
    def extract_features_from_image(self, image):
        """
        Extract features from image array (already loaded)
        
        Args:
            image: Image array (preprocessed or raw)
        
        Returns:
            Feature vector
        """
        # Ensure proper format
        if image.dtype in [np.float32, np.float64]:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        # Segment using MRG
        segmented, mask, _ = self.mrg.apply_segmentation(image_uint8)
        
        # Extract GLCM features
        glcm_features = self.glcm.extract_features(segmented)
        
        # Adjust and normalize
        features = self._adjust_feature_length(glcm_features)
        features = self._normalize_features(features)
        
        return features
    
    def _adjust_feature_length(self, features):
        """
        Adjust feature vector to exactly n_features length
        
        If shorter: tile/repeat features
        If longer: truncate
        """
        current_length = len(features)
        
        if current_length < self.n_features:
            # Tile features to reach required length
            repeats = (self.n_features // current_length) + 1
            tiled = np.tile(features, repeats)
            adjusted = tiled[:self.n_features]
        else:
            # Truncate
            adjusted = features[:self.n_features]
        
        return adjusted
    
    def _normalize_features(self, features):
        """
        Normalize features to [0, 1] range
        """
        min_val = features.min()
        max_val = features.max()
        
        if max_val - min_val > 1e-10:
            normalized = (features - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(features)
        
        return normalized
    
    def batch_extract(self, image_paths):
        """
        Extract features from multiple images
        
        Args:
            image_paths: List of image paths
        
        Returns:
            Dictionary mapping image path to features
        """
        all_features = {}
        
        for path in image_paths:
            try:
                features = self.extract_features(path)
                all_features[path] = features
            except Exception as e:
                print(f"Error extracting features from {path}: {e}")
        
        return all_features


def test_hand_features():
    """Test combined hand feature extraction"""
    
    print("=" * 60)
    print("TESTING HAND FEATURE EXTRACTION (MRG + GLCM)")
    print("=" * 60)
    
    extractor = HandFeatureExtractor()
    
    # Load sample images
    hand_dir = Config.HAND_DATA_DIR
    images = sorted([f for f in os.listdir(hand_dir) 
                     if f.lower().endswith(Config.HAND_IMAGE_FORMATS)])[:5]
    
    if not images:
        print("No hand images found!")
        return
    
    print(f"\nExtracting features from {len(images)} images...")
    
    all_features = []
    
    for img_name in images:
        img_path = os.path.join(hand_dir, img_name)
        
        features = extractor.extract_features(img_path)
        all_features.append(features)
        
        print(f"\n  {img_name}:")
        print(f"    Features shape: {features.shape}")
        print(f"    Range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"    Mean: {features.mean():.4f}")
    
    # Analyze feature consistency
    all_features = np.array(all_features)
    
    print("\n" + "-" * 40)
    print("Feature Statistics Across Images:")
    print(f"  Shape: {all_features.shape}")
    print(f"  Mean per feature: {all_features.mean(axis=0).mean():.4f}")
    print(f"  Std per feature: {all_features.std(axis=0).mean():.4f}")
    
    # Compute pairwise distances
    print("\nPairwise Euclidean Distances:")
    for i in range(len(all_features)):
        for j in range(i + 1, len(all_features)):
            dist = np.sqrt(np.sum((all_features[i] - all_features[j]) ** 2))
            print(f"  Image {i+1} vs Image {j+1}: {dist:.4f}")
    
    # Visualize features
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Feature vectors
    for i, features in enumerate(all_features):
        axes[0].plot(features, label=f'Image {i+1}', alpha=0.7)
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Feature Value')
    axes[0].set_title('Hand Feature Vectors (C2)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Feature heatmap
    im = axes[1].imshow(all_features, aspect='auto', cmap='viridis')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Image')
    axes[1].set_title('Feature Heatmap')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    
    output_path = os.path.join(Config.LOG_DIR, 'hand_features_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("HAND FEATURE EXTRACTION TEST COMPLETE")
    print("=" * 60)
    
    return all_features


if __name__ == "__main__":
    test_hand_features()