# tests/test_feature_extraction.py
"""
Tests for feature extraction modules
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFaceFeatures:
    """Tests for face feature extraction."""
    
    def test_face_model_initialization(self):
        """Test face model initializes correctly."""
        from models.face_model import FaceRecognitionModel
        
        model = FaceRecognitionModel()
        assert model.feature_dim == 128
        assert model.input_size == (160, 160)
    
    def test_face_feature_extraction(self, sample_face_image):
        """Test face feature extraction produces correct output."""
        from models.face_model import FaceRecognitionModel
        
        model = FaceRecognitionModel()
        preprocessed = model.preprocess(sample_face_image)
        features = model.extract_features(preprocessed)
        
        assert features.shape == (128,)
        assert not np.isnan(features).any()
    
    def test_face_similarity(self, sample_face_image):
        """Test face similarity computation."""
        from models.face_model import FaceRecognitionModel
        
        model = FaceRecognitionModel()
        preprocessed = model.preprocess(sample_face_image)
        features1 = model.extract_features(preprocessed)
        features2 = model.extract_features(preprocessed)
        
        similarity = model.compute_similarity(features1, features2)
        
        assert 0 <= similarity <= 1
        assert similarity > 0.9  # Same image should have high similarity


class TestHandFeatures:
    """Tests for hand feature extraction."""
    
    def test_hand_model_initialization(self):
        """Test hand model initializes correctly."""
        from models.hand_model import HandFeatureModel
        
        model = HandFeatureModel()
        assert model.feature_dim == 64
        assert model.input_size == (128, 128)
    
    def test_hand_feature_extraction(self, sample_hand_image):
        """Test hand feature extraction produces correct output."""
        from models.hand_model import HandFeatureModel
        
        model = HandFeatureModel()
        preprocessed = model.preprocess(sample_hand_image)
        features = model.extract_features(preprocessed)
        
        assert features.shape == (64,)
        assert not np.isnan(features).any()
    
    def test_glcm_computation(self, sample_hand_image):
        """Test GLCM computation."""
        from models.hand_model import HandFeatureModel
        
        model = HandFeatureModel()
        preprocessed = model.preprocess(sample_hand_image)
        glcm = model.compute_glcm(preprocessed, distance=1, angle=0)
        
        assert glcm.shape == (8, 8)
        assert np.abs(np.sum(glcm) - 1.0) < 0.01  # Should be normalized


def test_feature_dimensions():
    """Test that combined features have correct dimensions."""
    from models.face_model import FaceRecognitionModel
    from models.hand_model import HandFeatureModel
    
    face_model = FaceRecognitionModel()
    hand_model = HandFeatureModel()
    
    # Create test images
    face_img = np.random.rand(160, 160, 3).astype(np.float32)
    hand_img = np.random.rand(128, 128).astype(np.float32)
    
    face_features = face_model.extract_features(face_model.preprocess(face_img))
    hand_features = hand_model.extract_features(hand_model.preprocess(hand_img))
    
    # Combined features
    combined = np.concatenate([face_features, hand_features])
    
    assert combined.shape == (192,)  # 128 + 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
