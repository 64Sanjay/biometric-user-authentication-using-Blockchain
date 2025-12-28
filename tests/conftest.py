# tests/conftest.py
"""
Pytest configuration and fixtures for testing
"""

import pytest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_face_image():
    """Generate sample face image for testing."""
    return np.random.rand(160, 160, 3).astype(np.float32)


@pytest.fixture
def sample_hand_image():
    """Generate sample hand image for testing."""
    return np.random.rand(128, 128).astype(np.float32)


@pytest.fixture
def sample_user_id():
    """Sample user ID for testing."""
    return "test_user_001"


@pytest.fixture
def sample_vault_data():
    """Sample vault data for testing."""
    return {
        'user_id': 'test_user_001',
        'fuzzy_vault': [0.1, 0.2, 0.3, 0.4, 0.5],
        'bio_token': {
            'face_features': [0.1] * 128,
            'hand_features': [0.2] * 64
        },
        'biometric_type': 'face_hand'
    }
