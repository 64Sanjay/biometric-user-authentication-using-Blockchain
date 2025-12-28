# models/__init__.py
"""
Pre-trained models and model utilities for Biometric Authentication System
"""

from .face_model import FaceRecognitionModel
from .hand_model import HandFeatureModel

__all__ = ['FaceRecognitionModel', 'HandFeatureModel']
