# models/model_utils.py
"""
Model utilities for saving, loading, and managing biometric models
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model_info() -> Dict:
    """Get information about available models."""
    info = {
        'model_dir': MODEL_DIR,
        'available_models': [],
        'last_updated': datetime.now().isoformat()
    }
    
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(('.pth', '.pkl', '.h5')):
            filepath = os.path.join(MODEL_DIR, filename)
            info['available_models'].append({
                'name': filename,
                'path': filepath,
                'size': os.path.getsize(filepath),
                'modified': datetime.fromtimestamp(
                    os.path.getmtime(filepath)
                ).isoformat()
            })
    
    return info


def save_model_metadata(model_name: str, metadata: Dict) -> str:
    """Save model metadata to JSON file."""
    filepath = os.path.join(MODEL_DIR, f"{model_name}_metadata.json")
    
    metadata['saved_at'] = datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return filepath


def load_model_metadata(model_name: str) -> Optional[Dict]:
    """Load model metadata from JSON file."""
    filepath = os.path.join(MODEL_DIR, f"{model_name}_metadata.json")
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    
    return None


if __name__ == "__main__":
    info = get_model_info()
    print("Model Directory:", info['model_dir'])
    print("Available Models:", len(info['available_models']))
