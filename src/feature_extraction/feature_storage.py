# src/feature_extraction/feature_storage.py

"""
Feature Storage Module
Extracts and stores biometric features in CSV and NPY formats
for faster evaluation and analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.feature_extraction.face_features import FaceFeatureExtractor
from src.feature_extraction.hand_features import HandFeatureExtractor


class FeatureStorage:
    """
    Extract and store biometric features
    """
    
    def __init__(self):
        self.face_extractor = FaceFeatureExtractor()
        self.hand_extractor = HandFeatureExtractor()
        
        # Storage directories
        self.storage_dir = os.path.join(Config.DATA_DIR, 'features')
        self.face_features_dir = os.path.join(self.storage_dir, 'face')
        self.hand_features_dir = os.path.join(self.storage_dir, 'hand')
        
        # Create directories
        os.makedirs(self.face_features_dir, exist_ok=True)
        os.makedirs(self.hand_features_dir, exist_ok=True)
        
        print(f"Feature storage initialized")
        print(f"  Face features: {self.face_features_dir}")
        print(f"  Hand features: {self.hand_features_dir}")
    
    def extract_all_face_features(self, num_users=None, save_format='both'):
        """
        Extract and save face features for all users
        
        Args:
            num_users: Number of users to process (None = all)
            save_format: 'csv', 'npy', or 'both'
        
        Returns:
            DataFrame with all features
        """
        print("\n" + "=" * 60)
        print("EXTRACTING FACE FEATURES")
        print("=" * 60)
        
        face_dir = Config.FACE_DATA_DIR
        
        # Get user folders
        user_folders = sorted([
            f for f in os.listdir(face_dir)
            if os.path.isdir(os.path.join(face_dir, f))
        ])
        
        if num_users is not None:
            user_folders = user_folders[:num_users]
        
        print(f"Processing {len(user_folders)} users...")
        
        all_features = []
        all_metadata = []
        
        for idx, user_id in enumerate(user_folders):
            user_path = os.path.join(face_dir, user_id)
            
            # Get all images for this user
            images = sorted([
                f for f in os.listdir(user_path)
                if f.lower().endswith(Config.FACE_IMAGE_FORMATS)
            ])
            
            for img_name in images:
                img_path = os.path.join(user_path, img_name)
                
                try:
                    # Extract features
                    features = self.face_extractor.extract_features(img_path)
                    
                    all_features.append(features)
                    all_metadata.append({
                        'user_id': user_id,
                        'image_name': img_name,
                        'image_path': img_path
                    })
                    
                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
            
            # Progress
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(user_folders)} users")
        
        print(f"\nTotal face features extracted: {len(all_features)}")
        
        # Convert to numpy array
        features_array = np.array(all_features)
        
        # Create DataFrame
        feature_columns = [f'face_feat_{i}' for i in range(features_array.shape[1])]
        
        df_features = pd.DataFrame(features_array, columns=feature_columns)
        df_metadata = pd.DataFrame(all_metadata)
        df_complete = pd.concat([df_metadata, df_features], axis=1)
        
        # Save
        if save_format in ['csv', 'both']:
            csv_path = os.path.join(self.face_features_dir, 'face_features.csv')
            df_complete.to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path}")
        
        if save_format in ['npy', 'both']:
            npy_path = os.path.join(self.face_features_dir, 'face_features.npy')
            np.save(npy_path, features_array)
            print(f"Saved NPY: {npy_path}")
            
            # Save metadata separately
            metadata_path = os.path.join(self.face_features_dir, 'face_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(all_metadata, f, indent=2)
            print(f"Saved metadata: {metadata_path}")
        
        # Save summary
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'num_users': len(user_folders),
            'num_images': len(all_features),
            'feature_dim': features_array.shape[1],
            'feature_mean': float(features_array.mean()),
            'feature_std': float(features_array.std())
        }
        
        summary_path = os.path.join(self.face_features_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary:")
        print(f"  Users: {summary['num_users']}")
        print(f"  Images: {summary['num_images']}")
        print(f"  Feature dimension: {summary['feature_dim']}")
        
        return df_complete
    
    def extract_all_hand_features(self, num_images=None, save_format='both'):
        """
        Extract and save hand features for all images
        
        Args:
            num_images: Number of images to process (None = all)
            save_format: 'csv', 'npy', or 'both'
        
        Returns:
            DataFrame with all features
        """
        print("\n" + "=" * 60)
        print("EXTRACTING HAND FEATURES")
        print("=" * 60)
        
        hand_dir = Config.HAND_DATA_DIR
        
        # Get all hand images
        images = sorted([
            f for f in os.listdir(hand_dir)
            if f.lower().endswith(Config.HAND_IMAGE_FORMATS)
        ])
        
        if num_images is not None:
            images = images[:num_images]
        
        print(f"Processing {len(images)} images...")
        
        all_features = []
        all_metadata = []
        
        for idx, img_name in enumerate(images):
            img_path = os.path.join(hand_dir, img_name)
            
            try:
                # Extract features
                features = self.hand_extractor.extract_features(img_path)
                
                all_features.append(features)
                all_metadata.append({
                    'image_name': img_name,
                    'image_path': img_path,
                    'image_index': idx
                })
                
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
            
            # Progress
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(images)} images")
        
        print(f"\nTotal hand features extracted: {len(all_features)}")
        
        # Convert to numpy array
        features_array = np.array(all_features)
        
        # Create DataFrame
        feature_columns = [f'hand_feat_{i}' for i in range(features_array.shape[1])]
        
        df_features = pd.DataFrame(features_array, columns=feature_columns)
        df_metadata = pd.DataFrame(all_metadata)
        df_complete = pd.concat([df_metadata, df_features], axis=1)
        
        # Save
        if save_format in ['csv', 'both']:
            csv_path = os.path.join(self.hand_features_dir, 'hand_features.csv')
            df_complete.to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path}")
        
        if save_format in ['npy', 'both']:
            npy_path = os.path.join(self.hand_features_dir, 'hand_features.npy')
            np.save(npy_path, features_array)
            print(f"Saved NPY: {npy_path}")
            
            # Save metadata separately
            metadata_path = os.path.join(self.hand_features_dir, 'hand_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(all_metadata, f, indent=2)
            print(f"Saved metadata: {metadata_path}")
        
        # Save summary
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'num_images': len(all_features),
            'feature_dim': features_array.shape[1],
            'feature_mean': float(features_array.mean()),
            'feature_std': float(features_array.std())
        }
        
        summary_path = os.path.join(self.hand_features_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary:")
        print(f"  Images: {summary['num_images']}")
        print(f"  Feature dimension: {summary['feature_dim']}")
        
        return df_complete
    
    def load_face_features(self):
        """Load saved face features"""
        
        csv_path = os.path.join(self.face_features_dir, 'face_features.csv')
        npy_path = os.path.join(self.face_features_dir, 'face_features.npy')
        metadata_path = os.path.join(self.face_features_dir, 'face_metadata.json')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Loaded face features from CSV: {df.shape}")
            return df
        elif os.path.exists(npy_path):
            features = np.load(npy_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded face features from NPY: {features.shape}")
            return features, metadata
        else:
            print("No saved face features found")
            return None
    
    def load_hand_features(self):
        """Load saved hand features"""
        
        csv_path = os.path.join(self.hand_features_dir, 'hand_features.csv')
        npy_path = os.path.join(self.hand_features_dir, 'hand_features.npy')
        metadata_path = os.path.join(self.hand_features_dir, 'hand_metadata.json')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Loaded hand features from CSV: {df.shape}")
            return df
        elif os.path.exists(npy_path):
            features = np.load(npy_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded hand features from NPY: {features.shape}")
            return features, metadata
        else:
            print("No saved hand features found")
            return None
    
    def get_user_features(self, user_id):
        """Get features for a specific user"""
        
        df = self.load_face_features()
        if df is not None:
            user_df = df[df['user_id'] == user_id]
            if len(user_df) > 0:
                feature_cols = [c for c in user_df.columns if c.startswith('face_feat_')]
                return user_df[feature_cols].values
        return None
    
    def analyze_features(self):
        """Analyze stored features"""
        
        print("\n" + "=" * 60)
        print("FEATURE ANALYSIS")
        print("=" * 60)
        
        # Face features
        face_csv = os.path.join(self.face_features_dir, 'face_features.csv')
        if os.path.exists(face_csv):
            df_face = pd.read_csv(face_csv)
            feature_cols = [c for c in df_face.columns if c.startswith('face_feat_')]
            features = df_face[feature_cols].values
            
            print("\nFace Features:")
            print(f"  Shape: {features.shape}")
            print(f"  Users: {df_face['user_id'].nunique()}")
            print(f"  Mean: {features.mean():.6f}")
            print(f"  Std: {features.std():.6f}")
            print(f"  Min: {features.min():.6f}")
            print(f"  Max: {features.max():.6f}")
        
        # Hand features
        hand_csv = os.path.join(self.hand_features_dir, 'hand_features.csv')
        if os.path.exists(hand_csv):
            df_hand = pd.read_csv(hand_csv)
            feature_cols = [c for c in df_hand.columns if c.startswith('hand_feat_')]
            features = df_hand[feature_cols].values
            
            print("\nHand Features:")
            print(f"  Shape: {features.shape}")
            print(f"  Mean: {features.mean():.6f}")
            print(f"  Std: {features.std():.6f}")
            print(f"  Min: {features.min():.6f}")
            print(f"  Max: {features.max():.6f}")


def main():
    """Extract and store all features"""
    
    print("=" * 70)
    print("BIOMETRIC FEATURE EXTRACTION AND STORAGE")
    print("=" * 70)
    
    storage = FeatureStorage()
    
    # Extract face features (first 50 users)
    print("\n>>> Extracting Face Features...")
    df_face = storage.extract_all_face_features(num_users=50, save_format='both')
    
    # Extract hand features (first 500 images)
    print("\n>>> Extracting Hand Features...")
    df_hand = storage.extract_all_hand_features(num_images=500, save_format='both')
    
    # Analyze
    storage.analyze_features()
    
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nFiles saved to: {storage.storage_dir}")
    print("\nFiles created:")
    print("  - data/features/face/face_features.csv")
    print("  - data/features/face/face_features.npy")
    print("  - data/features/face/face_metadata.json")
    print("  - data/features/face/summary.json")
    print("  - data/features/hand/hand_features.csv")
    print("  - data/features/hand/hand_features.npy")
    print("  - data/features/hand/hand_metadata.json")
    print("  - data/features/hand/summary.json")


if __name__ == "__main__":
    main()