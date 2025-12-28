# src/feature_extraction/face_features.py

"""
Face Feature Extraction using Deep Learning
Based on Section 4.3 of the paper

Steps:
1. Encode picture using HOG (Histogram of Oriented Gradients)
2. Find main landmarks using face landmark estimation
3. Pass centered face through neural network for 128 measurements
4. Calculate Euclidean distance for matching

This generates C1 features (biometric 2 features) used in the fuzzy vault.
"""

import os
import sys
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.preprocessing.face_preprocessing import FacePreprocessor

# Try to import face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not available. Using fallback method.")


class FaceFeatureExtractor:
    """
    Face feature extraction using Deep Learning
    As described in Section 4.3 of the paper
    
    This generates C1 (biometric 2 features) used in the bio token.
    """
    
    def __init__(self, n_features=Config.N_FEATURES):
        self.n_features = n_features
        self.preprocessor = FacePreprocessor()
        
        if not FACE_RECOGNITION_AVAILABLE:
            print("Using HOG-based fallback for face features")
    
    def _load_image_for_face_recognition(self, image_path):
        """
        Load image in format required by face_recognition library
        """
        # face_recognition expects RGB format
        image = face_recognition.load_image_file(image_path)
        return image
    
    def _detect_face_hog(self, image):
        """
        Step 1: Encode picture using HOG algorithm
        
        HOG (Histogram of Oriented Gradients) creates a simplified 
        representation of the face for detection.
        
        Args:
            image: RGB image array
        
        Returns:
            List of face locations as (top, right, bottom, left)
        """
        if FACE_RECOGNITION_AVAILABLE:
            # Use HOG-based face detection
            face_locations = face_recognition.face_locations(image, model="hog")
            return face_locations
        else:
            # Fallback: use OpenCV Haar cascade
            return self._detect_face_opencv(image)
    
    def _detect_face_opencv(self, image):
        """
        Fallback face detection using OpenCV Haar Cascade
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Load Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to (top, right, bottom, left) format
        face_locations = []
        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))
        
        return face_locations
    
    def _get_face_landmarks(self, image, face_location):
        """
        Step 2: Find main landmarks in face
        
        Landmarks include:
        - Eyes (left and right)
        - Nose
        - Mouth
        - Chin
        
        Args:
            image: RGB image
            face_location: (top, right, bottom, left)
        
        Returns:
            Dictionary of landmark positions
        """
        if FACE_RECOGNITION_AVAILABLE:
            landmarks = face_recognition.face_landmarks(image, [face_location])
            return landmarks[0] if landmarks else None
        else:
            return None
    
    def _align_face(self, image, face_location, landmarks=None):
        """
        Warp image so eyes and mouth are centered
        
        Args:
            image: RGB image
            face_location: (top, right, bottom, left)
            landmarks: Face landmarks (optional)
        
        Returns:
            Aligned and centered face image
        """
        top, right, bottom, left = face_location
        
        # Extract face region
        face_image = image[top:bottom, left:right]
        
        # Resize to standard size (160x160 for face_recognition)
        aligned = cv2.resize(face_image, (160, 160))
        
        return aligned
    
    def _extract_face_encoding(self, image, face_locations=None):
        """
        Step 3: Generate 128 measurements using neural network
        
        The neural network generates a 128-dimensional embedding
        that uniquely represents the face.
        
        Args:
            image: RGB image
            face_locations: Optional list of face locations
        
        Returns:
            128-dimensional face encoding
        """
        if FACE_RECOGNITION_AVAILABLE:
            # Get face encodings
            if face_locations:
                encodings = face_recognition.face_encodings(image, face_locations)
            else:
                encodings = face_recognition.face_encodings(image)
            
            if encodings:
                return encodings[0]  # Return first face encoding
            else:
                return None
        else:
            return self._extract_hog_features(image)
    
    def _extract_hog_features(self, image):
        """
        Fallback: Extract HOG features when face_recognition is unavailable
        """
        from skimage.feature import hog
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize to standard size
        resized = cv2.resize(gray, (128, 128))
        
        # Extract HOG features
        features = hog(
            resized,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True
        )
        
        return features
    
    def extract_features(self, image_path):
        """
        Complete feature extraction pipeline for face images
        
        Steps:
        1. Load image
        2. Detect face using HOG
        3. Get landmarks
        4. Align face
        5. Extract 128-dimensional encoding
        6. Adjust to n_features length
        
        Args:
            image_path: Path to face image
        
        Returns:
            C1: Feature vector of length n_features
        """
        if FACE_RECOGNITION_AVAILABLE:
            # Load image for face_recognition
            image = self._load_image_for_face_recognition(image_path)
        else:
            # Load with OpenCV
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 1 & 2: Detect face using HOG
        face_locations = self._detect_face_hog(image)
        
        if not face_locations:
            # No face detected - try with whole image
            encoding = self._extract_face_encoding(image)
            if encoding is not None:
                return self._adjust_feature_length(encoding)
            else:
                # Return zero vector as last resort
                print(f"Warning: No face detected in {image_path}")
                return np.zeros(self.n_features)
        
        # Use first detected face
        face_location = face_locations[0]
        
        # Step 3: Get landmarks
        landmarks = self._get_face_landmarks(image, face_location)
        
        # Step 4: Align face (currently just crop)
        aligned = self._align_face(image, face_location, landmarks)
        
        # Step 5: Extract 128-dimensional encoding
        encoding = self._extract_face_encoding(image, [face_location])
        
        if encoding is None:
            # Try encoding from aligned face
            encoding = self._extract_face_encoding(aligned)
        
        if encoding is not None:
            # Step 6: Adjust to n_features
            return self._adjust_feature_length(encoding)
        else:
            print(f"Warning: Could not extract encoding from {image_path}")
            return np.zeros(self.n_features)
    
    def extract_features_from_image(self, image):
        """
        Extract features from image array (already loaded)
        
        Args:
            image: Image array (BGR or RGB)
        
        Returns:
            Feature vector
        """
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR if loaded with OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Detect face
        face_locations = self._detect_face_hog(image_rgb)
        
        if face_locations:
            encoding = self._extract_face_encoding(image_rgb, face_locations)
        else:
            encoding = self._extract_face_encoding(image_rgb)
        
        if encoding is not None:
            return self._adjust_feature_length(encoding)
        else:
            return np.zeros(self.n_features)
    
    def _adjust_feature_length(self, features):
        """
        Adjust feature vector to exactly n_features length
        """
        features = np.array(features)
        current_length = len(features)
        
        if current_length < self.n_features:
            # Pad with zeros or tile
            padded = np.zeros(self.n_features)
            padded[:current_length] = features
            return padded
        else:
            return features[:self.n_features]
    
    def compute_euclidean_distance(self, features1, features2):
        """
        Step 4: Calculate Euclidean distance between two feature vectors
        
        Used for matching faces during authentication.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
        
        Returns:
            Euclidean distance (lower = more similar)
        """
        return np.sqrt(np.sum((features1 - features2) ** 2))
    
    def batch_extract(self, user_id):
        """
        Extract features from all images of a user
        
        Args:
            user_id: User folder name (e.g., '000')
        
        Returns:
            List of feature vectors
        """
        user_dir = os.path.join(Config.FACE_DATA_DIR, user_id)
        
        if not os.path.exists(user_dir):
            raise ValueError(f"User directory not found: {user_dir}")
        
        # Get all images
        image_files = sorted([
            f for f in os.listdir(user_dir)
            if f.lower().endswith(Config.FACE_IMAGE_FORMATS)
        ])
        
        features_list = []
        
        for img_file in image_files:
            img_path = os.path.join(user_dir, img_file)
            features = self.extract_features(img_path)
            features_list.append({
                'filename': img_file,
                'features': features
            })
        
        return features_list


def test_face_features():
    """Test face feature extraction"""
    
    print("=" * 60)
    print("TESTING FACE FEATURE EXTRACTION")
    print("=" * 60)
    
    print(f"\nface_recognition available: {FACE_RECOGNITION_AVAILABLE}")
    
    extractor = FaceFeatureExtractor()
    
    # Test with first user's images
    test_user = "000"
    user_dir = os.path.join(Config.FACE_DATA_DIR, test_user)
    
    if not os.path.exists(user_dir):
        print(f"Test user directory not found: {user_dir}")
        return
    
    # Get image files
    image_files = sorted([
        f for f in os.listdir(user_dir)
        if f.lower().endswith(Config.FACE_IMAGE_FORMATS)
    ])
    
    print(f"\nExtracting features from user '{test_user}' ({len(image_files)} images)...")
    
    all_features = []
    
    for img_file in image_files:
        img_path = os.path.join(user_dir, img_file)
        
        features = extractor.extract_features(img_path)
        all_features.append(features)
        
        print(f"\n  {img_file}:")
        print(f"    Features shape: {features.shape}")
        print(f"    Range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"    Mean: {features.mean():.4f}")
        print(f"    Non-zero: {np.sum(features != 0)}")
    
    all_features = np.array(all_features)
    
    # Analyze intra-user distances (same person, different images)
    print("\n" + "-" * 40)
    print("Intra-User Distances (Same Person):")
    
    intra_distances = []
    for i in range(len(all_features)):
        for j in range(i + 1, len(all_features)):
            dist = extractor.compute_euclidean_distance(
                all_features[i], all_features[j]
            )
            intra_distances.append(dist)
            print(f"  Image {i+1} vs Image {j+1}: {dist:.4f}")
    
    print(f"\nIntra-user distance stats:")
    print(f"  Mean: {np.mean(intra_distances):.4f}")
    print(f"  Std: {np.std(intra_distances):.4f}")
    print(f"  Min: {np.min(intra_distances):.4f}")
    print(f"  Max: {np.max(intra_distances):.4f}")
    
    # Test with different user for inter-user distance
    print("\n" + "-" * 40)
    print("Inter-User Distances (Different Persons):")
    
    other_user = "001"
    other_user_dir = os.path.join(Config.FACE_DATA_DIR, other_user)
    
    if os.path.exists(other_user_dir):
        other_images = sorted([
            f for f in os.listdir(other_user_dir)
            if f.lower().endswith(Config.FACE_IMAGE_FORMATS)
        ])[:2]  # Just compare with first 2 images
        
        inter_distances = []
        for other_img in other_images:
            other_path = os.path.join(other_user_dir, other_img)
            other_features = extractor.extract_features(other_path)
            
            # Compare with first image of test user
            dist = extractor.compute_euclidean_distance(
                all_features[0], other_features
            )
            inter_distances.append(dist)
            print(f"  User {test_user} vs User {other_user} ({other_img}): {dist:.4f}")
        
        print(f"\nInter-user distance stats:")
        print(f"  Mean: {np.mean(inter_distances):.4f}")
    
    # Visualize
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Feature vectors
    for i, features in enumerate(all_features):
        axes[0].plot(features, label=f'Image {i+1}', alpha=0.7)
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Feature Value')
    axes[0].set_title(f'Face Feature Vectors (C1) - User {test_user}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Feature heatmap
    im = axes[1].imshow(all_features, aspect='auto', cmap='viridis')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Image')
    axes[1].set_title('Feature Heatmap')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    
    output_path = os.path.join(Config.LOG_DIR, 'face_features_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("FACE FEATURE EXTRACTION TEST COMPLETE")
    print("=" * 60)
    
    return all_features


if __name__ == "__main__":
    test_face_features()