# src/preprocessing/face_preprocessing.py

"""
Face Image Preprocessing Module
Based on Section 4.1 of the paper

Uses median filter to:
- Reduce noise
- Preserve edges
- Smooth appearance while keeping sharp features
"""

import os
import sys
import cv2
import numpy as np
from scipy.ndimage import median_filter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config


class FacePreprocessor:
    """
    Face image preprocessing using median filter
    As described in Section 4.1 of the paper
    """
    
    def __init__(self):
        self.image_size = Config.FACE_IMAGE_SIZE  # (640, 480)
        self.filter_size = Config.MEDIAN_FILTER_SIZE  # 3
        self.output_dir = Config.PROCESSED_FACE_DIR
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_image(self, image_path):
        """Load image from path"""
        
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return image
    
    def apply_median_filter(self, image):
        """
        Apply median filter to reduce noise while preserving edges
        
        As per Section 4.1:
        "For a given pixel (i, j) in the image, the median value is calculated as:
        Median = median(I(i-1:i+1, j-1:j+1))"
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Filtered image
        """
        
        if len(image.shape) == 3:
            # Color image - apply filter to each channel
            filtered = np.zeros_like(image)
            for channel in range(3):
                filtered[:, :, channel] = median_filter(
                    image[:, :, channel], 
                    size=self.filter_size
                )
        else:
            # Grayscale image
            filtered = median_filter(image, size=self.filter_size)
        
        return filtered
    
    def convert_to_grayscale(self, image):
        """Convert BGR image to grayscale"""
        
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def resize_image(self, image, target_size=None):
        """Resize image to target size"""
        
        if target_size is None:
            target_size = self.image_size
        
        return cv2.resize(image, target_size)
    
    def normalize_image(self, image):
        """Normalize pixel values to [0, 1] range"""
        
        return image.astype(np.float32) / 255.0
    
    def preprocess(self, image_path, return_color=False):
        """
        Complete preprocessing pipeline for face images
        
        Steps:
        1. Load image
        2. Apply median filter (noise reduction)
        3. Convert to grayscale (optional)
        4. Normalize
        
        Args:
            image_path: Path to input image
            return_color: If True, return color image; else grayscale
        
        Returns:
            Preprocessed image (normalized, float32)
        """
        
        # Step 1: Load image
        image = self.load_image(image_path)
        original_shape = image.shape
        
        # Step 2: Apply median filter
        filtered = self.apply_median_filter(image)
        
        # Step 3: Convert to grayscale if needed
        if not return_color:
            processed = self.convert_to_grayscale(filtered)
        else:
            processed = filtered
        
        # Step 4: Normalize to [0, 1]
        normalized = self.normalize_image(processed)
        
        return normalized, original_shape
    
    def preprocess_and_save(self, image_path, output_path=None):
        """
        Preprocess image and save to disk
        
        Args:
            image_path: Input image path
            output_path: Output path (optional)
        
        Returns:
            Output path where image was saved
        """
        
        # Preprocess
        processed, _ = self.preprocess(image_path, return_color=True)
        
        # Convert back to uint8 for saving
        processed_uint8 = (processed * 255).astype(np.uint8)
        
        # Generate output path if not provided
        if output_path is None:
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(self.output_dir, f"{name}_processed.png")
        
        # Save
        cv2.imwrite(output_path, processed_uint8)
        
        return output_path
    
    def preprocess_user_images(self, user_id):
        """
        Preprocess all images for a specific user
        
        Args:
            user_id: User folder name (e.g., '000', '001')
        
        Returns:
            List of preprocessed images
        """
        
        user_dir = os.path.join(Config.FACE_DATA_DIR, user_id)
        
        if not os.path.exists(user_dir):
            raise ValueError(f"User directory not found: {user_dir}")
        
        # Get all images
        image_files = sorted([
            f for f in os.listdir(user_dir)
            if f.lower().endswith(Config.FACE_IMAGE_FORMATS)
        ])
        
        preprocessed_images = []
        
        for img_file in image_files:
            img_path = os.path.join(user_dir, img_file)
            processed, _ = self.preprocess(img_path)
            preprocessed_images.append({
                'filename': img_file,
                'image': processed
            })
        
        return preprocessed_images
    
    def batch_preprocess(self, num_users=None, save_to_disk=False):
        """
        Preprocess images for multiple users
        
        Args:
            num_users: Number of users to process (None = all)
            save_to_disk: If True, save processed images
        
        Returns:
            Dictionary of user_id -> list of processed images
        """
        
        # Get user folders
        user_folders = sorted([
            f for f in os.listdir(Config.FACE_DATA_DIR)
            if os.path.isdir(os.path.join(Config.FACE_DATA_DIR, f))
        ])
        
        if num_users is not None:
            user_folders = user_folders[:num_users]
        
        print(f"Processing {len(user_folders)} users...")
        
        all_processed = {}
        
        for idx, user_id in enumerate(user_folders):
            try:
                processed_images = self.preprocess_user_images(user_id)
                all_processed[user_id] = processed_images
                
                if save_to_disk:
                    # Create user output directory
                    user_output_dir = os.path.join(self.output_dir, user_id)
                    os.makedirs(user_output_dir, exist_ok=True)
                    
                    for item in processed_images:
                        output_path = os.path.join(
                            user_output_dir, 
                            item['filename'].replace('.bmp', '_processed.png')
                        )
                        img_uint8 = (item['image'] * 255).astype(np.uint8)
                        cv2.imwrite(output_path, img_uint8)
                
                # Progress indicator
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(user_folders)} users")
                    
            except Exception as e:
                print(f"  Error processing user {user_id}: {e}")
        
        print(f"Completed processing {len(all_processed)} users")
        
        return all_processed


def test_face_preprocessing():
    """Test the face preprocessing module"""
    
    print("=" * 60)
    print("TESTING FACE PREPROCESSING")
    print("=" * 60)
    
    preprocessor = FacePreprocessor()
    
    # Get first user's first image
    first_user = "000"
    user_dir = os.path.join(Config.FACE_DATA_DIR, first_user)
    first_image = os.path.join(user_dir, "000_0.bmp")
    
    if not os.path.exists(first_image):
        print(f"Test image not found: {first_image}")
        return
    
    print(f"\nTest image: {first_image}")
    
    # Load original
    original = cv2.imread(first_image)
    print(f"Original shape: {original.shape}")
    print(f"Original dtype: {original.dtype}")
    print(f"Original range: [{original.min()}, {original.max()}]")
    
    # Preprocess
    processed, orig_shape = preprocessor.preprocess(first_image)
    print(f"\nProcessed shape: {processed.shape}")
    print(f"Processed dtype: {processed.dtype}")
    print(f"Processed range: [{processed.min():.4f}, {processed.max():.4f}]")
    
    # Save comparison
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Median filtered (color)
    filtered_color, _ = preprocessor.preprocess(first_image, return_color=True)
    axes[1].imshow(filtered_color)
    axes[1].set_title('Median Filtered (Color)')
    axes[1].axis('off')
    
    # Grayscale processed
    axes[2].imshow(processed, cmap='gray')
    axes[2].set_title('Processed (Grayscale)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(Config.LOG_DIR, 'face_preprocessing_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to: {output_path}")
    
    plt.close()
    
    # Test batch processing (first 5 users)
    print("\n" + "-" * 40)
    print("Testing batch preprocessing (5 users)...")
    
    batch_results = preprocessor.batch_preprocess(num_users=5, save_to_disk=True)
    
    print(f"\nBatch processing complete!")
    print(f"Users processed: {len(batch_results)}")
    for user_id, images in batch_results.items():
        print(f"  {user_id}: {len(images)} images")
    
    print("\n" + "=" * 60)
    print("FACE PREPROCESSING TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_face_preprocessing()