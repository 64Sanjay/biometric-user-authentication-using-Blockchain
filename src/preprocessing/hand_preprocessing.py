# src/preprocessing/hand_preprocessing.py

"""
Hand Image Preprocessing Module
Based on Section 4.1 of the paper

Uses median filter and includes:
- Noise reduction
- Edge preservation
- Resizing to standard size
- Hand region segmentation (optional)
"""

import os
import sys
import cv2
import numpy as np
from scipy.ndimage import median_filter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config


class HandPreprocessor:
    """
    Hand image preprocessing using median filter
    As described in Section 4.1 of the paper
    """
    
    def __init__(self):
        self.image_size = Config.HAND_IMAGE_SIZE  # (224, 224)
        self.filter_size = Config.MEDIAN_FILTER_SIZE  # 3
        self.output_dir = Config.PROCESSED_HAND_DIR
        
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
    
    def resize_image(self, image, target_size=None):
        """
        Resize image to target size
        Hand images are 1200x1600, need to resize to 224x224
        """
        
        if target_size is None:
            target_size = self.image_size
        
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def convert_to_grayscale(self, image):
        """Convert BGR image to grayscale"""
        
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def normalize_image(self, image):
        """Normalize pixel values to [0, 1] range"""
        
        return image.astype(np.float32) / 255.0
    
    def segment_hand(self, image):
        """
        Segment hand region from background
        Uses color-based segmentation for skin detection
        
        Args:
            image: BGR image
        
        Returns:
            Segmented image and mask
        """
        
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to image
        segmented = cv2.bitwise_and(image, image, mask=mask)
        
        return segmented, mask
    
    def enhance_contrast(self, image):
        """
        Enhance contrast using CLAHE
        Useful for bringing out vein patterns
        """
        
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge and convert back
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def preprocess(self, image_path, return_color=False, segment=False):
        """
        Complete preprocessing pipeline for hand images
        
        Steps:
        1. Load image
        2. Resize to standard size (224x224)
        3. Apply median filter (noise reduction)
        4. Enhance contrast (optional)
        5. Convert to grayscale (optional)
        6. Normalize
        
        Args:
            image_path: Path to input image
            return_color: If True, return color image; else grayscale
            segment: If True, apply hand segmentation
        
        Returns:
            Preprocessed image (normalized, float32)
        """
        
        # Step 1: Load image
        image = self.load_image(image_path)
        original_shape = image.shape
        
        # Step 2: Resize to standard size
        resized = self.resize_image(image)
        
        # Step 3: Apply median filter
        filtered = self.apply_median_filter(resized)
        
        # Step 4: Optional segmentation
        if segment:
            filtered, mask = self.segment_hand(filtered)
        
        # Step 5: Convert to grayscale if needed
        if not return_color:
            processed = self.convert_to_grayscale(filtered)
        else:
            processed = filtered
        
        # Step 6: Normalize to [0, 1]
        normalized = self.normalize_image(processed)
        
        return normalized, original_shape
    
    def preprocess_and_save(self, image_path, output_path=None):
        """
        Preprocess image and save to disk
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
    
    def get_image_list(self):
        """Get list of all hand images"""
        
        images = sorted([
            f for f in os.listdir(Config.HAND_DATA_DIR)
            if f.lower().endswith(Config.HAND_IMAGE_FORMATS)
        ])
        
        return images
    
    def batch_preprocess(self, num_images=None, save_to_disk=False):
        """
        Preprocess multiple hand images
        
        Args:
            num_images: Number of images to process (None = all)
            save_to_disk: If True, save processed images
        
        Returns:
            List of dictionaries with filename and processed image
        """
        
        # Get image list
        image_files = self.get_image_list()
        
        if num_images is not None:
            image_files = image_files[:num_images]
        
        print(f"Processing {len(image_files)} hand images...")
        
        all_processed = []
        
        for idx, img_file in enumerate(image_files):
            try:
                img_path = os.path.join(Config.HAND_DATA_DIR, img_file)
                processed, orig_shape = self.preprocess(img_path)
                
                all_processed.append({
                    'filename': img_file,
                    'image': processed,
                    'original_shape': orig_shape
                })
                
                if save_to_disk:
                    output_path = os.path.join(
                        self.output_dir,
                        img_file.replace('.jpg', '_processed.png')
                    )
                    img_uint8 = (processed * 255).astype(np.uint8)
                    cv2.imwrite(output_path, img_uint8)
                
                # Progress indicator
                if (idx + 1) % 500 == 0:
                    print(f"  Processed {idx + 1}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
        
        print(f"Completed processing {len(all_processed)} images")
        
        return all_processed


def test_hand_preprocessing():
    """Test the hand preprocessing module"""
    
    print("=" * 60)
    print("TESTING HAND PREPROCESSING")
    print("=" * 60)
    
    preprocessor = HandPreprocessor()
    
    # Get first hand image
    image_files = preprocessor.get_image_list()
    
    if not image_files:
        print("No hand images found!")
        return
    
    first_image = os.path.join(Config.HAND_DATA_DIR, image_files[0])
    
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
    
    # Original (resized for display)
    original_display = cv2.resize(original, (400, 300))
    axes[0].imshow(cv2.cvtColor(original_display, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original Image\n{original.shape}')
    axes[0].axis('off')
    
    # Median filtered (color)
    filtered_color, _ = preprocessor.preprocess(first_image, return_color=True)
    axes[1].imshow(cv2.cvtColor(
        (filtered_color * 255).astype(np.uint8), 
        cv2.COLOR_BGR2RGB
    ))
    axes[1].set_title(f'Resized + Filtered\n{filtered_color.shape}')
    axes[1].axis('off')
    
    # Grayscale processed
    axes[2].imshow(processed, cmap='gray')
    axes[2].set_title(f'Processed (Grayscale)\n{processed.shape}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(Config.LOG_DIR, 'hand_preprocessing_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to: {output_path}")
    
    plt.close()
    
    # Test batch processing (first 10 images)
    print("\n" + "-" * 40)
    print("Testing batch preprocessing (10 images)...")
    
    batch_results = preprocessor.batch_preprocess(num_images=10, save_to_disk=True)
    
    print(f"\nBatch processing complete!")
    print(f"Images processed: {len(batch_results)}")
    
    print("\n" + "=" * 60)
    print("HAND PREPROCESSING TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_hand_preprocessing()