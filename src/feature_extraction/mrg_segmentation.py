# src/feature_extraction/mrg_segmentation.py

"""
Modified Region Growing (MRG) Segmentation
Based on Section 4.2.1 of the paper

MRG is used to segment hand images by:
1. Automatic seed point selection
2. Region growing based on intensity similarity
3. Edge-aware segmentation to follow biometric features
"""

import os
import sys
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config


class ModifiedRegionGrowing:
    """
    Modified Region Growing (MRG) for hand image segmentation
    As described in Section 4.2.1 of the paper
    """
    
    def __init__(self):
        self.threshold = Config.MRG_THRESHOLD  # 10
        self.connectivity = Config.MRG_CONNECTIVITY  # 8
    
    def _ensure_grayscale_uint8(self, image):
        """Convert image to grayscale uint8"""
        
        if image.dtype in [np.float32, np.float64]:
            image = (image * 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image
    
    def find_seed_points(self, image, num_seeds=5):
        """
        Automatically find seed points near the center of biometric features
        
        Uses edge detection to find feature boundaries, then selects
        seed points at contour centers.
        
        Args:
            image: Input image
            num_seeds: Number of seed points to find
        
        Returns:
            List of (y, x) seed point coordinates
        """
        gray = self._ensure_grayscale_uint8(image)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        seed_points = []
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:num_seeds]:
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                seed_points.append((cy, cx))  # (row, col) format
        
        # If no contours found, use image center
        if not seed_points:
            h, w = gray.shape
            seed_points = [(h // 2, w // 2)]
        
        return seed_points
    
    def region_growing(self, image, seed_point):
        """
        Perform region growing from a seed point
        
        Args:
            image: Input grayscale image
            seed_point: (y, x) starting point
        
        Returns:
            Binary mask of grown region
        """
        gray = self._ensure_grayscale_uint8(image)
        h, w = gray.shape
        
        # Initialize output mask
        segmented = np.zeros_like(gray)
        visited = np.zeros_like(gray, dtype=bool)
        
        # Get seed value
        seed_y, seed_x = seed_point
        
        # Boundary check
        if not (0 <= seed_y < h and 0 <= seed_x < w):
            return segmented
        
        seed_value = int(gray[seed_y, seed_x])
        
        # Initialize queue with seed point
        queue = [seed_point]
        
        # Define neighbor offsets based on connectivity
        if self.connectivity == 8:
            neighbors = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),           (0, 1),
                         (1, -1),  (1, 0),  (1, 1)]
        else:  # 4-connectivity
            neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        
        while queue:
            y, x = queue.pop(0)
            
            # Skip if already visited
            if visited[y, x]:
                continue
            
            visited[y, x] = True
            
            # Check intensity similarity
            pixel_value = int(gray[y, x])
            if abs(pixel_value - seed_value) <= self.threshold:
                segmented[y, x] = 255
                
                # Add unvisited neighbors to queue
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    
                    if (0 <= ny < h and 0 <= nx < w and 
                        not visited[ny, nx]):
                        queue.append((ny, nx))
        
        return segmented
    
    def segment_image(self, image, num_seeds=5):
        """
        Segment image using MRG with multiple automatic seed points
        
        Args:
            image: Input image
            num_seeds: Number of seed points to use
        
        Returns:
            Combined segmentation mask
        """
        gray = self._ensure_grayscale_uint8(image)
        
        # Find seed points
        seed_points = self.find_seed_points(image, num_seeds)
        
        # Initialize combined mask
        combined_mask = np.zeros_like(gray)
        
        # Grow regions from each seed
        for seed in seed_points:
            mask = self.region_growing(image, seed)
            combined_mask = np.maximum(combined_mask, mask)
        
        return combined_mask, seed_points
    
    def apply_segmentation(self, image):
        """
        Apply segmentation and return masked image
        
        Args:
            image: Input image
        
        Returns:
            Segmented image with background removed
        """
        mask, seeds = self.segment_image(image)
        
        # Convert image for masking
        if len(image.shape) == 2:
            # Grayscale
            if image.dtype in [np.float32, np.float64]:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image
            segmented = cv2.bitwise_and(image_uint8, image_uint8, mask=mask)
        else:
            # Color
            if image.dtype in [np.float32, np.float64]:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image
            segmented = cv2.bitwise_and(image_uint8, image_uint8, mask=mask)
        
        return segmented, mask, seeds


def test_mrg_segmentation():
    """Test MRG segmentation"""
    
    print("=" * 60)
    print("TESTING MODIFIED REGION GROWING (MRG)")
    print("=" * 60)
    
    mrg = ModifiedRegionGrowing()
    
    # Load a sample hand image
    hand_dir = Config.HAND_DATA_DIR
    images = sorted([f for f in os.listdir(hand_dir) 
                     if f.lower().endswith(Config.HAND_IMAGE_FORMATS)])
    
    if not images:
        print("No hand images found!")
        return
    
    sample_path = os.path.join(hand_dir, images[0])
    print(f"\nTest image: {sample_path}")
    
    # Load and resize image
    image = cv2.imread(sample_path)
    image = cv2.resize(image, Config.HAND_IMAGE_SIZE)
    print(f"Image shape: {image.shape}")
    
    # Find seed points
    seeds = mrg.find_seed_points(image)
    print(f"\nFound {len(seeds)} seed points:")
    for i, (y, x) in enumerate(seeds):
        print(f"  Seed {i+1}: ({y}, {x})")
    
    # Apply segmentation
    segmented, mask, seeds = mrg.apply_segmentation(image)
    
    print(f"\nSegmentation Results:")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Segmented pixels: {np.sum(mask > 0)}")
    print(f"  Coverage: {np.sum(mask > 0) / mask.size * 100:.2f}%")
    
    # Visualize and save
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Seed points
    image_with_seeds = image.copy()
    for (y, x) in seeds:
        cv2.circle(image_with_seeds, (x, y), 5, (0, 255, 0), -1)
    axes[1].imshow(cv2.cvtColor(image_with_seeds, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Seed Points ({len(seeds)})')
    axes[1].axis('off')
    
    # Mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Segmentation Mask')
    axes[2].axis('off')
    
    # Segmented
    axes[3].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Segmented Image')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(Config.LOG_DIR, 'mrg_segmentation_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("MRG SEGMENTATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_mrg_segmentation()