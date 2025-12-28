# src/preprocessing/explore_datasets.py

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config


def explore_face_dataset():
    """Explore CASIA-FaceV5 dataset structure"""
    
    print("\n" + "=" * 60)
    print("EXPLORING FACE DATASET (CASIA-FaceV5)")
    print("=" * 60)
    
    face_dir = Config.FACE_DATA_DIR
    
    if not os.path.exists(face_dir):
        print(f"ERROR: Directory not found: {face_dir}")
        return None
    
    # Get all user folders
    user_folders = sorted([f for f in os.listdir(face_dir) 
                           if os.path.isdir(os.path.join(face_dir, f))])
    
    print(f"\nTotal Users: {len(user_folders)}")
    print(f"First 5 users: {user_folders[:5]}")
    print(f"Last 5 users: {user_folders[-5:]}")
    
    # Analyze first user
    first_user = user_folders[0]
    first_user_path = os.path.join(face_dir, first_user)
    images = sorted(os.listdir(first_user_path))
    
    print(f"\nSample user '{first_user}' images:")
    for img in images:
        print(f"  - {img}")
    
    # Load and analyze a sample image
    sample_img_path = os.path.join(first_user_path, images[0])
    sample_img = cv2.imread(sample_img_path)
    
    print(f"\nSample Image Analysis:")
    print(f"  Path: {sample_img_path}")
    print(f"  Shape: {sample_img.shape}")
    print(f"  Data Type: {sample_img.dtype}")
    print(f"  Min Value: {sample_img.min()}")
    print(f"  Max Value: {sample_img.max()}")
    print(f"  Mean Value: {sample_img.mean():.2f}")
    
    return sample_img, sample_img_path


def explore_hand_dataset():
    """Explore 11k Hands dataset structure"""
    
    print("\n" + "=" * 60)
    print("EXPLORING HAND DATASET (11k Hands)")
    print("=" * 60)
    
    hand_dir = Config.HAND_DATA_DIR
    
    if not os.path.exists(hand_dir):
        print(f"ERROR: Directory not found: {hand_dir}")
        return None
    
    # Get all image files
    all_images = sorted([f for f in os.listdir(hand_dir) 
                         if f.lower().endswith(Config.HAND_IMAGE_FORMATS)])
    
    print(f"\nTotal Images: {len(all_images)}")
    print(f"First 5 images: {all_images[:5]}")
    print(f"Last 5 images: {all_images[-5:]}")
    
    # Analyze image naming pattern
    print("\nImage Naming Pattern Analysis:")
    sample_names = all_images[:10]
    for name in sample_names:
        print(f"  {name}")
    
    # Load and analyze a sample image
    sample_img_path = os.path.join(hand_dir, all_images[0])
    sample_img = cv2.imread(sample_img_path)
    
    print(f"\nSample Image Analysis:")
    print(f"  Path: {sample_img_path}")
    print(f"  Shape: {sample_img.shape}")
    print(f"  Data Type: {sample_img.dtype}")
    print(f"  Min Value: {sample_img.min()}")
    print(f"  Max Value: {sample_img.max()}")
    print(f"  Mean Value: {sample_img.mean():.2f}")
    
    return sample_img, sample_img_path


def display_sample_images(face_img, hand_img, save_path=None):
    """Display sample face and hand images side by side"""
    
    print("\n" + "=" * 60)
    print("DISPLAYING SAMPLE IMAGES")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display face image
    if face_img is not None:
        # Convert BGR to RGB for matplotlib
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(face_rgb)
        axes[0].set_title(f'Face Image\nShape: {face_img.shape}')
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'Face image not available', 
                     ha='center', va='center')
        axes[0].set_title('Face Image')
    
    # Display hand image
    if hand_img is not None:
        hand_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(hand_rgb)
        axes[1].set_title(f'Hand Image\nShape: {hand_img.shape}')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Hand image not available', 
                     ha='center', va='center')
        axes[1].set_title('Hand Image')
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(Config.LOG_DIR, 'sample_images.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSample images saved to: {save_path}")
    
    # Also try to display (will work if you have display)
    try:
        plt.show()
    except:
        print("(Display not available, image saved to file)")
    
    plt.close()


def analyze_dataset_statistics():
    """Analyze and print dataset statistics"""
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    stats = {
        'face': {
            'users': 0,
            'total_images': 0,
            'images_per_user': [],
            'image_sizes': set()
        },
        'hand': {
            'total_images': 0,
            'image_sizes': set()
        }
    }
    
    # Face dataset statistics
    face_dir = Config.FACE_DATA_DIR
    if os.path.exists(face_dir):
        for folder in os.listdir(face_dir):
            folder_path = os.path.join(face_dir, folder)
            if os.path.isdir(folder_path):
                stats['face']['users'] += 1
                images = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(Config.FACE_IMAGE_FORMATS)]
                stats['face']['total_images'] += len(images)
                stats['face']['images_per_user'].append(len(images))
                
                # Check size of first image
                if images:
                    img = cv2.imread(os.path.join(folder_path, images[0]))
                    if img is not None:
                        stats['face']['image_sizes'].add(img.shape)
    
    # Hand dataset statistics
    hand_dir = Config.HAND_DATA_DIR
    if os.path.exists(hand_dir):
        images = [f for f in os.listdir(hand_dir) 
                  if f.lower().endswith(Config.HAND_IMAGE_FORMATS)]
        stats['hand']['total_images'] = len(images)
        
        # Check sizes of a few images
        for img_name in images[:5]:
            img = cv2.imread(os.path.join(hand_dir, img_name))
            if img is not None:
                stats['hand']['image_sizes'].add(img.shape)
    
    # Print statistics
    print("\nFACE DATASET (CASIA-FaceV5):")
    print(f"  Total Users: {stats['face']['users']}")
    print(f"  Total Images: {stats['face']['total_images']}")
    if stats['face']['images_per_user']:
        print(f"  Images per User: {min(stats['face']['images_per_user'])} - {max(stats['face']['images_per_user'])}")
    print(f"  Image Sizes: {stats['face']['image_sizes']}")
    
    print("\nHAND DATASET (11k Hands):")
    print(f"  Total Images: {stats['hand']['total_images']}")
    print(f"  Sample Image Sizes: {stats['hand']['image_sizes']}")
    
    print("\n" + "=" * 60)
    
    return stats


def main():
    """Main function to explore datasets"""
    
    print("=" * 60)
    print("BIOMETRIC DATASET EXPLORATION")
    print("=" * 60)
    
    # Create log directory if not exists
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Explore face dataset
    face_result = explore_face_dataset()
    face_img = face_result[0] if face_result else None
    
    # Explore hand dataset
    hand_result = explore_hand_dataset()
    hand_img = hand_result[0] if hand_result else None
    
    # Display sample images
    display_sample_images(face_img, hand_img)
    
    # Analyze statistics
    analyze_dataset_statistics()
    
    print("\nDataset exploration complete!")
    print(f"Check {Config.LOG_DIR} for saved visualizations.")


if __name__ == "__main__":
    main()