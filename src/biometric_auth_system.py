# src/biometric_auth_system.py

"""
Complete Biometric Authentication System
Integration of all components as described in the paper

Components:
1. Preprocessing (Median Filter) - Section 4.1
2. Feature Extraction (GLCM+MRG for hand, DL for face) - Sections 4.2, 4.3
3. Fuzzy Vault (Encoding/Decoding) - Section 4.4
4. Blockchain & IPFS (Decentralized Storage) - Phase 2

This module provides the complete enrollment and authentication pipeline.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config

# Import all components
from src.preprocessing.face_preprocessing import FacePreprocessor
from src.preprocessing.hand_preprocessing import HandPreprocessor
from src.feature_extraction.face_features import FaceFeatureExtractor
from src.feature_extraction.hand_features import HandFeatureExtractor
from src.fuzzy_vault.vault_encoder import ImprovedFuzzyVaultEncoder
from src.fuzzy_vault.vault_decoder import ImprovedFuzzyVaultDecoder
from src.blockchain.ipfs_handler import IPFSHandler
from src.blockchain.blockchain_handler import BlockchainHandler


class BiometricAuthSystem:
    """
    Complete Multimodal Biometric Authentication System
    Using Decentralized Fuzzy Vault on Blockchain
    """
    
    def __init__(self):
        print("=" * 60)
        print("INITIALIZING BIOMETRIC AUTHENTICATION SYSTEM")
        print("=" * 60)
        
        # Initialize preprocessing modules
        print("\n[1/6] Loading Face Preprocessor...")
        self.face_preprocessor = FacePreprocessor()
        
        print("[2/6] Loading Hand Preprocessor...")
        self.hand_preprocessor = HandPreprocessor()
        
        # Initialize feature extractors
        print("[3/6] Loading Face Feature Extractor...")
        self.face_extractor = FaceFeatureExtractor()
        
        print("[4/6] Loading Hand Feature Extractor...")
        self.hand_extractor = HandFeatureExtractor()
        
        # Initialize fuzzy vault
        print("[5/6] Loading Fuzzy Vault...")
        self.vault_encoder = ImprovedFuzzyVaultEncoder()
        self.vault_decoder = ImprovedFuzzyVaultDecoder()
        
        # Initialize blockchain/IPFS
        print("[6/6] Loading Blockchain & IPFS...")
        self.ipfs = IPFSHandler()
        self.blockchain = BlockchainHandler(use_simulation=True)
        
        print("\n" + "=" * 60)
        print("SYSTEM INITIALIZED SUCCESSFULLY")
        print("=" * 60)
    
    def enroll_user(self, user_id, face_image_path, hand_image_path):
        """
        Enroll a new user into the system
        
        Implements Algorithm 1 from the paper:
        1. Preprocess biometric images
        2. Extract features
        3. Create fuzzy vault
        4. Store in IPFS
        5. Record address in blockchain
        
        Args:
            user_id: Unique user identifier
            face_image_path: Path to face image
            hand_image_path: Path to hand image
        
        Returns:
            enrollment_result: Dictionary with enrollment details
        """
        print(f"\n{'='*60}")
        print(f"ENROLLING USER: {user_id}")
        print("=" * 60)
        
        result = {
            'user_id': user_id,
            'status': 'failed',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Preprocess images
            print("\n[Step 1] Preprocessing images...")
            
            face_processed, _ = self.face_preprocessor.preprocess(face_image_path)
            print(f"  Face image preprocessed: {face_processed.shape}")
            
            hand_processed, _ = self.hand_preprocessor.preprocess(hand_image_path)
            print(f"  Hand image preprocessed: {hand_processed.shape}")
            
            # Step 2: Extract features
            print("\n[Step 2] Extracting features...")
            
            face_features = self.face_extractor.extract_features(face_image_path)
            print(f"  Face features (C1): {face_features.shape}")
            
            hand_features = self.hand_extractor.extract_features(hand_image_path)
            print(f"  Hand features (C2): {hand_features.shape}")
            
            # Step 3: Create fuzzy vault
            print("\n[Step 3] Creating fuzzy vault...")
            
            fuzzy_vault, bio_token, _, _ = self.vault_encoder.encode(
                hand_features, face_features, user_id
            )
            print(f"  Vault created: {len(fuzzy_vault)} values")
            
            # Step 4: Store in IPFS
            print("\n[Step 4] Storing in IPFS...")
            
            vault_package = {
                'user_id': user_id,
                'fuzzy_vault': fuzzy_vault.tolist(),
                'bio_token': bio_token,
                'enrolled_at': datetime.now().isoformat()
            }
            
            ipfs_hash = self.ipfs.store_vault(vault_package)
            print(f"  IPFS hash: {ipfs_hash}")
            
            # Step 5: Record in blockchain
            print("\n[Step 5] Recording in blockchain...")
            
            tx_hash = self.blockchain.store_vault_address(user_id, ipfs_hash)
            print(f"  Transaction hash: {tx_hash[:20]}...")
            
            # Update result
            result['status'] = 'success'
            result['ipfs_hash'] = ipfs_hash
            result['tx_hash'] = tx_hash
            result['vault_size'] = len(fuzzy_vault)
            
            print("\n" + "=" * 60)
            print(f"ENROLLMENT SUCCESSFUL for {user_id}")
            print("=" * 60)
            
        except Exception as e:
            result['error'] = str(e)
            print(f"\nENROLLMENT FAILED: {e}")
        
        return result
    
    def authenticate_user(self, user_id, face_image_path, hand_image_path):
        """
        Authenticate a user
        
        Implements Algorithm 2 from the paper:
        1. Retrieve vault address from blockchain
        2. Get vault from IPFS
        3. Preprocess and extract features
        4. Verify using fuzzy vault decoder
        
        Args:
            user_id: User identifier to authenticate
            face_image_path: Path to face image
            hand_image_path: Path to hand image
        
        Returns:
            auth_result: Dictionary with authentication result
        """
        print(f"\n{'='*60}")
        print(f"AUTHENTICATING USER: {user_id}")
        print("=" * 60)
        
        result = {
            'user_id': user_id,
            'authenticated': False,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Get vault address from blockchain
            print("\n[Step 1] Retrieving from blockchain...")
            
            ipfs_hash = self.blockchain.get_vault_address(user_id)
            
            if ipfs_hash is None:
                result['error'] = 'User not found in blockchain'
                print(f"  Error: User {user_id} not enrolled")
                return result
            
            print(f"  IPFS hash: {ipfs_hash}")
            
            # Step 2: Get vault from IPFS
            print("\n[Step 2] Retrieving vault from IPFS...")
            
            vault_package = self.ipfs.retrieve_vault(ipfs_hash)
            
            if vault_package is None:
                result['error'] = 'Vault not found in IPFS'
                print("  Error: Vault not found")
                return result
            
            print(f"  Vault retrieved successfully")
            
            # Step 3: Extract features from verification images
            print("\n[Step 3] Extracting features...")
            
            face_features = self.face_extractor.extract_features(face_image_path)
            print(f"  Face features extracted: {face_features.shape}")
            
            hand_features = self.hand_extractor.extract_features(hand_image_path)
            print(f"  Hand features extracted: {hand_features.shape}")
            
            # Step 4: Verify using fuzzy vault
            print("\n[Step 4] Verifying with fuzzy vault...")
            
            is_genuine, score, message = self.vault_decoder.verify(
                vault_package['fuzzy_vault'],
                vault_package['bio_token'],
                hand_features,
                face_features
            )
            
            print(f"  Score: {score:.4f}")
            print(f"  Result: {message}")
            
            # Update result
            result['authenticated'] = is_genuine
            result['score'] = float(score)
            result['message'] = message
            
            print("\n" + "=" * 60)
            if is_genuine:
                print(f"AUTHENTICATION SUCCESSFUL for {user_id}")
            else:
                print(f"AUTHENTICATION FAILED for {user_id}")
            print("=" * 60)
            
        except Exception as e:
            result['error'] = str(e)
            print(f"\nAUTHENTICATION ERROR: {e}")
        
        return result
    
    def revoke_user(self, user_id):
        """
        Revoke a user's enrollment (for revocability)
        
        Args:
            user_id: User identifier
        
        Returns:
            revoke_result: Dictionary with revocation details
        """
        print(f"\nRevoking enrollment for user: {user_id}")
        
        # Revoke in blockchain
        tx_hash = self.blockchain.revoke_vault(user_id)
        
        if tx_hash:
            print(f"  Revocation recorded: {tx_hash[:20]}...")
            return {'status': 'revoked', 'tx_hash': tx_hash}
        else:
            return {'status': 'failed', 'error': 'User not found'}
    
    def get_system_stats(self):
        """Get system statistics"""
        blockchain_info = self.blockchain.get_info()
        ipfs_vaults = self.ipfs.list_vaults()
        
        return {
            'blockchain': blockchain_info,
            'ipfs_vaults': len(ipfs_vaults),
            'system_status': 'operational'
        }


def test_complete_system():
    """Test the complete biometric authentication system"""
    
    print("\n" + "=" * 70)
    print("TESTING COMPLETE BIOMETRIC AUTHENTICATION SYSTEM")
    print("=" * 70)
    
    # Initialize system
    system = BiometricAuthSystem()
    
    # Get sample images
    face_dir = Config.FACE_DATA_DIR
    hand_dir = Config.HAND_DATA_DIR
    
    # Get first user's face images
    user_folder = "000"
    face_user_dir = os.path.join(face_dir, user_folder)
    
    if not os.path.exists(face_user_dir):
        print(f"Error: Face directory not found: {face_user_dir}")
        return
    
    face_images = sorted([
        f for f in os.listdir(face_user_dir)
        if f.lower().endswith(Config.FACE_IMAGE_FORMATS)
    ])
    
    if not face_images:
        print("Error: No face images found")
        return
    
    # Get hand images
    hand_images = sorted([
        f for f in os.listdir(hand_dir)
        if f.lower().endswith(Config.HAND_IMAGE_FORMATS)
    ])
    
    if not hand_images:
        print("Error: No hand images found")
        return
    
    # Use first face and hand images for enrollment
    face_enroll = os.path.join(face_user_dir, face_images[0])
    hand_enroll = os.path.join(hand_dir, hand_images[0])
    
    print(f"\nUsing images:")
    print(f"  Face (enroll): {face_enroll}")
    print(f"  Hand (enroll): {hand_enroll}")
    
    # TEST 1: Enroll user
    print("\n" + "-" * 70)
    print("TEST 1: USER ENROLLMENT")
    print("-" * 70)
    
    enroll_result = system.enroll_user(
        user_id="test_user_real",
        face_image_path=face_enroll,
        hand_image_path=hand_enroll
    )
    
    print(f"\nEnrollment Result: {enroll_result['status']}")
    
    if enroll_result['status'] != 'success':
        print("Enrollment failed, cannot continue tests")
        return
    
    # TEST 2: Authenticate with same images (should succeed)
    print("\n" + "-" * 70)
    print("TEST 2: AUTHENTICATE GENUINE USER (Same Images)")
    print("-" * 70)
    
    auth_result_1 = system.authenticate_user(
        user_id="test_user_real",
        face_image_path=face_enroll,
        hand_image_path=hand_enroll
    )
    
    print(f"\nAuthentication: {'SUCCESS ✓' if auth_result_1['authenticated'] else 'FAILED ✗'}")
    print(f"Score: {auth_result_1.get('score', 'N/A')}")
    
    # TEST 3: Authenticate with different face image of same user
    print("\n" + "-" * 70)
    print("TEST 3: AUTHENTICATE WITH DIFFERENT IMAGE (Same User)")
    print("-" * 70)
    
    if len(face_images) > 1:
        face_verify = os.path.join(face_user_dir, face_images[1])
        print(f"Using different face image: {face_images[1]}")
        
        auth_result_2 = system.authenticate_user(
            user_id="test_user_real",
            face_image_path=face_verify,
            hand_image_path=hand_enroll  # Same hand
        )
        
        print(f"\nAuthentication: {'SUCCESS ✓' if auth_result_2['authenticated'] else 'FAILED ✗'}")
        print(f"Score: {auth_result_2.get('score', 'N/A')}")
    
    # TEST 4: Authenticate with different user's face (should fail)
    print("\n" + "-" * 70)
    print("TEST 4: AUTHENTICATE IMPOSTER (Different User's Face)")
    print("-" * 70)
    
    # Get different user's face
    other_user_folder = "001"
    other_face_dir = os.path.join(face_dir, other_user_folder)
    
    if os.path.exists(other_face_dir):
        other_face_images = sorted([
            f for f in os.listdir(other_face_dir)
            if f.lower().endswith(Config.FACE_IMAGE_FORMATS)
        ])
        
        if other_face_images:
            other_face = os.path.join(other_face_dir, other_face_images[0])
            print(f"Using imposter face: {other_face}")
            
            auth_result_3 = system.authenticate_user(
                user_id="test_user_real",
                face_image_path=other_face,
                hand_image_path=hand_enroll
            )
            
            print(f"\nAuthentication: {'SUCCESS ✓' if auth_result_3['authenticated'] else 'REJECTED ✓ (Expected)'}")
            print(f"Score: {auth_result_3.get('score', 'N/A')}")
    
    # TEST 5: System statistics
    print("\n" + "-" * 70)
    print("TEST 5: SYSTEM STATISTICS")
    print("-" * 70)
    
    stats = system.get_system_stats()
    print(f"\nBlockchain:")
    for key, value in stats['blockchain'].items():
        print(f"  {key}: {value}")
    print(f"IPFS Vaults: {stats['ipfs_vaults']}")
    print(f"System Status: {stats['system_status']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("""
    ┌────────────────────────────────────────┬──────────────┐
    │ Test                                   │ Status       │
    ├────────────────────────────────────────┼──────────────┤
    │ 1. User Enrollment                     │ Check above  │
    │ 2. Authenticate (same images)          │ Check above  │
    │ 3. Authenticate (different image)      │ Check above  │
    │ 4. Imposter Detection                  │ Check above  │
    │ 5. System Statistics                   │ ✓ Working    │
    └────────────────────────────────────────┴──────────────┘
    """)
    
    print("=" * 70)
    print("COMPLETE SYSTEM TEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    test_complete_system()