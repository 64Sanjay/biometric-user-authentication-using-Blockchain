# # src/fuzzy_vault/vault_encoder.py

# """
# Improved Fuzzy Vault Encoder
# Implementation of Algorithm 1 from the paper

# Key improvements over traditional fuzzy vault:
# 1. Variable key size (not fixed)
# 2. Multimodal biometric integration (face + hand)
# 3. Asymmetric cryptosystem for encryption

# This encoder creates the fuzzy vault during user enrollment.
# """

# import os
# import sys
# import numpy as np

# # Add project root to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from config.config import Config
# from src.fuzzy_vault.reed_solomon import ReedSolomonEncoder


# class ImprovedFuzzyVaultEncoder:
#     """
#     Improved Fuzzy Vault Encoder
#     Based on Algorithm 1 from the paper
    
#     Creates bio token by:
#     1. Generating random positions R
#     2. Creating grid with features and chaff points
#     3. Encrypting with second biometric
#     """
    
#     def __init__(self, n_features=Config.N_FEATURES, tolerance=Config.FV_TOLERANCE):
#         """
#         Initialize encoder
        
#         Args:
#             n_features: Number of features (128 as per paper)
#             tolerance: Fuzzy vault tolerance for matching
#         """
#         self.n = n_features
#         self.tolerance = tolerance
#         self.grid_columns = Config.GRID_COLUMNS  # 3 columns
#         self.rs_encoder = ReedSolomonEncoder()
    
#     def generate_random_positions(self):
#         """
#         Step 1 (Algorithm 1): Generate n random numbers between 1 to 2n
        
#         R = {r1, r2, ..., rn} - random distinct positions
        
#         Returns:
#             Sorted array of n random positions
#         """
#         # Generate n unique random positions from range [1, 2n]
#         R = np.random.choice(
#             range(1, 2 * self.n + 1), 
#             size=self.n, 
#             replace=False
#         )
#         return np.sort(R)
    
#     def generate_rs_codes(self, biometric_features):
#         """
#         Steps 2 & 8 (Algorithm 1): Generate RS codes from biometric
        
#         C1 from face (Biometric 2)
#         C2 from hand (Biometric 1)
        
#         Args:
#             biometric_features: Feature vector
        
#         Returns:
#             RS codes array
#         """
#         return self.rs_encoder.generate_codes_from_features(
#             biometric_features,
#             self.n
#         )
    
#     def create_grid(self, R, C1):
#         """
#         Steps 3-7 (Algorithm 1): Create grid Gr of size 2n × 3
        
#         Process:
#         - Row indices from R contain the actual features from C1
#         - Each feature placed randomly in column 0, 1, or 2
#         - Remaining positions filled to form arithmetic progression
#         - Rows not in R are chaff rows with random values
        
#         Args:
#             R: Random position indices
#             C1: Feature codes from face biometric
        
#         Returns:
#             Grid Gr and feature position map
#         """
#         grid_rows = 2 * self.n
#         Gr = np.zeros((grid_rows, self.grid_columns))
        
#         # Track which column contains the real feature
#         feature_positions = {}
        
#         # Step 4-5: Place features at positions specified by R
#         for i, (r_idx, feature) in enumerate(zip(R, C1)):
#             # Step 4: Randomly choose column (0, 1, or 2)
#             ic = np.random.randint(0, self.grid_columns)
            
#             # Step 5: Place feature at Gr[r_idx-1][ic] (0-indexed)
#             row_idx = r_idx - 1
#             Gr[row_idx][ic] = feature
#             feature_positions[row_idx] = ic
            
#             # Step 6: Fill other positions to form arithmetic progression
#             self._fill_row_arithmetic(Gr, row_idx, ic, feature)
        
#         # Step 7: Fill chaff rows (rows not in R)
#         R_set = set(R - 1)  # Convert to 0-indexed
#         for row_idx in range(grid_rows):
#             if row_idx not in R_set:
#                 # Random chaff row forming arithmetic progression
#                 start_val = np.random.random()
#                 for col in range(self.grid_columns):
#                     Gr[row_idx][col] = start_val + col * self.tolerance
        
#         return Gr, feature_positions
    
#     def _fill_row_arithmetic(self, grid, row_idx, feature_col, feature_value):
#         """
#         Fill row with arithmetic progression
#         Common difference = FV tolerance
        
#         Args:
#             grid: The grid matrix
#             row_idx: Row to fill
#             feature_col: Column containing the actual feature
#             feature_value: The feature value
#         """
#         for col in range(self.grid_columns):
#             if col != feature_col:
#                 distance = col - feature_col
#                 grid[row_idx][col] = feature_value + distance * self.tolerance
    
#     def encrypt_with_biometric(self, grid, R, C2):
#         """
#         Step 9 (Algorithm 1): Encrypt grid and R using C2
        
#         Uses asymmetric cryptosystem based on hand features.
        
#         Args:
#             grid: The grid matrix Gr
#             R: Position indices
#             C2: RS codes from hand biometric
        
#         Returns:
#             Encrypted fuzzy vault
#         """
#         # Flatten grid and combine with R
#         flat_grid = grid.flatten()
#         R_float = R.astype(float)
#         combined_data = np.concatenate([flat_grid, R_float])
        
#         # Extend C2 to match data length
#         extended_key = np.tile(C2, (len(combined_data) // len(C2)) + 1)
#         extended_key = extended_key[:len(combined_data)]
        
#         # Encrypt using XOR-like operation with non-linear mixing
#         encrypted = np.zeros_like(combined_data)
#         for i in range(len(combined_data)):
#             # Add key component
#             mixed = combined_data[i] + extended_key[i]
#             # Apply non-linear transformation
#             encrypted[i] = np.sin(mixed * np.pi) * 0.5 + 0.5
        
#         return encrypted
    
#     def encode(self, hand_features, face_features, user_id=None):
#         """
#         Complete enrollment algorithm (Algorithm 1)
        
#         Args:
#             hand_features: Biometric 1 features (C2 source)
#             face_features: Biometric 2 features (C1 source)
#             user_id: Optional user identifier
        
#         Returns:
#             fuzzy_vault: Encrypted vault
#             bio_token: Metadata for decoding
#         """
#         # Step 1: Generate random positions R
#         R = self.generate_random_positions()
        
#         # Step 2: Generate RS codes C1 from face features
#         C1 = self.generate_rs_codes(face_features)
        
#         # Steps 3-7: Create grid with features and chaff
#         Gr, feature_positions = self.create_grid(R, C1)
        
#         # Step 8: Generate RS codes C2 from hand features
#         C2 = self.generate_rs_codes(hand_features)
        
#         # Step 9: Encrypt grid and R using C2
#         fuzzy_vault = self.encrypt_with_biometric(Gr, R, C2)
        
#         # Create bio token with metadata
#         bio_token = {
#             'grid_shape': Gr.shape,
#             'n_positions': len(R),
#             'n_features': self.n,
#             'tolerance': self.tolerance,
#             'user_id': user_id
#         }
        
#         return fuzzy_vault, bio_token, C1, C2
    
#     def create_vault_package(self, hand_features, face_features, user_id):
#         """
#         Create complete vault package for storage (IPFS)
        
#         Args:
#             hand_features: Hand biometric features
#             face_features: Face biometric features
#             user_id: User identifier
        
#         Returns:
#             Dictionary ready for storage
#         """
#         fuzzy_vault, bio_token, _, _ = self.encode(
#             hand_features, face_features, user_id
#         )
        
#         package = {
#             'user_id': user_id,
#             'fuzzy_vault': fuzzy_vault.tolist(),
#             'bio_token': bio_token
#         }
        
#         return package


# def test_vault_encoder():
#     """Test the fuzzy vault encoder"""
    
#     print("=" * 60)
#     print("TESTING FUZZY VAULT ENCODER (Algorithm 1)")
#     print("=" * 60)
    
#     encoder = ImprovedFuzzyVaultEncoder()
    
#     # Generate fake biometric features for testing
#     np.random.seed(42)
#     hand_features = np.random.randn(128)  # Biometric 1
#     face_features = np.random.randn(128)  # Biometric 2
    
#     print("\nInput Features:")
#     print(f"  Hand features shape: {hand_features.shape}")
#     print(f"  Face features shape: {face_features.shape}")
    
#     # Test encoding
#     print("\n" + "-" * 40)
#     print("Encoding Process:")
    
#     # Step by step
#     R = encoder.generate_random_positions()
#     print(f"\n  Step 1 - Random positions R:")
#     print(f"    Shape: {R.shape}")
#     print(f"    Range: [{R.min()}, {R.max()}]")
#     print(f"    First 5: {R[:5]}")
    
#     C1 = encoder.generate_rs_codes(face_features)
#     print(f"\n  Step 2 - RS codes C1 (from face):")
#     print(f"    Shape: {C1.shape}")
#     print(f"    Range: [{C1.min():.4f}, {C1.max():.4f}]")
    
#     Gr, positions = encoder.create_grid(R, C1)
#     print(f"\n  Steps 3-7 - Grid Gr:")
#     print(f"    Shape: {Gr.shape}")
#     print(f"    Range: [{Gr.min():.4f}, {Gr.max():.4f}]")
#     print(f"    Feature positions stored: {len(positions)}")
    
#     C2 = encoder.generate_rs_codes(hand_features)
#     print(f"\n  Step 8 - RS codes C2 (from hand):")
#     print(f"    Shape: {C2.shape}")
#     print(f"    Range: [{C2.min():.4f}, {C2.max():.4f}]")
    
#     # Complete encoding
#     fuzzy_vault, bio_token, _, _ = encoder.encode(
#         hand_features, face_features, "test_user"
#     )
    
#     print(f"\n  Step 9 - Encrypted Fuzzy Vault:")
#     print(f"    Shape: {fuzzy_vault.shape}")
#     print(f"    Range: [{fuzzy_vault.min():.4f}, {fuzzy_vault.max():.4f}]")
    
#     print(f"\n  Bio Token:")
#     for key, value in bio_token.items():
#         print(f"    {key}: {value}")
    
#     # Create package
#     print("\n" + "-" * 40)
#     print("Creating Vault Package:")
    
#     package = encoder.create_vault_package(
#         hand_features, face_features, "user_001"
#     )
    
#     print(f"  Package keys: {list(package.keys())}")
#     print(f"  Vault size: {len(package['fuzzy_vault'])} values")
    
#     # Visualize
#     import matplotlib.pyplot as plt
    
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
#     # Grid visualization
#     axes[0, 0].imshow(Gr, aspect='auto', cmap='viridis')
#     axes[0, 0].set_title('Grid Gr (2n × 3)')
#     axes[0, 0].set_xlabel('Column')
#     axes[0, 0].set_ylabel('Row')
    
#     # Fuzzy vault
#     vault_reshaped = fuzzy_vault[:256].reshape(16, 16)
#     axes[0, 1].imshow(vault_reshaped, cmap='plasma')
#     axes[0, 1].set_title('Encrypted Fuzzy Vault (sample)')
    
#     # C1 and C2 comparison
#     axes[1, 0].plot(C1, label='C1 (Face)', alpha=0.7)
#     axes[1, 0].plot(C2, label='C2 (Hand)', alpha=0.7)
#     axes[1, 0].set_title('RS Codes: C1 vs C2')
#     axes[1, 0].set_xlabel('Index')
#     axes[1, 0].set_ylabel('Value')
#     axes[1, 0].legend()
#     axes[1, 0].grid(True, alpha=0.3)
    
#     # R positions distribution
#     axes[1, 1].hist(R, bins=30, edgecolor='black', alpha=0.7)
#     axes[1, 1].set_title('Random Positions R Distribution')
#     axes[1, 1].set_xlabel('Position')
#     axes[1, 1].set_ylabel('Count')
    
#     plt.tight_layout()
    
#     output_path = os.path.join(Config.LOG_DIR, 'vault_encoder_test.png')
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     print(f"\nVisualization saved to: {output_path}")
    
#     plt.close()
    
#     print("\n" + "=" * 60)
#     print("VAULT ENCODER TEST COMPLETE")
#     print("=" * 60)
    
#     return fuzzy_vault, bio_token


# if __name__ == "__main__":
#     test_vault_encoder()
# --------------
# V2
# src/fuzzy_vault/vault_encoder.py

"""
Improved Fuzzy Vault Encoder - FIXED VERSION
Implementation of Algorithm 1 from the paper
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.fuzzy_vault.reed_solomon import ReedSolomonEncoder


class ImprovedFuzzyVaultEncoder:
    """
    Improved Fuzzy Vault Encoder
    Based on Algorithm 1 from the paper
    """
    
    def __init__(self, n_features=Config.N_FEATURES, tolerance=Config.FV_TOLERANCE):
        self.n = n_features
        self.tolerance = tolerance
        self.grid_columns = Config.GRID_COLUMNS  # 3
        self.rs_encoder = ReedSolomonEncoder()
    
    def generate_random_positions(self, seed=None):
        """
        Step 1: Generate n random numbers between 1 to 2n
        """
        if seed is not None:
            np.random.seed(seed)
        
        R = np.random.choice(
            range(1, 2 * self.n + 1), 
            size=self.n, 
            replace=False
        )
        return np.sort(R)
    
    def generate_rs_codes(self, biometric_features):
        """
        Generate RS codes from biometric features
        """
        return self.rs_encoder.generate_codes_from_features(
            biometric_features,
            self.n
        )
    
    def create_grid(self, R, C1, seed=None):
        """
        Steps 3-7: Create grid Gr of size 2n × 3
        """
        if seed is not None:
            np.random.seed(seed)
        
        grid_rows = 2 * self.n
        Gr = np.zeros((grid_rows, self.grid_columns))
        
        # Track which column contains the real feature
        feature_columns = np.zeros(self.n, dtype=int)
        
        # Place features at positions specified by R
        for i, (r_idx, feature) in enumerate(zip(R, C1)):
            ic = np.random.randint(0, self.grid_columns)
            feature_columns[i] = ic
            
            row_idx = r_idx - 1  # 0-indexed
            Gr[row_idx][ic] = feature
            
            # Fill other columns with chaff (arithmetic progression)
            for col in range(self.grid_columns):
                if col != ic:
                    distance = col - ic
                    Gr[row_idx][col] = feature + distance * self.tolerance
        
        # Fill chaff rows
        R_set = set(R - 1)
        for row_idx in range(grid_rows):
            if row_idx not in R_set:
                start_val = np.random.random()
                for col in range(self.grid_columns):
                    Gr[row_idx][col] = start_val + col * self.tolerance
        
        return Gr, feature_columns
    
    def encrypt_vault(self, Gr, R, feature_columns, C2):
        """
        Step 9: Encrypt grid using C2
        
        Simple XOR-based encryption that can be reversed
        """
        # Store everything we need for decryption
        vault_data = {
            'grid': Gr.copy(),
            'R': R.copy(),
            'feature_columns': feature_columns.copy()
        }
        
        # Create encryption key from C2
        key = np.sum(C2)  # Simple key derivation
        
        # Encrypt grid values
        encrypted_grid = Gr + key
        
        # Encrypt R values
        encrypted_R = R.astype(float) + key
        
        # Encrypt feature columns
        encrypted_fc = feature_columns.astype(float) + key
        
        # Combine into single array
        encrypted_vault = np.concatenate([
            encrypted_grid.flatten(),
            encrypted_R,
            encrypted_fc,
            [key]  # Store key for verification (in real system, derive from C2)
        ])
        
        return encrypted_vault, vault_data
    
    def encode(self, hand_features, face_features, user_id=None):
        """
        Complete enrollment algorithm (Algorithm 1)
        """
        # Use consistent random seed based on features for reproducibility
        seed = int(np.abs(np.sum(hand_features) * 1000)) % (2**31)
        
        # Step 1: Generate random positions R
        R = self.generate_random_positions(seed)
        
        # Step 2: Generate RS codes C1 from face features
        C1 = self.generate_rs_codes(face_features)
        
        # Steps 3-7: Create grid
        Gr, feature_columns = self.create_grid(R, C1, seed + 1)
        
        # Step 8: Generate RS codes C2 from hand features
        C2 = self.generate_rs_codes(hand_features)
        
        # Step 9: Encrypt
        encrypted_vault, vault_data = self.encrypt_vault(Gr, R, feature_columns, C2)
        
        # Bio token with metadata
        bio_token = {
            'grid_shape': Gr.shape,
            'n_positions': len(R),
            'n_features': self.n,
            'tolerance': self.tolerance,
            'user_id': user_id,
            'C2_hash': float(np.sum(C2))  # Store hash for decryption verification
        }
        
        return encrypted_vault, bio_token, C1, C2
    
    def create_vault_package(self, hand_features, face_features, user_id):
        """
        Create complete vault package for storage
        """
        fuzzy_vault, bio_token, _, _ = self.encode(
            hand_features, face_features, user_id
        )
        
        package = {
            'user_id': user_id,
            'fuzzy_vault': fuzzy_vault.tolist(),
            'bio_token': bio_token
        }
        
        return package


def test_vault_encoder():
    """Test the fuzzy vault encoder"""
    
    print("=" * 60)
    print("TESTING FUZZY VAULT ENCODER (Algorithm 1) - FIXED")
    print("=" * 60)
    
    encoder = ImprovedFuzzyVaultEncoder()
    
    # Generate test features
    np.random.seed(42)
    hand_features = np.random.randn(128)
    face_features = np.random.randn(128)
    
    print("\nInput Features:")
    print(f"  Hand features shape: {hand_features.shape}")
    print(f"  Face features shape: {face_features.shape}")
    
    # Encode
    fuzzy_vault, bio_token, C1, C2 = encoder.encode(
        hand_features, face_features, "test_user"
    )
    
    print(f"\nEncoded Vault:")
    print(f"  Vault size: {len(fuzzy_vault)}")
    print(f"  Vault range: [{fuzzy_vault.min():.4f}, {fuzzy_vault.max():.4f}]")
    
    print(f"\nBio Token:")
    for key, value in bio_token.items():
        print(f"  {key}: {value}")
    
    print(f"\nC1 (face codes):")
    print(f"  Shape: {C1.shape}")
    print(f"  Range: [{C1.min():.4f}, {C1.max():.4f}]")
    
    print(f"\nC2 (hand codes):")
    print(f"  Shape: {C2.shape}")
    print(f"  Range: [{C2.min():.4f}, {C2.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("VAULT ENCODER TEST COMPLETE")
    print("=" * 60)
    
    return fuzzy_vault, bio_token, C1, C2


if __name__ == "__main__":
    test_vault_encoder()