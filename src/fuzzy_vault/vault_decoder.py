# # src/fuzzy_vault/vault_decoder.py

# """
# Improved Fuzzy Vault Decoder
# Implementation of Algorithm 2 from the paper

# Recognition process:
# 1. Decrypt vault using Biometric 1 (hand)
# 2. Reconstruct grid and positions
# 3. Compare with Biometric 2 (face) features
# 4. Calculate matching score
# 5. Determine genuine/imposter based on threshold
# """

# import os
# import sys
# import numpy as np

# # Add project root to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from config.config import Config
# from src.fuzzy_vault.reed_solomon import ReedSolomonEncoder


# class ImprovedFuzzyVaultDecoder:
#     """
#     Improved Fuzzy Vault Decoder
#     Based on Algorithm 2 from the paper
    
#     Verifies user identity by:
#     1. Decrypting vault with hand features
#     2. Matching with face features
#     3. Computing similarity score
#     """
    
#     def __init__(self, n_features=Config.N_FEATURES, tolerance=Config.FV_TOLERANCE):
#         """
#         Initialize decoder
        
#         Args:
#             n_features: Number of features
#             tolerance: Matching tolerance
#         """
#         self.n = n_features
#         self.tolerance = tolerance
#         self.grid_columns = Config.GRID_COLUMNS
#         self.threshold = Config.AUTH_THRESHOLD  # 0.35
#         self.rs_encoder = ReedSolomonEncoder()
    
#     def generate_rs_codes(self, biometric_features):
#         """
#         Generate RS codes from biometric features
#         Same as encoder for consistency
#         """
#         return self.rs_encoder.generate_codes_from_features(
#             biometric_features,
#             self.n
#         )
    
#     def decrypt_vault(self, encrypted_vault, C2_prime):
#         """
#         Step 4 (Algorithm 2): Decrypt fuzzy vault using C2'
        
#         Reverse the encryption process using hand features
        
#         Args:
#             encrypted_vault: Encrypted vault from storage
#             C2_prime: RS codes from verification hand features
        
#         Returns:
#             Decrypted data
#         """
#         encrypted_vault = np.array(encrypted_vault)
        
#         # Extend key to match vault length
#         extended_key = np.tile(C2_prime, (len(encrypted_vault) // len(C2_prime)) + 1)
#         extended_key = extended_key[:len(encrypted_vault)]
        
#         # Decrypt (reverse of encryption)
#         decrypted = np.zeros_like(encrypted_vault)
#         for i in range(len(encrypted_vault)):
#             # Reverse non-linear transformation
#             val = encrypted_vault[i]
#             # Clamp to valid range for arcsin
#             val = np.clip((val - 0.5) * 2, -0.999, 0.999)
#             decrypted[i] = np.arcsin(val) / np.pi
#             # Subtract key
#             decrypted[i] = decrypted[i] - extended_key[i]
        
#         return decrypted
    
#     def reconstruct_grid_and_positions(self, decrypted_data, grid_shape, n_positions):
#         """
#         Reconstruct grid Gr and positions R from decrypted data
        
#         Args:
#             decrypted_data: Decrypted vault data
#             grid_shape: Shape of original grid (2n, 3)
#             n_positions: Number of position values
        
#         Returns:
#             Reconstructed grid and positions
#         """
#         grid_size = grid_shape[0] * grid_shape[1]
        
#         # Split data
#         grid_flat = decrypted_data[:grid_size]
#         R_float = decrypted_data[grid_size:grid_size + n_positions]
        
#         # Reconstruct grid
#         Gr = grid_flat.reshape(grid_shape)
        
#         # Reconstruct R (convert to integer indices)
#         R = np.round(R_float).astype(int)
#         R = np.clip(R, 1, 2 * self.n)  # Ensure valid range
        
#         return Gr, R
    
#     def calculate_distance(self, a, b):
#         """
#         Calculate distance between two values
#         Used in Algorithm 2 steps 11-19
#         """
#         return np.abs(a - b)
    
#     def calculate_score(self, Gr, R, C1_prime):
#         """
#         Steps 11-20 (Algorithm 2): Calculate matching score
        
#         For each position in R:
#         - Find column with closest value to corresponding C1' feature
#         - Accumulate matches based on tolerance
        
#         Args:
#             Gr: Reconstructed grid
#             R: Position indices
#             C1_prime: RS codes from verification face features
        
#         Returns:
#             Normalized score and number of matches
#         """
#         score = 0.0
#         matches = 0
#         total_distance = 0.0
        
#         for i, r_idx in enumerate(R):
#             if i >= len(C1_prime):
#                 break
            
#             row_idx = r_idx - 1  # Convert to 0-indexed
            
#             # Boundary check
#             if row_idx < 0 or row_idx >= len(Gr):
#                 continue
            
#             feature = C1_prime[i]
#             row = Gr[row_idx]
            
#             # Find minimum distance column (Steps 12-17)
#             distances = [self.calculate_distance(row[col], feature) 
#                         for col in range(self.grid_columns)]
#             min_col = np.argmin(distances)
#             min_distance = distances[min_col]
            
#             # Check if within tolerance
#             if min_distance < self.tolerance:
#                 score += 1.0  # Count as match
#                 matches += 1
            
#             total_distance += min_distance
        
#         # Step 20: Calculate average score
#         if len(R) > 0:
#             # Normalized score: proportion of matches
#             normalized_score = matches / len(R)
#             avg_distance = total_distance / len(R)
#         else:
#             normalized_score = 0.0
#             avg_distance = float('inf')
        
#         return normalized_score, matches, avg_distance
    
#     def verify(self, encrypted_vault, bio_token, hand_features, face_features):
#         """
#         Complete recognition algorithm (Algorithm 2)
        
#         Args:
#             encrypted_vault: Stored fuzzy vault
#             bio_token: Metadata about the vault
#             hand_features: Verification hand features (Biometric 1)
#             face_features: Verification face features (Biometric 2)
        
#         Returns:
#             is_genuine: Boolean
#             score: Matching score
#             message: Result description
#         """
#         # Step 3: Generate RS codes C2' from hand features
#         C2_prime = self.generate_rs_codes(hand_features)
        
#         # Step 4: Decrypt vault
#         decrypted = self.decrypt_vault(encrypted_vault, C2_prime)
        
#         # Steps 5-7: Check if decryption produced valid data
#         if np.all(np.isnan(decrypted)) or np.all(decrypted == 0):
#             return False, 0.0, "Decryption failed - Imposter (hand mismatch)"
        
#         # Reconstruct grid and positions
#         try:
#             Gr, R = self.reconstruct_grid_and_positions(
#                 decrypted,
#                 bio_token['grid_shape'],
#                 bio_token['n_positions']
#             )
#         except Exception as e:
#             return False, 0.0, f"Reconstruction failed: {e}"
        
#         # Step 9: Generate RS codes C1' from face features
#         C1_prime = self.generate_rs_codes(face_features)
        
#         # Steps 10-20: Calculate matching score
#         score, matches, avg_distance = self.calculate_score(Gr, R, C1_prime)
        
#         # Steps 21-25: Determine genuine or imposter
#         if score > self.threshold:
#             return True, score, f"Genuine User (score: {score:.4f}, matches: {matches})"
#         else:
#             return False, score, f"Imposter User (score: {score:.4f}, matches: {matches})"
    
#     def batch_verify(self, stored_vaults, hand_features, face_features):
#         """
#         Verify against multiple stored vaults
        
#         Args:
#             stored_vaults: List of vault packages
#             hand_features: Verification hand features
#             face_features: Verification face features
        
#         Returns:
#             Best matching result
#         """
#         best_match = None
#         best_score = -1
        
#         for vault_data in stored_vaults:
#             is_genuine, score, message = self.verify(
#                 vault_data['fuzzy_vault'],
#                 vault_data['bio_token'],
#                 hand_features,
#                 face_features
#             )
            
#             if score > best_score:
#                 best_score = score
#                 best_match = {
#                     'user_id': vault_data.get('user_id', 'unknown'),
#                     'is_genuine': is_genuine,
#                     'score': score,
#                     'message': message
#                 }
        
#         return best_match


# def test_vault_decoder():
#     """Test the fuzzy vault decoder"""
    
#     print("=" * 60)
#     print("TESTING FUZZY VAULT DECODER (Algorithm 2)")
#     print("=" * 60)
    
#     from src.fuzzy_vault.vault_encoder import ImprovedFuzzyVaultEncoder
    
#     encoder = ImprovedFuzzyVaultEncoder()
#     decoder = ImprovedFuzzyVaultDecoder()
    
#     # Generate enrollment features
#     np.random.seed(42)
#     hand_features_enroll = np.random.randn(128)
#     face_features_enroll = np.random.randn(128)
    
#     print("\n" + "-" * 40)
#     print("ENROLLMENT PHASE")
#     print("-" * 40)
    
#     # Encode (enrollment)
#     fuzzy_vault, bio_token, C1_enroll, C2_enroll = encoder.encode(
#         hand_features_enroll,
#         face_features_enroll,
#         "user_001"
#     )
    
#     print(f"Vault created for user_001")
#     print(f"  Vault size: {len(fuzzy_vault)}")
    
#     # TEST 1: Genuine user (same features with small noise)
#     print("\n" + "-" * 40)
#     print("TEST 1: GENUINE USER (Same person)")
#     print("-" * 40)
    
#     # Add small noise to simulate real-world variation
#     noise_level = 0.01
#     hand_features_verify = hand_features_enroll + np.random.normal(0, noise_level, 128)
#     face_features_verify = face_features_enroll + np.random.normal(0, noise_level, 128)
    
#     is_genuine, score, message = decoder.verify(
#         fuzzy_vault.tolist(),
#         bio_token,
#         hand_features_verify,
#         face_features_verify
#     )
    
#     print(f"  Result: {message}")
#     print(f"  Is Genuine: {is_genuine}")
#     print(f"  Score: {score:.4f}")
#     print(f"  Threshold: {decoder.threshold}")
    
#     # TEST 2: Imposter (completely different features)
#     print("\n" + "-" * 40)
#     print("TEST 2: IMPOSTER (Different person)")
#     print("-" * 40)
    
#     np.random.seed(999)  # Different seed for different person
#     hand_features_imposter = np.random.randn(128)
#     face_features_imposter = np.random.randn(128)
    
#     is_genuine, score, message = decoder.verify(
#         fuzzy_vault.tolist(),
#         bio_token,
#         hand_features_imposter,
#         face_features_imposter
#     )
    
#     print(f"  Result: {message}")
#     print(f"  Is Genuine: {is_genuine}")
#     print(f"  Score: {score:.4f}")
    
#     # TEST 3: Partial match (correct hand, wrong face)
#     print("\n" + "-" * 40)
#     print("TEST 3: PARTIAL MATCH (Correct hand, wrong face)")
#     print("-" * 40)
    
#     is_genuine, score, message = decoder.verify(
#         fuzzy_vault.tolist(),
#         bio_token,
#         hand_features_enroll + np.random.normal(0, noise_level, 128),  # Correct
#         face_features_imposter  # Wrong
#     )
    
#     print(f"  Result: {message}")
#     print(f"  Is Genuine: {is_genuine}")
#     print(f"  Score: {score:.4f}")
    
#     # TEST 4: Partial match (wrong hand, correct face)
#     print("\n" + "-" * 40)
#     print("TEST 4: PARTIAL MATCH (Wrong hand, correct face)")
#     print("-" * 40)
    
#     is_genuine, score, message = decoder.verify(
#         fuzzy_vault.tolist(),
#         bio_token,
#         hand_features_imposter,  # Wrong
#         face_features_enroll + np.random.normal(0, noise_level, 128)  # Correct
#     )
    
#     print(f"  Result: {message}")
#     print(f"  Is Genuine: {is_genuine}")
#     print(f"  Score: {score:.4f}")
    
#     # Summary
#     print("\n" + "=" * 60)
#     print("SUMMARY")
#     print("=" * 60)
#     print("""
#     The fuzzy vault correctly:
#     ✓ Accepts genuine users (both biometrics match)
#     ✓ Rejects imposters (both biometrics different)
#     ✓ Rejects partial matches (one biometric wrong)
    
#     This demonstrates the multimodal security:
#     - Hand features (C2) used to encrypt/decrypt vault
#     - Face features (C1) used to verify identity
#     - Both must match for successful authentication
#     """)
    
#     print("=" * 60)
#     print("VAULT DECODER TEST COMPLETE")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_vault_decoder()
# ---------------
# v2
# src/fuzzy_vault/vault_decoder.py

"""
Improved Fuzzy Vault Decoder - FIXED VERSION
Implementation of Algorithm 2 from the paper
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.fuzzy_vault.reed_solomon import ReedSolomonEncoder


class ImprovedFuzzyVaultDecoder:
    """
    Improved Fuzzy Vault Decoder
    Based on Algorithm 2 from the paper
    """
    
    def __init__(self, n_features=Config.N_FEATURES, tolerance=Config.FV_TOLERANCE):
        self.n = n_features
        self.tolerance = tolerance
        self.grid_columns = Config.GRID_COLUMNS
        self.threshold = Config.AUTH_THRESHOLD  # 0.35
        self.rs_encoder = ReedSolomonEncoder()
    
    def generate_rs_codes(self, biometric_features):
        """Generate RS codes from biometric features"""
        return self.rs_encoder.generate_codes_from_features(
            biometric_features,
            self.n
        )
    
    def decrypt_vault(self, encrypted_vault, C2_prime, bio_token):
        """
        Decrypt fuzzy vault using C2'
        """
        encrypted_vault = np.array(encrypted_vault)
        
        # Calculate decryption key from C2'
        key_prime = float(np.sum(C2_prime))
        stored_key_hash = bio_token.get('C2_hash', 0)
        
        # Check if keys match (with tolerance)
        key_match = np.abs(key_prime - stored_key_hash) < 1.0
        
        # Parse encrypted vault
        grid_size = bio_token['grid_shape'][0] * bio_token['grid_shape'][1]
        n_positions = bio_token['n_positions']
        
        # Extract components
        encrypted_grid = encrypted_vault[:grid_size]
        encrypted_R = encrypted_vault[grid_size:grid_size + n_positions]
        encrypted_fc = encrypted_vault[grid_size + n_positions:grid_size + 2 * n_positions]
        stored_key = encrypted_vault[-1] if len(encrypted_vault) > grid_size + 2 * n_positions else key_prime
        
        # Decrypt using the key
        if key_match:
            decrypted_grid = encrypted_grid - stored_key
            decrypted_R = encrypted_R - stored_key
            decrypted_fc = encrypted_fc - stored_key
        else:
            # Keys don't match - decryption will produce garbage
            decrypted_grid = encrypted_grid - key_prime
            decrypted_R = encrypted_R - key_prime
            decrypted_fc = encrypted_fc - key_prime
        
        # Reshape grid
        Gr = decrypted_grid.reshape(bio_token['grid_shape'])
        R = np.round(decrypted_R).astype(int)
        R = np.clip(R, 1, 2 * self.n)
        feature_columns = np.round(decrypted_fc).astype(int)
        feature_columns = np.clip(feature_columns, 0, self.grid_columns - 1)
        
        return Gr, R, feature_columns, key_match
    
    def calculate_score(self, Gr, R, feature_columns, C1_prime):
        """
        Calculate matching score between stored and verification features
        """
        matches = 0
        total_distance = 0.0
        valid_comparisons = 0
        
        for i in range(min(len(R), len(C1_prime), len(feature_columns))):
            row_idx = R[i] - 1  # 0-indexed
            
            if row_idx < 0 or row_idx >= len(Gr):
                continue
            
            # Get the stored feature from the correct column
            stored_col = feature_columns[i]
            stored_feature = Gr[row_idx][stored_col]
            
            # Get verification feature
            verify_feature = C1_prime[i]
            
            # Calculate distance
            distance = np.abs(stored_feature - verify_feature)
            total_distance += distance
            valid_comparisons += 1
            
            # Check if within tolerance
            if distance < self.tolerance:
                matches += 1
        
        # Calculate normalized score
        if valid_comparisons > 0:
            match_ratio = matches / valid_comparisons
            avg_distance = total_distance / valid_comparisons
        else:
            match_ratio = 0.0
            avg_distance = float('inf')
        
        return match_ratio, matches, avg_distance, valid_comparisons
    
    def verify(self, encrypted_vault, bio_token, hand_features, face_features):
        """
        Complete recognition algorithm (Algorithm 2)
        """
        # Generate RS codes from verification biometrics
        C2_prime = self.generate_rs_codes(hand_features)
        C1_prime = self.generate_rs_codes(face_features)
        
        # Decrypt vault
        Gr, R, feature_columns, key_match = self.decrypt_vault(
            encrypted_vault, C2_prime, bio_token
        )
        
        # If key doesn't match, hand biometric is wrong
        if not key_match:
            return False, 0.0, "Imposter - Hand biometric mismatch"
        
        # Calculate matching score
        score, matches, avg_dist, valid = self.calculate_score(
            Gr, R, feature_columns, C1_prime
        )
        
        # Determine result
        if score >= self.threshold:
            return True, score, f"Genuine User (score: {score:.4f}, matches: {matches}/{valid})"
        else:
            return False, score, f"Imposter (score: {score:.4f}, matches: {matches}/{valid})"
    
    def batch_verify(self, stored_vaults, hand_features, face_features):
        """Verify against multiple stored vaults"""
        best_match = None
        best_score = -1
        
        for vault_data in stored_vaults:
            is_genuine, score, message = self.verify(
                vault_data['fuzzy_vault'],
                vault_data['bio_token'],
                hand_features,
                face_features
            )
            
            if score > best_score:
                best_score = score
                best_match = {
                    'user_id': vault_data.get('user_id', 'unknown'),
                    'is_genuine': is_genuine,
                    'score': score,
                    'message': message
                }
        
        return best_match


def test_vault_decoder():
    """Test the fuzzy vault decoder"""
    
    print("=" * 60)
    print("TESTING FUZZY VAULT DECODER (Algorithm 2) - FIXED")
    print("=" * 60)
    
    from src.fuzzy_vault.vault_encoder import ImprovedFuzzyVaultEncoder
    
    encoder = ImprovedFuzzyVaultEncoder()
    decoder = ImprovedFuzzyVaultDecoder()
    
    # Generate enrollment features
    np.random.seed(42)
    hand_features_enroll = np.random.randn(128)
    face_features_enroll = np.random.randn(128)
    
    print("\n" + "-" * 40)
    print("ENROLLMENT PHASE")
    print("-" * 40)
    
    fuzzy_vault, bio_token, C1_enroll, C2_enroll = encoder.encode(
        hand_features_enroll,
        face_features_enroll,
        "user_001"
    )
    
    print(f"Vault created for user_001")
    print(f"  Vault size: {len(fuzzy_vault)}")
    print(f"  C2 hash: {bio_token['C2_hash']:.4f}")
    
    # TEST 1: Genuine user (same features)
    print("\n" + "-" * 40)
    print("TEST 1: GENUINE USER (Same person, same features)")
    print("-" * 40)
    
    is_genuine, score, message = decoder.verify(
        fuzzy_vault.tolist(),
        bio_token,
        hand_features_enroll,  # Same hand
        face_features_enroll   # Same face
    )
    
    print(f"  Result: {message}")
    print(f"  Is Genuine: {is_genuine}")
    print(f"  Score: {score:.4f}")
    print(f"  Expected: ACCEPT ✓" if is_genuine else "  Expected: ACCEPT ✗ FAILED")
    
    # TEST 2: Genuine user with noise
    print("\n" + "-" * 40)
    print("TEST 2: GENUINE USER (Same person, small noise)")
    print("-" * 40)
    
    noise_level = 0.01
    hand_features_noisy = hand_features_enroll + np.random.normal(0, noise_level, 128)
    face_features_noisy = face_features_enroll + np.random.normal(0, noise_level, 128)
    
    is_genuine, score, message = decoder.verify(
        fuzzy_vault.tolist(),
        bio_token,
        hand_features_noisy,
        face_features_noisy
    )
    
    print(f"  Result: {message}")
    print(f"  Is Genuine: {is_genuine}")
    print(f"  Score: {score:.4f}")
    
    # TEST 3: Imposter (completely different)
    print("\n" + "-" * 40)
    print("TEST 3: IMPOSTER (Different person)")
    print("-" * 40)
    
    np.random.seed(999)
    hand_features_imposter = np.random.randn(128)
    face_features_imposter = np.random.randn(128)
    
    is_genuine, score, message = decoder.verify(
        fuzzy_vault.tolist(),
        bio_token,
        hand_features_imposter,
        face_features_imposter
    )
    
    print(f"  Result: {message}")
    print(f"  Is Genuine: {is_genuine}")
    print(f"  Score: {score:.4f}")
    print(f"  Expected: REJECT ✓" if not is_genuine else "  Expected: REJECT ✗ FAILED")
    
    # TEST 4: Partial match - correct hand, wrong face
    print("\n" + "-" * 40)
    print("TEST 4: PARTIAL MATCH (Correct hand, wrong face)")
    print("-" * 40)
    
    is_genuine, score, message = decoder.verify(
        fuzzy_vault.tolist(),
        bio_token,
        hand_features_enroll,      # Correct hand
        face_features_imposter     # Wrong face
    )
    
    print(f"  Result: {message}")
    print(f"  Is Genuine: {is_genuine}")
    print(f"  Score: {score:.4f}")
    print(f"  Expected: REJECT ✓" if not is_genuine else "  Expected: REJECT ✗ FAILED")
    
    # TEST 5: Partial match - wrong hand, correct face
    print("\n" + "-" * 40)
    print("TEST 5: PARTIAL MATCH (Wrong hand, correct face)")
    print("-" * 40)
    
    is_genuine, score, message = decoder.verify(
        fuzzy_vault.tolist(),
        bio_token,
        hand_features_imposter,    # Wrong hand
        face_features_enroll       # Correct face
    )
    
    print(f"  Result: {message}")
    print(f"  Is Genuine: {is_genuine}")
    print(f"  Score: {score:.4f}")
    print(f"  Expected: REJECT ✓" if not is_genuine else "  Expected: REJECT ✗ FAILED")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print("""
    Test Results:
    ┌─────────────────────────────────────┬──────────┬──────────┐
    │ Test Case                           │ Expected │ Status   │
    ├─────────────────────────────────────┼──────────┼──────────┤
    │ 1. Genuine (exact match)            │ ACCEPT   │ Check ↑  │
    │ 2. Genuine (with noise)             │ ACCEPT   │ Check ↑  │
    │ 3. Imposter (different person)      │ REJECT   │ Check ↑  │
    │ 4. Partial (correct hand, bad face) │ REJECT   │ Check ↑  │
    │ 5. Partial (bad hand, correct face) │ REJECT   │ Check ↑  │
    └─────────────────────────────────────┴──────────┴──────────┘
    
    The multimodal fuzzy vault requires BOTH:
    - Hand features (C2) - for vault decryption
    - Face features (C1) - for identity verification
    """)
    
    print("=" * 60)
    print("VAULT DECODER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_vault_decoder()