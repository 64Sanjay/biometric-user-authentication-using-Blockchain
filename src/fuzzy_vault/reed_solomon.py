# src/fuzzy_vault/reed_solomon.py

"""
Reed-Solomon Error Correction Codes
Based on Section 3 of the paper

Reed-Solomon codes are used to:
- Bind biometric template with cryptographic key
- Allow error tolerance during matching
- Generate RS codes from biometric features (C1 and C2)
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

try:
    from reedsolo import RSCodec, ReedSolomonError
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False
    print("Warning: reedsolo not available")


class ReedSolomonEncoder:
    """
    Reed-Solomon encoding for fuzzy vault
    
    Used to generate RS codes from biometric features
    as mentioned in Algorithm 1 & 2 of the paper.
    """
    
    def __init__(self, nsym=10):
        """
        Initialize RS encoder
        
        Args:
            nsym: Number of error correction symbols (controls error tolerance)
        """
        self.nsym = nsym
        
        if RS_AVAILABLE:
            self.rs = RSCodec(nsym)
        else:
            self.rs = None
    
    def encode(self, data):
        """
        Encode data with Reed-Solomon codes
        
        Args:
            data: Input data (bytes, string, or numpy array)
        
        Returns:
            Encoded data as numpy array
        """
        if not RS_AVAILABLE:
            return self._fallback_encode(data)
        
        # Convert to bytes
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif isinstance(data, str):
            data = data.encode('utf-8')
        
        # Encode
        encoded = self.rs.encode(data)
        
        return np.frombuffer(encoded, dtype=np.uint8)
    
    def decode(self, encoded_data):
        """
        Decode RS encoded data
        
        Args:
            encoded_data: Encoded data
        
        Returns:
            Decoded data or None if decoding fails
        """
        if not RS_AVAILABLE:
            return self._fallback_decode(encoded_data)
        
        try:
            if isinstance(encoded_data, np.ndarray):
                encoded_data = bytes(encoded_data)
            
            decoded = self.rs.decode(encoded_data)
            return np.frombuffer(decoded[0], dtype=np.uint8)
        
        except Exception as e:
            print(f"RS decoding error: {e}")
            return None
    
    def _fallback_encode(self, data):
        """Fallback encoding when reedsolo not available"""
        if isinstance(data, np.ndarray):
            return data.astype(np.uint8)
        elif isinstance(data, str):
            return np.frombuffer(data.encode('utf-8'), dtype=np.uint8)
        else:
            return np.frombuffer(data, dtype=np.uint8)
    
    def _fallback_decode(self, data):
        """Fallback decoding"""
        return data
    
    def generate_codes_from_features(self, features, n_codes):
        """
        Generate RS codes from biometric features
        
        This is used to generate C1 and C2 in the paper.
        
        Args:
            features: Biometric feature vector
            n_codes: Number of codes to generate
        
        Returns:
            Array of n_codes values in range [0, 1]
        """
        features = np.array(features).flatten()
        
        # Normalize features to [0, 255] for byte representation
        f_min, f_max = features.min(), features.max()
        if f_max - f_min > 1e-10:
            normalized = (features - f_min) / (f_max - f_min)
        else:
            normalized = np.zeros_like(features)
        
        scaled = (normalized * 255).astype(np.uint8)
        
        # Generate codes by hashing chunks of features
        codes = []
        chunk_size = max(1, len(scaled) // n_codes)
        
        for i in range(n_codes):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(scaled))
            
            if start_idx < len(scaled):
                chunk = scaled[start_idx:end_idx]
                
                # Create a hash-like value from chunk
                if len(chunk) > 0:
                    # Use weighted sum with position encoding
                    weights = np.arange(1, len(chunk) + 1)
                    code_value = np.sum(chunk * weights) % 256
                    codes.append(code_value / 255.0)
                else:
                    codes.append(0.0)
            else:
                codes.append(0.0)
        
        # Ensure exactly n_codes
        while len(codes) < n_codes:
            codes.append(0.0)
        
        return np.array(codes[:n_codes])


def test_reed_solomon():
    """Test Reed-Solomon encoding"""
    
    print("=" * 60)
    print("TESTING REED-SOLOMON ENCODER")
    print("=" * 60)
    
    print(f"\nreedsolo available: {RS_AVAILABLE}")
    
    encoder = ReedSolomonEncoder(nsym=10)
    
    # Test 1: Basic encode/decode
    print("\n" + "-" * 40)
    print("Test 1: Basic Encode/Decode")
    
    test_data = "Hello, Fuzzy Vault!"
    encoded = encoder.encode(test_data)
    print(f"  Original: '{test_data}'")
    print(f"  Encoded length: {len(encoded)}")
    
    if RS_AVAILABLE:
        decoded = encoder.decode(encoded)
        if decoded is not None:
            decoded_str = bytes(decoded).decode('utf-8')
            print(f"  Decoded: '{decoded_str}'")
            print(f"  Match: {decoded_str == test_data}")
    
    # Test 2: Generate codes from features
    print("\n" + "-" * 40)
    print("Test 2: Generate Codes from Features")
    
    # Simulate biometric features
    np.random.seed(42)
    fake_features = np.random.randn(128)
    
    codes = encoder.generate_codes_from_features(fake_features, n_codes=128)
    
    print(f"  Input features shape: {fake_features.shape}")
    print(f"  Output codes shape: {codes.shape}")
    print(f"  Codes range: [{codes.min():.4f}, {codes.max():.4f}]")
    print(f"  Codes mean: {codes.mean():.4f}")
    
    # Test 3: Consistency check
    print("\n" + "-" * 40)
    print("Test 3: Consistency Check")
    
    codes1 = encoder.generate_codes_from_features(fake_features, n_codes=128)
    codes2 = encoder.generate_codes_from_features(fake_features, n_codes=128)
    
    print(f"  Same features produce same codes: {np.allclose(codes1, codes2)}")
    
    # Different features
    different_features = np.random.randn(128)
    codes3 = encoder.generate_codes_from_features(different_features, n_codes=128)
    
    distance = np.sqrt(np.sum((codes1 - codes3) ** 2))
    print(f"  Distance between different feature codes: {distance:.4f}")
    
    print("\n" + "=" * 60)
    print("REED-SOLOMON TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_reed_solomon()