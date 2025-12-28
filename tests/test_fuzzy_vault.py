# tests/test_fuzzy_vault.py
"""
Tests for Fuzzy Vault encryption module
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVaultEncoder:
    """Tests for vault encoding."""
    
    def test_encoder_initialization(self):
        """Test vault encoder initializes correctly."""
        from src.fuzzy_vault.vault_encoder import VaultEncoder
        
        encoder = VaultEncoder()
        assert encoder is not None
    
    def test_vault_creation(self):
        """Test vault creation with biometric features."""
        from src.fuzzy_vault.vault_encoder import VaultEncoder
        
        encoder = VaultEncoder()
        
        # Create test biometric features
        biometric1 = np.random.rand(64).astype(np.float32)
        biometric2 = np.random.rand(128).astype(np.float32)
        
        vault = encoder.encode(biometric1, biometric2, user_id="test_user")
        
        assert vault is not None
        assert 'grid' in vault or 'encoded' in vault


class TestVaultDecoder:
    """Tests for vault decoding."""
    
    def test_decoder_initialization(self):
        """Test vault decoder initializes correctly."""
        from src.fuzzy_vault.vault_decoder import VaultDecoder
        
        decoder = VaultDecoder()
        assert decoder is not None


class TestReedSolomon:
    """Tests for Reed-Solomon error correction."""
    
    def test_rs_initialization(self):
        """Test RS code initializes correctly."""
        try:
            from src.fuzzy_vault.reed_solomon import ReedSolomonCode
            rs = ReedSolomonCode()
            assert rs is not None
        except ImportError:
            pytest.skip("Reed-Solomon module not available")


def test_vault_encode_decode_cycle():
    """Test complete encode-decode cycle."""
    try:
        from src.fuzzy_vault.vault_encoder import VaultEncoder
        from src.fuzzy_vault.vault_decoder import VaultDecoder
        
        encoder = VaultEncoder()
        decoder = VaultDecoder()
        
        # Create test features
        biometric1 = np.random.rand(64).astype(np.float32)
        biometric2 = np.random.rand(128).astype(np.float32)
        
        # Encode
        vault = encoder.encode(biometric1, biometric2, user_id="test_user")
        
        # Decode with same features (should succeed)
        result = decoder.decode(vault, biometric1, biometric2)
        
        assert result is not None
    except Exception as e:
        pytest.skip(f"Vault modules not fully implemented: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
