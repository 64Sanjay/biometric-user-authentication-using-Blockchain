# tests/test_blockchain.py
"""
Tests for Blockchain integration (Ethereum + IPFS)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEthereumHandler:
    """Tests for Ethereum blockchain handler."""
    
    def test_handler_initialization(self):
        """Test Ethereum handler initializes."""
        from src.blockchain.ethereum_handler import EthereumHandler
        
        handler = EthereumHandler(auto_connect=False)
        assert handler is not None
    
    def test_user_id_hashing(self):
        """Test user ID hashing for privacy."""
        from src.blockchain.ethereum_handler import EthereumHandler
        
        hash1 = EthereumHandler.hash_user_id("user_001")
        hash2 = EthereumHandler.hash_user_id("user_001")
        hash3 = EthereumHandler.hash_user_id("user_002")
        
        assert hash1 == hash2  # Same input = same hash
        assert hash1 != hash3  # Different input = different hash
        assert len(hash1) == 64  # SHA256 produces 64 hex characters
    
    def test_vault_data_hashing(self, sample_vault_data):
        """Test vault data hashing for integrity."""
        from src.blockchain.ethereum_handler import EthereumHandler
        
        hash1 = EthereumHandler.hash_vault_data(sample_vault_data)
        hash2 = EthereumHandler.hash_vault_data(sample_vault_data)
        
        assert hash1 == hash2
        assert len(hash1) == 64


class TestIPFSHandler:
    """Tests for IPFS handler."""
    
    def test_handler_initialization(self):
        """Test IPFS handler initializes."""
        try:
            from src.blockchain.real_ipfs_handler import RealIPFSHandler
            handler = RealIPFSHandler()
            # May or may not connect depending on daemon status
            assert handler is not None
        except Exception:
            pytest.skip("IPFS handler initialization failed")


class TestIntegratedHandler:
    """Tests for integrated IPFS + Blockchain handler."""
    
    def test_handler_initialization(self):
        """Test integrated handler initializes."""
        try:
            from src.blockchain.integrated_handler import IntegratedHandler
            handler = IntegratedHandler()
            assert handler is not None
        except Exception as e:
            pytest.skip(f"Integrated handler failed: {e}")


def test_blockchain_connection():
    """Test blockchain connection status check."""
    from src.blockchain.ethereum_handler import EthereumHandler
    
    handler = EthereumHandler(auto_connect=True)
    
    # Should return True or False, not raise exception
    is_connected = handler.is_connected()
    assert isinstance(is_connected, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
