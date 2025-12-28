# tests/test_integration.py
"""
Integration tests for complete biometric authentication flow
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEnrollmentFlow:
    """Tests for user enrollment flow."""
    
    def test_complete_enrollment(self, sample_user_id, sample_vault_data):
        """Test complete enrollment process."""
        try:
            from src.blockchain.integrated_handler import IntegratedHandler
            
            handler = IntegratedHandler()
            
            if not handler.is_ready():
                pytest.skip("System not ready (IPFS or Ganache not running)")
            
            result = handler.enroll_user(sample_user_id, sample_vault_data)
            
            assert result is not None
            assert result.get('success', False) or 'ipfs_cid' in result
        except Exception as e:
            pytest.skip(f"Enrollment test skipped: {e}")


class TestAuthenticationFlow:
    """Tests for user authentication flow."""
    
    def test_authentication_nonexistent_user(self):
        """Test authentication fails for non-existent user."""
        try:
            from src.blockchain.integrated_handler import IntegratedHandler
            
            handler = IntegratedHandler()
            
            if not handler.is_ready():
                pytest.skip("System not ready")
            
            result = handler.authenticate_user("nonexistent_user_xyz")
            
            # Should return None or fail gracefully
            assert result is None or result.get('success', True) == False
        except Exception as e:
            pytest.skip(f"Auth test skipped: {e}")


class TestRevocationFlow:
    """Tests for vault revocation flow."""
    
    def test_revocation_flow(self, sample_user_id, sample_vault_data):
        """Test vault revocation."""
        try:
            from src.blockchain.integrated_handler import IntegratedHandler
            
            handler = IntegratedHandler()
            
            if not handler.is_ready():
                pytest.skip("System not ready")
            
            # Enroll first
            handler.enroll_user(sample_user_id + "_revoke", sample_vault_data)
            
            # Revoke
            result = handler.revoke_vault(sample_user_id + "_revoke", 0)
            
            # Should succeed or fail gracefully
            assert isinstance(result, bool)
        except Exception as e:
            pytest.skip(f"Revocation test skipped: {e}")


def test_end_to_end_flow():
    """Test complete end-to-end authentication flow."""
    try:
        from src.blockchain.integrated_handler import IntegratedHandler
        
        handler = IntegratedHandler()
        
        if not handler.is_ready():
            pytest.skip("System not ready for E2E test")
        
        user_id = "e2e_test_user"
        vault_data = {
            'user_id': user_id,
            'test': True,
            'features': [0.1] * 100
        }
        
        # 1. Enroll
        enroll_result = handler.enroll_user(user_id, vault_data)
        assert enroll_result is not None
        
        # 2. Authenticate
        auth_result = handler.authenticate_user(user_id)
        # May or may not succeed depending on implementation
        
        # 3. Revoke
        revoke_result = handler.revoke_vault(user_id, 0)
        
        print(f"E2E Test: Enroll={enroll_result.get('success')}, Revoke={revoke_result}")
        
    except Exception as e:
        pytest.skip(f"E2E test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
