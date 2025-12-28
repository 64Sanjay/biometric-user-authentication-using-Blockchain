# src/blockchain/integrated_handler.py
"""
Integrated IPFS + Blockchain Handler
Combines IPFS storage with Ethereum blockchain verification

Based on: Sharma et al. (2024) - Multimodal Biometric Authentication
Architecture:
    1. Vault data → IPFS (distributed storage)
    2. IPFS CID → Blockchain (immutable reference)
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.blockchain.real_ipfs_handler import RealIPFSHandler
from src.blockchain.ethereum_handler import EthereumHandler


class IntegratedHandler:
    """
    Integrated handler for decentralized biometric vault storage.
    
    This implements the Phase-2 architecture from the paper:
    - IPFS: Distributed storage for encrypted biometric vaults
    - Blockchain: Immutable reference storage and access control
    """
    
    def __init__(self,
                 ipfs_api_url: str = 'http://127.0.0.1:5001/api/v0',
                 ethereum_url: str = 'http://127.0.0.1:8545'):
        """
        Initialize integrated handler.
        
        Args:
            ipfs_api_url: IPFS API endpoint
            ethereum_url: Ethereum node URL (Ganache)
        """
        print("=" * 60)
        print("INITIALIZING INTEGRATED IPFS + BLOCKCHAIN HANDLER")
        print("=" * 60)
        
        # Initialize IPFS
        print("\n📦 Setting up IPFS...")
        self.ipfs = RealIPFSHandler(api_url=ipfs_api_url)
        
        # Initialize Ethereum
        print("\n⛓️  Setting up Ethereum (Ganache)...")
        self.ethereum = EthereumHandler(provider_url=ethereum_url)
        
        # Status
        self.ipfs_ready = self.ipfs.is_connected()
        self.ethereum_ready = self.ethereum.is_connected()
        
        print("\n" + "-" * 40)
        print("STATUS:")
        print(f"  IPFS:       {'✅ Ready' if self.ipfs_ready else '❌ Not Ready'}")
        print(f"  Ethereum:   {'✅ Ready' if self.ethereum_ready else '❌ Not Ready'}")
        print("-" * 40)
    
    def is_ready(self) -> bool:
        """Check if both systems are ready."""
        return self.ipfs_ready and self.ethereum_ready
    
    def enroll_user(self, user_id: str, vault_data: Dict) -> Optional[Dict]:
        """
        Enroll a user with their biometric vault.
        
        Process (from paper Algorithm 1):
        1. Register user on blockchain
        2. Store vault data in IPFS
        3. Store IPFS CID on blockchain
        
        Args:
            user_id: Unique user identifier
            vault_data: Encrypted biometric vault data
            
        Returns:
            Enrollment result dictionary
        """
        print(f"\n🔐 ENROLLING USER: {user_id}")
        print("-" * 40)
        
        result = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Step 1: Register user on blockchain
            print("\n1️⃣  Registering on blockchain...")
            reg_tx = self.ethereum.register_user(user_id)
            result['registration_tx'] = reg_tx
            
            # Step 2: Store vault in IPFS
            print("\n2️⃣  Storing vault in IPFS...")
            ipfs_cid = self.ipfs.store_vault(vault_data, user_id=user_id)
            
            if not ipfs_cid:
                print("❌ Failed to store in IPFS")
                return result
            
            result['ipfs_cid'] = ipfs_cid
            
            # Step 3: Store CID reference on blockchain
            print("\n3️⃣  Recording CID on blockchain...")
            store_result = self.ethereum.store_vault(
                user_id=user_id,
                ipfs_cid=ipfs_cid,
                vault_data=vault_data,
                biometric_type=1  # Face + Hand
            )
            
            if store_result:
                tx_hash, vault_index = store_result
                result['storage_tx'] = tx_hash
                result['vault_index'] = vault_index
                result['success'] = True
                
                print(f"\n✅ ENROLLMENT SUCCESSFUL!")
                print(f"   User ID: {user_id}")
                print(f"   IPFS CID: {ipfs_cid[:30]}...")
                print(f"   Vault Index: {vault_index}")
            else:
                print("❌ Failed to store on blockchain")
            
            return result
            
        except Exception as e:
            print(f"❌ Enrollment error: {e}")
            result['error'] = str(e)
            return result
    
    def authenticate_user(self, 
                          user_id: str, 
                          vault_index: int = 0) -> Optional[Dict]:
        """
        Retrieve user's vault for authentication.
        
        Process (from paper Algorithm 2):
        1. Get CID from blockchain
        2. Verify vault is active
        3. Retrieve vault data from IPFS
        
        Args:
            user_id: User identifier
            vault_index: Vault index
            
        Returns:
            Vault data for authentication
        """
        print(f"\n🔍 AUTHENTICATING USER: {user_id}")
        print("-" * 40)
        
        try:
            # Step 1: Get vault info from blockchain
            print("\n1️⃣  Getting vault info from blockchain...")
            vault_info = self.ethereum.get_vault(user_id, vault_index)
            
            if not vault_info:
                print("❌ Vault not found on blockchain")
                return None
            
            # Step 2: Check if active (revocability check)
            if not vault_info.get('is_active', False):
                print("⚠️  Vault has been revoked - Access denied")
                return None
            
            ipfs_cid = vault_info.get('ipfs_cid')
            print(f"   IPFS CID: {ipfs_cid[:30]}...")
            
            # Step 3: Retrieve from IPFS
            print("\n2️⃣  Retrieving vault data from IPFS...")
            vault_data = self.ipfs.retrieve_vault(ipfs_cid)
            
            if not vault_data:
                print("❌ Failed to retrieve from IPFS")
                return None
            
            result = {
                'user_id': user_id,
                'vault_index': vault_index,
                'blockchain_info': vault_info,
                'vault_data': vault_data,
                'retrieved_at': datetime.now().isoformat()
            }
            
            print(f"\n✅ VAULT RETRIEVED SUCCESSFULLY!")
            
            return result
            
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return None
    
    def revoke_vault(self, user_id: str, vault_index: int = 0) -> bool:
        """
        Revoke a user's vault (ISO/IEC 24745 revocability).
        
        Args:
            user_id: User identifier
            vault_index: Vault index to revoke
            
        Returns:
            Success status
        """
        print(f"\n🚫 REVOKING VAULT: {user_id} (index: {vault_index})")
        print("-" * 40)
        
        try:
            # Get vault info first
            vault_info = self.ethereum.get_vault(user_id, vault_index)
            
            if not vault_info:
                print("❌ Vault not found")
                return False
            
            # Unpin from IPFS (optional - keeps data but removes local pin)
            ipfs_cid = vault_info.get('ipfs_cid')
            if ipfs_cid:
                print("1️⃣  Unpinning from IPFS...")
                self.ipfs.unpin_vault(ipfs_cid)
            
            # Revoke on blockchain
            print("2️⃣  Revoking on blockchain...")
            tx_hash = self.ethereum.revoke_vault(user_id, vault_index)
            
            if tx_hash:
                print(f"\n✅ VAULT REVOKED SUCCESSFULLY!")
                return True
            else:
                print("❌ Revocation failed")
                return False
            
        except Exception as e:
            print(f"❌ Revocation error: {e}")
            return False
    
    def get_system_stats(self) -> Dict:
        """Get combined system statistics."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'system_ready': self.is_ready()
        }
        
        # IPFS stats
        if self.ipfs_ready:
            ipfs_stats = self.ipfs.get_stats()
            stats['ipfs'] = ipfs_stats or {}
            stats['ipfs']['connected'] = True
            stats['ipfs']['pinned_count'] = len(self.ipfs.list_pins())
        else:
            stats['ipfs'] = {'connected': False}
        
        # Ethereum stats
        if self.ethereum_ready:
            eth_stats = self.ethereum.get_stats()
            stats['ethereum'] = eth_stats
        else:
            stats['ethereum'] = {'connected': False}
        
        return stats


def test_integrated_handler():
    """Test the integrated IPFS + Blockchain handler."""
    
    print("\n" + "=" * 60)
    print("TESTING INTEGRATED IPFS + BLOCKCHAIN HANDLER")
    print("=" * 60)
    
    # Initialize
    handler = IntegratedHandler()
    
    if not handler.is_ready():
        print("\n⚠️  System not fully ready!")
        print("   Check: docker ps  (for Ganache)")
        print("   Check: IPFS daemon status")
        return None
    
    # Test 1: User Enrollment
    print("\n" + "=" * 60)
    print("TEST 1: USER ENROLLMENT")
    print("=" * 60)
    
    test_vault = {
        'fuzzy_vault': {
            'grid': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            'random_indices': [5, 12, 28, 45],
            'encrypted': True
        },
        'bio_token': {
            'face_features': [0.1] * 128,
            'hand_features': [0.2] * 64
        },
        'metadata': {
            'biometric_type': 'face_hand',
            'algorithm': 'improved_fuzzy_vault',
            'version': '1.0'
        }
    }
    
    enrollment = handler.enroll_user(
        user_id="integrated_test_user_001",
        vault_data=test_vault
    )
    
    if enrollment and enrollment.get('success'):
        
        # Test 2: Authentication (Vault Retrieval)
        print("\n" + "=" * 60)
        print("TEST 2: USER AUTHENTICATION")
        print("=" * 60)
        
        auth_result = handler.authenticate_user(
            user_id="integrated_test_user_001",
            vault_index=0
        )
        
        if auth_result:
            print("\n📋 Authentication Result:")
            print(f"   User ID: {auth_result['user_id']}")
            print(f"   Vault Active: {auth_result['blockchain_info']['is_active']}")
        
        # Test 3: Vault Revocation
        print("\n" + "=" * 60)
        print("TEST 3: VAULT REVOCATION")
        print("=" * 60)
        
        handler.revoke_vault(
            user_id="integrated_test_user_001",
            vault_index=0
        )
        
        # Test 4: Access Revoked Vault
        print("\n" + "=" * 60)
        print("TEST 4: ACCESS REVOKED VAULT")
        print("=" * 60)
        
        auth_result = handler.authenticate_user(
            user_id="integrated_test_user_001",
            vault_index=0
        )
        
        if not auth_result:
            print("✅ Correctly blocked access to revoked vault")
    
    # System Statistics
    print("\n" + "=" * 60)
    print("SYSTEM STATISTICS")
    print("=" * 60)
    
    stats = handler.get_system_stats()
    
    print(f"\n📊 System Ready: {stats['system_ready']}")
    
    print(f"\n📦 IPFS:")
    print(f"   Connected: {stats['ipfs'].get('connected')}")
    if 'pinned_count' in stats['ipfs']:
        print(f"   Pinned Items: {stats['ipfs']['pinned_count']}")
    
    print(f"\n⛓️  Ethereum:")
    print(f"   Connected: {stats['ethereum'].get('connected')}")
    print(f"   Total Users: {stats['ethereum'].get('total_users', 'N/A')}")
    print(f"   Total Vaults: {stats['ethereum'].get('total_vaults', 'N/A')}")
    if 'block_number' in stats['ethereum']:
        print(f"   Block Number: {stats['ethereum']['block_number']}")
    
    print("\n" + "=" * 60)
    print("INTEGRATED TEST COMPLETE ✅")
    print("=" * 60)
    
    return handler


if __name__ == "__main__":
    test_integrated_handler()
