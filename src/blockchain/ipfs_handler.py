# src/blockchain/ipfs_handler.py

"""
IPFS Handler for Distributed Storage
Based on Phase-2, Section 4.4 of the paper

IPFS (InterPlanetary File System) is used to:
- Store encrypted fuzzy vaults in a distributed manner
- Eliminate single point of failure
- Provide content-addressed storage

Step 10 (Algorithm 1): Store FV in IPFS
Step 1 (Algorithm 2): Retrieve FV from IPFS
"""

import os
import sys
import json
import hashlib
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

# Try to import IPFS client
try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False
    print("Warning: ipfshttpclient not available. Using local storage fallback.")


class IPFSHandler:
    """
    IPFS Handler for storing fuzzy vaults
    
    If IPFS daemon is not running, uses local file storage as fallback.
    """
    
    def __init__(self):
        self.client = None
        self.use_local = True
        self.local_storage_dir = os.path.join(Config.DATA_DIR, 'ipfs_local')
        
        # Create local storage directory
        os.makedirs(self.local_storage_dir, exist_ok=True)
        
        # Try to connect to IPFS
        if IPFS_AVAILABLE:
            try:
                self.client = ipfshttpclient.connect(Config.IPFS_URL, timeout=5)
                self.use_local = False
                print(f"Connected to IPFS at {Config.IPFS_URL}")
            except Exception as e:
                print(f"Could not connect to IPFS: {e}")
                print("Using local storage fallback.")
        
        # In-memory index for local storage
        self.local_index = {}
        self._load_local_index()
    
    def _load_local_index(self):
        """Load local storage index"""
        index_path = os.path.join(self.local_storage_dir, 'index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self.local_index = json.load(f)
    
    def _save_local_index(self):
        """Save local storage index"""
        index_path = os.path.join(self.local_storage_dir, 'index.json')
        with open(index_path, 'w') as f:
            json.dump(self.local_index, f, indent=2)
    
    def _generate_hash(self, data):
        """Generate IPFS-like hash for local storage"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        hash_bytes = hashlib.sha256(data_str.encode()).hexdigest()
        # Create IPFS-like CID (Content Identifier)
        return f"Qm{hash_bytes[:44]}"
    
    def store_vault(self, vault_package):
        """
        Store fuzzy vault in IPFS (or local storage)
        
        Args:
            vault_package: Dictionary containing vault data
                - user_id: User identifier
                - fuzzy_vault: Encrypted vault data
                - bio_token: Vault metadata
        
        Returns:
            ipfs_hash: Content identifier (CID) for the stored vault
        """
        # Add timestamp
        vault_package['stored_at'] = datetime.now().isoformat()
        
        if not self.use_local and self.client:
            try:
                # Store in IPFS
                result = self.client.add_json(vault_package)
                return result
            except Exception as e:
                print(f"IPFS storage failed: {e}, using local fallback")
        
        # Local storage fallback
        ipfs_hash = self._generate_hash(vault_package)
        
        # Save to file
        file_path = os.path.join(self.local_storage_dir, f"{ipfs_hash}.json")
        with open(file_path, 'w') as f:
            json.dump(vault_package, f, indent=2, default=str)
        
        # Update index
        self.local_index[ipfs_hash] = {
            'user_id': vault_package.get('user_id'),
            'stored_at': vault_package['stored_at'],
            'file_path': file_path
        }
        self._save_local_index()
        
        return ipfs_hash
    
    def retrieve_vault(self, ipfs_hash):
        """
        Retrieve fuzzy vault from IPFS (or local storage)
        
        Args:
            ipfs_hash: Content identifier of the vault
        
        Returns:
            vault_package: The stored vault data, or None if not found
        """
        if not self.use_local and self.client:
            try:
                # Retrieve from IPFS
                vault_data = self.client.get_json(ipfs_hash)
                return vault_data
            except Exception as e:
                print(f"IPFS retrieval failed: {e}, trying local storage")
        
        # Local storage fallback
        file_path = os.path.join(self.local_storage_dir, f"{ipfs_hash}.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def list_vaults(self):
        """
        List all stored vaults
        
        Returns:
            List of vault metadata
        """
        vaults = []
        
        for ipfs_hash, info in self.local_index.items():
            vaults.append({
                'ipfs_hash': ipfs_hash,
                'user_id': info.get('user_id'),
                'stored_at': info.get('stored_at')
            })
        
        return vaults
    
    def delete_vault(self, ipfs_hash):
        """
        Delete vault from storage (for revocability)
        
        Note: In real IPFS, content cannot be truly deleted,
        but we can unpin it. In local storage, we delete the file.
        
        Args:
            ipfs_hash: Content identifier of the vault
        
        Returns:
            success: Boolean indicating success
        """
        if not self.use_local and self.client:
            try:
                self.client.pin.rm(ipfs_hash)
                print(f"Unpinned from IPFS: {ipfs_hash}")
            except Exception as e:
                print(f"IPFS unpin failed: {e}")
        
        # Remove from local storage
        file_path = os.path.join(self.local_storage_dir, f"{ipfs_hash}.json")
        
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if ipfs_hash in self.local_index:
            del self.local_index[ipfs_hash]
            self._save_local_index()
        
        return True
    
    def get_vault_by_user(self, user_id):
        """
        Get vault by user ID
        
        Args:
            user_id: User identifier
        
        Returns:
            vault_package or None
        """
        for ipfs_hash, info in self.local_index.items():
            if info.get('user_id') == user_id:
                return self.retrieve_vault(ipfs_hash), ipfs_hash
        
        return None, None


def test_ipfs_handler():
    """Test IPFS handler"""
    
    print("=" * 60)
    print("TESTING IPFS HANDLER")
    print("=" * 60)
    
    handler = IPFSHandler()
    
    print(f"\nUsing local storage: {handler.use_local}")
    print(f"Storage directory: {handler.local_storage_dir}")
    
    # Create test vault package
    test_vault = {
        'user_id': 'test_user_001',
        'fuzzy_vault': [0.1, 0.2, 0.3, 0.4, 0.5],  # Simplified for testing
        'bio_token': {
            'grid_shape': (256, 3),
            'n_features': 128,
            'tolerance': 0.1
        }
    }
    
    # Test 1: Store vault
    print("\n" + "-" * 40)
    print("TEST 1: Store Vault")
    print("-" * 40)
    
    ipfs_hash = handler.store_vault(test_vault)
    print(f"  Stored vault with hash: {ipfs_hash}")
    
    # Test 2: Retrieve vault
    print("\n" + "-" * 40)
    print("TEST 2: Retrieve Vault")
    print("-" * 40)
    
    retrieved = handler.retrieve_vault(ipfs_hash)
    print(f"  Retrieved user_id: {retrieved.get('user_id')}")
    print(f"  Retrieved vault length: {len(retrieved.get('fuzzy_vault', []))}")
    print(f"  Match: {retrieved.get('user_id') == test_vault['user_id']}")
    
    # Test 3: List vaults
    print("\n" + "-" * 40)
    print("TEST 3: List Vaults")
    print("-" * 40)
    
    vaults = handler.list_vaults()
    print(f"  Total vaults stored: {len(vaults)}")
    for v in vaults:
        print(f"    - {v['user_id']}: {v['ipfs_hash'][:20]}...")
    
    # Test 4: Get by user ID
    print("\n" + "-" * 40)
    print("TEST 4: Get Vault by User ID")
    print("-" * 40)
    
    vault, hash_found = handler.get_vault_by_user('test_user_001')
    if vault:
        print(f"  Found vault for test_user_001")
        print(f"  Hash: {hash_found}")
    else:
        print("  Vault not found")
    
    # Test 5: Store another vault
    print("\n" + "-" * 40)
    print("TEST 5: Store Second Vault")
    print("-" * 40)
    
    test_vault_2 = {
        'user_id': 'test_user_002',
        'fuzzy_vault': [0.5, 0.6, 0.7, 0.8, 0.9],
        'bio_token': {
            'grid_shape': (256, 3),
            'n_features': 128
        }
    }
    
    ipfs_hash_2 = handler.store_vault(test_vault_2)
    print(f"  Stored second vault: {ipfs_hash_2}")
    
    vaults = handler.list_vaults()
    print(f"  Total vaults now: {len(vaults)}")
    
    print("\n" + "=" * 60)
    print("IPFS HANDLER TEST COMPLETE")
    print("=" * 60)
    
    return handler


if __name__ == "__main__":
    test_ipfs_handler()