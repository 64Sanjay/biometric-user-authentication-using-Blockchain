# src/blockchain/real_ipfs_handler.py

"""
Real IPFS Handler using Direct HTTP API
Compatible with all IPFS versions
"""

import os
import sys
import json
import requests
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config


class RealIPFSHandler:
    """
    Real IPFS Handler using direct HTTP API calls
    Compatible with IPFS v0.24.0 and newer
    """
    
    def __init__(self, api_url='http://127.0.0.1:5001/api/v0'):
        self.api_url = api_url
        self.connected = False
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to IPFS daemon"""
        try:
            response = requests.post(f'{self.api_url}/version', timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.connected = True
                print("✅ Connected to IPFS!")
                print(f"   Version: {data.get('Version', 'unknown')}")
                print(f"   Commit: {data.get('Commit', 'unknown')[:8]}...")
            else:
                print(f"❌ IPFS returned status {response.status_code}")
                self.connected = False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to IPFS daemon")
            print("   Make sure IPFS daemon is running: ipfs daemon &")
            self.connected = False
        except Exception as e:
            print(f"❌ Error connecting to IPFS: {e}")
            self.connected = False
    
    def is_connected(self):
        """Check if connected to IPFS"""
        return self.connected
    
    def get_id(self):
        """Get IPFS node ID"""
        if not self.connected:
            return None
        
        try:
            response = requests.post(f'{self.api_url}/id', timeout=10)
            return response.json()
        except Exception as e:
            print(f"❌ Error getting ID: {e}")
            return None
    
    def store_vault(self, vault_data, user_id=None):
        """
        Store vault data in IPFS
        
        Args:
            vault_data: Dictionary containing vault information
            user_id: Optional user identifier
        
        Returns:
            IPFS CID (Content Identifier)
        """
        if not self.connected:
            raise ConnectionError("Not connected to IPFS")
        
        # Add metadata
        vault_data['stored_at'] = datetime.now().isoformat()
        vault_data['storage_type'] = 'ipfs_real'
        
        if user_id:
            vault_data['user_id'] = user_id
        
        # Convert to JSON
        json_data = json.dumps(vault_data, default=str)
        
        try:
            # Add to IPFS using files API
            files = {'file': ('vault.json', json_data)}
            response = requests.post(
                f'{self.api_url}/add',
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                cid = result.get('Hash')
                
                # Pin the content
                self.pin_vault(cid)
                
                print(f"✅ Stored in IPFS: {cid}")
                return cid
            else:
                print(f"❌ Failed to store: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error storing to IPFS: {e}")
            return None
    
    def retrieve_vault(self, cid):
        """
        Retrieve vault data from IPFS
        
        Args:
            cid: IPFS Content Identifier
        
        Returns:
            Vault data dictionary
        """
        if not self.connected:
            raise ConnectionError("Not connected to IPFS")
        
        try:
            response = requests.post(
                f'{self.api_url}/cat',
                params={'arg': cid},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Retrieved from IPFS: {cid[:20]}...")
                return data
            else:
                print(f"❌ Failed to retrieve: {response.text}")
                return None
                
        except json.JSONDecodeError:
            # Response might be raw text
            return {'raw': response.text}
        except Exception as e:
            print(f"❌ Error retrieving from IPFS: {e}")
            return None
    
    def pin_vault(self, cid):
        """Pin content to prevent garbage collection"""
        if not self.connected:
            return False
        
        try:
            response = requests.post(
                f'{self.api_url}/pin/add',
                params={'arg': cid},
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✅ Pinned: {cid[:20]}...")
                return True
            else:
                print(f"⚠️ Failed to pin: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error pinning: {e}")
            return False
    
    def unpin_vault(self, cid):
        """Unpin content (for revocability)"""
        if not self.connected:
            return False
        
        try:
            response = requests.post(
                f'{self.api_url}/pin/rm',
                params={'arg': cid},
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✅ Unpinned: {cid[:20]}...")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Error unpinning: {e}")
            return False
    
    def list_pins(self):
        """List all pinned content"""
        if not self.connected:
            return []
        
        try:
            response = requests.post(
                f'{self.api_url}/pin/ls',
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return list(data.get('Keys', {}).keys())
            else:
                return []
                
        except Exception as e:
            print(f"❌ Error listing pins: {e}")
            return []
    
    def get_stats(self):
        """Get IPFS node statistics"""
        if not self.connected:
            return None
        
        try:
            # Get ID
            id_response = requests.post(f'{self.api_url}/id', timeout=10)
            id_data = id_response.json() if id_response.status_code == 200 else {}
            
            # Get version
            version_response = requests.post(f'{self.api_url}/version', timeout=10)
            version_data = version_response.json() if version_response.status_code == 200 else {}
            
            # Get repo stats
            repo_response = requests.post(f'{self.api_url}/repo/stat', timeout=10)
            repo_data = repo_response.json() if repo_response.status_code == 200 else {}
            
            # Get swarm peers
            peers_response = requests.post(f'{self.api_url}/swarm/peers', timeout=10)
            peers_data = peers_response.json() if peers_response.status_code == 200 else {}
            
            stats = {
                'id': id_data.get('ID', 'unknown'),
                'version': version_data.get('Version', 'unknown'),
                'repo_size': repo_data.get('RepoSize', 0),
                'num_objects': repo_data.get('NumObjects', 0),
                'peers': len(peers_data.get('Peers', []) or [])
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return None


def test_real_ipfs():
    """Test real IPFS integration"""
    
    print("=" * 60)
    print("TESTING REAL IPFS INTEGRATION")
    print("=" * 60)
    
    # Initialize handler
    handler = RealIPFSHandler()
    
    if not handler.is_connected():
        print("\n⚠️  IPFS daemon not running or not accessible!")
        print("Please ensure IPFS daemon is running:")
        print("  $ ipfs daemon &")
        return None
    
    # Get stats
    print("\n" + "-" * 40)
    print("IPFS Node Statistics")
    print("-" * 40)
    
    stats = handler.get_stats()
    if stats:
        print(f"  Node ID: {stats['id'][:30]}...")
        print(f"  Version: {stats['version']}")
        print(f"  Repo Size: {stats['repo_size'] / 1024 / 1024:.2f} MB")
        print(f"  Objects: {stats['num_objects']}")
        print(f"  Connected Peers: {stats['peers']}")
    
    # Test store and retrieve
    print("\n" + "-" * 40)
    print("Testing Store and Retrieve")
    print("-" * 40)
    
    # Create test vault
    test_vault = {
        'user_id': 'test_user_ipfs_001',
        'fuzzy_vault': [0.1, 0.2, 0.3, 0.4, 0.5],
        'bio_token': {
            'n_features': 128,
            'threshold': 0.9,
            'grid_shape': [256, 3]
        },
        'test': True
    }
    
    # Store
    print("\n1. Storing vault in IPFS...")
    cid = handler.store_vault(test_vault, user_id='test_user_ipfs_001')
    
    if cid:
        print(f"   CID: {cid}")
        
        # Retrieve
        print("\n2. Retrieving vault from IPFS...")
        retrieved = handler.retrieve_vault(cid)
        
        if retrieved:
            print(f"   User ID: {retrieved.get('user_id', 'N/A')}")
            print(f"   Stored At: {retrieved.get('stored_at', 'N/A')}")
            print(f"   Storage Type: {retrieved.get('storage_type', 'N/A')}")
            
            # Verify match
            match = retrieved.get('user_id') == test_vault['user_id']
            print(f"   Data Match: {'✅ Yes' if match else '❌ No'}")
    
    # List pins
    print("\n3. Listing pinned content...")
    pins = handler.list_pins()
    print(f"   Total pins: {len(pins)}")
    if pins:
        for i, pin in enumerate(pins[:5]):  # Show first 5
            print(f"     {i+1}. {pin}")
        if len(pins) > 5:
            print(f"     ... and {len(pins) - 5} more")
    
    print("\n" + "=" * 60)
    print("REAL IPFS TEST COMPLETE ✅")
    print("=" * 60)
    
    return handler


if __name__ == "__main__":
    test_real_ipfs()
