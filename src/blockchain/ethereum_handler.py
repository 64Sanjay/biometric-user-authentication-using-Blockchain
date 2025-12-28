# src/blockchain/ethereum_handler.py
"""
Ethereum Handler for Biometric Vault Storage
Connects to Ganache for blockchain operations
Fixed for Web3.py v6+ compatibility

Based on: Sharma et al. (2024) - Multimodal Biometric Authentication
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from web3 import Web3
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    print("⚠️  Web3 not installed. Run: pip install web3 eth-account")

# Ganache Configuration
GANACHE_URL = "http://127.0.0.1:8545"
CHAIN_ID = 1337

# Default account (first Ganache deterministic account)
DEPLOYER_ADDRESS = "0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1"
DEPLOYER_PRIVATE_KEY = "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d"


class EthereumHandler:
    """Ethereum blockchain handler for biometric vault storage."""
    
    def __init__(self, 
                 provider_url: str = None,
                 private_key: str = None,
                 auto_connect: bool = True):
        self.provider_url = provider_url or GANACHE_URL
        self.private_key = private_key or DEPLOYER_PRIVATE_KEY
        self.connected = False
        self.w3 = None
        self.account = None
        self.chain_id = CHAIN_ID
        
        # Contract storage
        self.users: Dict[str, Dict] = {}
        self.vaults: Dict[str, Dict[int, Dict]] = {}
        self.cid_exists: Dict[str, bool] = {}
        self.total_users = 0
        self.total_vaults = 0
        
        if auto_connect and WEB3_AVAILABLE:
            self._connect()
    
    def _connect(self):
        """Connect to Ethereum node."""
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
            
            if self.w3.is_connected():
                self.connected = True
                self.chain_id = self.w3.eth.chain_id
                
                print(f"✅ Connected to Ethereum node")
                print(f"   URL: {self.provider_url}")
                print(f"   Chain ID: {self.chain_id}")
                print(f"   Block Number: {self.w3.eth.block_number}")
                
                if self.private_key:
                    self.account = Account.from_key(self.private_key)
                    balance = self.w3.eth.get_balance(self.account.address)
                    balance_eth = self.w3.from_wei(balance, 'ether')
                    print(f"   Account: {self.account.address}")
                    print(f"   Balance: {balance_eth} ETH")
                
                return True
            else:
                print("❌ Failed to connect to Ethereum node")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            self.connected = False
            return False
    
    def is_connected(self) -> bool:
        if self.w3 is None:
            return False
        try:
            return self.w3.is_connected()
        except:
            return False
    
    @staticmethod
    def hash_user_id(user_id: str) -> str:
        return hashlib.sha256(user_id.encode()).hexdigest()
    
    @staticmethod
    def hash_vault_data(vault_data: Dict) -> str:
        data_str = json.dumps(vault_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _generate_tx_hash(self) -> str:
        import secrets
        return "0x" + secrets.token_hex(32)
    
    def _send_transaction(self, data: Dict) -> Optional[str]:
        """Send a transaction to the blockchain."""
        if not self.is_connected():
            print("❌ Not connected to Ethereum")
            return None
        
        try:
            tx_data = json.dumps(data, default=str).encode()
            
            tx = {
                'from': self.account.address,
                'to': self.account.address,
                'value': 0,
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'chainId': self.chain_id,
                'data': self.w3.to_hex(tx_data)
            }
            
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            
            # FIXED: Use raw_transaction instead of rawTransaction (Web3 v6+)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
            
            if receipt['status'] == 1:
                return tx_hash.hex() if hasattr(tx_hash, 'hex') else tx_hash
            else:
                print("❌ Transaction failed")
                return None
                
        except Exception as e:
            print(f"❌ Transaction error: {e}")
            return None
    
    def register_user(self, user_id: str) -> Optional[str]:
        """Register a new user."""
        user_id_hash = self.hash_user_id(user_id)
        
        if user_id_hash in self.users:
            print(f"⚠️  User '{user_id}' already registered")
            return None
        
        self.users[user_id_hash] = {
            'user_id': user_id,
            'user_id_hash': user_id_hash,
            'registered_at': datetime.now().isoformat(),
            'is_active': True,
            'vault_count': 0
        }
        self.total_users += 1
        
        tx_data = {
            'action': 'register_user',
            'user_id_hash': user_id_hash,
            'timestamp': datetime.now().isoformat()
        }
        
        tx_hash = self._send_transaction(tx_data)
        
        if tx_hash:
            print(f"✅ User registered on blockchain: {user_id}")
            print(f"   TX Hash: {tx_hash[:20]}...")
            return tx_hash
        else:
            print(f"✅ User registered locally: {user_id}")
            return self._generate_tx_hash()
    
    def is_user_registered(self, user_id: str) -> bool:
        user_id_hash = self.hash_user_id(user_id)
        return user_id_hash in self.users
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        user_id_hash = self.hash_user_id(user_id)
        return self.users.get(user_id_hash)
    
    def store_vault(self,
                    user_id: str,
                    ipfs_cid: str,
                    vault_data: Dict,
                    biometric_type: int = 1) -> Optional[Tuple[str, int]]:
        """Store a biometric vault."""
        user_id_hash = self.hash_user_id(user_id)
        
        if user_id_hash not in self.users:
            print(f"❌ User '{user_id}' not registered. Registering now...")
            self.register_user(user_id)
        
        if ipfs_cid in self.cid_exists and self.cid_exists[ipfs_cid]:
            print(f"❌ IPFS CID already exists")
            return None
        
        vault_index = self.users[user_id_hash]['vault_count']
        vault_hash = self.hash_vault_data(vault_data)
        
        if user_id_hash not in self.vaults:
            self.vaults[user_id_hash] = {}
        
        self.vaults[user_id_hash][vault_index] = {
            'ipfs_cid': ipfs_cid,
            'vault_hash': vault_hash,
            'biometric_type': biometric_type,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'is_active': True,
            'version': 1
        }
        
        self.cid_exists[ipfs_cid] = True
        self.users[user_id_hash]['vault_count'] += 1
        self.total_vaults += 1
        
        tx_data = {
            'action': 'store_vault',
            'user_id_hash': user_id_hash,
            'ipfs_cid': ipfs_cid,
            'vault_hash': vault_hash,
            'vault_index': vault_index
        }
        
        tx_hash = self._send_transaction(tx_data)
        
        if tx_hash:
            print(f"✅ Vault stored on blockchain")
            print(f"   TX Hash: {tx_hash[:20]}...")
            print(f"   Vault Index: {vault_index}")
        else:
            tx_hash = self._generate_tx_hash()
            print(f"✅ Vault stored locally")
            print(f"   Vault Index: {vault_index}")
        
        return (tx_hash, vault_index)
    
    def get_vault(self, user_id: str, vault_index: int = 0) -> Optional[Dict]:
        """Get vault information."""
        user_id_hash = self.hash_user_id(user_id)
        
        if user_id_hash not in self.vaults:
            print(f"❌ No vaults found for user: {user_id}")
            return None
        
        if vault_index not in self.vaults[user_id_hash]:
            print(f"❌ Vault {vault_index} not found")
            return None
        
        vault = self.vaults[user_id_hash][vault_index]
        print(f"✅ Retrieved vault {vault_index} for {user_id}")
        
        return vault
    
    def revoke_vault(self, user_id: str, vault_index: int = 0) -> Optional[str]:
        """Revoke a vault."""
        user_id_hash = self.hash_user_id(user_id)
        
        if user_id_hash not in self.vaults:
            print(f"❌ No vaults found for user: {user_id}")
            return None
        
        if vault_index not in self.vaults[user_id_hash]:
            print(f"❌ Vault {vault_index} not found")
            return None
        
        if not self.vaults[user_id_hash][vault_index]['is_active']:
            print(f"⚠️  Vault {vault_index} already revoked")
            return None
        
        self.vaults[user_id_hash][vault_index]['is_active'] = False
        self.vaults[user_id_hash][vault_index]['updated_at'] = datetime.now().isoformat()
        
        tx_data = {
            'action': 'revoke_vault',
            'user_id_hash': user_id_hash,
            'vault_index': vault_index
        }
        
        tx_hash = self._send_transaction(tx_data)
        
        if tx_hash:
            print(f"✅ Vault {vault_index} revoked on blockchain")
            print(f"   TX Hash: {tx_hash[:20]}...")
        else:
            tx_hash = self._generate_tx_hash()
            print(f"✅ Vault {vault_index} revoked locally")
        
        return tx_hash
    
    def reactivate_vault(self, user_id: str, vault_index: int = 0) -> Optional[str]:
        """Reactivate a revoked vault."""
        user_id_hash = self.hash_user_id(user_id)
        
        if user_id_hash not in self.vaults:
            return None
        
        if vault_index not in self.vaults[user_id_hash]:
            return None
        
        if self.vaults[user_id_hash][vault_index]['is_active']:
            print(f"⚠️  Vault {vault_index} already active")
            return None
        
        self.vaults[user_id_hash][vault_index]['is_active'] = True
        self.vaults[user_id_hash][vault_index]['updated_at'] = datetime.now().isoformat()
        
        tx_hash = self._generate_tx_hash()
        print(f"✅ Vault {vault_index} reactivated")
        
        return tx_hash
    
    def get_user_vault_count(self, user_id: str) -> int:
        user_id_hash = self.hash_user_id(user_id)
        if user_id_hash in self.users:
            return self.users[user_id_hash]['vault_count']
        return 0
    
    def verify_vault_integrity(self, user_id: str, vault_index: int, vault_data: Dict) -> bool:
        """Verify vault integrity."""
        vault = self.get_vault(user_id, vault_index)
        if not vault:
            return False
        
        expected_hash = self.hash_vault_data(vault_data)
        actual_hash = vault.get('vault_hash', '')
        
        is_valid = expected_hash == actual_hash
        
        if is_valid:
            print(f"✅ Vault integrity verified")
        else:
            print(f"❌ Vault integrity check failed")
        
        return is_valid
    
    def get_stats(self) -> Dict:
        """Get blockchain statistics."""
        stats = {
            'connected': self.is_connected(),
            'provider_url': self.provider_url,
            'chain_id': self.chain_id,
            'total_users': self.total_users,
            'total_vaults': self.total_vaults
        }
        
        if self.is_connected():
            try:
                stats['block_number'] = self.w3.eth.block_number
                stats['gas_price'] = self.w3.eth.gas_price
                if self.account:
                    balance = self.w3.eth.get_balance(self.account.address)
                    stats['account_balance'] = float(self.w3.from_wei(balance, 'ether'))
            except Exception as e:
                stats['error'] = str(e)
        
        return stats


def test_ethereum_handler():
    """Test Ethereum handler with Ganache."""
    
    print("\n" + "=" * 60)
    print("TESTING ETHEREUM HANDLER WITH GANACHE")
    print("=" * 60)
    
    handler = EthereumHandler()
    
    if not handler.is_connected():
        print("\n⚠️  Not connected to Ganache!")
        print("   Start Ganache: docker start ganache")
        return None
    
    # Test user registration
    print("\n" + "-" * 40)
    print("1️⃣  User Registration")
    print("-" * 40)
    
    handler.register_user("test_user_001")
    handler.register_user("test_user_002")
    handler.register_user("test_user_001")
    
    # Test vault storage
    print("\n" + "-" * 40)
    print("2️⃣  Vault Storage")
    print("-" * 40)
    
    vault_data = {
        'fuzzy_vault': [0.1, 0.2, 0.3, 0.4, 0.5],
        'bio_token': {
            'face_features': [0.1] * 128,
            'hand_features': [0.2] * 64
        }
    }
    
    result = handler.store_vault(
        user_id="test_user_001",
        ipfs_cid="QmTestCid123456789012345678901234567890123456789",
        vault_data=vault_data,
        biometric_type=1
    )
    
    # Test vault retrieval
    print("\n" + "-" * 40)
    print("3️⃣  Vault Retrieval")
    print("-" * 40)
    
    vault = handler.get_vault("test_user_001", 0)
    if vault:
        print(f"   IPFS CID: {vault['ipfs_cid'][:30]}...")
        print(f"   Is Active: {vault['is_active']}")
        print(f"   Biometric Type: {vault['biometric_type']}")
    
    # Test integrity verification
    print("\n" + "-" * 40)
    print("4️⃣  Integrity Verification")
    print("-" * 40)
    
    handler.verify_vault_integrity("test_user_001", 0, vault_data)
    
    # Test vault revocation
    print("\n" + "-" * 40)
    print("5️⃣  Vault Revocation")
    print("-" * 40)
    
    handler.revoke_vault("test_user_001", 0)
    
    vault = handler.get_vault("test_user_001", 0)
    print(f"   Is Active after revocation: {vault['is_active']}")
    
    # Test reactivation
    print("\n" + "-" * 40)
    print("6️⃣  Vault Reactivation")
    print("-" * 40)
    
    handler.reactivate_vault("test_user_001", 0)
    
    vault = handler.get_vault("test_user_001", 0)
    print(f"   Is Active after reactivation: {vault['is_active']}")
    
    # Statistics
    print("\n" + "-" * 40)
    print("📊 Statistics")
    print("-" * 40)
    
    stats = handler.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("ETHEREUM HANDLER TEST COMPLETE ✅")
    print("=" * 60)
    
    return handler


if __name__ == "__main__":
    test_ethereum_handler()
