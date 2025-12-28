# src/blockchain/blockchain_handler.py

"""
Blockchain Handler for Storing Vault Addresses
Based on Phase-2, Section 4.4 of the paper

The blockchain is used to:
- Store IPFS addresses of fuzzy vaults
- Provide immutable record of enrollments
- Enable decentralized access control

Step 11 (Algorithm 1): Store location of FV from IPFS in Blockchain
"""

import os
import sys
import json
import hashlib
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

# Try to import Web3
try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    print("Warning: web3 not available. Using simulated blockchain.")


class SimulatedBlockchain:
    """
    Simulated blockchain for testing without actual Ethereum node
    
    Mimics the behavior of a real blockchain:
    - Stores transactions in blocks
    - Provides immutability through hashing
    - Maintains a chain of blocks
    """
    
    def __init__(self, storage_dir=None):
        if storage_dir is None:
            storage_dir = os.path.join(Config.DATA_DIR, 'blockchain_local')
        
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.chain = []
        self.pending_transactions = []
        self.vault_registry = {}  # user_id -> ipfs_hash mapping
        
        # Load existing chain
        self._load_chain()
        
        # Create genesis block if chain is empty
        if not self.chain:
            self._create_genesis_block()
    
    def _load_chain(self):
        """Load blockchain from disk"""
        chain_path = os.path.join(self.storage_dir, 'chain.json')
        registry_path = os.path.join(self.storage_dir, 'registry.json')
        
        if os.path.exists(chain_path):
            with open(chain_path, 'r') as f:
                self.chain = json.load(f)
        
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                self.vault_registry = json.load(f)
    
    def _save_chain(self):
        """Save blockchain to disk"""
        chain_path = os.path.join(self.storage_dir, 'chain.json')
        registry_path = os.path.join(self.storage_dir, 'registry.json')
        
        with open(chain_path, 'w') as f:
            json.dump(self.chain, f, indent=2)
        
        with open(registry_path, 'w') as f:
            json.dump(self.vault_registry, f, indent=2)
    
    def _create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_block = {
            'index': 0,
            'timestamp': datetime.now().isoformat(),
            'transactions': [],
            'previous_hash': '0' * 64,
            'hash': self._calculate_hash({
                'index': 0,
                'timestamp': datetime.now().isoformat(),
                'transactions': [],
                'previous_hash': '0' * 64
            })
        }
        self.chain.append(genesis_block)
        self._save_chain()
    
    def _calculate_hash(self, block):
        """Calculate SHA256 hash of a block"""
        block_string = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _get_last_block(self):
        """Get the last block in the chain"""
        return self.chain[-1] if self.chain else None
    
    def add_transaction(self, transaction):
        """Add a transaction to pending transactions"""
        transaction['timestamp'] = datetime.now().isoformat()
        transaction['tx_hash'] = hashlib.sha256(
            json.dumps(transaction, sort_keys=True).encode()
        ).hexdigest()
        
        self.pending_transactions.append(transaction)
        return transaction['tx_hash']
    
    def mine_block(self):
        """Mine a new block with pending transactions"""
        if not self.pending_transactions:
            return None
        
        last_block = self._get_last_block()
        
        new_block = {
            'index': len(self.chain),
            'timestamp': datetime.now().isoformat(),
            'transactions': self.pending_transactions.copy(),
            'previous_hash': last_block['hash']
        }
        new_block['hash'] = self._calculate_hash(new_block)
        
        self.chain.append(new_block)
        self.pending_transactions = []
        self._save_chain()
        
        return new_block
    
    def store_vault_address(self, user_id, ipfs_hash):
        """
        Store vault IPFS address in blockchain
        
        Args:
            user_id: User identifier
            ipfs_hash: IPFS content identifier
        
        Returns:
            Transaction hash
        """
        transaction = {
            'type': 'STORE_VAULT',
            'user_id': user_id,
            'ipfs_hash': ipfs_hash
        }
        
        tx_hash = self.add_transaction(transaction)
        
        # Update registry
        self.vault_registry[user_id] = {
            'ipfs_hash': ipfs_hash,
            'tx_hash': tx_hash,
            'active': True
        }
        
        # Auto-mine for simplicity
        self.mine_block()
        
        return tx_hash
    
    def get_vault_address(self, user_id):
        """
        Get vault IPFS address from blockchain
        
        Args:
            user_id: User identifier
        
        Returns:
            ipfs_hash or None
        """
        if user_id in self.vault_registry:
            entry = self.vault_registry[user_id]
            if entry.get('active', True):
                return entry.get('ipfs_hash')
        return None
    
    def revoke_vault(self, user_id):
        """
        Revoke a vault (mark as inactive)
        
        Args:
            user_id: User identifier
        
        Returns:
            Transaction hash
        """
        if user_id not in self.vault_registry:
            return None
        
        transaction = {
            'type': 'REVOKE_VAULT',
            'user_id': user_id
        }
        
        tx_hash = self.add_transaction(transaction)
        self.vault_registry[user_id]['active'] = False
        self.mine_block()
        
        return tx_hash
    
    def get_chain_info(self):
        """Get blockchain information"""
        return {
            'chain_length': len(self.chain),
            'total_users': len(self.vault_registry),
            'active_vaults': sum(1 for v in self.vault_registry.values() if v.get('active')),
            'pending_transactions': len(self.pending_transactions)
        }


class BlockchainHandler:
    """
    Blockchain Handler for EVM-based networks
    
    Uses simulated blockchain for testing, can be upgraded to real Ethereum.
    """
    
    def __init__(self, use_simulation=True):
        self.use_simulation = use_simulation
        self.web3 = None
        self.blockchain = None
        
        if use_simulation or not WEB3_AVAILABLE:
            print("Using simulated blockchain")
            self.blockchain = SimulatedBlockchain()
        else:
            try:
                self.web3 = Web3(Web3.HTTPProvider(Config.BLOCKCHAIN_URL))
                if self.web3.is_connected():
                    print(f"Connected to Ethereum at {Config.BLOCKCHAIN_URL}")
                else:
                    print("Could not connect to Ethereum, using simulation")
                    self.blockchain = SimulatedBlockchain()
                    self.use_simulation = True
            except Exception as e:
                print(f"Ethereum connection failed: {e}")
                self.blockchain = SimulatedBlockchain()
                self.use_simulation = True
    
    def store_vault_address(self, user_id, ipfs_hash):
        """
        Store vault address in blockchain
        
        Step 11 of Algorithm 1
        """
        if self.use_simulation:
            return self.blockchain.store_vault_address(user_id, ipfs_hash)
        else:
            # Real Ethereum implementation would go here
            pass
    
    def get_vault_address(self, user_id):
        """
        Get vault address from blockchain
        
        Step 1 of Algorithm 2
        """
        if self.use_simulation:
            return self.blockchain.get_vault_address(user_id)
        else:
            # Real Ethereum implementation would go here
            pass
    
    def revoke_vault(self, user_id):
        """
        Revoke vault (for revocability requirement)
        
        As described in Section 6.3
        """
        if self.use_simulation:
            return self.blockchain.revoke_vault(user_id)
        else:
            pass
    
    def get_info(self):
        """Get blockchain information"""
        if self.use_simulation:
            return self.blockchain.get_chain_info()
        else:
            return {'status': 'connected to Ethereum'}


def test_blockchain_handler():
    """Test blockchain handler"""
    
    print("=" * 60)
    print("TESTING BLOCKCHAIN HANDLER")
    print("=" * 60)
    
    handler = BlockchainHandler(use_simulation=True)
    
    print(f"\nUsing simulation: {handler.use_simulation}")
    
    # Test 1: Store vault address
    print("\n" + "-" * 40)
    print("TEST 1: Store Vault Address")
    print("-" * 40)
    
    tx_hash = handler.store_vault_address(
        'user_001',
        'QmTestHash123456789'
    )
    print(f"  Transaction hash: {tx_hash[:20]}...")
    
    # Test 2: Retrieve vault address
    print("\n" + "-" * 40)
    print("TEST 2: Retrieve Vault Address")
    print("-" * 40)
    
    ipfs_hash = handler.get_vault_address('user_001')
    print(f"  Retrieved IPFS hash: {ipfs_hash}")
    
    # Test 3: Store another vault
    print("\n" + "-" * 40)
    print("TEST 3: Store Second Vault")
    print("-" * 40)
    
    tx_hash_2 = handler.store_vault_address(
        'user_002',
        'QmAnotherTestHash987654'
    )
    print(f"  Transaction hash: {tx_hash_2[:20]}...")
    
    # Test 4: Get blockchain info
    print("\n" + "-" * 40)
    print("TEST 4: Blockchain Info")
    print("-" * 40)
    
    info = handler.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test 5: Revoke vault
    print("\n" + "-" * 40)
    print("TEST 5: Revoke Vault")
    print("-" * 40)
    
    revoke_tx = handler.revoke_vault('user_001')
    print(f"  Revoke transaction: {revoke_tx[:20]}...")
    
    # Try to retrieve revoked vault
    revoked_hash = handler.get_vault_address('user_001')
    print(f"  Vault after revocation: {revoked_hash}")
    
    info = handler.get_info()
    print(f"  Active vaults now: {info['active_vaults']}")
    
    print("\n" + "=" * 60)
    print("BLOCKCHAIN HANDLER TEST COMPLETE")
    print("=" * 60)
    
    return handler


if __name__ == "__main__":
    test_blockchain_handler()