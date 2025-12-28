# config/ganache_config.py
"""
Ganache Configuration for Biometric Blockchain Authentication
These are deterministic accounts - same every time Ganache starts with --deterministic flag
"""

# Ganache RPC Configuration
GANACHE_URL = "http://127.0.0.1:8545"
CHAIN_ID = 1337

# Deterministic Accounts (from Ganache with --deterministic flag)
ACCOUNTS = [
    {
        "address": "0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1",
        "private_key": "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d",
        "balance": "1000 ETH"
    },
    {
        "address": "0xFFcf8FDEE72ac11b5c542428B35EEF5769C409f0",
        "private_key": "0x6cbed15c793ce57650b9877cf6fa156fbef513c4e6134f022a85b1ffdd59b2a1",
        "balance": "1000 ETH"
    },
    {
        "address": "0x22d491Bde2303f2f43325b2108D26f1eAbA1e32b",
        "private_key": "0x6370fd033278c143179d81c5526140625662b8daa446c22ee2d73db3707e620c",
        "balance": "1000 ETH"
    },
    {
        "address": "0xE11BA2b4D45Eaed5996Cd0823791E0C93114882d",
        "private_key": "0x646f1ce2fdad0e6deeeb5c7e8e5543bdde65e86029e2fd9fc169899c440a7913",
        "balance": "1000 ETH"
    },
    {
        "address": "0xd03ea8624C8C5987235048901fB614fDcA89b117",
        "private_key": "0xadd53f9a7e588d003326d1cbf9e4a43c061aadd9bc938c843a79e7b4fd2ad743",
        "balance": "1000 ETH"
    },
    {
        "address": "0x95cED938F7991cd0dFcb48F0a06a40FA1aF46EBC",
        "private_key": "0x395df67f0c2d2d9fe1ad08d1bc8b6627011959b79c53d7dd6a3536a33ab8a4fd",
        "balance": "1000 ETH"
    },
    {
        "address": "0x3E5e9111Ae8eB78Fe1CC3bb8915d5D461F3Ef9A9",
        "private_key": "0xe485d098507f54e7733a205420dfddbe58db035fa577fc294ebd14db90767a52",
        "balance": "1000 ETH"
    },
    {
        "address": "0x28a8746e75304c0780E011BEd21C72cD78cd535E",
        "private_key": "0xa453611d9419d0e56f499079478fd72c37b251a94bfde4d19872c44cf65386e3",
        "balance": "1000 ETH"
    },
    {
        "address": "0xACa94ef8bD5ffEE41947b4585a84BdA5a3d3DA6E",
        "private_key": "0x829e924fdf021ba3dbbc4225edfece9aca04b929d6e75613329ca6f1d31c0bb4",
        "balance": "1000 ETH"
    },
    {
        "address": "0x1dF62f291b2E969fB0849d99D9Ce41e2F137006e",
        "private_key": "0xb0057716d5917badaf911b193b12b910811c1497b5bada8d7711f758981c3773",
        "balance": "1000 ETH"
    }
]

# Default deployer account (first account)
DEPLOYER_ADDRESS = ACCOUNTS[0]["address"]
DEPLOYER_PRIVATE_KEY = ACCOUNTS[0]["private_key"]

# HD Wallet Mnemonic
MNEMONIC = "myth like bonus scare over problem client lizard pioneer submit female collect"

# Gas settings
DEFAULT_GAS_PRICE = 2000000000  # 2 Gwei
BLOCK_GAS_LIMIT = 30000000

def get_account(index=0):
    """Get account by index."""
    if 0 <= index < len(ACCOUNTS):
        return ACCOUNTS[index]
    return None

def get_private_key(index=0):
    """Get private key by account index."""
    account = get_account(index)
    return account["private_key"] if account else None
