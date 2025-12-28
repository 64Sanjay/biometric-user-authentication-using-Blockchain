// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title BiometricVault
 * @author Based on Sharma et al. (2024) - IIT Jodhpur
 * @notice Decentralized Biometric Template Storage using Blockchain
 * @dev Stores IPFS CIDs of encrypted fuzzy vaults on Ethereum
 * 
 * Paper: "Multimodal biometric user authentication using improved 
 *         decentralized fuzzy vault scheme based on Blockchain network"
 * Journal: Journal of Information Security and Applications (2024)
 */

contract BiometricVault {
    
    // ═══════════════════════════════════════════════════════════════
    //                         STATE VARIABLES
    // ═══════════════════════════════════════════════════════════════
    
    address public owner;
    uint256 public totalUsers;
    uint256 public totalVaults;
    uint256 public contractDeployedAt;
    
    // Biometric Types (as per paper)
    uint8 public constant BIOMETRIC_FACE_HAND = 1;      // Face + Dorsal Hand
    uint8 public constant BIOMETRIC_FINGERPRINT_IRIS = 2;
    uint8 public constant BIOMETRIC_MULTIMODAL = 3;
    
    // ═══════════════════════════════════════════════════════════════
    //                           STRUCTS
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @dev Vault structure storing encrypted biometric template reference
     */
    struct Vault {
        string ipfsCid;           // IPFS Content Identifier (e.g., "Qm...")
        uint256 createdAt;        // Unix timestamp of creation
        uint256 updatedAt;        // Unix timestamp of last update
        bool isActive;            // Revocability flag (ISO/IEC 24745)
        uint8 biometricType;      // Type of biometric used
        bytes32 vaultHash;        // Keccak256 hash for integrity verification
        uint256 version;          // Vault version (for re-enrollment)
    }
    
    /**
     * @dev User structure for identity management
     */
    struct User {
        address walletAddress;    // Ethereum wallet address
        bytes32 userIdHash;       // Hashed user identifier (privacy)
        uint256 registeredAt;     // Registration timestamp
        bool isRegistered;        // Registration status
        bool isActive;            // Account active status
        uint256 vaultCount;       // Number of vaults stored
        uint256 lastActivity;     // Last activity timestamp
    }
    
    /**
     * @dev Authentication attempt logging
     */
    struct AuthAttempt {
        uint256 timestamp;
        bool success;
        bytes32 attemptHash;
    }
    
    // ═══════════════════════════════════════════════════════════════
    //                          MAPPINGS
    // ═══════════════════════════════════════════════════════════════
    
    // userIdHash => User struct
    mapping(bytes32 => User) public users;
    
    // userIdHash => vaultIndex => Vault struct
    mapping(bytes32 => mapping(uint256 => Vault)) public vaults;
    
    // walletAddress => userIdHash
    mapping(address => bytes32) public addressToUserId;
    
    // ipfsCid => exists (prevent duplicates)
    mapping(string => bool) public cidExists;
    
    // userIdHash => AuthAttempt[] (audit trail)
    mapping(bytes32 => AuthAttempt[]) private authHistory;
    
    // ═══════════════════════════════════════════════════════════════
    //                           EVENTS
    // ═══════════════════════════════════════════════════════════════
    
    event UserRegistered(
        bytes32 indexed userIdHash,
        address indexed walletAddress,
        uint256 timestamp
    );
    
    event UserDeactivated(
        bytes32 indexed userIdHash,
        uint256 timestamp
    );
    
    event VaultStored(
        bytes32 indexed userIdHash,
        uint256 indexed vaultIndex,
        string ipfsCid,
        uint8 biometricType,
        uint256 timestamp
    );
    
    event VaultUpdated(
        bytes32 indexed userIdHash,
        uint256 indexed vaultIndex,
        string oldCid,
        string newCid,
        uint256 timestamp
    );
    
    event VaultRevoked(
        bytes32 indexed userIdHash,
        uint256 indexed vaultIndex,
        uint256 timestamp
    );
    
    event VaultReactivated(
        bytes32 indexed userIdHash,
        uint256 indexed vaultIndex,
        uint256 timestamp
    );
    
    event AuthenticationAttempt(
        bytes32 indexed userIdHash,
        bool success,
        uint256 timestamp
    );
    
    // ═══════════════════════════════════════════════════════════════
    //                          MODIFIERS
    // ═══════════════════════════════════════════════════════════════
    
    modifier onlyOwner() {
        require(msg.sender == owner, "BiometricVault: caller is not owner");
        _;
    }
    
    modifier onlyRegisteredUser() {
        bytes32 userId = addressToUserId[msg.sender];
        require(users[userId].isRegistered, "BiometricVault: user not registered");
        require(users[userId].isActive, "BiometricVault: user account deactivated");
        _;
    }
    
    modifier validCid(string memory _cid) {
        require(bytes(_cid).length >= 46, "BiometricVault: CID too short");
        require(bytes(_cid).length <= 100, "BiometricVault: CID too long");
        _;
    }
    
    modifier validBiometricType(uint8 _type) {
        require(
            _type >= BIOMETRIC_FACE_HAND && _type <= BIOMETRIC_MULTIMODAL,
            "BiometricVault: invalid biometric type"
        );
        _;
    }
    
    // ═══════════════════════════════════════════════════════════════
    //                         CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════
    
    constructor() {
        owner = msg.sender;
        totalUsers = 0;
        totalVaults = 0;
        contractDeployedAt = block.timestamp;
    }
    
    // ═══════════════════════════════════════════════════════════════
    //                    USER MANAGEMENT FUNCTIONS
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @notice Register a new user in the system
     * @param _userIdHash Keccak256 hash of the user identifier
     * @dev User ID is hashed off-chain for privacy (unlinkability)
     */
    function registerUser(bytes32 _userIdHash) external {
        require(!users[_userIdHash].isRegistered, "BiometricVault: user already registered");
        require(
            addressToUserId[msg.sender] == bytes32(0),
            "BiometricVault: address already registered"
        );
        
        users[_userIdHash] = User({
            walletAddress: msg.sender,
            userIdHash: _userIdHash,
            registeredAt: block.timestamp,
            isRegistered: true,
            isActive: true,
            vaultCount: 0,
            lastActivity: block.timestamp
        });
        
        addressToUserId[msg.sender] = _userIdHash;
        totalUsers++;
        
        emit UserRegistered(_userIdHash, msg.sender, block.timestamp);
    }
    
    /**
     * @notice Check if a user is registered
     * @param _userIdHash User ID hash to check
     * @return bool Registration status
     */
    function isUserRegistered(bytes32 _userIdHash) external view returns (bool) {
        return users[_userIdHash].isRegistered;
    }
    
    /**
     * @notice Deactivate a user account
     * @param _userIdHash User ID hash to deactivate
     * @dev Only owner can deactivate users
     */
    function deactivateUser(bytes32 _userIdHash) external onlyOwner {
        require(users[_userIdHash].isRegistered, "BiometricVault: user not found");
        require(users[_userIdHash].isActive, "BiometricVault: user already deactivated");
        
        users[_userIdHash].isActive = false;
        users[_userIdHash].lastActivity = block.timestamp;
        
        emit UserDeactivated(_userIdHash, block.timestamp);
    }
    
    // ═══════════════════════════════════════════════════════════════
    //                    VAULT MANAGEMENT FUNCTIONS
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @notice Store a new biometric vault
     * @param _ipfsCid IPFS Content Identifier of the encrypted vault
     * @param _biometricType Type of biometric (1=Face+Hand, 2=Fingerprint+Iris, 3=Multimodal)
     * @param _vaultHash Keccak256 hash of vault data for integrity verification
     * @return vaultIndex Index of the stored vault
     */
    function storeVault(
        string calldata _ipfsCid,
        uint8 _biometricType,
        bytes32 _vaultHash
    )
        external
        onlyRegisteredUser
        validCid(_ipfsCid)
        validBiometricType(_biometricType)
        returns (uint256 vaultIndex)
    {
        require(!cidExists[_ipfsCid], "BiometricVault: CID already exists");
        
        bytes32 userId = addressToUserId[msg.sender];
        vaultIndex = users[userId].vaultCount;
        
        vaults[userId][vaultIndex] = Vault({
            ipfsCid: _ipfsCid,
            createdAt: block.timestamp,
            updatedAt: block.timestamp,
            isActive: true,
            biometricType: _biometricType,
            vaultHash: _vaultHash,
            version: 1
        });
        
        cidExists[_ipfsCid] = true;
        users[userId].vaultCount++;
        users[userId].lastActivity = block.timestamp;
        totalVaults++;
        
        emit VaultStored(userId, vaultIndex, _ipfsCid, _biometricType, block.timestamp);
        
        return vaultIndex;
    }
    
    /**
     * @notice Update an existing vault (re-enrollment)
     * @param _vaultIndex Index of vault to update
     * @param _newIpfsCid New IPFS CID
     * @param _newVaultHash New vault hash
     * @dev Implements revocability requirement from ISO/IEC 24745
     */
    function updateVault(
        uint256 _vaultIndex,
        string calldata _newIpfsCid,
        bytes32 _newVaultHash
    )
        external
        onlyRegisteredUser
        validCid(_newIpfsCid)
    {
        bytes32 userId = addressToUserId[msg.sender];
        require(_vaultIndex < users[userId].vaultCount, "BiometricVault: vault not found");
        require(vaults[userId][_vaultIndex].isActive, "BiometricVault: vault is revoked");
        require(!cidExists[_newIpfsCid], "BiometricVault: new CID already exists");
        
        string memory oldCid = vaults[userId][_vaultIndex].ipfsCid;
        
        // Remove old CID from existence mapping
        cidExists[oldCid] = false;
        
        // Update vault
        vaults[userId][_vaultIndex].ipfsCid = _newIpfsCid;
        vaults[userId][_vaultIndex].vaultHash = _newVaultHash;
        vaults[userId][_vaultIndex].updatedAt = block.timestamp;
        vaults[userId][_vaultIndex].version++;
        
        cidExists[_newIpfsCid] = true;
        users[userId].lastActivity = block.timestamp;
        
        emit VaultUpdated(userId, _vaultIndex, oldCid, _newIpfsCid, block.timestamp);
    }
    
    /**
     * @notice Revoke a vault (implements revocability from ISO/IEC 24745)
     * @param _vaultIndex Index of vault to revoke
     */
    function revokeVault(uint256 _vaultIndex) external onlyRegisteredUser {
        bytes32 userId = addressToUserId[msg.sender];
        require(_vaultIndex < users[userId].vaultCount, "BiometricVault: vault not found");
        require(vaults[userId][_vaultIndex].isActive, "BiometricVault: vault already revoked");
        
        vaults[userId][_vaultIndex].isActive = false;
        vaults[userId][_vaultIndex].updatedAt = block.timestamp;
        users[userId].lastActivity = block.timestamp;
        
        emit VaultRevoked(userId, _vaultIndex, block.timestamp);
    }
    
    /**
     * @notice Reactivate a previously revoked vault
     * @param _vaultIndex Index of vault to reactivate
     */
    function reactivateVault(uint256 _vaultIndex) external onlyRegisteredUser {
        bytes32 userId = addressToUserId[msg.sender];
        require(_vaultIndex < users[userId].vaultCount, "BiometricVault: vault not found");
        require(!vaults[userId][_vaultIndex].isActive, "BiometricVault: vault already active");
        
        vaults[userId][_vaultIndex].isActive = true;
        vaults[userId][_vaultIndex].updatedAt = block.timestamp;
        users[userId].lastActivity = block.timestamp;
        
        emit VaultReactivated(userId, _vaultIndex, block.timestamp);
    }
    
    // ═══════════════════════════════════════════════════════════════
    //                      GETTER FUNCTIONS
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @notice Get vault information
     * @param _userIdHash User ID hash
     * @param _vaultIndex Vault index
     * @return Vault details
     */
    function getVault(bytes32 _userIdHash, uint256 _vaultIndex)
        external
        view
        returns (
            string memory ipfsCid,
            uint256 createdAt,
            uint256 updatedAt,
            bool isActive,
            uint8 biometricType,
            bytes32 vaultHash,
            uint256 version
        )
    {
        require(users[_userIdHash].isRegistered, "BiometricVault: user not found");
        require(_vaultIndex < users[_userIdHash].vaultCount, "BiometricVault: vault not found");
        
        Vault memory vault = vaults[_userIdHash][_vaultIndex];
        
        return (
            vault.ipfsCid,
            vault.createdAt,
            vault.updatedAt,
            vault.isActive,
            vault.biometricType,
            vault.vaultHash,
            vault.version
        );
    }
    
    /**
     * @notice Get user's vault count
     * @param _userIdHash User ID hash
     * @return count Number of vaults
     */
    function getUserVaultCount(bytes32 _userIdHash) external view returns (uint256 count) {
        return users[_userIdHash].vaultCount;
    }
    
    /**
     * @notice Get user information
     * @param _userIdHash User ID hash
     * @return User details
     */
    function getUser(bytes32 _userIdHash)
        external
        view
        returns (
            address walletAddress,
            uint256 registeredAt,
            bool isRegistered,
            bool isActive,
            uint256 vaultCount,
            uint256 lastActivity
        )
    {
        User memory user = users[_userIdHash];
        return (
            user.walletAddress,
            user.registeredAt,
            user.isRegistered,
            user.isActive,
            user.vaultCount,
            user.lastActivity
        );
    }
    
    /**
     * @notice Get all active vault CIDs for a user
     * @param _userIdHash User ID hash
     * @return cids Array of active IPFS CIDs
     */
    function getActiveVaultCids(bytes32 _userIdHash)
        external
        view
        returns (string[] memory cids)
    {
        require(users[_userIdHash].isRegistered, "BiometricVault: user not found");
        
        uint256 vaultCount = users[_userIdHash].vaultCount;
        uint256 activeCount = 0;
        
        // Count active vaults
        for (uint256 i = 0; i < vaultCount; i++) {
            if (vaults[_userIdHash][i].isActive) {
                activeCount++;
            }
        }
        
        // Create array of active CIDs
        cids = new string[](activeCount);
        uint256 index = 0;
        
        for (uint256 i = 0; i < vaultCount; i++) {
            if (vaults[_userIdHash][i].isActive) {
                cids[index] = vaults[_userIdHash][i].ipfsCid;
                index++;
            }
        }
        
        return cids;
    }
    
    // ═══════════════════════════════════════════════════════════════
    //                   VERIFICATION FUNCTIONS
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @notice Verify vault integrity using hash
     * @param _userIdHash User ID hash
     * @param _vaultIndex Vault index
     * @param _expectedHash Expected vault hash
     * @return isValid Whether hash matches
     */
    function verifyVaultIntegrity(
        bytes32 _userIdHash,
        uint256 _vaultIndex,
        bytes32 _expectedHash
    )
        external
        view
        returns (bool isValid)
    {
        require(users[_userIdHash].isRegistered, "BiometricVault: user not found");
        require(_vaultIndex < users[_userIdHash].vaultCount, "BiometricVault: vault not found");
        
        return vaults[_userIdHash][_vaultIndex].vaultHash == _expectedHash;
    }
    
    /**
     * @notice Log authentication attempt (for audit trail)
     * @param _userIdHash User ID hash
     * @param _success Whether authentication was successful
     * @param _attemptHash Hash of attempt details
     */
    function logAuthAttempt(
        bytes32 _userIdHash,
        bool _success,
        bytes32 _attemptHash
    ) external onlyRegisteredUser {
        authHistory[_userIdHash].push(AuthAttempt({
            timestamp: block.timestamp,
            success: _success,
            attemptHash: _attemptHash
        }));
        
        emit AuthenticationAttempt(_userIdHash, _success, block.timestamp);
    }
    
    /**
     * @notice Get authentication history count
     * @param _userIdHash User ID hash
     * @return count Number of auth attempts
     */
    function getAuthHistoryCount(bytes32 _userIdHash) external view returns (uint256 count) {
        return authHistory[_userIdHash].length;
    }
    
    // ═══════════════════════════════════════════════════════════════
    //                     STATISTICS FUNCTIONS
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @notice Get contract statistics
     * @return Contract stats
     */
    function getStats()
        external
        view
        returns (
            uint256 _totalUsers,
            uint256 _totalVaults,
            address _owner,
            uint256 _deployedAt
        )
    {
        return (totalUsers, totalVaults, owner, contractDeployedAt);
    }
    
    /**
     * @notice Get contract version
     * @return version string
     */
    function getVersion() external pure returns (string memory) {
        return "BiometricVault v1.0.0 - Sharma et al. (2024)";
    }
}