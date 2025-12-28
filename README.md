# Multimodal Biometric Authentication — Blockchain & Improved Fuzzy Vault

Implementation of the framework from the paper:  
"Multimodal biometric user authentication using improved decentralized fuzzy vault scheme based on Blockchain network" 

This repository is a research prototype that demonstrates:
- Multimodal biometric authentication (Face + Dorsal Hand)
- An improved Fuzzy Vault scheme for template protection (locking/unlocking with chaff points)
- Decentralized vault storage on IPFS (Kubo)
- Smart-contract based indexing on an Ethereum-compatible network (Ganache for local testing)
- A Flask web UI and REST API with camera integration

This README has been tailored to the repository layout and code at commit `2dbd760ff67652ac3a5e739bce8791d05c9bb699`.

Contents
- Project overview
- What’s in this repo (actual files & folders)
- Quick start (run locally)
- Environment & configuration
- Web UI & API (precise fields & endpoints)
- Camera endpoints
- IPFS / Ethereum integration details
- Running tests & integration helpers
- Security & privacy notes
- Troubleshooting & tips
- Contributing, citation & license

Project overview
--------------
High-level flows implemented in the code:
- Enrollment: capture images → extract features → build multimodal template → lock into fuzzy vault → store vault on IPFS → record IPFS CID and metadata on Ethereum (via handlers).
- Authentication: fetch CID from blockchain → retrieve vault from IPFS → attempt to unlock with probe features → return authentication decision.

folders
----------------------------------------
Files and folders present at the time of tailoring:
- run_web.py — application runner / entrypoint that starts the Flask server (prints common URLs).
- web/app.py — Flask application: HTML UI routes, REST API routes, camera integration (uses handlers in src/).
- src/ — core modules (biometric pipeline, fuzzy vault, blockchain & IPFS handlers, camera handler).
  - src/blockchain/integrated_handler.py — integrates IPFS + Ethereum handlers and provides enroll/authenticate/revoke functions.
  - src/blockchain/ethereum_handler.py — in-memory/simple Ethereum handler (talks to Ganache via Web3.py where available).
  - src/blockchain/real_ipfs_handler.py — IPFS client used by IntegratedHandler (refer to src for details).
  - src/camera/camera_handler.py — camera capture utilities used by web UI and API.
- requirements.txt — Python dependencies for the main app.
- requirements_blockchain.txt — blockchain-related dependencies (web3, eth-account, etc.)
- contracts/ — solidity sources / placeholders (compile separately to produce artifacts).
- models/ — placeholder for pretrained face/hand models (if required).
- kubo/ — IPFS helper scripts / configs (optional).
- data/ — runtime uploads (`data/uploads` is created by web/app.py).
- verify_structure.py, test_setup.py — helper scripts to verify environment and run quick tests.
- .env.example — example environment variables.

## Dataset Setup

This project requires two datasets (not included due to size):

### CASIA-FaceV5
- **Source**: [CASIA Biometrics](http://biometrics.idealtest.org/)
- **Size**: ~2.2 GB
- **Structure**: 500 subjects, 5 images each
- **Path**: Place in `data/face_images/CASIA-FaceV5/`

### 11k Hands Dataset
- **Source**: [11k Hands](https://sites.google.com/view/11khands)
- **Size**: ~873 MB
- **Structure**: 11,076 hand images from 190 subjects
- **Path**: Place in `data/hand_images/Hands/`

Quick start (local)
-------------------
1. Clone:
   git clone https://github.com/sanjay423bel/biometric-user-authentication-using-Blockchain.git
   cd biometric-user-authentication-using-Blockchain

2. Create & activate virtualenv:
   - macOS / Linux
     python3 -m venv venv
     source venv/bin/activate
   - Windows (PowerShell)
     python -m venv venv
     .\venv\Scripts\Activate.ps1

3. Install dependencies:
   pip install -r requirements.txt
   pip install -r requirements_blockchain.txt

4. Start Ganache (local Ethereum):
   - Desktop: open Ganache app and start a workspace
   - CLI:
     ganache-cli --deterministic --port 8545
   Note: ethereum_handler.py contains default Ganache address & private key for deterministic testing. Replace private keys before using public networks.

5. Start IPFS (Kubo):
   ipfs init        # first time only
   ipfs daemon

6. Copy .env.example to .env and edit:
   cp .env.example .env
   Edit `.env` to set:
   - ETH_RPC_URL (default: http://127.0.0.1:8545)
   - ETH_PRIVATE_KEY (use a Ganache account private key for local testing)
   - CONTRACT_ADDRESS (set after deployment)
   - IPFS_API_URL (default: http://127.0.0.1:5001)
   - FLASK_SECRET_KEY (optional to override default)

7. Deploy smart contract (optional for production-like flow). See "Smart contract / deployment" section below.

8. Start the server:
   python run_web.py
   or (explicit Flask)
   FLASK_ENV=development python run_web.py

9. Open the UI (default):
   - Server: http://127.0.0.1:5000
   - Dashboard: http://127.0.0.1:5000/dashboard
   - Enroll page: http://127.0.0.1:5000/enroll
   - Authenticate: http://127.0.0.1:5000/authenticate
   - Camera UI: http://127.0.0.1:5000/camera

Environment & configuration
---------------------------
Important variables (from `.env.example`):
- FLASK_ENV=development
- FLASK_APP=run_web.py
- FLASK_SECRET_KEY=biometric_auth_secret_key_2024 (change for deployment)
- ETH_RPC_URL=http://127.0.0.1:8545
- ETH_PRIVATE_KEY=0x...
- CONTRACT_ADDRESS=0x...
- IPFS_API_URL=http://127.0.0.1:5001
- UPLOAD_FOLDER=./data/uploads

Note: IntegratedHandler defaults:
- IPFS API URL default in code: 'http://127.0.0.1:5001/api/v0'
- Ethereum URL default in code: 'http://127.0.0.1:8545'

Web UI & API — exact fields & endpoints
--------------------------------------
The code in web/app.py implements both HTML routes (used by the web UI) and JSON REST endpoints. Below are the exact routes and parameters as implemented:

Web (HTML) forms
- GET /enroll — enrollment form
- POST /enroll — form fields:
  - user_id — required (string)
  - face_image — file input (field name exactly: face_image)
  - hand_image — file input (field name exactly: hand_image)
  Notes: uploaded files are saved to `data/uploads` using secure filenames with timestamps.

- GET /authenticate — authentication form
- POST /authenticate — form fields:
  - user_id — required (string)
  The server will call IntegratedHandler.authenticate_user(user_id)

HTML pages:
- / (index)
- /dashboard (system status)
- /vault-management (shows vaults from Ethereum handler)
- /camera (camera UI)

REST API (JSON endpoints)
- GET /api/status
  - Response: { success: true/false, data: handler.get_system_stats() | error }

- POST /api/enroll
  - Expects JSON body:
    {
      "user_id": "alice",
      "vault_data": { ... }        # optional structure describing vault (if files uploaded via camera endpoints, vault_data can contain filepaths)
    }
  - Behavior: calls IntegratedHandler.enroll_user(user_id, vault_data)
  - Response: { success: bool, data: result | error }

- POST /api/authenticate
  - JSON body:
    { "user_id": "alice" }
  - Behavior: calls IntegratedHandler.authenticate_user(user_id)
  - Response: { success: bool, data: result | error }

- POST /api/revoke
  - JSON body:
    { "user_id": "alice", "vault_index": 0 }
  - Behavior: calls IntegratedHandler.revoke_vault(user_id, vault_index)
  - Response: { success: bool }

Camera endpoints (exact)
- POST /api/camera/start
  - Starts camera capture loop
  - Returns: { success: true/false }

- POST /api/camera/stop
  - Stops camera capture
  - Returns: { success: true/false }

- GET /api/camera/frame
  - Query params:
    - detect_face=true|false (default true)
    - detect_hand=true|false (default false)
  - Returns: { success: true/false, frame: "<base64 image>" }

- POST /api/camera/capture/face
  - JSON body: { "user_id": "alice" } (optional)
  - Returns: { success: true/false, filepath: "...", quality: {...} }

- POST /api/camera/capture/hand
  - JSON body: { "user_id": "alice" } (optional)
  - Returns: { success: true/false, filepath: "...", quality: {...} }

Important handler behavior (IntegratedHandler & EthereumHandler)
---------------------------------------------------------------
- IntegratedHandler (src/blockchain/integrated_handler.py):
  - Initializes RealIPFSHandler(api_url) and EthereumHandler(provider_url).
  - enroll_user(user_id, vault_data):
    1. calls ethereum.register_user(user_id)
    2. calls ipfs.store_vault(vault_data, user_id=user_id) → returns ipfs_cid
    3. calls ethereum.store_vault(user_id, ipfs_cid, vault_data, biometric_type=1) → returns (tx_hash, vault_index)
    4. returns a result dict with registration_tx, ipfs_cid, storage_tx, vault_index, success flag
  - authenticate_user(user_id, vault_index=0):
    - calls ethereum.get_vault(user_id, vault_index) → checks is_active → obtains ipfs_cid
    - calls ipfs.retrieve_vault(ipfs_cid) → returns vault_data for authentication
  - revoke_vault(user_id, vault_index=0): unpins IPFS (optional) then calls ethereum.revoke_vault(user_id, vault_index)

- EthereumHandler (src/blockchain/ethereum_handler.py):
  - Provides a light in-memory blockchain abstraction and also sends a JSON-encoded transaction via Web3 (if Web3 installed).
  - Default Ganache config (for local testing):
    - GANACHE_URL = http://127.0.0.1:8545
    - CHAIN_ID = 1337
    - Default deployer address & private key (present in code for deterministic local testing). Do NOT use these keys on any public network.
  - Useful functions:
    - register_user(user_id)
    - store_vault(user_id, ipfs_cid, vault_data, biometric_type)
    - get_vault(user_id, vault_index)
    - revoke_vault(user_id, vault_index)
    - verify_vault_integrity(user_id, vault_index, vault_data)
  - The handler stores users and vaults in memory (persisting to real on-chain contract would require deploying & connecting to the on-chain contract).

IPFS details
------------
- The RealIPFSHandler is used by IntegratedHandler. The default IPFS API URL used in IntegratedHandler is 'http://127.0.0.1:5001/api/v0'.
- Use `ipfs daemon` to run the IPFS node locally.
- The code uses `ipfs.store_vault(...)` and `ipfs.retrieve_vault(ipfs_cid)` wrappers — review `src/blockchain/real_ipfs_handler.py` for exact serialization/encryption behavior if you plan to change storage format.

Smart contract / deployment (recommended)
----------------------------------------
This repository contains a `contracts/` folder for solidity sources. The Python handlers are currently designed to store references in-memory and optionally send JSON transactions via Web3.py.

To deploy a real solidity contract and integrate:
1. Compile contracts using solc / Truffle / Hardhat to produce ABI + bytecode (artifact JSON).
2. Use a deploy script (Web3.py example below) to deploy to local Ganache and set CONTRACT_ADDRESS in `.env`.
3. Update `src/blockchain/ethereum_handler.py` to use the deployed contract ABI & address (or create a new handler that calls contract functions rather than storing in-memory).

Example (minimal) deploy script skeleton (create scripts/deploy_contract.py):
```python
# See README earlier or repository scripts for a full example. You will need compiled artifact JSON (ABI + bytecode).
```

Running tests & integration helpers
----------------------------------
- Verify repository structure:
  python verify_structure.py

- Quick integration test for Ethereum handler:
  python src/blockchain/ethereum_handler.py
  (This runs test_ethereum_handler() if executed directly — it uses Ganache for connectivity.)

- Quick integration test for IntegratedHandler (IPFS + Ethereum):
  python src/blockchain/integrated_handler.py
  (Runs test_integrated_handler() which exercises enroll/authenticate/revoke flows; requires Ganache and IPFS daemon.)

Troubleshooting & tips
----------------------
- IPFS connection refused: ensure `ipfs daemon` is running and IPFS_API_URL in `.env` matches the daemon's API address (default: http://127.0.0.1:5001).
- Ganache not reachable: ensure Ganache is running and ETH_RPC_URL in `.env` points to it. Use Ganache accounts/private keys for signing in local tests.
- Missing models: if a module expects pretrained models in `models/`, download the specified weights and add them to that folder (see model-loading code in src).
- Camera errors: verify camera index and OS permissions; if camera unavailable, use sample images via /enroll web form.
- If Web3.py is not installed, EthereumHandler will fall back to local in-memory operations (but will print a warning).

Security & privacy notes
------------------------
- This repository is a research prototype. Biometric data is extremely sensitive — do not use with real users without encryption, access controls, and a security/privacy review.
- IPFS data is addressable publicly via CID — encrypt vaults before publishing to public IPFS or use private IPFS networks.
- Never commit private keys. The repository contains a test private key in `src/blockchain/ethereum_handler.py` for deterministic local testing only; replace it before any network usage.

Contributing
--------------------------------
- Contributions: fork → feature branch → PR. Include tests and documentation for algorithmic changes.

Appendix — handy commands
-------------------------
- Start Ganache CLI:
  ganache-cli --deterministic --port 8545
- Start IPFS:
  ipfs init
  ipfs daemon
- Run server:
  python run_web.py
- Test Ethereum handler:
  python src/blockchain/ethereum_handler.py
- Test Integrated handler (requires IPFS + Ganache):
  python src/blockchain/integrated_handler.py

