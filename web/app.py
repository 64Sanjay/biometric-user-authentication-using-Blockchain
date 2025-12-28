# web/app.py
"""
Flask Web Application for Biometric Authentication System
With Real-time Camera Integration
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blockchain.integrated_handler import IntegratedHandler
from src.blockchain.real_ipfs_handler import RealIPFSHandler
from src.blockchain.ethereum_handler import EthereumHandler

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'biometric_auth_secret_key_2024'
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Handlers
_integrated_handler = None
_ipfs_handler = None
_ethereum_handler = None


def get_integrated_handler():
    global _integrated_handler
    if _integrated_handler is None:
        try:
            _integrated_handler = IntegratedHandler()
        except Exception as e:
            print(f"Error: {e}")
    return _integrated_handler


def get_ipfs_handler():
    global _ipfs_handler
    if _ipfs_handler is None:
        try:
            _ipfs_handler = RealIPFSHandler()
        except Exception as e:
            print(f"Error: {e}")
    return _ipfs_handler


def get_ethereum_handler():
    global _ethereum_handler
    if _ethereum_handler is None:
        try:
            _ethereum_handler = EthereumHandler()
        except Exception as e:
            print(f"Error: {e}")
    return _ethereum_handler


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ════════════════════════════════════════════════════════════════
#                         ROUTES
# ════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'POST':
        try:
            user_id = request.form.get('user_id', '').strip()
            if not user_id:
                flash('User ID is required', 'error')
                return render_template('enroll.html')
            
            face_file = request.files.get('face_image')
            hand_file = request.files.get('hand_image')
            face_path = None
            hand_path = None
            
            if face_file and allowed_file(face_file.filename):
                filename = secure_filename(f"{user_id}_face_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
                face_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                face_file.save(face_path)
            
            if hand_file and allowed_file(hand_file.filename):
                filename = secure_filename(f"{user_id}_hand_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
                hand_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                hand_file.save(hand_path)
            
            vault_data = {
                'user_id': user_id,
                'enrollment_time': datetime.now().isoformat(),
                'face_image': face_path,
                'hand_image': hand_path,
                'biometric_type': 'face_hand'
            }
            
            handler = get_integrated_handler()
            if handler and handler.is_ready():
                result = handler.enroll_user(user_id, vault_data)
                if result and result.get('success'):
                    flash(f'User "{user_id}" enrolled successfully!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Enrollment failed.', 'error')
            else:
                flash('System not ready.', 'error')
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('enroll.html')


@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate():
    if request.method == 'POST':
        try:
            user_id = request.form.get('user_id', '').strip()
            if not user_id:
                flash('User ID is required', 'error')
                return render_template('authenticate.html')
            
            handler = get_integrated_handler()
            if handler and handler.is_ready():
                result = handler.authenticate_user(user_id)
                if result:
                    flash(f'Authentication successful!', 'success')
                    return render_template('authenticate.html', auth_result=result, authenticated=True)
                else:
                    flash(f'Authentication failed.', 'error')
            else:
                flash('System not ready.', 'error')
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('authenticate.html')


@app.route('/dashboard')
def dashboard():
    stats = {'ipfs': {'connected': False}, 'ethereum': {'connected': False}, 'system_ready': False}
    try:
        handler = get_integrated_handler()
        if handler:
            stats = handler.get_system_stats()
    except Exception as e:
        print(f"Error: {e}")
    return render_template('dashboard.html', stats=stats)


@app.route('/vault-management')
def vault_management():
    handler = get_ethereum_handler()
    users = []
    if handler:
        for user_hash, user_data in handler.users.items():
            user_info = {
                'user_id': user_data.get('user_id', 'Unknown'),
                'registered_at': user_data.get('registered_at', 'N/A'),
                'vault_count': user_data.get('vault_count', 0),
                'vaults': []
            }
            if user_hash in handler.vaults:
                for vault_idx, vault_data in handler.vaults[user_hash].items():
                    user_info['vaults'].append({
                        'index': vault_idx,
                        'ipfs_cid': vault_data.get('ipfs_cid', 'N/A'),
                        'is_active': vault_data.get('is_active', False),
                        'created_at': vault_data.get('created_at', 'N/A'),
                        'biometric_type': vault_data.get('biometric_type', 1)
                    })
            users.append(user_info)
    return render_template('vault_management.html', users=users)


@app.route('/camera')
def camera():
    return render_template('camera.html')


# ════════════════════════════════════════════════════════════════
#                         API ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.route('/api/status')
def api_status():
    try:
        handler = get_integrated_handler()
        if handler:
            return jsonify({'success': True, 'data': handler.get_system_stats()})
        return jsonify({'success': False, 'error': 'Handler not initialized'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id required'})
        
        vault_data = data.get('vault_data', {'user_id': user_id})
        handler = get_integrated_handler()
        if handler and handler.is_ready():
            result = handler.enroll_user(user_id, vault_data)
            return jsonify({'success': result.get('success', False) if result else False, 'data': result})
        return jsonify({'success': False, 'error': 'System not ready'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/authenticate', methods=['POST'])
def api_authenticate():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id required'})
        
        handler = get_integrated_handler()
        if handler and handler.is_ready():
            result = handler.authenticate_user(user_id)
            return jsonify({'success': result is not None, 'data': result})
        return jsonify({'success': False, 'error': 'System not ready'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/revoke', methods=['POST'])
def api_revoke():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        vault_index = data.get('vault_index', 0)
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id required'})
        
        handler = get_integrated_handler()
        if handler and handler.is_ready():
            result = handler.revoke_vault(user_id, vault_index)
            return jsonify({'success': result})
        return jsonify({'success': False, 'error': 'System not ready'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ════════════════════════════════════════════════════════════════
#                    CAMERA API ENDPOINTS
# ════════════════════════════════════════════════════════════════

_camera_handler = None

def get_camera_handler():
    global _camera_handler
    if _camera_handler is None:
        try:
            from src.camera.camera_handler import CameraHandler
            _camera_handler = CameraHandler(camera_id=0)
        except Exception as e:
            print(f"Camera error: {e}")
    return _camera_handler


@app.route('/api/camera/start', methods=['POST'])
def api_camera_start():
    handler = get_camera_handler()
    if handler:
        success = handler.start()
        return jsonify({'success': success})
    return jsonify({'success': False, 'error': 'Camera not available'})


@app.route('/api/camera/stop', methods=['POST'])
def api_camera_stop():
    handler = get_camera_handler()
    if handler:
        handler.stop()
        return jsonify({'success': True})
    return jsonify({'success': False})


@app.route('/api/camera/frame')
def api_camera_frame():
    handler = get_camera_handler()
    if handler and handler.is_running:
        detect_face = request.args.get('detect_face', 'true').lower() == 'true'
        detect_hand = request.args.get('detect_hand', 'false').lower() == 'true'
        frame = handler.get_frame_base64(detect_face=detect_face, detect_hand=detect_hand)
        if frame:
            return jsonify({'success': True, 'frame': frame})
    return jsonify({'success': False})


@app.route('/api/camera/capture/face', methods=['POST'])
def api_camera_capture_face():
    handler = get_camera_handler()
    if handler:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'unknown')
        filepath = handler.capture_face(user_id)
        if filepath:
            quality = handler.check_image_quality(filepath)
            return jsonify({'success': True, 'filepath': filepath, 'quality': quality})
    return jsonify({'success': False, 'error': 'Capture failed'})


@app.route('/api/camera/capture/hand', methods=['POST'])
def api_camera_capture_hand():
    handler = get_camera_handler()
    if handler:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'unknown')
        filepath = handler.capture_hand(user_id)
        if filepath:
            quality = handler.check_image_quality(filepath)
            return jsonify({'success': True, 'filepath': filepath, 'quality': quality})
    return jsonify({'success': False, 'error': 'Capture failed'})


if __name__ == '__main__':
    print("=" * 60)
    print("🌐 BIOMETRIC AUTHENTICATION WEB SERVER")
    print("=" * 60)
    print(f"📍 Server: http://127.0.0.1:5000")
    print(f"📷 Camera: http://127.0.0.1:5000/camera")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)


@app.route('/architecture')
def architecture():
    """System architecture page."""
    return render_template('architecture.html')
