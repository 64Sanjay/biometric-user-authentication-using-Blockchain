# web/routes/camera.py
"""
Camera routes for real-time biometric capture
"""

from flask import Blueprint, render_template, jsonify, request, Response
import cv2
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.camera.camera_handler import CameraHandler

camera_bp = Blueprint('camera', __name__)

# Global camera handler
_camera_handler = None


def get_camera_handler():
    """Get or create camera handler."""
    global _camera_handler
    if _camera_handler is None:
        _camera_handler = CameraHandler(camera_id=0)
    return _camera_handler


@camera_bp.route('/camera')
def camera_page():
    """Camera capture page."""
    return render_template('camera.html')


@camera_bp.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start the camera."""
    handler = get_camera_handler()
    success = handler.start()
    return jsonify({
        'success': success,
        'message': 'Camera started' if success else 'Failed to start camera'
    })


@camera_bp.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop the camera."""
    handler = get_camera_handler()
    handler.stop()
    return jsonify({
        'success': True,
        'message': 'Camera stopped'
    })


@camera_bp.route('/api/camera/frame')
def get_frame():
    """Get current camera frame as base64."""
    handler = get_camera_handler()
    
    if not handler.is_running:
        return jsonify({
            'success': False,
            'error': 'Camera not running'
        })
    
    detect_face = request.args.get('detect_face', 'true').lower() == 'true'
    detect_hand = request.args.get('detect_hand', 'false').lower() == 'true'
    
    frame_b64 = handler.get_frame_base64(detect_face=detect_face, detect_hand=detect_hand)
    
    if frame_b64:
        return jsonify({
            'success': True,
            'frame': frame_b64
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to get frame'
        })


@camera_bp.route('/api/camera/capture/face', methods=['POST'])
def capture_face():
    """Capture face image."""
    handler = get_camera_handler()
    
    data = request.get_json() or {}
    user_id = data.get('user_id', 'unknown')
    
    filepath = handler.capture_face(user_id)
    
    if filepath:
        quality = handler.check_image_quality(filepath)
        return jsonify({
            'success': True,
            'filepath': filepath,
            'quality': quality
        })
    else:
        return jsonify({
            'success': False,
            'error': 'No face detected or camera not running'
        })


@camera_bp.route('/api/camera/capture/hand', methods=['POST'])
def capture_hand():
    """Capture hand image."""
    handler = get_camera_handler()
    
    data = request.get_json() or {}
    user_id = data.get('user_id', 'unknown')
    
    filepath = handler.capture_hand(user_id)
    
    if filepath:
        quality = handler.check_image_quality(filepath)
        return jsonify({
            'success': True,
            'filepath': filepath,
            'quality': quality
        })
    else:
        return jsonify({
            'success': False,
            'error': 'No hand detected or camera not running'
        })


def generate_frames():
    """Generator for video streaming."""
    handler = get_camera_handler()
    
    if not handler.is_running:
        handler.start()
    
    while handler.is_running:
        frame = handler.read_frame()
        if frame is None:
            continue
        
        # Detect face
        frame, _ = handler.detect_face(frame)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@camera_bp.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
