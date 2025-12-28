# src/camera/camera_handler.py
"""
Real-time Camera Handler for Biometric Capture
Supports face and hand image capture with detection
"""

import cv2
import numpy as np
import base64
import os
from datetime import datetime
from typing import Optional, Tuple, Dict
import threading

# Try to import MediaPipe for face/hand detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not installed. Run: pip install mediapipe")


class CameraHandler:
    """
    Handles camera operations for biometric capture.
    Supports face detection and hand detection.
    """
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize camera handler.
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
        """
        self.camera_id = camera_id
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        
        # Initialize MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=0.5
            )
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
        
        # Capture storage
        self.captured_face = None
        self.captured_hand = None
        self.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'captures'
        )
        os.makedirs(self.save_dir, exist_ok=True)
    
    def start(self) -> bool:
        """Start the camera."""
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                print(f"❌ Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            print(f"✅ Camera {self.camera_id} started")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop the camera."""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        print("✅ Camera stopped")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the camera."""
        if not self.is_running or self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        if ret:
            with self.lock:
                self.current_frame = frame.copy()
            return frame
        return None
    
    def detect_face(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Detect faces in frame and draw bounding boxes.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (annotated frame, list of face detections)
        """
        if not MEDIAPIPE_AVAILABLE:
            return frame, []
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        detections = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Draw confidence
                confidence = detection.score[0]
                cv2.putText(frame, f'Face: {confidence:.2f}', 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                detections.append({
                    'x': x, 'y': y, 
                    'width': width, 'height': height,
                    'confidence': confidence
                })
        
        return frame, detections
    
    def detect_hand(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Detect hands in frame and draw landmarks.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (annotated frame, list of hand detections)
        """
        if not MEDIAPIPE_AVAILABLE:
            return frame, []
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = self.hands.process(rgb_frame)
        
        detections = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get bounding box
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min - 20, y_min - 20), 
                             (x_max + 20, y_max + 20), (255, 0, 0), 2)
                cv2.putText(frame, 'Hand Detected', 
                           (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 0, 0), 2)
                
                detections.append({
                    'x': x_min - 20,
                    'y': y_min - 20,
                    'width': x_max - x_min + 40,
                    'height': y_max - y_min + 40
                })
        
        return frame, detections
    
    def capture_face(self, user_id: str = None) -> Optional[str]:
        """
        Capture current frame as face image.
        
        Args:
            user_id: User identifier for filename
            
        Returns:
            Path to saved image or None
        """
        with self.lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        
        # Detect face
        _, detections = self.detect_face(frame)
        
        if not detections:
            print("⚠️  No face detected")
            return None
        
        # Use the first detected face
        det = detections[0]
        
        # Crop face with padding
        h, w = frame.shape[:2]
        padding = 50
        x1 = max(0, det['x'] - padding)
        y1 = max(0, det['y'] - padding)
        x2 = min(w, det['x'] + det['width'] + padding)
        y2 = min(h, det['y'] + det['height'] + padding)
        
        face_img = frame[y1:y2, x1:x2]
        
        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"face_{user_id}_{timestamp}.jpg" if user_id else f"face_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        cv2.imwrite(filepath, face_img)
        self.captured_face = filepath
        
        print(f"✅ Face captured: {filepath}")
        return filepath
    
    def capture_hand(self, user_id: str = None) -> Optional[str]:
        """
        Capture current frame as hand image.
        
        Args:
            user_id: User identifier for filename
            
        Returns:
            Path to saved image or None
        """
        with self.lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        
        # Detect hand
        _, detections = self.detect_hand(frame)
        
        if not detections:
            print("⚠️  No hand detected")
            return None
        
        # Use the first detected hand
        det = detections[0]
        
        # Crop hand with padding
        h, w = frame.shape[:2]
        padding = 30
        x1 = max(0, det['x'] - padding)
        y1 = max(0, det['y'] - padding)
        x2 = min(w, det['x'] + det['width'] + padding)
        y2 = min(h, det['y'] + det['height'] + padding)
        
        hand_img = frame[y1:y2, x1:x2]
        
        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"hand_{user_id}_{timestamp}.jpg" if user_id else f"hand_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        cv2.imwrite(filepath, hand_img)
        self.captured_hand = filepath
        
        print(f"✅ Hand captured: {filepath}")
        return filepath
    
    def get_frame_base64(self, detect_face: bool = True, 
                         detect_hand: bool = False) -> Optional[str]:
        """
        Get current frame as base64 encoded JPEG.
        
        Args:
            detect_face: Whether to detect and annotate faces
            detect_hand: Whether to detect and annotate hands
            
        Returns:
            Base64 encoded image string
        """
        frame = self.read_frame()
        if frame is None:
            return None
        
        # Apply detections
        if detect_face:
            frame, _ = self.detect_face(frame)
        if detect_hand:
            frame, _ = self.detect_hand(frame)
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        # Convert to base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        return jpg_as_text
    
    def check_image_quality(self, image_path: str) -> Dict:
        """
        Check quality of captured image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Quality metrics dictionary
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'valid': False, 'error': 'Cannot read image'}
        
        # Calculate metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Size
        height, width = img.shape[:2]
        
        # Quality assessment
        quality = {
            'valid': True,
            'width': width,
            'height': height,
            'brightness': float(brightness),
            'contrast': float(contrast),
            'sharpness': float(sharpness),
            'issues': []
        }
        
        # Check thresholds
        if brightness < 50:
            quality['issues'].append('Image too dark')
        elif brightness > 200:
            quality['issues'].append('Image too bright')
        
        if contrast < 30:
            quality['issues'].append('Low contrast')
        
        if sharpness < 100:
            quality['issues'].append('Image may be blurry')
        
        if width < 100 or height < 100:
            quality['issues'].append('Image too small')
        
        quality['valid'] = len(quality['issues']) == 0
        
        return quality


def test_camera():
    """Test camera functionality."""
    print("=" * 60)
    print("TESTING CAMERA HANDLER")
    print("=" * 60)
    
    handler = CameraHandler(camera_id=0)
    
    if not handler.start():
        print("❌ Failed to start camera")
        return
    
    print("\nPress 'f' to capture face")
    print("Press 'h' to capture hand")
    print("Press 'q' to quit")
    print("")
    
    try:
        while True:
            frame = handler.read_frame()
            if frame is None:
                continue
            
            # Detect face and hand
            frame, faces = handler.detect_face(frame)
            frame, hands = handler.detect_hand(frame)
            
            # Show frame
            cv2.imshow('Biometric Capture', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('f'):
                path = handler.capture_face('test_user')
                if path:
                    quality = handler.check_image_quality(path)
                    print(f"   Quality: {quality}")
            
            elif key == ord('h'):
                path = handler.capture_hand('test_user')
                if path:
                    quality = handler.check_image_quality(path)
                    print(f"   Quality: {quality}")
            
            elif key == ord('q'):
                break
    
    finally:
        handler.stop()
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("CAMERA TEST COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_camera()
