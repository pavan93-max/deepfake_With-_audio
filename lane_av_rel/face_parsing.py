"""
Face parsing and landmark extraction with ROI crops.
Extracts fixed ROIs: {L-eye, R-eye, mouth (teeth/lip split), nose, cheeks, jawline}.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Face detection backends (in order of preference):
# 1. InsightFace (best accuracy, no protobuf conflicts)
# 2. MediaPipe (fallback, may have protobuf issues)
# 3. OpenCV DNN (last resort, basic detection only)

INSIGHTFACE_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
mp = None

# Try InsightFace first (best option)
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    pass

# Try MediaPipe as fallback
try:
    import mediapipe as mp
    from mediapipe import solutions
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_USE_TASKS = False
except (ImportError, AttributeError):
    pass

# OpenCV DNN is always available (built into opencv-python)


@dataclass
class FaceROI:
    """Represents a face region of interest."""
    name: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: Optional[np.ndarray] = None  # Nx2 array of landmark points (optional for OpenCV fallback)
    confidence: float = 0.0


class OpenCVDNNFaceDetector:
    """
    OpenCV DNN-based face detector as fallback when MediaPipe is unavailable.
    Uses pre-trained Caffe models for face detection.
    """
    
    def __init__(self):
        # Try to load OpenCV DNN face detection model
        # Note: These model files need to be downloaded separately
        # For now, use OpenCV's built-in Haar Cascade as a simpler fallback
        try:
            # Try to use OpenCV's DNN module with Caffe model
            self.net = None
            # For simplicity, use OpenCV's built-in face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise RuntimeError("Could not load Haar Cascade classifier")
        except Exception as e:
            raise RuntimeError(f"OpenCV face detector initialization failed: {e}")
    
    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face bounding box.
        
        Args:
            image: Input image (H, W, 3) in RGB
            
        Returns:
            Bounding box (x, y, w, h) or None
        """
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Return the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)


class FaceParser:
    """
    Face parsing using InsightFace (preferred) or MediaPipe (fallback).
    Extracts fixed ROIs with scale-normalized padding.
    InsightFace provides higher accuracy for deepfake detection.
    """
    
    # ROI definitions
    # InsightFace uses 106 landmarks (or 68 in some models)
    # MediaPipe uses 468 landmarks
    # We'll map based on which backend is used
    ROI_INDICES_INSIGHTFACE = {
        # InsightFace 68-landmark format (similar to dlib)
        'left_eye': list(range(36, 42)),  # 6 points
        'right_eye': list(range(42, 48)),  # 6 points
        'mouth': list(range(48, 68)),  # 20 points (outer + inner)
        'nose': list(range(27, 36)),  # 9 points
        'left_cheek': [1, 2, 3, 4, 5, 48, 31],  # Approximate cheek region
        'right_cheek': [13, 14, 15, 16, 17, 54, 35],  # Approximate cheek region
        'jawline': list(range(0, 17))  # 17 points along jaw
    }
    
    ROI_INDICES_MEDIAPIPE = {
        # MediaPipe 468-landmark format
        'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
        'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 363, 360],
        'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207],
        'right_cheek': [345, 346, 347, 348, 349, 350, 451, 452, 453, 276, 283, 282],
        'jawline': [10, 151, 9, 175, 199, 200, 18]
    }
    
    def __init__(self, 
                 roi_size: int = 64,
                 padding_ratio: float = 0.2,
                 min_face_confidence: float = 0.5):
        """
        Args:
            roi_size: Output size for each ROI crop
            padding_ratio: Padding around ROI bounding box
            min_face_confidence: Minimum confidence for face detection
        """
        self.roi_size = roi_size
        self.padding_ratio = padding_ratio
        self.min_face_confidence = min_face_confidence
        self.use_insightface = False
        self.face_analyzer = None
        self.face_mesh = None
        self.use_tasks_api = False
        self.opencv_detector = None
        
        # Try InsightFace first (better accuracy)
        if INSIGHTFACE_AVAILABLE:
            try:
                # Initialize InsightFace FaceAnalysis
                # Check version to use correct API
                if hasattr(insightface, '__version__'):
                    version = insightface.__version__
                else:
                    version = "0.2.1"  # Default to old version
                
                # Newer versions (0.7+) use providers parameter
                # Older versions (0.2.x) require name as positional argument
                try:
                    if version.startswith('0.2'):
                        # Old API (0.2.x) - name is positional, not keyword
                        # Common model names: 'antelope', 'buffalo_l', 'buffalo_s', 'buffalo_m'
                        self.face_analyzer = insightface.app.FaceAnalysis('antelope')
                        self.face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
                    else:
                        # New API (0.7+) - try with providers
                        try:
                            self.face_analyzer = insightface.app.FaceAnalysis(
                                name='buffalo_l',
                                providers=['CPUExecutionProvider']
                            )
                            self.face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
                        except TypeError:
                            # Fallback: try without providers
                            self.face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l')
                            self.face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
                    self.use_insightface = True
                    print(f"Using InsightFace {version} for face detection and landmarks (higher accuracy)")
                except Exception as e:
                    # Try alternative model names for old API
                    if version.startswith('0.2'):
                        try:
                            self.face_analyzer = insightface.app.FaceAnalysis('buffalo_l')
                            self.face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
                            self.use_insightface = True
                            print(f"Using InsightFace {version} for face detection and landmarks (higher accuracy)")
                        except Exception:
                            raise e  # Re-raise original error
                    else:
                        raise e  # Re-raise original error
            except Exception as e:
                error_msg = str(e) if str(e) else repr(e)
                if 'AssertionError' in error_msg or 'detection' in error_msg.lower():
                    print(f"Warning: InsightFace 0.2.1 requires model files to be downloaded manually.")
                    print(f"  Please download models from: https://github.com/deepinsight/insightface")
                    print(f"  Or install Visual C++ Build Tools and upgrade to InsightFace 0.7+")
                    print(f"  Falling back to MediaPipe/OpenCV DNN.")
                else:
                    print(f"Warning: InsightFace initialization failed: {error_msg}. Falling back to MediaPipe.")
                self.use_insightface = False
        
        # Fallback to MediaPipe if InsightFace unavailable
        if not self.use_insightface:
            if MEDIAPIPE_AVAILABLE and mp is not None:
                try:
                    # Use old solutions API
                    self.mp_face_mesh = mp.solutions.face_mesh
                    self.face_mesh = self.mp_face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=min_face_confidence,
                        min_tracking_confidence=0.5
                    )
                    self.use_tasks_api = False
                    print("Using MediaPipe for face detection and landmarks (fallback)")
                except (AttributeError, RuntimeError, ImportError, ValueError) as e:
                    print(f"Warning: MediaPipe initialization failed: {e}. Using OpenCV DNN fallback.")
                    self.use_insightface = False
                    self.face_mesh = None
            
            # Final fallback: OpenCV DNN (basic face detection, no landmarks)
            if not self.use_insightface and (self.face_mesh is None or not MEDIAPIPE_AVAILABLE):
                self.opencv_detector = OpenCVDNNFaceDetector()
                print("Using OpenCV DNN for face detection (last resort - no landmarks available)")
                self.use_insightface = False
                self.face_mesh = None
        
    def detect_face(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect face and extract landmarks using InsightFace (preferred) or MediaPipe.
        
        Args:
            image: Input image (H, W, 3) in RGB
            
        Returns:
            Dictionary with landmarks and face detection info, or None
        """
        if self.use_insightface and self.face_analyzer is not None:
            # Use InsightFace (higher accuracy)
            # InsightFace expects BGR, but we have RGB, so convert
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            faces = self.face_analyzer.get(img_bgr)
            
            if len(faces) == 0:
                return None
            
            # Get the largest face
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            # InsightFace provides landmarks - check which version
            landmarks = None
            if hasattr(face, 'landmark_2d_106'):
                landmarks = face.landmark_2d_106.astype(np.float32)  # 106 landmarks (newer versions)
            elif hasattr(face, 'landmark_2d_68'):
                landmarks = face.landmark_2d_68.astype(np.float32)  # 68 landmarks
            elif hasattr(face, 'landmark'):
                landmarks = face.landmark.astype(np.float32)  # Old API (0.2.x)
            else:
                # Fallback: use bbox corners
                bbox = face.bbox
                landmarks = np.array([
                    [bbox[0], bbox[1]],  # top-left
                    [bbox[2], bbox[1]],  # top-right
                    [bbox[2], bbox[3]],  # bottom-right
                    [bbox[0], bbox[3]]   # bottom-left
                ], dtype=np.float32)
            
            # Get confidence score
            confidence = 0.9
            if hasattr(face, 'det_score'):
                confidence = float(face.det_score)
            elif hasattr(face, 'score'):
                confidence = float(face.score)
            
            return {
                'landmarks': landmarks,
                'confidence': confidence,
                'num_landmarks': len(landmarks),
                'bbox': face.bbox  # [x1, y1, x2, y2]
            }
        elif self.face_mesh is not None:
            # Use MediaPipe (fallback)
            results = self.face_mesh.process(image)
            
            if not results.multi_face_landmarks:
                return None
                
            face_landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # Convert normalized landmarks to pixel coordinates
            landmarks = np.array([
                [lm.x * w, lm.y * h] for lm in face_landmarks.landmark
            ])
            
            return {
                'landmarks': landmarks,
                'confidence': 1.0,
                'num_landmarks': len(landmarks)
            }
        else:
            # Use OpenCV DNN (last resort - only bounding box, no landmarks)
            bbox = self.opencv_detector.detect(image)
            if bbox is None:
                return None
            
            x, y, w, h = bbox
            # Create dummy landmarks from bounding box corners (for compatibility)
            landmarks = np.array([
                [x, y],           # top-left
                [x + w, y],      # top-right
                [x + w, y + h],  # bottom-right
                [x, y + h]       # bottom-left
            ], dtype=np.float32)
            
            return {
                'landmarks': landmarks,
                'confidence': 0.9,  # Lower confidence for basic detection
                'num_landmarks': 4,
                'bbox': bbox
            }
    
    def extract_roi(self, 
                    image: np.ndarray, 
                    landmark_indices: List[int],
                    landmarks: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract ROI crop with padding.
        
        Args:
            image: Input image
            landmark_indices: Indices of landmarks for this ROI
            landmarks: All face landmarks
            
        Returns:
            Cropped ROI image and bounding box (x, y, w, h)
        """
        roi_landmarks = landmarks[landmark_indices]
        
        # Compute bounding box
        x_min = int(roi_landmarks[:, 0].min())
        y_min = int(roi_landmarks[:, 1].min())
        x_max = int(roi_landmarks[:, 0].max())
        y_max = int(roi_landmarks[:, 1].max())
        
        # Add padding
        w = x_max - x_min
        h = y_max - y_min
        pad_w = int(w * self.padding_ratio)
        pad_h = int(h * self.padding_ratio)
        
        x_min = max(0, x_min - pad_w)
        y_min = max(0, y_min - pad_h)
        x_max = min(image.shape[1], x_max + pad_w)
        y_max = min(image.shape[0], y_max + pad_h)
        
        # Crop and resize
        roi = image[y_min:y_max, x_min:x_max]
        roi_resized = cv2.resize(roi, (self.roi_size, self.roi_size))
        
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        return roi_resized, bbox
    
    def parse_face(self, image: np.ndarray) -> Optional[Dict[str, FaceROI]]:
        """
        Parse face into ROIs.
        
        Args:
            image: Input image (H, W, 3) in RGB
            
        Returns:
            Dictionary mapping ROI names to FaceROI objects, or None if no face detected
        """
        face_data = self.detect_face(image)
        if face_data is None:
            return None
        
        landmarks = face_data['landmarks']
        rois = {}
        
        # Select appropriate ROI indices based on backend
        roi_indices = self.ROI_INDICES_INSIGHTFACE if self.use_insightface else self.ROI_INDICES_MEDIAPIPE
        
        for roi_name, indices in roi_indices.items():
            # Filter valid indices
            valid_indices = [i for i in indices if i < len(landmarks)]
            if len(valid_indices) < 3:  # Need at least 3 points
                continue
                
            roi_image, bbox = self.extract_roi(image, valid_indices, landmarks)
            roi_landmarks = landmarks[valid_indices]
            
            rois[roi_name] = FaceROI(
                name=roi_name,
                bbox=bbox,
                landmarks=roi_landmarks,
                confidence=face_data['confidence']
            )
        
        return rois
    
    def parse_face_batch(self, images: List[np.ndarray]) -> List[Optional[Dict[str, FaceROI]]]:
        """Parse multiple faces."""
        return [self.parse_face(img) for img in images]
    
    def split_mouth_roi(self, mouth_roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split mouth ROI into upper lip (teeth visible) and lower lip regions.
        Simple vertical split for now.
        """
        h = mouth_roi.shape[0]
        split_point = h // 2
        
        upper_lip = mouth_roi[:split_point, :]
        lower_lip = mouth_roi[split_point:, :]
        
        # Resize back to original size
        upper_lip = cv2.resize(upper_lip, (self.roi_size, self.roi_size))
        lower_lip = cv2.resize(lower_lip, (self.roi_size, self.roi_size))
        
        return upper_lip, lower_lip


class FaceParserTorch(nn.Module):
    """
    PyTorch wrapper for face parsing with batching support.
    """
    
    def __init__(self, roi_size: int = 64, padding_ratio: float = 0.2):
        super().__init__()
        self.parser = FaceParser(roi_size=roi_size, padding_ratio=padding_ratio)
        self.roi_size = roi_size
        
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, C, H, W) tensor in RGB, values [0, 1]
            
        Returns:
            Dictionary of ROI tensors: (B, C, roi_size, roi_size)
        """
        batch_size = images.shape[0]
        images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        
        roi_dict = {
            'left_eye': [],
            'right_eye': [],
            'mouth': [],
            'nose': [],
            'left_cheek': [],
            'right_cheek': [],
            'jawline': []
        }
        
        for img in images_np:
            rois = self.parser.parse_face(img)
            
            if rois is None:
                # Fallback: use whole face or zeros
                for key in roi_dict.keys():
                    roi_dict[key].append(torch.zeros(3, self.roi_size, self.roi_size))
            else:
                for key in roi_dict.keys():
                    if key in rois:
                        face_roi = rois[key]  # FaceROI object
                        # Extract ROI image from original image using bbox
                        x, y, w, h = face_roi.bbox
                        # Ensure bbox is within image bounds
                        x = max(0, min(x, img.shape[1] - 1))
                        y = max(0, min(y, img.shape[0] - 1))
                        w = min(w, img.shape[1] - x)
                        h = min(h, img.shape[0] - y)
                        
                        if w > 0 and h > 0:
                            roi_crop = img[y:y+h, x:x+w]
                            # Validate roi_crop is a valid numpy array
                            if not isinstance(roi_crop, np.ndarray) or roi_crop.size == 0:
                                roi_tensor = torch.zeros(3, self.roi_size, self.roi_size)
                            else:
                                # Ensure roi_crop has 3 channels (handle grayscale edge cases)
                                if len(roi_crop.shape) == 2:
                                    roi_crop = cv2.cvtColor(roi_crop, cv2.COLOR_GRAY2RGB)
                                elif len(roi_crop.shape) == 3 and roi_crop.shape[2] != 3:
                                    # Handle unexpected channel count
                                    roi_tensor = torch.zeros(3, self.roi_size, self.roi_size)
                                else:
                                    # Resize to roi_size
                                    roi_img = cv2.resize(roi_crop, (self.roi_size, self.roi_size))
                                    
                                    # Validate roi_img is a valid numpy array before transpose
                                    if not isinstance(roi_img, np.ndarray) or roi_img.size == 0:
                                        roi_tensor = torch.zeros(3, self.roi_size, self.roi_size)
                                    else:
                                        # Convert to tensor format (C, H, W) - already RGB, no need to convert
                                        roi_tensor = torch.from_numpy(
                                            roi_img.transpose(2, 0, 1)
                                        ).float() / 255.0
                        else:
                            roi_tensor = torch.zeros(3, self.roi_size, self.roi_size)
                        roi_dict[key].append(roi_tensor)
                    else:
                        roi_dict[key].append(torch.zeros(3, self.roi_size, self.roi_size))
        
        # Stack into batches
        return {k: torch.stack(v) for k, v in roi_dict.items()}

