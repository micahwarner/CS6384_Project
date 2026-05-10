"""
vision.py — Webcam capture, MediaPipe landmark detection, feature extraction,
             and FER emotion classification via MobileNetV3 (PyTorch).
"""

import os
import cv2
import json
import numpy as np
import mediapipe as mp
try:
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION, FACEMESH_CONTOURS
except (ModuleNotFoundError, ImportError):
    try:
        from mediapipe.solutions.face_mesh_connections import FACEMESH_TESSELATION, FACEMESH_CONTOURS
    except (ModuleNotFoundError, ImportError, AttributeError):
        # Hardcoded from mediapipe source — stable across versions
        _OVAL  = frozenset([(10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),(356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),(379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),(149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),(234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10)])
        _LIPS  = frozenset([(61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),(321,375),(375,291),(61,185),(185,40),(40,39),(39,37),(37,0),(0,267),(267,269),(269,270),(270,409),(409,291),(78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),(402,318),(318,324),(324,308),(78,191),(191,80),(80,81),(81,82),(82,13),(13,312),(312,311),(311,310),(310,415),(415,308)])
        _LEYE  = frozenset([(263,249),(249,390),(390,373),(373,374),(374,380),(380,381),(381,382),(382,362),(263,466),(466,388),(388,387),(387,386),(386,385),(385,384),(384,398),(398,362)])
        _REYE  = frozenset([(33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),(33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),(173,133)])
        _LBROW = frozenset([(276,283),(283,282),(282,295),(295,285),(300,293),(293,334),(334,296),(296,336)])
        _RBROW = frozenset([(46,53),(53,52),(52,65),(65,55),(70,63),(63,105),(105,66),(66,107)])
        FACEMESH_CONTOURS    = frozenset().union(_OVAL, _LIPS, _LEYE, _REYE, _LBROW, _RBROW)
        FACEMESH_TESSELATION = None
import torch
import torchvision.transforms as T
import time
import warnings
import logging
from collections import deque, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import timm

warnings.filterwarnings("ignore")
logging.getLogger("py.warnings").setLevel(logging.ERROR)


# Paths

MODEL_DIR       = Path("models")
MODEL_PATH      = MODEL_DIR / "best_model.pth"
CLASS_IDX_PATH  = MODEL_DIR / "class_to_idx.json"
CONFIG_PATH     = MODEL_DIR / "config.json"


# Data Structures

@dataclass
class FaceFeatures:
    """Normalized facial feature values, all in [0.0, 1.0] unless noted"""
    mouth_openness: float = 0.0
    smile_width:    float = 0.0
    eyebrow_raise:  float = 0.0
    eye_openness:   float = 0.0
    head_tilt:      float = 0.5      # 0=left, 0.5=neutral, 1=right
    emotion:        str   = "neutral"
    emotion_scores: dict  = field(default_factory=dict)
    face_detected:  bool  = False
    timestamp:      float = 0.0


# MediaPipe Landmark Indices

MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291

LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT   = 33
LEFT_EYE_RIGHT  = 133

RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263

LEFT_BROW_INNER  = 107
LEFT_BROW_OUTER  = 70
RIGHT_BROW_INNER = 336
RIGHT_BROW_OUTER = 300

NOSE_TIP   = 4
CHIN       = 152
FACE_LEFT  = 234
FACE_RIGHT = 454


# MediaPipe Tasks face landmarker model

_FACE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
_FACE_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

def _ensure_face_model():
    if not os.path.exists(_FACE_MODEL_PATH):
        import urllib.request
        print("[vision] Downloading face_landmarker.task …")
        urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)
        print("[vision] Model ready.")


# PyTorch Model Loader

def _load_model(config: dict, num_classes: int, device: torch.device) -> torch.nn.Module:
    m = timm.create_model(
        'tf_efficientnetv2_s',
        pretrained=False,
        num_classes=0
    )

    in_features = m.num_features
    m.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features, 512),
        torch.nn.Hardswish(),
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(512, num_classes)
    )

    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    m.load_state_dict(state)
    m.to(device).eval()
    return m


# Temporal Smoother

class TemporalSmoother:
    """Exponential moving average for stable feature values"""

    def __init__(self, alpha: float = 0.3, history_size: int = 5):
        self.alpha        = alpha
        self.history      = {}
        self.smoothed     = {}
        self.history_size = history_size

    def smooth(self, features: FaceFeatures) -> FaceFeatures:
        float_fields = ["mouth_openness", "smile_width", "eyebrow_raise",
                        "eye_openness", "head_tilt"]
        for f in float_fields:
            raw = getattr(features, f)
            if f not in self.smoothed:
                self.smoothed[f] = raw
                self.history[f]  = deque([raw], maxlen=self.history_size)
            else:
                self.history[f].append(raw)
                self.smoothed[f] = (
                    self.alpha * raw + (1 - self.alpha) * self.smoothed[f]
                )
            setattr(features, f, round(self.smoothed[f], 4))
        return features


# FaceProcessor

class FaceProcessor:

    def __init__(self,
                 camera_index:      int   = 0,
                 target_fps:        int   = 30,
                 smooth_alpha:      float = 0.3,
                 emotion_smoothing: int   = 6):

        # Load config, class map, model
        with open(CONFIG_PATH)    as f: self._config       = json.load(f)
        with open(CLASS_IDX_PATH) as f: self._class_to_idx = json.load(f)

        self._idx_to_class = {v: k for k, v in self._class_to_idx.items()}
        self._num_classes  = len(self._class_to_idx)
        self._img_size     = self._config.get("img_size", 112)
        self._device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = _load_model(self._config, self._num_classes, self._device)

        mean = self._config.get("mean", [0.485, 0.456, 0.406])
        std  = self._config.get("std",  [0.229, 0.224, 0.225])

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(num_output_channels=3),
            T.Resize((self._img_size, self._img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        print(f"[Model] Loaded {self._config.get('model_name', 'efficientnet_b2')} "
              f"({self._num_classes} classes) on {self._device}")
        print(f"[Model] Classes: {self._idx_to_class}")

        # Camera
        print("[DEBUG] Opening camera...")
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FPS,          target_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[DEBUG] Camera opened: {self.cap.isOpened()}")


        # MediaPipe Face Landmarker (Tasks API - works on Windows and Linux)
        _ensure_face_model()
        print("[DEBUG] face_landmarker.task ready")
        _opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_FACE_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[DEBUG] MediaPipe landmarker ready")
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(_opts)
        self._detect_ts_ms   = 0
        self.show_dots       = False
        self.show_lines      = False
        self.show_confidence = False

        # Smoothing
        self.smoother        = TemporalSmoother(alpha=smooth_alpha)
        self.emotion_history = deque(maxlen=emotion_smoothing)

        # State
        self.last_features       = FaceFeatures()
        self.frame: Optional[np.ndarray] = None
        self._last_mesh_results  = None
        self._last_crop_box      = None

        # Calibration
        self._calib_frames   = 0
        self._raw_brow_vals  = []
        self._raw_eye_vals   = []
        self._raw_mouth_vals = []
        self._raw_smile_vals = []
        self._baselines = {
            "eyebrow_raise": None, "eye_openness": None,
            "eyebrow_std":   None, "eye_std":      None,
            "mouth":         None, "mouth_std":    None,
            "smile":         None, "smile_std":    None,
        }

        # Inference throttle — every 6 frames (~5 Hz @ 30 fps)
        self._frame_count  = 0
        self._fer_every    = 10
        self._last_emotion = "neutral"
        self._last_scores  = {}


    # Public API

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def update(self) -> FaceFeatures:
        ret, frame = self.cap.read()
        if not ret:
            return self.last_features

        frame = cv2.flip(frame, 1)
        self.frame = frame.copy()
        self._frame_count += 1

        features = self._process_frame(frame)
        self._auto_calibrate(features)
        features = self.smoother.smooth(features)

        self.last_features = features
        return features

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        if self.frame is None:
            return None
        return self._draw_debug(self.frame.copy(), self.last_features)

    def release(self):
        self.cap.release()
        self.face_landmarker.close()

    def reset_calibration(self):
        self._calib_frames   = 0
        self._raw_brow_vals  = []
        self._raw_eye_vals   = []
        self._raw_mouth_vals = []
        self._raw_smile_vals = []
        self._baselines = {k: None for k in self._baselines}
        print("[Calibration] Reset — sit neutral and face the camera.")


    # Internal: EfficientNetv2S Inference

    def _classify_crop(self, frame: np.ndarray, lm, w: int, h: int):
        """
        Crop face using MediaPipe bounding box, run model inference.
        Updates self._last_emotion, self._last_scores, self._last_crop_box.
        """
        try:
            xs = [lm[i].x * w for i in range(468)]
            ys = [lm[i].y * h for i in range(468)]
            pad = 20
            x1 = int(max(0, min(xs) - pad))
            x2 = int(min(w, max(xs) + pad))
            y1 = int(max(0, min(ys) - pad))
            y2 = int(min(h, max(ys) + pad))

            self._last_crop_box = (x1, y1, x2, y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return

            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor   = self._transform(rgb_crop).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs  = torch.softmax(logits, dim=1).squeeze(0)

            scores = {
                self._idx_to_class[i]: float(probs[i]) * 100
                for i in range(self._num_classes)
            }
            self._last_emotion = max(scores, key=scores.get)
            self._last_scores  = scores

        except Exception:
            pass


    # Internal: Frame Processing

    def _process_frame(self, frame: np.ndarray) -> FaceFeatures:
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self._detect_ts_ms += 33
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.face_landmarker.detect_for_video(mp_img, self._detect_ts_ms)
        self._last_mesh_results = results

        features = FaceFeatures(timestamp=time.time())

        if not results.face_landmarks:
            features.emotion = self._get_majority_emotion("neutral")
            return features

        lm = results.face_landmarks[0]

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        features.face_detected = True

        face_height = np.linalg.norm(pt(CHIN) - pt(NOSE_TIP)) * 2.0 + 1e-6
        face_width  = np.linalg.norm(pt(FACE_RIGHT) - pt(FACE_LEFT)) + 1e-6

        # Mouth openness
        mouth_gap       = np.linalg.norm(pt(MOUTH_BOTTOM) - pt(MOUTH_TOP))
        raw_mouth_ratio = mouth_gap / (face_height * 0.15)

        if self._calib_frames < 30:
            self._raw_mouth_vals.append(raw_mouth_ratio)

        if self._baselines["mouth"] is not None:
            baseline  = self._baselines["mouth"]
            spread    = self._baselines["mouth_std"]
            raw_mouth = (raw_mouth_ratio - baseline) / max(spread, 0.01)
            raw_mouth = (raw_mouth + 1.0) / 2.0
        else:
            raw_mouth = raw_mouth_ratio
        features.mouth_openness = float(np.clip(raw_mouth, 0, 1))

        # Smile width
        mouth_w         = np.linalg.norm(pt(MOUTH_RIGHT) - pt(MOUTH_LEFT))
        raw_smile_ratio = mouth_w / face_width

        if self._calib_frames < 30:
            self._raw_smile_vals.append(raw_smile_ratio)

        if self._baselines["smile"] is not None:
            baseline  = self._baselines["smile"]
            spread    = self._baselines["smile_std"]
            raw_smile = (raw_smile_ratio - baseline) / max(spread, 0.01)
            raw_smile = (raw_smile + 1.0) / 2.0
        else:
            raw_smile = (raw_smile_ratio - 0.30) / 0.25
        features.smile_width = float(np.clip(raw_smile, 0, 1))

        # Eyebrow raise
        left_brow_mid  = (pt(LEFT_BROW_INNER)  + pt(LEFT_BROW_OUTER))  / 2
        right_brow_mid = (pt(RIGHT_BROW_INNER) + pt(RIGHT_BROW_OUTER)) / 2
        left_eye_mid   = (pt(LEFT_EYE_TOP)     + pt(LEFT_EYE_BOTTOM))  / 2
        right_eye_mid  = (pt(RIGHT_EYE_TOP)    + pt(RIGHT_EYE_BOTTOM)) / 2

        left_brow_dist  = np.linalg.norm(left_brow_mid  - left_eye_mid)
        right_brow_dist = np.linalg.norm(right_brow_mid - right_eye_mid)
        avg_brow_dist   = (left_brow_dist + right_brow_dist) / 2
        raw_brow_ratio  = avg_brow_dist / (face_height * 0.12)

        if self._calib_frames < 30:
            self._raw_brow_vals.append(raw_brow_ratio)

        if self._baselines["eyebrow_raise"] is not None:
            baseline = self._baselines["eyebrow_raise"]
            spread   = self._baselines["eyebrow_std"]
            raw_brow = (raw_brow_ratio - baseline) / max(spread, 0.01)
            raw_brow = (raw_brow + 1.0) / 2.0
        else:
            raw_brow = raw_brow_ratio
        features.eyebrow_raise = float(np.clip(raw_brow, 0, 1))

        # Eye openness
        def ear(top, bottom, left, right):
            vertical   = np.linalg.norm(pt(top)  - pt(bottom))
            horizontal = np.linalg.norm(pt(left) - pt(right))
            return vertical / (horizontal + 1e-6)

        avg_ear = (ear(LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,  LEFT_EYE_LEFT,  LEFT_EYE_RIGHT) +
                   ear(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)) / 2

        if self._calib_frames < 30:
            self._raw_eye_vals.append(avg_ear)

        if self._baselines["eye_openness"] is not None:
            baseline = self._baselines["eye_openness"]
            spread   = self._baselines["eye_std"]
            raw_eye  = (avg_ear - baseline) / max(spread, 0.005)
            raw_eye  = (raw_eye + 1.0) / 2.0
        else:
            raw_eye = (avg_ear - 0.15) / 0.25
        features.eye_openness = float(np.clip(raw_eye, 0, 1))

        # Head tilt
        left_x  = lm[FACE_LEFT].x
        right_x = lm[FACE_RIGHT].x
        nose_x  = lm[NOSE_TIP].x
        features.head_tilt = float(
            np.clip((nose_x - left_x) / (right_x - left_x + 1e-6), 0, 1)
        )

        # EfficientNet v2 inference every N frames
        if self._frame_count % self._fer_every == 0:
            self._classify_crop(frame, lm, w, h)

        features.emotion_scores = self._last_scores
        features.emotion        = self._get_majority_emotion(self._last_emotion)
        return features

    def _get_majority_emotion(self, raw: str) -> str:
        self.emotion_history.append(raw)
        return Counter(self.emotion_history).most_common(1)[0][0]

    def _auto_calibrate(self, features: FaceFeatures):
        if self._calib_frames >= 30:
            return
        if features.face_detected:
            self._calib_frames += 1
            if self._calib_frames == 30:
                self._baselines["eyebrow_raise"] = float(np.mean(self._raw_brow_vals))
                self._baselines["eyebrow_std"]   = float(max(np.std(self._raw_brow_vals)  * 2, 0.02))
                self._baselines["eye_openness"]  = float(np.mean(self._raw_eye_vals))
                self._baselines["eye_std"]       = float(max(np.std(self._raw_eye_vals)   * 2, 0.008))
                self._baselines["mouth"]         = float(np.mean(self._raw_mouth_vals))
                self._baselines["mouth_std"]     = float(max(np.std(self._raw_mouth_vals) * 2, 0.02))
                self._baselines["smile"]         = float(np.mean(self._raw_smile_vals))
                self._baselines["smile_std"]     = float(max(np.std(self._raw_smile_vals) * 2, 0.015))

                print(f"[Calibration] Brow  baseline={self._baselines['eyebrow_raise']:.3f} "
                      f"spread={self._baselines['eyebrow_std']:.3f}")
                print(f"[Calibration] Eye   baseline={self._baselines['eye_openness']:.3f} "
                      f"spread={self._baselines['eye_std']:.3f}")
                print(f"[Calibration] Mouth baseline={self._baselines['mouth']:.3f} "
                      f"spread={self._baselines['mouth_std']:.3f}")
                print(f"[Calibration] Smile baseline={self._baselines['smile']:.3f} "
                      f"spread={self._baselines['smile_std']:.3f}")


    # Draw

    def _draw_debug(self, frame: np.ndarray, f: FaceFeatures) -> np.ndarray:
        h, w = frame.shape[:2]

        # Calibration progress bar
        if self._calib_frames < 30:
            progress = self._calib_frames / 30.0
            bar_w = int(w * progress)
            cv2.rectangle(frame, (0, h - 6), (bar_w, h), (0, 255, 150), -1)
            cv2.putText(frame, f"Calibrating to your face... {self._calib_frames}/30",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (0, 255, 150), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Calibrated", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 100), 1, cv2.LINE_AA)

        # FER crop box with corner brackets
        if self._last_crop_box is not None and f.face_detected:
            x1, y1, x2, y2 = self._last_crop_box
            emo_color = {
                "happy":    (0,   200, 100),
                "sad":      (200, 100, 50),
                "angry":    (0,   50,  220),
                "surprise": (0,   200, 220),
                "fear":     (150, 0,   200),
                "disgust":  (50,  180, 50),
                "neutral":  (160, 160, 160),
            }.get(f.emotion, (160, 160, 160))

            corner, thickness = 14, 2
            for (cx, cy, dx, dy) in [
                (x1, y1,  1,  1),
                (x2, y1, -1,  1),
                (x1, y2,  1, -1),
                (x2, y2, -1, -1),
            ]:
                cv2.line(frame, (cx, cy), (cx + dx * corner, cy),               emo_color, thickness)
                cv2.line(frame, (cx, cy), (cx,               cy + dy * corner), emo_color, thickness)

            cv2.putText(frame, "FER crop", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, emo_color, 1, cv2.LINE_AA)

        # Face mesh overlay - uses cached result, no second MP inference
        if (self.show_dots or self.show_lines) and self._last_mesh_results is not None:
            if self._last_mesh_results.face_landmarks:
                for face_lms in self._last_mesh_results.face_landmarks:
                    pts = [(int(lm_pt.x * w), int(lm_pt.y * h)) for lm_pt in face_lms]
                    if self.show_lines:
                        if FACEMESH_TESSELATION is not None:
                            for a, b in FACEMESH_TESSELATION:
                                cv2.line(frame, pts[a], pts[b], (0, 255, 100), 1, cv2.LINE_AA)
                        for a, b in FACEMESH_CONTOURS:
                            cv2.line(frame, pts[a], pts[b], (0, 180, 255), 1, cv2.LINE_AA)
                    if self.show_dots:
                        for cx, cy in pts:
                            cv2.circle(frame, (cx, cy), 1, (0, 255, 100), -1)

        # Feature text overlay
        lines = [
            f"Emotion:      {f.emotion}",
            f"Mouth open:   {f.mouth_openness:.2f}",
            f"Smile width:  {f.smile_width:.2f}",
            f"Brow raise:   {f.eyebrow_raise:.2f}",
            f"Eye open:     {f.eye_openness:.2f}",
            f"Head tilt:    {f.head_tilt:.2f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 24 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 100), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Dots[M]: {'ON' if self.show_dots  else 'OFF'}", (w - 140, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Lines[L]: {'ON' if self.show_lines else 'OFF'}", (w - 140, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

        # Confidence bars
        if self.show_confidence and f.emotion_scores:
            bar_x       = w - 200
            bar_y_start = 50
            bar_max_w   = 150
            bar_h       = 16
            gap         = 22

            cv2.rectangle(frame,
                          (bar_x - 8, bar_y_start - 20),
                          (w - 4,     bar_y_start + 7 * gap),
                          (20, 20, 20), -1)
            cv2.rectangle(frame,
                          (bar_x - 8, bar_y_start - 20),
                          (w - 4,     bar_y_start + 7 * gap),
                          (60, 60, 60), 1)
            cv2.putText(frame, "Confidence", (bar_x, bar_y_start - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            emotions_ordered = ["happy", "sad", "angry", "surprise",
                                "fear", "disgust", "neutral"]
            colors = {
                "happy":    (0,   200, 100),
                "sad":      (200, 100, 50),
                "angry":    (0,   50,  220),
                "surprise": (0,   200, 220),
                "fear":     (150, 0,   200),
                "disgust":  (50,  180, 50),
                "neutral":  (160, 160, 160),
            }

            for i, emo in enumerate(emotions_ordered):
                score  = f.emotion_scores.get(emo, 0.0)
                norm   = float(np.clip(score / 100.0, 0, 1))
                fill_w = int(norm * bar_max_w)
                y      = bar_y_start + i * gap
                color  = colors.get(emo, (160, 160, 160))
                is_top = (emo == f.emotion)

                cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_w, y + bar_h),
                              (45, 45, 45), -1)
                if fill_w > 0:
                    cv2.rectangle(frame, (bar_x, y), (bar_x + fill_w, y + bar_h),
                                  color, -1)
                if is_top:
                    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_w, y + bar_h),
                                  color, 1)

                label      = f"{emo[:4]}  {score:.1f}%"
                text_color = (255, 255, 255) if is_top else (170, 170, 170)
                cv2.putText(frame, label, (bar_x - 48, y + bar_h - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)

        conf_status = "ON" if self.show_confidence else "OFF"
        cv2.putText(frame, f"Conf[C]: {conf_status}", (w - 140, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

        return frame
