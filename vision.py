"""
vision.py — Webcam capture, MediaPipe landmark detection, feature extraction,
             and FER emotion classification.
"""


import os
import cv2
import numpy as np
import mediapipe as mp
import time
import warnings
import logging
from fer import FER
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Optional


warnings.filterwarnings("ignore", category=UserWarning, module="keras")
logging.getLogger("py.warnings").setLevel(logging.ERROR)



# Data Structures


@dataclass
class FaceFeatures:
    """Normalized facial feature values, all in [0.0, 1.0] unless noted."""
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
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

def _ensure_face_model():
    if not os.path.exists(_MODEL_PATH):
        import urllib.request
        print("[vision] Downloading face_landmarker.task …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[vision] Model ready.")


# Keras model emotion order (matches FER-2013 training label indices)
_KERAS_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]



# Temporal Smoother


class TemporalSmoother:
    """Exponential moving average for stable feature values"""


    def __init__(self, alpha: float = 0.3, history_size: int = 5):
        self.alpha = alpha
        self.history = {}
        self.smoothed = {}
        self.history_size = history_size


    def smooth(self, features: FaceFeatures) -> FaceFeatures:
        float_fields = ["mouth_openness", "smile_width", "eyebrow_raise",
                        "eye_openness", "head_tilt"]
        for f in float_fields:
            raw = getattr(features, f)
            if f not in self.smoothed:
                self.smoothed[f] = raw
                self.history[f] = deque([raw], maxlen=self.history_size)
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


        # Camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FPS,          target_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


        # MediaPipe Face Landmarker (Tasks API — works on Windows and Linux)
        _ensure_face_model()
        _opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(_opts)
        self._detect_ts_ms   = 0
        self.show_mesh        = False
        self.show_confidence  = False


        # FER — init once, extract Keras model immediately so we never
        # run face detection on live frames (we pass pre-cropped faces)
        self._fer         = FER(mtcnn=False)
        self._keras_model = self._fer._FER__emotion_classifier


        # Smoothing
        self.smoother        = TemporalSmoother(alpha=smooth_alpha)
        self.emotion_history = deque(maxlen=emotion_smoothing)


        # State
        self.last_features       = FaceFeatures()
        self.frame: Optional[np.ndarray] = None
        self._last_mesh_results  = None   # cached MediaPipe result for mesh draw
        self._last_crop_box      = None   # cached (x1, y1, x2, y2) of last FER crop


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


        # FER performance control — run every 6 frames (~5 Hz at 30 fps)
        self._frame_count  = 0
        self._fer_every    = 6
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
        """Reset calibration so the next 30 detected frames recalibrate."""
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
        print("[Calibration] Reset — sit neutral and face the camera.")


    # Internal: FER inference on face crop


    def _classify_crop(self, frame: np.ndarray, lm, w: int, h: int):
        """
        Crop face from frame using MediaPipe landmark bounding box,
        then classify directly with the Keras model — no face detection step.
        Updates self._last_emotion, self._last_scores, and self._last_crop_box.
        """
        if self._keras_model is None:
            return


        try:
            xs = [lm[i].x * w for i in range(468)]
            ys = [lm[i].y * h for i in range(468)]
            pad = 20
            x1 = int(max(0, min(xs) - pad))
            x2 = int(min(w, max(xs) + pad))
            y1 = int(max(0, min(ys) - pad))
            y2 = int(min(h, max(ys) + pad))

            # Cache crop box so _draw_debug can visualize it
            self._last_crop_box = (x1, y1, x2, y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return


            gray  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray  = cv2.resize(gray, (64, 64)).astype("float32") / 255.0
            inp   = gray.reshape(1, 64, 64, 1)
            preds = self._keras_model.predict(inp, verbose=0)


            scores = {
                e: float(preds[0][i]) * 100
                for i, e in enumerate(_KERAS_EMOTIONS)
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
        self._last_mesh_results = results          # cache for mesh overlay draw


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
            vertical   = np.linalg.norm(pt(top)   - pt(bottom))
            horizontal = np.linalg.norm(pt(left)  - pt(right))
            return vertical / (horizontal + 1e-6)


        left_ear_val  = ear(LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,  LEFT_EYE_LEFT,  LEFT_EYE_RIGHT)
        right_ear_val = ear(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
        avg_ear = (left_ear_val + right_ear_val) / 2


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


        # FER emotion: face crop → Keras direct, every N frames
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


        # FER crop box — shows the region fed into the Keras classifier
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

            # Dashed-style crop box using corner brackets
            corner = 14
            thickness = 2
            for (cx, cy, dx, dy) in [
                (x1, y1,  1,  1),   # top-left
                (x2, y1, -1,  1),   # top-right
                (x1, y2,  1, -1),   # bottom-left
                (x2, y2, -1, -1),   # bottom-right
            ]:
                cv2.line(frame, (cx, cy), (cx + dx * corner, cy),           emo_color, thickness)
                cv2.line(frame, (cx, cy), (cx,               cy + dy * corner), emo_color, thickness)

            # "FER crop" label just above the box
            cv2.putText(frame, "FER crop", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, emo_color, 1, cv2.LINE_AA)


        # Face mesh overlay — uses cached result, no second MP inference
        if self.show_mesh and self._last_mesh_results is not None:
            if self._last_mesh_results.face_landmarks:
                for face_lms in self._last_mesh_results.face_landmarks:
                    for lm_pt in face_lms:
                        cx = int(lm_pt.x * w)
                        cy = int(lm_pt.y * h)
                        cv2.circle(frame, (cx, cy), 1, (0, 255, 100), -1)


        # Text overlay
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


        mesh_status = "ON" if self.show_mesh else "OFF"
        cv2.putText(frame, f"Mesh[M]: {mesh_status}", (w - 140, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)


        # Emotion confidence bars
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


        # Status indicators
        conf_status = "ON" if self.show_confidence else "OFF"
        cv2.putText(frame, f"Conf[C]: {conf_status}", (w - 140, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)


        return frame