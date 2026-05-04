"""
test_audio.py — Audio-only test: switch emotions manually, no camera needed.

Usage
─────
    python test_audio.py           # system MIDI
    python test_audio.py --synth   # software synth fallback

Keys
────
  1–7    happy / sad / angry / surprise / fear / disgust / neutral
  +/-    raise / lower simulated confidence (10–100 %)
  P      pause / resume
  Q/ESC  quit
"""

import json
import sys
import time
import math

import cv2
import numpy as np

from vision import FaceFeatures
from mapping import ExpressionMusicMapper
from audio import AudioEngine

EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]

EMOTION_COLORS = {
    "happy":    (0, 200, 100),
    "sad":      (200, 100, 50),
    "angry":    (0, 50, 220),
    "surprise": (200, 180, 0),
    "fear":     (150, 0, 200),
    "disgust":  (50, 150, 50),
    "neutral":  (180, 180, 180),
}


def make_features(emotion: str, confidence: float, t: float) -> FaceFeatures:
    """Synthesize slowly-oscillating facial features so the music varies naturally."""
    f = FaceFeatures()
    f.emotion = emotion
    f.face_detected = True
    f.timestamp = t

    f.smile_width    = float(np.clip(0.5 + 0.35 * math.sin(t * 0.7),         0.0, 1.0))
    f.mouth_openness = float(np.clip(0.4 + 0.30 * math.sin(t * 0.5 + 1.0),   0.0, 1.0))
    f.eyebrow_raise  = float(np.clip(0.5 + 0.25 * math.sin(t * 0.3 + 2.0),   0.0, 1.0))
    f.eye_openness   = float(np.clip(0.5 + 0.20 * math.sin(t * 0.4 + 0.5),   0.0, 1.0))
    f.head_tilt      = float(np.clip(0.5 + 0.15 * math.sin(t * 0.2 + 1.5),   0.0, 1.0))

    # Distribute the remaining probability evenly across other emotions
    other_score = max(0.0, (1.0 - confidence) * 100.0 / (len(EMOTIONS) - 1))
    f.emotion_scores = {
        e: (confidence * 100.0 if e == emotion else other_score)
        for e in EMOTIONS
    }
    return f


def draw_panel(emotion: str, confidence: float, params, paused: bool) -> np.ndarray:
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    color = EMOTION_COLORS.get(emotion, (180, 180, 180))

    tint = img.copy()
    cv2.rectangle(tint, (0, 0), (500, 300), color, -1)
    cv2.addWeighted(tint, 0.15, img, 0.85, 0, img)

    label = f"{'[PAUSED]  ' if paused else ''}{emotion.upper()}"
    cv2.putText(img, label, (18, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

    lines = [
        f"Confidence : {confidence * 100:.0f}%",
        f"BPM        : {params.bpm:.1f}",
        f"Instrument : {params.instrument_name}",
        f"Note / Vel : {params.midi_note}  /  {params.velocity}",
        f"Duration   : {params.note_duration:.2f} s",
    ]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (18, 90 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(img, "1:happy  2:sad  3:angry  4:surprise  5:fear", (18, 252),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(img, "6:disgust  7:neutral    +/-:confidence   P:pause   Q:quit",
                (18, 274), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1, cv2.LINE_AA)

    return img


def main():
    force_synth = "--synth" in sys.argv

    print("[test_audio] Initializing…")
    mapper = ExpressionMusicMapper(note_trigger_interval=0.4)
    engine = AudioEngine(force_synth=force_synth)

    emotion_idx      = 0
    confidence       = 0.90
    paused           = False
    t0               = time.time()
    session_start    = time.time()
    emotion_start    = time.time()
    emotion_durations: dict = {}

    # Bootstrap so draw_panel has valid params immediately
    params = mapper.map(make_features(EMOTIONS[emotion_idx], confidence, 0.0))

    cv2.namedWindow("Audio Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Audio Test", 500, 300)
    print("[test_audio] Ready — press 1–7 to switch emotions, Q to quit.")

    while True:
        t = time.time() - t0
        features = make_features(EMOTIONS[emotion_idx], confidence, t)
        params   = mapper.map(features)

        if not paused:
            engine.play(params)

        cv2.imshow("Audio Test", draw_panel(EMOTIONS[emotion_idx], confidence, params, paused))
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif ord('1') <= key <= ord('7'):
            # Flush time on the outgoing emotion before switching
            _now = time.time()
            emotion_durations[EMOTIONS[emotion_idx]] = (
                emotion_durations.get(EMOTIONS[emotion_idx], 0.0) + _now - emotion_start
            )
            emotion_start = _now
            emotion_idx = key - ord('1')
            print(f"[test_audio] → {EMOTIONS[emotion_idx]}")
        elif key in (ord('+'), ord('=')):
            confidence = min(1.0, round(confidence + 0.10, 2))
            print(f"[test_audio] Confidence: {confidence * 100:.0f}%")
        elif key == ord('-'):
            confidence = max(0.10, round(confidence - 0.10, 2))
            print(f"[test_audio] Confidence: {confidence * 100:.0f}%")
        elif key == ord('p'):
            paused = not paused
            print(f"[test_audio] {'Paused' if paused else 'Resumed'}")

        try:
            if cv2.getWindowProperty("Audio Test", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        time.sleep(0.01)

    print("[test_audio] Shutting down…")
    cv2.destroyAllWindows()
    engine.close()

    # Flush final emotion and write session stats for launcher
    try:
        _now = time.time()
        emotion_durations[EMOTIONS[emotion_idx]] = (
            emotion_durations.get(EMOTIONS[emotion_idx], 0.0) + _now - emotion_start
        )
        stats = {
            "mode": "audio",
            "session_duration": round(_now - session_start, 1),
            "emotion_durations": {
                k: round(v, 2)
                for k, v in sorted(emotion_durations.items(), key=lambda x: -x[1])
                if v >= 0.5
            },
        }
        with open("session_stats.json", "w") as _f:
            json.dump(stats, _f, indent=2)
    except Exception as _e:
        print(f"[test_audio] Could not write session stats: {_e}")


if __name__ == "__main__":
    main()
