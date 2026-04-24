"""
main.py — Entry point for the Real-Time Facial Expression Music Generator.

Run
───
    python main.py                   # normal mode
    python main.py --synth           # force software synth (no system MIDI)
    python main.py --eval            # enable emotion evaluation mode
    python main.py --latency         # print latency stats every 5 seconds
    python main.py --no-audio        # vision-only (debug CV without sound)

Controls (OpenCV window):
    Q / ESC   — quit
    P         — pause / resume audio
    R         — reset evaluator & profiler
    S         — print latency summary now
    E         — print emotion evaluation report now
    L         — cycle emotion labels (for building ground-truth dataset)
    M         — toggle overlay mesh
    C         — toggle emotion confidence bars
    F         — toggle fullscreen on/off
    TAB       — recalibrate
"""

import argparse
import sys
import time
import threading

import cv2
import numpy as np

from vision import FaceProcessor
from mapping import ExpressionMusicMapper
from audio import AudioEngine
from evaluation import EmotionEvaluator, LatencyProfiler


# Arguments

def parse_args():
    p = argparse.ArgumentParser(description="Facial Expression Music Generator")
    p.add_argument("--synth",    action="store_true",
                   help="Force software synthesizer (no system MIDI required)")
    p.add_argument("--eval",     action="store_true",
                   help="Enable emotion evaluation (manual ground-truth labelling)")
    p.add_argument("--latency",  action="store_true",
                   help="Print latency profiler summary every 5 seconds")
    p.add_argument("--no-audio", action="store_true",
                   help="Disable audio output (CV debug mode)")
    p.add_argument("--camera",   type=int, default=0,
                   help="Camera device index (default: 0)")
    p.add_argument("--fps",      type=int, default=30,
                   help="Target frame rate (default: 30)")
    p.add_argument("--smooth",   type=float, default=0.3,
                   help="Feature smoothing alpha 0–1 (lower=smoother, default: 0.3)")
    p.add_argument("--interval", type=float, default=0.4,
                   help="Minimum seconds between note triggers (default: 0.4)")
    return p.parse_args()


# Render HUD

EMOTION_COLORS = {
    "happy":    (0, 200, 100),
    "sad":      (200, 100, 50),
    "angry":    (0, 50, 220),
    "surprise": (200, 180, 0),
    "fear":     (150, 0, 200),
    "disgust":  (50, 150, 50),
    "neutral":  (180, 180, 180),
}

def draw_hud(frame: np.ndarray, features, params, backend: str,
             fps: float, paused: bool, gt_label: str = "") -> np.ndarray:
    """Draw info overlay on the bottom of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Translucent bottom bar
    bar_h = 110
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    emo_color = EMOTION_COLORS.get(features.emotion, (180, 180, 180))

    # Emotion label
    cv2.putText(frame,
                f"{'[PAUSED] ' if paused else ''}{features.emotion.upper()}",
                (14, h - bar_h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, emo_color, 2, cv2.LINE_AA)

    # Instrument & BPM
    cv2.putText(frame,
                f"{params.instrument_name}  |  {params.bpm:.0f} BPM  |  "
                f"Note: {params.midi_note}  |  {backend}",
                (14, h - bar_h + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA)

    # Feature mini bars
    bar_fields = [
        ("Mouth", features.mouth_openness),
        ("Smile", features.smile_width),
        ("Brow",  features.eyebrow_raise),
        ("Eye",   features.eye_openness),
    ]
    bar_x0 = 14
    bar_y  = h - bar_h + 80
    bar_w  = 70
    bar_max_h = 16
    spacing = 90
    for label, val in bar_fields:
        fill = int(val * bar_w)
        cv2.rectangle(frame, (bar_x0, bar_y),
                      (bar_x0 + bar_w, bar_y + bar_max_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x0, bar_y),
                      (bar_x0 + fill, bar_y + bar_max_h), emo_color, -1)
        cv2.putText(frame, label, (bar_x0, bar_y + bar_max_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        bar_x0 += spacing

    # FPS
    cv2.putText(frame, f"{fps:.0f} FPS",
                (w - 75, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180),
                1, cv2.LINE_AA)

    # Ground truth label during eval mode
    if gt_label:
        cv2.putText(frame, f"GT: {gt_label}",
                    (w - 130, h - bar_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 200, 255), 2, cv2.LINE_AA)

    return frame


# main loop

GT_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def main():
    args = parse_args()

    #Initialize components
    print("[main] Initializing Face Processor…")
    processor = FaceProcessor(
        camera_index=args.camera,
        target_fps=args.fps,
        smooth_alpha=args.smooth,
    )
    if not processor.is_opened():
        print("[main] ERROR: Cannot open camera. Check --camera index.")
        sys.exit(1)

    print("[main] Initializing Mapper…")
    mapper = ExpressionMusicMapper(note_trigger_interval=args.interval)

    print("[main] Initializing Audio Engine…")
    if args.no_audio:
        engine = None
        print("[main] Audio disabled (--no-audio).")
    else:
        engine = AudioEngine(force_synth=args.synth)

    profiler = LatencyProfiler(window=90)
    evaluator = EmotionEvaluator()

    # State
    paused = False
    gt_idx = 0          # index into GT_LABELS for eval mode
    last_latency_print = time.time()
    frame_count = 0
    fps_acc = 0.0
    fps_display = 0.0
    t_prev = time.time()
    fullscreen = False

    print("\n[main] Starting main loop. Press Q/ESC to quit.\n")

    cv2.namedWindow("Facial Expression Music Generator", cv2.WINDOW_NORMAL)

    # Main loop
    while True:
        try:
            t_loop_start = time.perf_counter()

            # 1. Capture & Vision
            with profiler.measure("total"):
                with profiler.measure("vision"):
                    features = processor.update()

                frame = processor.get_annotated_frame()
                if frame is None:
                    continue

                # 2. Mapping
                with profiler.measure("mapping"):
                    params = mapper.map(features)

                # 3. Audio
                if engine and not paused:
                    with profiler.measure("audio"):
                        engine.play(params)

                # 4. Evaluation recording
                if args.eval and features.face_detected:
                    evaluator.record(
                        predicted=features.emotion,
                        ground_truth=GT_LABELS[gt_idx]
                    )

            # FPS calculation
            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps_acc = 0.9 * fps_acc + 0.1 * (1.0 / dt)
                fps_display = fps_acc
            frame_count += 1

            # Draw HUD
            gt_str = GT_LABELS[gt_idx] if args.eval else ""
            frame = draw_hud(frame, features, params,
                            engine.backend_name if engine else "off",
                            fps_display, paused, gt_str)

            cv2.imshow("Facial Expression Music Generator", frame)

            # Keyboard Handling
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):       # Q or ESC: quit
                break

            # Check if window was closed with X button
            try:
                if cv2.getWindowProperty("Facial Expression Music Generator",
                                        cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key == ord('p'):           # P: pause/resume
                paused = not paused
                print(f"[main] {'Paused' if paused else 'Resumed'}.")
            elif key == ord('r'):           # R: reset stats
                profiler = LatencyProfiler(window=90)
                evaluator = EmotionEvaluator()
                print("[main] Stats reset.")
            elif key == ord('s'):           # S: print latency
                profiler.print_summary()
            elif key == ord('e'):           # E: print emotion eval
                if evaluator.records:
                    evaluator.report()
                else:
                    print("[main] No evaluation records yet.")
            elif key == ord('l') and args.eval:  # L: cycle GT label
                gt_idx = (gt_idx + 1) % len(GT_LABELS)
                print(f"[main] Ground-truth label → {GT_LABELS[gt_idx]}")
            elif key == ord('m'):           # M: toggle mesh overlay
                processor.face_mesh  # just access to confirm alive
                processor.show_mesh = not processor.show_mesh
                print(f"[main] Mesh overlay {'ON' if processor.show_mesh else 'OFF'}.")
            elif key == ord('c'):           # C: toggle confidence bars
                processor.show_confidence = not processor.show_confidence
                print(f"[main] Confidence bars {'ON' if processor.show_confidence else 'OFF'}.")
            elif key == ord('f'):           # F: toggle fullscreen
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty("Facial Expression Music Generator",
                                        cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty("Facial Expression Music Generator",
                                        cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print(f"[main] Fullscreen {'ON' if fullscreen else 'OFF'}.")
            elif key == 9:                  # Tab: recalibrate
                processor.reset_calibration()

            # periodic latency print
            if args.latency and (now - last_latency_print) >= 5.0:
                profiler.print_summary()
                last_latency_print = now

            # Frame-rate cap
            elapsed = time.perf_counter() - t_loop_start
            target  = 1.0 / args.fps
            if elapsed < target:
                time.sleep(target - elapsed)
                
        except Exception as e:
            print(f"[main] Frame error (skipping): {e}")
            continue

    #Cleanup
    print("\n[main] Shutting down…")

    print("[main] destroyAllWindows()...")
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print("[main] processor.release()...")
    processor.release()
    print("[main] processor.release() done.")

    if engine:
        print("[main] engine.close()...")
        engine.close()
        print("[main] engine.close() done.")

    if evaluator.records:
        print("[main] Final evaluation report:")
        evaluator.report()

    print("[main] profiler.print_summary()...")
    profiler.print_summary()
    print("[main] Done.")


if __name__ == "__main__":
    main()
