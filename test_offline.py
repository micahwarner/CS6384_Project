"""
test_offline.py — Offline evaluation using FER-2013 dataset

Usage
─────
    # Against a local FER-2013 CSV (column: 'emotion' int, 'pixels' str, 'Usage' str)
    python test_offline.py --fer2013 /path/to/fer2013.csv --split PublicTest

    # Against a folder-structured dataset (subfolders named by emotion)
    python test_offline.py --testdir /path/to/test

    # Quick smoke-test with synthetic data (no dataset needed)
    python test_offline.py --smoke
"""

import argparse
import logging
import os
import random
import warnings
import numpy as np
import cv2
from evaluation import EmotionEvaluator, LatencyProfiler

warnings.filterwarnings("ignore", category=UserWarning, module="keras")
logging.getLogger("py.warnings").setLevel(logging.ERROR)

FER2013_LABELS = {0:"angry", 1:"disgust", 2:"fear", 3:"happy",
                  4:"sad", 5:"surprise", 6:"neutral"}

KERAS_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

LABEL_MAP = {
    "angry":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happy",
    "sad":      "sad",
    "surprise": "surprise",
    "neutral":  "neutral",
}


# helpers

def load_keras_model(fer_model):
    """Extract the raw Keras classifier from a FER instance (handles name mangling)."""
    # Try mangled private name first, then public fallback
    model = getattr(fer_model, "_FER__emotion_classifier", None)
    if model is None:
        model = getattr(fer_model, "emotion_classifier", None)
    return model


def predict_direct(keras_model, img_bgr: np.ndarray) -> str:
    """
    Classify emotion by feeding directly into the Keras model,
    bypassing face detection entirely

    img_bgr: any size BGR image (already cropped face)
    returns: emotion label string
    """
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray   = cv2.resize(gray, (64, 64)).astype("float32") / 255.0
    inp    = gray.reshape(1, 64, 64, 1)
    preds  = keras_model.predict(inp, verbose=0)
    return KERAS_EMOTIONS[int(np.argmax(preds))]


# Smoke Test

def smoke_test():
    """Verify the evaluation module is working with synthetic predictions."""
    print("\n[smoke] Running synthetic evaluation smoke test…")
    evaluator = EmotionEvaluator()
    np.random.seed(42)
    labels = list(FER2013_LABELS.values())
    for _ in range(500):
        gt   = np.random.choice(labels)
        pred = gt if np.random.rand() < 0.65 else np.random.choice(labels)
        evaluator.record(predicted=pred, ground_truth=gt)
    evaluator.report()


# CSV Mode

def eval_on_fer2013(csv_path: str, split: str = "PublicTest", max_samples: int = 1000):
    """Run the FER model on FER-2013 images from a CSV export and compute metrics."""
    from fer import FER
    import pandas as pd

    print(f"\n[eval] Loading FER-2013 from {csv_path} (split={split})…")
    df = pd.read_csv(csv_path)
    df = df[df["Usage"] == split].reset_index(drop=True)
    df = df.head(max_samples)
    print(f"[eval] {len(df)} samples loaded.")

    fer_model   = FER(mtcnn=False)
    keras_model = load_keras_model(fer_model)
    evaluator   = EmotionEvaluator()
    profiler    = LatencyProfiler(window=200)

    if keras_model is None:
        print("[eval] ERROR: Could not extract Keras model from FER instance.")
        return

    for i, row in df.iterrows():
        pixels  = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img_bgr = cv2.cvtColor(cv2.resize(pixels, (64, 64)), cv2.COLOR_GRAY2BGR)

        with profiler.measure("fer_inference"):
            pred = predict_direct(keras_model, img_bgr)

        gt = FER2013_LABELS[int(row["emotion"])]
        evaluator.record(predicted=pred, ground_truth=gt)

        if (i + 1) % 100 == 0:
            print(f"  …processed {i + 1}/{len(df)}")

    evaluator.report()
    profiler.print_summary()
    evaluator.save_csv("fer2013_eval_results.csv")
    print("\n[eval] Results saved to fer2013_eval_results.csv")


# ─Folder Mode

def eval_on_folder(data_dir: str, max_samples: int = 1000):
    """
    Run FER model on a folder-structured dataset.

    Expected layout:
        data_dir/
            angry/    *.jpg / *.png
            disgust/  *.jpg / *.png
            fear/     *.jpg / *.png
            happy/    *.jpg / *.png
            neutral/  *.jpg / *.png
            sad/      *.jpg / *.png
            surprise/ *.jpg / *.png
    """
    from fer import FER
    from collections import Counter

    fer_model   = FER(mtcnn=False)
    keras_model = load_keras_model(fer_model)
    evaluator   = EmotionEvaluator()
    profiler    = LatencyProfiler(window=200)

    if keras_model is None:
        print("[eval] ERROR: Could not extract Keras model from FER instance.")
        return

    # Collect all (image_path, gt_label) pairs
    samples = []
    for emotion_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, emotion_folder)
        if not os.path.isdir(folder_path):
            continue
        gt_label = LABEL_MAP.get(emotion_folder.lower())
        if gt_label is None:
            print(f"[eval] Skipping unrecognized folder: {emotion_folder}")
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(folder_path, fname), gt_label))

    if not samples:
        print(f"[eval] ERROR: No images found in {data_dir}")
        return

    random.shuffle(samples)
    samples = samples[:max_samples]

    class_counts = Counter(gt for _, gt in samples)
    print(f"\n[eval] {len(samples)} images selected from: {data_dir}")
    for label, count in sorted(class_counts.items()):
        print(f"  {label:<10} {count}")
    print()

    skipped = 0
    for i, (img_path, gt) in enumerate(samples):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            skipped += 1
            continue

        with profiler.measure("fer_inference"):
            pred = predict_direct(keras_model, img_bgr)

        evaluator.record(predicted=pred, ground_truth=gt)

        if (i + 1) % 100 == 0:
            print(f"  …processed {i + 1}/{len(samples)}")

    if skipped:
        print(f"[eval] Skipped {skipped} unreadable images.")

    evaluator.report()
    profiler.print_summary()
    evaluator.save_csv("fer2013_eval_results.csv")
    print("\n[eval] Results saved to fer2013_eval_results.csv")


# entry point

def main():
    parser = argparse.ArgumentParser(description="Offline FER emotion evaluation")
    parser.add_argument("--smoke",   action="store_true",
                        help="Run synthetic smoke test (no dataset needed)")
    parser.add_argument("--fer2013", type=str, default=None,
                        help="Path to FER-2013 CSV file")
    parser.add_argument("--testdir", type=str, default=None,
                        help="Path to folder-structured dataset (subfolders per emotion)")
    parser.add_argument("--split",   type=str, default="PublicTest",
                        help="CSV split: Training | PublicTest | PrivateTest (default: PublicTest)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Max images to evaluate (default: 1000, use 9999 for full set)")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    elif args.testdir:
        eval_on_folder(args.testdir, args.samples)
    elif args.fer2013:
        eval_on_fer2013(args.fer2013, args.split, args.samples)
    else:
        print("[test_offline] No mode specified. Use --smoke, --testdir, or --fer2013.")
        print("               Run with --help for usage details.")


if __name__ == "__main__":
    main()