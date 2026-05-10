"""
test_offline.py — Offline evaluation using the MobileNetV3 PyTorch model.

Usage
─────
    # Against a local FER-2013 CSV (columns: 'emotion' int, 'pixels' str, 'Usage' str)
    python test_offline.py --fer2013 /path/to/fer2013.csv --split PublicTest

    # Against a folder-structured dataset (subfolders named by emotion)
    python test_offline.py --testdir /path/to/test

    # Quick smoke-test with synthetic data (no dataset needed)
    python test_offline.py --smoke
"""

import argparse
import json
import logging
import os
import random
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import timm

from evaluation import EmotionEvaluator, LatencyProfiler

warnings.filterwarnings("ignore")
logging.getLogger("py.warnings").setLevel(logging.ERROR)


# Paths

MODEL_DIR      = Path("models")
MODEL_PATH     = MODEL_DIR / "best_model.pth"
CLASS_IDX_PATH = MODEL_DIR / "class_to_idx.json"
CONFIG_PATH    = MODEL_DIR / "config.json"

FER2013_LABELS = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "sad",   5: "surprise", 6: "neutral"
}

LABEL_MAP = {e: e for e in FER2013_LABELS.values()}


# Model Loading
def load_model():
    with open(CONFIG_PATH)    as f: config       = json.load(f)
    with open(CLASS_IDX_PATH) as f: class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes  = len(class_to_idx)
    img_size     = config.get("img_size", 112)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch = config.get("model_name", "tf_efficientnetv2_s") 
    m    = timm.create_model(arch, num_classes=0, pretrained=False) 

    # Match the custom head from Colab training
    in_features  = m.num_features  
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

    mean = config.get("mean", [0.485, 0.456, 0.406])
    std  = config.get("std",  [0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    print(f"[Model] {arch} ({num_classes} classes) on {device}")
    return m, transform, idx_to_class, device


# Inference

def predict(model, transform, idx_to_class, device, img_bgr: np.ndarray) -> str:
    """
    Classify a BGR face crop using efficient net v2 s
    Returns the predicted emotion label string.
    """
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)
    return idx_to_class[int(probs.argmax())]


# Smoke Test

def smoke_test():
    """Verify the evaluation module works with synthetic predictions"""
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
    """Run EfficientNetv2S on FER-2013 images from a CSV export and compute metrics."""
    import pandas as pd

    print(f"\n[eval] Loading FER-2013 from {csv_path} (split={split})…")
    df = pd.read_csv(csv_path)
    df = df[df["Usage"] == split].reset_index(drop=True)
    df = df.head(max_samples)
    print(f"[eval] {len(df)} samples loaded.")

    model, transform, idx_to_class, device = load_model()
    evaluator = EmotionEvaluator()
    profiler  = LatencyProfiler(window=200)

    for i, row in df.iterrows():
        # FER-2013 pixels are 48×48 grayscale — convert to BGR for consistency
        pixels  = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img_bgr = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)

        with profiler.measure("fer_inference"):
            pred = predict(model, transform, idx_to_class, device, img_bgr)

        gt = FER2013_LABELS[int(row["emotion"])]
        evaluator.record(predicted=pred, ground_truth=gt)

        if (i + 1) % 100 == 0:
            print(f"  …processed {i + 1}/{len(df)}")

    evaluator.report()
    profiler.print_summary()
    evaluator.save_csv("fer2013_eval_results.csv")
    print("\n[eval] Results saved to fer2013_eval_results.csv")


# Folder Mode

def eval_on_folder(data_dir: str, max_samples: int = 1000):
    """model on a folder-structured dataset

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
    model, transform, idx_to_class, device = load_model()
    evaluator = EmotionEvaluator()
    profiler  = LatencyProfiler(window=200)

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
            pred = predict(model, transform, idx_to_class, device, img_bgr)

        evaluator.record(predicted=pred, ground_truth=gt)

        if (i + 1) % 100 == 0:
            print(f"  …processed {i + 1}/{len(samples)}")

    if skipped:
        print(f"[eval] Skipped {skipped} unreadable images.")

    evaluator.report()
    profiler.print_summary()
    evaluator.save_csv("folder_eval_results.csv")
    print("\n[eval] Results saved to folder_eval_results.csv")


# Entry Point

def main():
    parser = argparse.ArgumentParser(description="Offline MobileNetV3 emotion evaluation")
    parser.add_argument("--smoke",   action="store_true",
                        help="Run synthetic smoke test (no dataset needed)")
    parser.add_argument("--fer2013", type=str, default=None,
                        help="Path to FER-2013 CSV file")
    parser.add_argument("--testdir", type=str, default=None,
                        help="Path to folder-structured dataset (subfolders per emotion)")
    parser.add_argument("--split",   type=str, default="PublicTest",
                        help="CSV split: Training | PublicTest | PrivateTest  (default: PublicTest)")
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