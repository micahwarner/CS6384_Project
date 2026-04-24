"""
evaluation.py: Accuracy, precision, recall, F1 for emotion classification
                and latency measurements for real-time performance profiling

Usage
    from evaluation import EmotionEvaluator, LatencyProfiler

    # Emotion accuracy
    evaluator = EmotionEvaluator()
    evaluator.record(predicted="happy", ground_truth="happy")
    ...
    report = evaluator.report()

    # Latency
    profiler = LatencyProfiler()
    with profiler.measure("vision"):
        ...run vision code...
    profiler.summary()
"""

import time
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional
from contextlib import contextmanager


# Emotion Evaluator

class EmotionEvaluator:
    """
    Collects (predicted, ground_truth) pairs and computes:
      - Per-class precision, recall, F1
      - Macro-averaged precision, recall, F1
      - Overall accuracy
    """

    EMOTION_CLASSES = ["angry", "disgust", "fear", "happy",
                       "neutral", "sad", "surprise"]

    def __init__(self):
        self.records: List[tuple] = []   # (predicted, ground_truth)

    def record(self, predicted: str, ground_truth: str):
        self.records.append((predicted.lower(), ground_truth.lower()))

    def report(self, print_report: bool = True) -> Dict:
        if not self.records:
            return {}

        classes = self.EMOTION_CLASSES
        n = len(classes)

        # Build confusion matrix
        cm = np.zeros((n, n), dtype=int)
        cls_idx = {c: i for i, c in enumerate(classes)}

        for pred, gt in self.records:
            if pred in cls_idx and gt in cls_idx:
                cm[cls_idx[gt]][cls_idx[pred]] += 1

        # Derive per-class metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        precision = np.divide(tp, tp + fp,
                              out=np.zeros(n), where=(tp + fp) != 0)
        recall    = np.divide(tp, tp + fn,
                              out=np.zeros(n), where=(tp + fn) != 0)
        f1        = np.divide(2 * precision * recall,
                              precision + recall,
                              out=np.zeros(n),
                              where=(precision + recall) != 0)

        total = sum(r[0] == r[1] for r in self.records)
        accuracy = total / len(self.records)

        result = {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(float(np.mean(precision)), 4),
            "macro_recall":    round(float(np.mean(recall)), 4),
            "macro_f1":        round(float(np.mean(f1)), 4),
            "per_class": {
                c: {
                    "precision": round(float(precision[i]), 4),
                    "recall":    round(float(recall[i]), 4),
                    "f1":        round(float(f1[i]), 4),
                    "support":   int(cm.sum(axis=1)[i]),
                }
                for i, c in enumerate(classes)
            },
            "confusion_matrix": cm.tolist(),
            "n_samples": len(self.records),
        }

        if print_report:
            self._print(result, classes)

        return result

    @staticmethod
    def _print(result: Dict, classes: List[str]):
        print("\n" + "─" * 60)
        print(f"  EMOTION CLASSIFICATION REPORT  (n={result['n_samples']})")
        print("─" * 60)
        print(f"  Overall Accuracy : {result['accuracy']:.2%}")
        print(f"  Macro Precision  : {result['macro_precision']:.4f}")
        print(f"  Macro Recall     : {result['macro_recall']:.4f}")
        print(f"  Macro F1-Score   : {result['macro_f1']:.4f}")
        print()
        print(f"  {'Class':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
        print("  " + "-" * 44)
        for c in classes:
            m = result["per_class"][c]
            print(f"  {c:<12} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1']:>6.3f} {m['support']:>8}")
        print("─" * 60 + "\n")

    def save_csv(self, path: str = "emotion_eval.csv"):
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["predicted", "ground_truth"])
            writer.writerows(self.records)
        print(f"[Evaluator] Records saved → {path}")


#Latency Profiler

class LatencyProfiler:
    """
    Context-manager based latency profiler.  Tracks per-stage timing in the
    processing pipeline and prints a rolling summary.

    Stages in project:
      "capture"    — cv2 frame read
      "vision"     — MediaPipe + FER inference
      "mapping"    — ExpressionMusicMapper.map()
      "audio"      — AudioEngine.play()
      "total"      — entire loop iteration
    """

    def __init__(self, window: int = 60):
        self.window = window
        self._history: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - t0) * 1000
        hist = self._history[stage]
        hist.append(elapsed_ms)
        if len(hist) > self.window:
            hist.pop(0)

    def summary(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for stage, times in self._history.items():
            arr = np.array(times)
            result[stage] = {
                "mean_ms":   round(float(np.mean(arr)), 2),
                "p95_ms":    round(float(np.percentile(arr, 95)), 2),
                "p99_ms":    round(float(np.percentile(arr, 99)), 2),
                "max_ms":    round(float(np.max(arr)), 2),
                "n_frames":  len(arr),
            }
        return result

    def print_summary(self):
        data = self.summary()
        if not data:
            print("[Profiler] No data collected yet.")
            return
        print("\n" + "─" * 62)
        print("  LATENCY PROFILER SUMMARY")
        print("─" * 62)
        print(f"  {'Stage':<12} {'Mean':>8} {'p95':>8} {'p99':>8} {'Max':>8}  n")
        print("  " + "-" * 58)
        for stage, m in data.items():
            print(f"  {stage:<12} {m['mean_ms']:>7.1f}ms {m['p95_ms']:>7.1f}ms "
                  f"{m['p99_ms']:>7.1f}ms {m['max_ms']:>7.1f}ms  {m['n_frames']}")
        # FPS estimate from total stage
        if "total" in data and data["total"]["mean_ms"] > 0:
            fps = 1000 / data["total"]["mean_ms"]
            print(f"\n  Estimated FPS: {fps:.1f}")
        print("─" * 62 + "\n")
