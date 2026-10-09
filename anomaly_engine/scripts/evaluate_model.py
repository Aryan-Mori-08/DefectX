#!/usr/bin/env python3
"""Evaluates PatchCore model on validation dataset with normal and defect splits.

Usage:
    python scripts/evaluate_model.py --product pcb_rev_b --val-dir data/validation/pcb_rev_b/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import AnomalyEvaluator
from src.patchcore_engine import PatchCoreEngine

try:
    import cv2
except ImportError:
    cv2 = None


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection performance on validation set.")
    parser.add_argument("--product", type=str, required=True, help="Product ID to evaluate")
    parser.add_argument("--val-dir", type=str, required=True, help="Directory with 'good/' and 'defect/' subfolders")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--threshold", type=float, default=None, help="Explicit decision threshold (uses bank default if None)")
    parser.add_argument("--report-file", type=str, default="evaluation_report.json", help="Path to write JSON report")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    val_root = Path(args.val_dir)
    good_dir = val_root / "good"
    defect_dir = val_root / "defect"

    if not good_dir.exists() or not defect_dir.exists():
        print(f"[ERROR] Validation directory must contain 'good/' and 'defect/' subdirectories: {val_root}")
        sys.exit(1)

    image_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    good_images = []
    for ext in image_exts:
        good_images.extend(list(good_dir.glob(ext)))
    defect_images = []
    for ext in image_exts:
        defect_images.extend(list(defect_dir.glob(ext)))

    print(f"=== Evaluating '{args.product}' on {len(good_images)} normal & {len(defect_images)} defect samples ===")

    engine = PatchCoreEngine(config=config)
    bank = engine.get_memory_bank(args.product)
    threshold = args.threshold if args.threshold is not None else bank.manifest.image_threshold

    y_true = []
    y_scores = []
    identifiers = []
    latencies = []

    all_samples = [(p, 0) for p in good_images] + [(p, 1) for p in defect_images]

    for p, label in all_samples:
        if cv2 is not None:
            raw_bgr = cv2.imread(str(p))
        else:
            from PIL import Image
            raw_bgr = np.array(Image.open(p))[:, :, ::-1]

        if raw_bgr is None:
            continue

        result = engine.inspect(
            image=raw_bgr,
            product_id_override=args.product,
            image_threshold_override=threshold,
            is_bgr=True,
        )

        y_true.append(label)
        y_scores.append(result.image_anomaly_score)
        identifiers.append(p.name)
        latencies.append(result.total_latency_ms)

    evaluator = AnomalyEvaluator(product_id=args.product)
    report = evaluator.evaluate(
        y_true=y_true,
        y_scores=y_scores,
        threshold=threshold,
        image_identifiers=identifiers,
        latencies_ms=latencies,
    )

    print("\n" + report.summary())

    out_json = Path(args.report_file)
    with open(out_json, "w", encoding="utf-8") as f:
        # Convert report to serializable dict
        report_dict = {
            "product_id": report.product_id,
            "total_images": report.total_images,
            "num_normal": report.num_normal,
            "num_defective": report.num_defective,
            "image_auroc": report.image_auroc,
            "image_aupr": report.image_aupr,
            "precision": report.precision,
            "recall": report.recall,
            "f1_score": report.f1_score,
            "false_positive_rate": report.false_positive_rate,
            "false_negative_rate": report.false_negative_rate,
            "confusion_matrix": report.confusion_matrix,
            "mean_latency_ms": report.mean_latency_ms,
            "fps": report.fps,
            "calibrated_threshold": report.calibrated_threshold,
            "false_positives": report.false_positive_samples,
            "false_negatives": report.false_negative_samples,
            "threshold_sensitivity": report.threshold_sensitivity,
        }
        json.dump(report_dict, f, indent=2)

    print(f"\n[SUCCESS] Wrote full evaluation report to: {out_json}")


if __name__ == "__main__":
    main()
