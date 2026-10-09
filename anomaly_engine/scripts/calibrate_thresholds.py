#!/usr/bin/env python3
"""Calibrates optimal image and pixel decision thresholds on a validation set.

Usage:
    python scripts/calibrate_thresholds.py --product pcb_rev_b --val-dir data/validation/pcb_rev_b/ --method target_recall
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

from src.patchcore_engine import PatchCoreEngine
from src.thresholding import ThresholdCalibrator

try:
    import cv2
except ImportError:
    cv2 = None


def main():
    parser = argparse.ArgumentParser(description="Calibrate decision thresholds using validation scores.")
    parser.add_argument("--product", type=str, required=True, help="Product profile ID")
    parser.add_argument("--val-dir", type=str, required=True, help="Directory containing 'good/' and 'defect/'")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--method",
        type=str,
        default="target_recall",
        choices=["target_recall", "youden_j", "f_beta"],
        help="Calibration objective criterion",
    )
    parser.add_argument("--target-recall", type=float, default=0.999, help="Target defect recall for zero-escape policy")
    parser.add_argument("--update-bank", action="store_true", help="Update the active memory bank manifest with new thresholds")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    val_root = Path(args.val_dir)
    good_dir = val_root / "good"
    defect_dir = val_root / "defect"

    if not good_dir.exists() or not defect_dir.exists():
        print(f"[ERROR] Validation folder must contain 'good/' and 'defect/' subdirectories: {val_root}")
        sys.exit(1)

    image_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    good_images = sorted([p for ext in image_exts for p in good_dir.glob(ext)])
    defect_images = sorted([p for ext in image_exts for p in defect_dir.glob(ext)])

    print(f"=== Calibrating Thresholds for '{args.product}' ({len(good_images)} normal, {len(defect_images)} defect) ===")

    engine = PatchCoreEngine(config=config)
    normal_scores = []
    normal_heatmaps = []
    defect_scores = []

    for p in good_images:
        raw_bgr = cv2.imread(str(p)) if cv2 is not None else None
        if raw_bgr is None:
            continue
        res = engine.inspect(raw_bgr, product_id_override=args.product, is_bgr=True)
        normal_scores.append(res.image_anomaly_score)
        if res.raw_heatmap is not None:
            normal_heatmaps.append(res.raw_heatmap)

    for p in defect_images:
        raw_bgr = cv2.imread(str(p)) if cv2 is not None else None
        if raw_bgr is None:
            continue
        res = engine.inspect(raw_bgr, product_id_override=args.product, is_bgr=True)
        defect_scores.append(res.image_anomaly_score)

    calibrator = ThresholdCalibrator(target_defect_recall=args.target_recall)
    report = calibrator.calibrate_image_threshold(
        normal_scores=normal_scores,
        defect_scores=defect_scores,
        method=args.method,
    )

    if normal_heatmaps:
        pixel_t = calibrator.calibrate_pixel_threshold(normal_heatmaps, k_sigma=3.0)
    else:
        pixel_t = round(report.recommended_image_threshold * 0.70, 4)

    print("\n--- Calibration Results ---")
    print(f"Recommended Image Threshold (PASS/REJECT): {report.recommended_image_threshold:.4f}")
    print(f"Recommended Pixel Threshold (Heatmap Mask):  {pixel_t:.4f}")
    print(f"Expected Recall at Threshold:               {report.recall_at_threshold * 100:.2f}%")
    print(f"Expected Precision at Threshold:            {report.precision_at_threshold * 100:.2f}%")
    print(f"Expected False Positive Rate (False Alarm): {report.fpr_at_threshold * 100:.2f}%")

    if args.update_bank:
        bank = engine.get_memory_bank(args.product)
        if bank.manifest is not None:
            bank.manifest.image_threshold = report.recommended_image_threshold
            bank.manifest.pixel_threshold = pixel_t
            bank.save(version=bank.manifest.version, make_active=True)
            print(f"[SUCCESS] Updated active Memory Bank manifest for '{args.product}' with new thresholds.")


if __name__ == "__main__":
    main()
