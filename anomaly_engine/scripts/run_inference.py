#!/usr/bin/env python3
"""Performs online PatchCore anomaly inspection on industrial images.

Usage:
    python scripts/run_inference.py --image path/to/sample.png --product my_product --save-vis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.patchcore_engine import PatchCoreEngine

try:
    import cv2
except ImportError:
    cv2 = None


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run industrial anomaly inspection on input image(s).")
    parser.add_argument("--image", type=str, required=True, help="Path to input inspection image or directory")
    parser.add_argument("--product", type=str, default=None, help="Product ID override (or auto-detected if omitted)")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--image-threshold", type=float, default=None, help="Runtime image threshold override")
    parser.add_argument("--pixel-threshold", type=float, default=None, help="Runtime pixel threshold override")
    parser.add_argument("--output-dir", type=str, default="inspection_results", help="Directory to save visual overlays")
    parser.add_argument("--save-vis", action="store_true", help="Save heatmap overlays and binary masks to disk")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    engine = PatchCoreEngine(config=config)

    input_path = Path(args.image)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            [p for p in input_path.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
        )
    else:
        print(f"[ERROR] Invalid input path: {input_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    if args.save_vis:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running Anomaly Inspection on {len(image_paths)} image(s) ===")

    for p in image_paths:
        if cv2 is not None:
            raw_bgr = cv2.imread(str(p))
        else:
            from PIL import Image
            raw_bgr = np.array(Image.open(p))[:, :, ::-1]

        if raw_bgr is None:
            print(f"[ERROR] Failed to read {p.name}")
            continue

        result = engine.inspect(
            image=raw_bgr,
            product_id_override=args.product,
            image_threshold_override=args.image_threshold,
            pixel_threshold_override=args.pixel_threshold,
            is_bgr=True,
        )

        status_str = f"[{result.decision}]"
        print(
            f"{status_str} File: {p.name} | Score: {result.image_anomaly_score:.4f} "
            f"(Threshold: {result.image_threshold:.4f}) | Defects: {result.num_defects} "
            f"| Latency: {result.total_latency_ms:.1f}ms"
        )

        if result.num_defects > 0:
            for d in result.defect_regions:
                print(f"   -> Region #{d.region_id}: BBox={d.bbox_xywh}, Area={d.area_pixels}px, MaxScore={d.max_anomaly_score:.3f}")

        if args.save_vis and result.heatmap_overlay is not None and cv2 is not None:
            vis_bgr = cv2.cvtColor(result.heatmap_overlay, cv2.COLOR_RGB2BGR)
            save_path = out_dir / f"{p.stem}_{result.decision}_overlay.png"
            cv2.imwrite(str(save_path), vis_bgr)

            mask_path = out_dir / f"{p.stem}_mask.png"
            cv2.imwrite(str(mask_path), result.binary_mask)


if __name__ == "__main__":
    main()
