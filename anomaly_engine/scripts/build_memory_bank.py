#!/usr/bin/env python3
"""Builds a calibrated Coreset Memory Bank from 10-30 defect-free reference images.

Usage:
    python scripts/build_memory_bank.py --product pcb_rev_b --ref-dir data/reference/pcb_rev_b/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture_quality import CaptureQualityGate
from src.feature_extractor import FeatureExtractor
from src.memory_bank import ProductMemoryBank
from src.preprocessing import ImagePreprocessor
from src.product_classifier import ProductClassifier

try:
    import cv2
except ImportError:
    cv2 = None


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Profile a product type and construct its PatchCore Memory Bank."
    )
    parser.add_argument("--product", type=str, required=True, help="Unique product profile ID")
    parser.add_argument("--ref-dir", type=str, required=True, help="Path to defect-free normal images")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Version tag for new snapshot")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Root directory for artifacts")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    ref_path = Path(args.ref_dir)
    if not ref_path.exists():
        print(f"[ERROR] Reference directory does not exist: {ref_path}")
        sys.exit(1)

    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(ref_path.glob(ext)))
    image_files = sorted(image_files)

    if len(image_files) < 1:
        print(f"[ERROR] Found {len(image_files)} reference images. At least 5-10 images are recommended.")
        sys.exit(1)

    print(f"=== Profiling Product: '{args.product}' with {len(image_files)} reference images ===")

    # Initialize Modules
    quality_gate = CaptureQualityGate(
        min_laplacian_variance=config["quality_gate"]["min_laplacian_variance"],
        min_mean_intensity=config["quality_gate"]["min_mean_intensity"],
        max_mean_intensity=config["quality_gate"]["max_mean_intensity"],
    )

    preprocessor = ImagePreprocessor(
        target_size=tuple(config["preprocessing"]["target_size"]),
        apply_clahe=config["preprocessing"]["apply_clahe"],
        clahe_clip_limit=config["preprocessing"]["clahe_clip_limit"],
        apply_white_balance=config["preprocessing"]["apply_white_balance"],
    )

    feature_extractor = FeatureExtractor(
        backbone_name=config["model"]["backbone_name"],
        extracted_layers=config["model"]["extracted_layers"],
        device=config["system"]["device"],
        half_precision=config["system"]["precision"] == "fp16",
    )

    all_patch_features = []
    global_descriptors = []
    accepted_images = 0

    for img_path in image_files:
        if cv2 is not None:
            raw_img = cv2.imread(str(img_path))
        else:
            from PIL import Image
            raw_img = np.array(Image.open(img_path))[:, :, ::-1]  # BGR

        if raw_img is None:
            print(f"[WARN] Skipping corrupted image: {img_path.name}")
            continue

        report = quality_gate.evaluate(raw_img)
        if not report.is_valid_capture:
            print(f"[WARN] Rejected capture {img_path.name}: {report.rejection_reasons}")
            continue

        tensor, _ = preprocessor.process(raw_img, is_bgr=True)
        patch_feats, global_feat, spatial_shape = feature_extractor.extract_features(tensor)

        all_patch_features.append(patch_feats.detach().cpu().numpy())
        global_descriptors.append(global_feat.detach().cpu().numpy().flatten())
        accepted_images += 1

    if accepted_images == 0:
        print("[ERROR] All candidate reference images failed the capture quality gate.")
        sys.exit(1)

    stacked_patches = np.concatenate(all_patch_features, axis=0)
    print(f"Extracted {stacked_patches.shape[0]} candidate patch vectors (dim: {stacked_patches.shape[1]}).")

    # Compute prototype centroid
    mean_prototype = np.mean(np.stack(global_descriptors, axis=0), axis=0)
    mean_prototype /= np.linalg.norm(mean_prototype)

    # Subsample Coreset and construct Memory Bank
    memory_bank = ProductMemoryBank(
        product_id=args.product,
        artifacts_dir=Path(args.artifacts_dir) / "memory_banks",
    )

    coreset_ratio = config["memory_bank"]["coreset_sampling_ratio"]
    proj_dim = config["memory_bank"]["projection_dimension"]

    print(f"Applying greedy coreset subsampling (ratio: {coreset_ratio}, target: ~{int(stacked_patches.shape[0] * coreset_ratio)})...")
    manifest = memory_bank.fit_from_patches(
        raw_patches=stacked_patches,
        reference_count=accepted_images,
        backbone_name=config["model"]["backbone_name"],
        layers=config["model"]["extracted_layers"],
        input_resolution=tuple(config["preprocessing"]["target_size"]),
        image_threshold=config["inference"]["image_threshold"],
        pixel_threshold=config["inference"]["pixel_threshold"],
        coreset_ratio=coreset_ratio,
        projection_dim=proj_dim,
        product_prototype=mean_prototype,
        version=args.version,
    )

    saved_dir = memory_bank.save(version=args.version, make_active=True)
    print(f"[SUCCESS] Saved active Memory Bank snapshot to: {saved_dir}")

    # Register product prototype for zero-shot classifier routing
    classifier = ProductClassifier(
        registry_path=Path(args.artifacts_dir) / "models/product_prototypes.json"
    )
    classifier.register_product(args.product, mean_prototype, overwrite=True)
    print(f"[SUCCESS] Registered product prototype for '{args.product}' in router registry.")


if __name__ == "__main__":
    main()
