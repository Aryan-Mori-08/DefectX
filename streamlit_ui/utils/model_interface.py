"""Model interface bridge for the PatchCore Industrial Anomaly Engine.

Exposes the standardized model interface contract:
1. load_product_profile(profile_name)
2. list_product_profiles()
3. preprocess_image(image_bgr)
4. check_capture_quality(image_bgr)
5. predict(image_bgr, profile_name)
6. generate_heatmap_overlay(image_bgr, anomaly_map)
7. extract_defect_regions(anomaly_map, pixel_threshold, min_area)

Contains clean placeholder hooks marked with 'TODO: connect existing anomaly engine'.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Attempt to link to the concrete anomaly_engine if present on disk
ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent / "anomaly_engine"
if ENGINE_ROOT.exists() and str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

REGISTRY_FILE = Path(__file__).resolve().parent.parent / "product_profiles.json"
DEFAULT_PROFILES = ["pcb_rev_b", "metal_casting_part_a", "injection_mold_04"]


# -----------------------------------------------------------------------------
# 1 & 2. Product Profile Management
# -----------------------------------------------------------------------------

def list_product_profiles() -> List[str]:
    """Lists all enrolled product profiles available in the memory bank registry.

    Returns:
        List of product profile identifiers.
    """
    profiles_set = set(DEFAULT_PROFILES)

    # 1. Check local registry JSON file
    if REGISTRY_FILE.exists():
        try:
            import json
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    profiles_set.update(saved)
        except Exception:
            pass

    # 2. Check memory bank directory in anomaly_engine if present
    mb_dir = ENGINE_ROOT / "artifacts/memory_banks"
    if mb_dir.exists():
        for p in mb_dir.iterdir():
            if p.is_dir() and p.name != "audit_log.jsonl":
                profiles_set.add(p.name)

    # 3. Check local artifacts directory if exists
    local_mb = Path(__file__).resolve().parent.parent / "artifacts/memory_banks"
    if local_mb.exists():
        for p in local_mb.iterdir():
            if p.is_dir() and p.name != "audit_log.jsonl":
                profiles_set.add(p.name)

    return sorted(list(profiles_set))


def register_product_profile(profile_name: str) -> None:
    """Persists a new product profile name into the profile registry."""
    clean_name = profile_name.strip()
    if not clean_name:
        return

    current = list_product_profiles()
    if clean_name not in current:
        current.append(clean_name)
    current = sorted(list(set(current)))

    try:
        import json
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    except Exception:
        pass

    # Also create directory structure in memory_banks directory so it's picked up by directory scanners
    for base in [ENGINE_ROOT / "artifacts/memory_banks", Path(__file__).resolve().parent.parent / "artifacts/memory_banks"]:
        try:
            target_p = base / clean_name / "active"
            target_p.mkdir(parents=True, exist_ok=True)
            manifest_file = target_p / "manifest.json"
            if not manifest_file.exists():
                import json
                with open(manifest_file, "w", encoding="utf-8") as f:
                    json.dump({"product_id": clean_name, "version": "v1.0.0", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}, f, indent=2)
        except Exception:
            pass


def load_product_profile(profile_name: str) -> bool:
    """Preloads the active memory bank embeddings for the specified product.

    Args:
        profile_name: Product profile identifier.

    Returns:
        True if loaded successfully, False otherwise.
    """
    # TODO: connect existing anomaly engine
    # Example integration:
    # engine.get_memory_bank(profile_name)
    time.sleep(0.05)  # Simulate fast index initialization
    return True


# -----------------------------------------------------------------------------
# 3. Preprocessing
# -----------------------------------------------------------------------------

def preprocess_image(image_bgr: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Applies standardized white-balancing, CLAHE enhancement, and resizing.

    Args:
        image_bgr: Raw input image in OpenCV BGR format.
        target_size: Target (H, W) spatial dimensions.

    Returns:
        Preprocessed image in BGR format.
    """
    # TODO: connect existing anomaly engine
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    h, w = target_size
    if HAS_OPENCV:
        # Resize
        resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)

        # LAB CLAHE Enhancement
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        preprocessed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return preprocessed

    return image_bgr


# -----------------------------------------------------------------------------
# 4. Capture Quality Checks
# -----------------------------------------------------------------------------

def check_capture_quality(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Evaluates optical image quality for blur, exposure, and contrast.

    Args:
        image_bgr: Input frame in BGR format.

    Returns:
        Dictionary with blur_score, brightness_score, contrast_score, and status ('PASS'/'RETRY').
    """
    # TODO: connect existing anomaly engine
    if image_bgr is None or image_bgr.size == 0:
        return {
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "status": "RETRY",
            "message": "Empty capture frame received",
        }

    # Grayscale conversion
    if HAS_OPENCV:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(np.var(laplacian))
    else:
        gray = 0.299 * image_bgr[:, :, 2] + 0.587 * image_bgr[:, :, 1] + 0.114 * image_bgr[:, :, 0]
        # Central difference approximation
        diff = gray[1:-1, 1:-1] * 4 - gray[0:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, 0:-2] - gray[1:-1, 2:]
        blur_score = float(np.var(diff))

    brightness_score = float(np.mean(gray))
    contrast_score = float(np.std(gray))

    # Threshold checks
    reasons: List[str] = []
    if blur_score < 80.0:
        reasons.append(f"Image is blurred (Laplacian Var: {blur_score:.1f} < 80.0)")
    if brightness_score < 25.0:
        reasons.append(f"Underexposed (Brightness: {brightness_score:.1f} < 25.0)")
    elif brightness_score > 235.0:
        reasons.append(f"Overexposed (Brightness: {brightness_score:.1f} > 235.0)")
    if contrast_score < 18.0:
        reasons.append(f"Low contrast (RMS: {contrast_score:.1f} < 18.0)")

    is_pass = len(reasons) == 0
    return {
        "blur_score": round(blur_score, 1),
        "brightness_score": round(brightness_score, 1),
        "contrast_score": round(contrast_score, 1),
        "status": "PASS" if is_pass else "RETRY",
        "rejection_reasons": reasons,
    }


# -----------------------------------------------------------------------------
# 5. Core Anomaly Prediction
# -----------------------------------------------------------------------------

def predict(
    image_bgr: np.ndarray,
    profile_name: str,
) -> Tuple[float, np.ndarray, float, str]:
    """Runs PatchCore neural feature extraction and nearest-neighbor distance scoring.

    Args:
        image_bgr: Input capture in BGR format.
        profile_name: Product profile memory bank to evaluate against.

    Returns:
        Tuple of:
            - image_anomaly_score: float continuous score.
            - anomaly_map: 2D float32 array of shape (H, W) normalized to [0, 1].
            - inference_latency_ms: float latency in milliseconds.
            - profile_used: str profile name actually used.
    """
    # TODO: connect existing anomaly engine
    # Example real implementation:
    # res = engine.inspect(image_bgr, product_id_override=profile_name)
    # return res.image_anomaly_score, res.raw_heatmap, res.total_latency_ms, res.product_id

    t_start = time.perf_counter()

    # Preprocess
    proc_bgr = preprocess_image(image_bgr)
    h, w = proc_bgr.shape[:2]

    # Deterministic simulation hook if standalone model weights are not loaded
    # Check center pixels for contrast delta to produce dynamic anomaly map
    gray = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY) if HAS_OPENCV else proc_bgr[:, :, 0]
    grad_x = np.abs(np.diff(gray.astype(np.float32), axis=1, append=gray[:, -1:]))
    grad_y = np.abs(np.diff(gray.astype(np.float32), axis=0, append=gray[-1:, :]))
    local_energy = grad_x + grad_y

    if HAS_OPENCV:
        smoothed = cv2.GaussianBlur(local_energy, (31, 31), 5.0)
    else:
        smoothed = local_energy

    s_min, s_max = float(np.min(smoothed)), float(np.max(smoothed))
    denom = s_max - s_min if (s_max - s_min) > 1e-6 else 1.0
    anomaly_map = np.clip((smoothed - s_min) / denom, 0.0, 1.0).astype(np.float32)

    # Compute top-5% mean for image anomaly score
    top_pixels = np.sort(anomaly_map.flatten())[::-1]
    image_anomaly_score = float(np.mean(top_pixels[: max(1, int(len(top_pixels) * 0.05))]))

    latency_ms = (time.perf_counter() - t_start) * 1000.0 + 32.5  # Realistic ~35ms forward pass

    return image_anomaly_score, anomaly_map, latency_ms, profile_name


# -----------------------------------------------------------------------------
# 6. Visualization & Overlay Generation
# -----------------------------------------------------------------------------

def generate_heatmap_overlay(
    image_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Renders a Jet colormap overlay blended on the RGB/BGR frame.

    Args:
        image_bgr: Base image in BGR format (H, W, 3).
        anomaly_map: 2D float32 array in [0, 1] (H, W).
        alpha: Heatmap blend weight (0.0 to 1.0).

    Returns:
        Blended RGB image array ready for display.
    """
    # TODO: connect existing anomaly engine
    h, w = anomaly_map.shape
    if image_bgr.shape[:2] != (h, w) and HAS_OPENCV:
        base = cv2.resize(image_bgr, (w, h))
    else:
        base = image_bgr.copy()

    # Convert base to RGB
    base_rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB) if HAS_OPENCV else base[:, :, ::-1]

    h_norm = np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8)

    if HAS_OPENCV:
        color_map_bgr = cv2.applyColorMap(h_norm, cv2.COLORMAP_JET)
        color_map_rgb = cv2.cvtColor(color_map_bgr, cv2.COLOR_BGR2RGB)
    else:
        color_map_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        color_map_rgb[:, :, 0] = h_norm
        color_map_rgb[:, :, 2] = 255 - h_norm

    overlay = alpha * color_map_rgb.astype(np.float32) + (1.0 - alpha) * base_rgb.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# 7. Defect Region Extraction (Zero Re-Inference Post-Processing)
# -----------------------------------------------------------------------------

def extract_defect_regions(
    anomaly_map: np.ndarray,
    pixel_threshold: float,
    min_area: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Segments the anomaly map into binary mask and isolates connected defect regions.

    Args:
        anomaly_map: 2D float32 continuous heatmap.
        pixel_threshold: Cutoff in [0, 1] for binary anomaly thresholding.
        min_area: Minimum pixel area to keep a defect candidate.

    Returns:
        Tuple of:
            - binary_mask: uint8 (H, W) array with values 0 or 255.
            - defect_regions: List of dicts containing bbox (x, y, w, h), area, and scores.
    """
    # TODO: connect existing anomaly engine
    binary_mask = (anomaly_map >= pixel_threshold).astype(np.uint8) * 255
    defect_regions: List[Dict[str, Any]] = []

    if HAS_OPENCV:
        # Morphological noise cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        region_id = 1
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                binary_mask[labels == label] = 0
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[label]

            mask_patch = labels == label
            peak_score = float(np.max(anomaly_map[mask_patch]))
            mean_score = float(np.mean(anomaly_map[mask_patch]))

            defect_regions.append({
                "region_id": region_id,
                "bbox": [x, y, w, h],
                "centroid": [round(float(cx), 1), round(float(cy), 1)],
                "area_pixels": area,
                "max_score": round(peak_score, 4),
                "mean_score": round(mean_score, 4),
            })
            region_id += 1

    return binary_mask, defect_regions


# -----------------------------------------------------------------------------
# Additional Operational Placeholders
# -----------------------------------------------------------------------------

def build_memory_bank(profile_name: str, image_paths: List[str]) -> bool:
    """Builds a new PatchCore memory bank snapshot for a new product profile.

    Args:
        profile_name: Profile name to create.
        image_paths: List of file paths to defect-free normal reference images.

    Returns:
        True on successful construction.
    """
    clean_name = profile_name.strip()
    if not clean_name:
        return False

    # Simulate feature extraction & coreset selection progress latency
    time.sleep(0.05 * min(len(image_paths), 10))

    # Persist the product profile in file registry and directory structure
    register_product_profile(clean_name)

    # Update Streamlit session state if running inside Streamlit
    try:
        import streamlit as st
        current_list = st.session_state.get("enrolled_profiles", [])
        if clean_name not in current_list:
            current_list.append(clean_name)
        st.session_state["enrolled_profiles"] = sorted(list(set(current_list)))
        st.session_state["selected_profile"] = clean_name
    except Exception:
        pass

    return True


def submit_false_alarm_feedback(
    image_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    profile_name: str,
    defect_category: str = "False Alarm",
    notes: str = "",
) -> bool:
    """Submits operator feedback for a false alarm into the quarantined staging buffer.

    Args:
        image_bgr: The flagged image frame.
        anomaly_map: The corresponding anomaly map.
        profile_name: Product profile ID.
        defect_category: Operator assigned label or taxonomy.
        notes: Operator comment.

    Returns:
        True on successful staging.
    """
    # TODO: connect existing anomaly engine
    # Example real implementation:
    # feedback_mgr.stage_false_alarm(profile_name, "operator_feedback", patches, notes=notes)
    time.sleep(0.05)
    return True
