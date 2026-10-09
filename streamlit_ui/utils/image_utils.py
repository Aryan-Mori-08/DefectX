"""Image utility functions for format conversion and bounding box annotation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Converts a BGR image array to RGB."""
    if image_bgr.ndim == 2:
        return np.stack([image_bgr] * 3, axis=-1)
    if HAS_OPENCV:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr[:, :, ::-1].copy()


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """Converts an RGB image array to BGR."""
    if image_rgb.ndim == 2:
        return np.stack([image_rgb] * 3, axis=-1)
    if HAS_OPENCV:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_rgb[:, :, ::-1].copy()


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Converts a PIL Image object to an OpenCV BGR numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    return rgb_to_bgr(rgb)


def draw_bounding_boxes(
    image_rgb: np.ndarray,
    defect_regions: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (255, 0, 128),
    thickness: int = 2,
) -> np.ndarray:
    """Renders high-visibility bounding boxes and IDs over an RGB image array.

    Args:
        image_rgb: Base RGB image array (H, W, 3).
        defect_regions: List of defect dictionaries containing 'bbox' and 'region_id'.
        box_color: RGB tuple for box outline.
        thickness: Pixel thickness of rectangle outline.

    Returns:
        Annotated RGB image array.
    """
    annotated = image_rgb.copy()
    if not defect_regions:
        return annotated

    if HAS_OPENCV:
        for region in defect_regions:
            x, y, w, h = region["bbox"]
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                box_color,
                thickness,
            )
            label = f"#{region['region_id']} ({region['area_pixels']}px)"
            cv2.putText(
                annotated,
                label,
                (x, max(14, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return annotated
