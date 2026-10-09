"""Defect localization and spatial anomaly mapping for industrial inspection.

Transforms 1D patch nearest-neighbor distances into smoothed pixel-level
heatmaps, extracts binary defect segmentations, and isolates connected
components into bounded defects with geometric and severity metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2  # type: ignore
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from scipy.ndimage import gaussian_filter, zoom  # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass(frozen=True)
class DefectRegion:
    """Localized industrial defect candidate extracted from connected-component analysis."""

    region_id: int
    bbox_xywh: Tuple[int, int, int, int]  # (x, y, width, height)
    centroid: Tuple[float, float]          # (cx, cy)
    area_pixels: int
    max_anomaly_score: float
    mean_anomaly_score: float
    severity_grade: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bbox": list(self.bbox_xywh),
            "centroid": [round(c, 2) for c in self.centroid],
            "area_pixels": self.area_pixels,
            "max_score": round(self.max_anomaly_score, 4),
            "mean_score": round(self.mean_anomaly_score, 4),
            "severity_grade": round(self.severity_grade, 4),
        }


@dataclass
class LocalizationArtifacts:
    """Complete localization outputs for visualization and defect reporting."""

    raw_heatmap: np.ndarray             # float32 (H, W) in [0, 1]
    binary_mask: np.ndarray             # uint8 (H, W) 0 or 255
    heatmap_overlay: np.ndarray         # uint8 (H, W, 3) blended RGB image
    defect_regions: List[DefectRegion]
    total_defect_area: int
    num_defects: int


class DefectLocalizer:
    """Generates continuous heatmaps, binary masks, and localized defect regions.

    Parameters:
        gaussian_sigma: Standard deviation for Gaussian smoothing kernel.
        min_defect_area_px: Minimum component pixel area to qualify as a defect.
        overlay_alpha: Blending weight for the Jet colormap overlay (0.0 to 1.0).
    """

    def __init__(
        self,
        gaussian_sigma: float = 4.0,
        min_defect_area_px: int = 12,
        overlay_alpha: float = 0.45,
    ) -> None:
        self.gaussian_sigma = float(gaussian_sigma)
        self.min_defect_area_px = int(min_defect_area_px)
        self.overlay_alpha = float(overlay_alpha)

    def generate_heatmap(
        self,
        patch_scores: np.ndarray,
        grid_shape: Tuple[int, int],
        target_size: Tuple[int, int],
        score_min: Optional[float] = None,
        score_max: Optional[float] = None,
    ) -> np.ndarray:
        """Upsamples and smooths patch scores into a dense 2D pixel anomaly map.

        Args:
            patch_scores: 1D array of length (H_p * W_p) containing anomaly distances.
            grid_shape: (H_p, W_p) dimensions of feature extractor patch grid.
            target_size: (H, W) target image resolution.
            score_min: Optional lower bound for min-max normalization.
            score_max: Optional upper bound for min-max normalization.

        Returns:
            Smooth 2D float32 heatmap array with values in [0, 1].
        """
        hp, wp = grid_shape
        h_target, w_target = target_size

        grid = patch_scores.reshape((hp, wp)).astype(np.float32)

        # 1. Bilinear upsampling to target image size
        if HAS_OPENCV:
            upsampled = cv2.resize(grid, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
        elif HAS_SCIPY:
            scale_y = h_target / hp
            scale_x = w_target / wp
            upsampled = zoom(grid, (scale_y, scale_x), order=1).astype(np.float32)
        else:
            upsampled = np.repeat(np.repeat(grid, h_target // hp, axis=0), w_target // wp, axis=1)

        # 2. Gaussian smoothing to remove patch tile boundary discontinuities
        if HAS_OPENCV:
            kernel_size = int(2 * np.ceil(2 * self.gaussian_sigma) + 1)
            smoothed = cv2.GaussianBlur(upsampled, (kernel_size, kernel_size), self.gaussian_sigma)
        elif HAS_SCIPY:
            smoothed = gaussian_filter(upsampled, sigma=self.gaussian_sigma)
        else:
            smoothed = upsampled

        # 3. Dynamic or Fixed Normalization to [0, 1]
        s_min = score_min if score_min is not None else float(np.min(smoothed))
        s_max = score_max if score_max is not None else float(np.max(smoothed))
        denom = s_max - s_min if (s_max - s_min) > 1e-6 else 1.0

        normalized = np.clip((smoothed - s_min) / denom, 0.0, 1.0)
        return normalized.astype(np.float32)

    def create_jet_overlay(
        self,
        heatmap: np.ndarray,
        base_rgb: np.ndarray,
    ) -> np.ndarray:
        """Renders a Jet colormap heatmap superimposed on the RGB inspection frame."""
        h_norm = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)

        if HAS_OPENCV:
            color_map = cv2.applyColorMap(h_norm, cv2.COLORMAP_JET)
            color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        else:
            # Simple pseudo-jet mapping fallback
            color_map_rgb = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
            color_map_rgb[:, :, 0] = h_norm  # Red
            color_map_rgb[:, :, 2] = 255 - h_norm  # Blue

        # Alpha blend: overlay = alpha * color_map + (1 - alpha) * base_image
        overlay = (
            self.overlay_alpha * color_map_rgb.astype(np.float32)
            + (1.0 - self.overlay_alpha) * base_rgb.astype(np.float32)
        )
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def extract_regions(
        self,
        heatmap: np.ndarray,
        pixel_threshold: float,
        base_rgb: np.ndarray,
    ) -> LocalizationArtifacts:
        """Extracts connected defect regions and generates overlay visualization.

        Args:
            heatmap: Smoothed float32 2D heatmap in [0, 1].
            pixel_threshold: Anomaly threshold for binary segmentation.
            base_rgb: Preprocessed RGB image (H, W, 3).

        Returns:
            LocalizationArtifacts containing masks, overlays, and detected regions.
        """
        h, w = heatmap.shape

        # 1. Binary segmentation
        binary_mask = (heatmap >= pixel_threshold).astype(np.uint8) * 255

        # 2. Morphological cleaning: Remove single-pixel noise, close tiny holes
        if HAS_OPENCV:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Connected-Component Analysis
        defect_regions: List[DefectRegion] = []
        total_defect_area = 0

        if HAS_OPENCV:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )

            region_counter = 1
            for label_id in range(1, num_labels):  # Skip 0 (background)
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < self.min_defect_area_px:
                    # Clear out small noise region from binary mask
                    binary_mask[labels == label_id] = 0
                    continue

                x = int(stats[label_id, cv2.CC_STAT_LEFT])
                y = int(stats[label_id, cv2.CC_STAT_TOP])
                width = int(stats[label_id, cv2.CC_STAT_WIDTH])
                height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
                cx, cy = centroids[label_id]

                comp_mask = labels == label_id
                comp_scores = heatmap[comp_mask]
                max_score = float(np.max(comp_scores))
                mean_score = float(np.mean(comp_scores))

                # Severity: (Peak Score / Threshold) * log10(1 + Area)
                severity = (max_score / max(pixel_threshold, 1e-4)) * float(np.log10(1.0 + area))

                region = DefectRegion(
                    region_id=region_counter,
                    bbox_xywh=(x, y, width, height),
                    centroid=(float(cx), float(cy)),
                    area_pixels=area,
                    max_anomaly_score=max_score,
                    mean_anomaly_score=mean_score,
                    severity_grade=severity,
                )
                defect_regions.append(region)
                total_defect_area += area
                region_counter += 1
        else:
            # Fallback simple region extraction if opencv not loaded
            active_pixels = np.sum(binary_mask > 0)
            if active_pixels >= self.min_defect_area_px:
                y_indices, x_indices = np.where(binary_mask > 0)
                x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
                y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
                comp_scores = heatmap[binary_mask > 0]

                defect_regions.append(
                    DefectRegion(
                        region_id=1,
                        bbox_xywh=(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1),
                        centroid=(float(np.mean(x_indices)), float(np.mean(y_indices))),
                        area_pixels=int(active_pixels),
                        max_anomaly_score=float(np.max(comp_scores)),
                        mean_anomaly_score=float(np.mean(comp_scores)),
                        severity_grade=float(np.max(comp_scores) * np.log10(1.0 + active_pixels)),
                    )
                )
                total_defect_area = int(active_pixels)

        # 4. Generate colored heatmap overlay
        overlay = self.create_jet_overlay(heatmap, base_rgb)

        # 5. Draw bounding boxes on overlay for visual confirmation
        if HAS_OPENCV:
            for d in defect_regions:
                x, y, bw, bh = d.bbox_xywh
                # Draw high-visibility magenta/cyan bounding box
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 0, 128), 2)
                tag = f"#{d.region_id} S:{d.max_anomaly_score:.2f}"
                cv2.putText(
                    overlay,
                    tag,
                    (x, max(15, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return LocalizationArtifacts(
            raw_heatmap=heatmap,
            binary_mask=binary_mask,
            heatmap_overlay=overlay,
            defect_regions=defect_regions,
            total_defect_area=total_defect_area,
            num_defects=len(defect_regions),
        )
