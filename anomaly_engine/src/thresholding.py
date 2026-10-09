"""Decoupled dual-threshold calibration engine for industrial visual inspection.

Supports separate image-level (PASS/REJECT) and pixel-level (segmentation)
thresholds, calibrated via Youden's J index, Target Zero Escape Rate, or F-beta.
Changing thresholds takes O(1) time and requires zero feature re-extraction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class ThresholdConfig:
    """Configurable threshold profile for a specific product inspection profile."""

    image_threshold: float
    pixel_threshold: float
    min_defect_area_px: int = 12
    calibration_method: str = "youden_j"
    target_defect_recall: float = 0.999
    expected_fpr: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ThresholdConfig:
        return cls(**data)


@dataclass
class ThresholdCalibrationReport:
    """Detailed summary of the statistical threshold calibration procedure."""

    recommended_image_threshold: float
    recommended_pixel_threshold: float
    best_metric_value: float
    metric_name: str
    precision_at_threshold: float
    recall_at_threshold: float
    fpr_at_threshold: float
    num_normal_samples: int
    num_defective_samples: int
    sweep_data: Dict[str, List[float]]


class ThresholdCalibrator:
    """Calibrates optimal image and pixel thresholds using validation distributions.

    Parameters:
        target_defect_recall: Target recall on defect samples for zero-escape policy.
        beta: Weighting of recall vs precision in F-beta scoring (default 2.0).
    """

    def __init__(
        self,
        target_defect_recall: float = 0.999,
        beta: float = 2.0,
    ) -> None:
        self.target_defect_recall = target_defect_recall
        self.beta = beta

    def calibrate_image_threshold(
        self,
        normal_scores: Sequence[float],
        defect_scores: Sequence[float],
        method: str = "target_recall",  # "target_recall" | "youden_j" | "f_beta"
    ) -> ThresholdCalibrationReport:
        """Computes optimal image-level decision boundary T_image.

        Args:
            normal_scores: Image-level anomaly scores from defect-free validation samples.
            defect_scores: Image-level anomaly scores from defective validation samples.
            method: Optimization criterion.

        Returns:
            ThresholdCalibrationReport with recommended boundary and metrics.
        """
        normals = np.sort(np.asarray(normal_scores, dtype=np.float64))
        defects = np.sort(np.asarray(defect_scores, dtype=np.float64))

        n_norm = len(normals)
        n_def = len(defects)

        if n_norm == 0 or n_def == 0:
            raise ValueError("Both normal and defective scores are required for calibration.")

        all_scores = np.unique(np.concatenate([normals, defects]))
        # Evaluate candidate thresholds between consecutive scores
        threshold_candidates = np.linspace(
            float(np.min(all_scores)),
            float(np.max(all_scores)),
            num=min(500, len(all_scores) * 2),
        )

        best_t = float(threshold_candidates[0])
        best_metric = -1e9
        best_p, best_r, best_fpr = 0.0, 0.0, 0.0

        sweep_recalls: List[float] = []
        sweep_fprs: List[float] = []
        sweep_thresholds: List[float] = []

        for t in threshold_candidates:
            tp = float(np.sum(defects >= t))
            fn = float(np.sum(defects < t))
            fp = float(np.sum(normals >= t))
            tn = float(np.sum(normals < t))

            recall = tp / (tp + fn + 1e-9)
            precision = tp / (tp + fp + 1e-9)
            fpr = fp / (fp + tn + 1e-9)
            specificity = tn / (tn + fp + 1e-9)

            sweep_recalls.append(recall)
            sweep_fprs.append(fpr)
            sweep_thresholds.append(float(t))

            if method == "target_recall":
                # Maximize precision / minimize false alarms while satisfying recall >= target
                if recall >= self.target_defect_recall:
                    metric = -fpr  # minimize FPR
                else:
                    metric = -1e6 - (self.target_defect_recall - recall) * 1e4
            elif method == "youden_j":
                # J = Sensitivity + Specificity - 1
                metric = recall + specificity - 1.0
            elif method == "f_beta":
                # F-beta = (1 + beta^2) * (P * R) / (beta^2 * P + R)
                b2 = self.beta ** 2
                metric = (1 + b2) * (precision * recall) / (b2 * precision + recall + 1e-9)
            else:
                raise ValueError(f"Unknown calibration method: {method}")

            if metric > best_metric:
                best_metric = metric
                best_t = float(t)
                best_p = precision
                best_r = recall
                best_fpr = fpr

        return ThresholdCalibrationReport(
            recommended_image_threshold=round(best_t, 4),
            recommended_pixel_threshold=round(best_t * 0.70, 4),  # Initial heuristic pairing
            best_metric_value=round(best_metric, 4),
            metric_name=method,
            precision_at_threshold=round(best_p, 4),
            recall_at_threshold=round(best_r, 4),
            fpr_at_threshold=round(best_fpr, 4),
            num_normal_samples=n_norm,
            num_defective_samples=n_def,
            sweep_data={
                "thresholds": sweep_thresholds,
                "recalls": sweep_recalls,
                "fprs": sweep_fprs,
            },
        )

    def calibrate_pixel_threshold(
        self,
        normal_heatmaps: Sequence[np.ndarray],
        k_sigma: float = 3.0,
    ) -> float:
        """Calibrates pixel-level threshold using normal validation heatmaps.

        Sets T_pixel to mu + k_sigma * sigma of normal pixel anomaly distribution,
        guaranteeing that 99.7% of defect-free pixels remain unsegmented.
        """
        all_pixels = np.concatenate([h.flatten() for h in normal_heatmaps])
        mean_val = float(np.mean(all_pixels))
        std_val = float(np.std(all_pixels))
        calibrated_t = mean_val + k_sigma * std_val
        return round(float(np.clip(calibrated_t, 0.05, 0.95)), 4)
