"""Comprehensive evaluation suite for industrial anomaly detection.

Computes:
- Image-level AUROC and Average Precision (AUPR)
- Pixel-level AUROC (when ground truth defect masks are supplied)
- Precision, Recall, F1-Score, False Positive Rate (FPR), False Negative Rate (FNR)
- Full Confusion Matrix
- Latency and throughput (FPS) benchmarks
- Threshold sensitivity sweeps
- Sample identification of False Positives and False Negatives
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    from sklearn.metrics import (  # type: ignore
        auc,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class AnomalyEvaluationReport:
    """Comprehensive performance assessment metrics report."""

    product_id: str
    total_images: int
    num_normal: int
    num_defective: int
    image_auroc: float
    image_aupr: float
    pixel_auroc: Optional[float]
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    mean_latency_ms: float
    p95_latency_ms: float
    fps: float
    calibrated_threshold: float
    false_positive_samples: List[str]
    false_negative_samples: List[str]
    threshold_sensitivity: Dict[str, List[float]]

    def summary(self) -> str:
        cm = self.confusion_matrix
        return (
            f"=== Evaluation Report for '{self.product_id}' ===\n"
            f"Images: {self.total_images} (Normal: {self.num_normal}, Defect: {self.num_defective})\n"
            f"Image AUROC: {self.image_auroc:.4f} | AUPR: {self.image_aupr:.4f}\n"
            f"Pixel AUROC: {f'{self.pixel_auroc:.4f}' if self.pixel_auroc is not None else 'N/A'}\n"
            f"Precision: {self.precision:.4f} | Recall: {self.recall:.4f} | F1: {self.f1_score:.4f}\n"
            f"FPR: {self.false_positive_rate * 100:.2f}% | FNR: {self.false_negative_rate * 100:.2f}%\n"
            f"Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}\n"
            f"Latency: Mean={self.mean_latency_ms:.1f}ms, P95={self.p95_latency_ms:.1f}ms, FPS={self.fps:.1f}\n"
            f"False Alarms (FP): {len(self.false_positive_samples)} | Escaped Defects (FN): {len(self.false_negative_samples)}"
        )


class AnomalyEvaluator:
    """Evaluates PatchCore engine predictions against ground-truth inspection sets."""

    def __init__(self, product_id: str) -> None:
        self.product_id = product_id

    def evaluate(
        self,
        y_true: Sequence[int],
        y_scores: Sequence[float],
        threshold: float,
        image_identifiers: Optional[Sequence[str]] = None,
        latencies_ms: Optional[Sequence[float]] = None,
        pixel_heatmaps: Optional[Sequence[np.ndarray]] = None,
        pixel_masks: Optional[Sequence[np.ndarray]] = None,
    ) -> AnomalyEvaluationReport:
        """Executes full diagnostic evaluation.

        Args:
            y_true: Ground truth binary labels (0 for normal, 1 for defect).
            y_scores: Continuous image-level anomaly scores.
            threshold: Image decision threshold applied.
            image_identifiers: Names or file paths of evaluated samples.
            latencies_ms: Recorded per-image inference runtimes in milliseconds.
            pixel_heatmaps: Optional collection of 2D anomaly heatmaps.
            pixel_masks: Optional collection of ground truth 2D binary defect masks.

        Returns:
            AnomalyEvaluationReport populated with metrics and sensitivity sweeps.
        """
        y_t = np.asarray(y_true, dtype=np.int32)
        y_s = np.asarray(y_scores, dtype=np.float64)
        n_total = len(y_t)
        n_norm = int(np.sum(y_t == 0))
        n_def = int(np.sum(y_t == 1))

        # Binary decisions based on current threshold
        y_pred = (y_s >= threshold).astype(np.int32)

        # 1. Image-Level AUROC & AUPR
        if HAS_SKLEARN and len(np.unique(y_t)) > 1:
            img_auroc = float(roc_auc_score(y_t, y_s))
            prec_curve, rec_curve, _ = precision_recall_curve(y_t, y_s)
            img_aupr = float(auc(rec_curve, prec_curve))
        else:
            img_auroc = 1.0 if np.all(y_s[y_t == 1] > y_s[y_t == 0]) else 0.5
            img_aupr = 1.0

        # 2. Binary Classification Metrics
        tp = int(np.sum((y_pred == 1) & (y_t == 1)))
        tn = int(np.sum((y_pred == 0) & (y_t == 0)))
        fp = int(np.sum((y_pred == 1) & (y_t == 0)))
        fn = int(np.sum((y_pred == 0) & (y_t == 1)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        cm = [[tn, fp], [fn, tp]]

        # 3. False Positive & False Negative Identification
        names = list(image_identifiers) if image_identifiers else [f"img_{i}" for i in range(n_total)]
        fp_samples = [names[i] for i in range(n_total) if y_pred[i] == 1 and y_t[i] == 0]
        fn_samples = [names[i] for i in range(n_total) if y_pred[i] == 0 and y_t[i] == 1]

        # 4. Latency Metrics
        if latencies_ms and len(latencies_ms) > 0:
            lats = np.array(latencies_ms, dtype=np.float64)
            mean_lat = float(np.mean(lats))
            p95_lat = float(np.percentile(lats, 95))
            fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
        else:
            mean_lat, p95_lat, fps = 0.0, 0.0, 0.0

        # 5. Pixel-Level AUROC
        pixel_auroc: Optional[float] = None
        if pixel_heatmaps is not None and pixel_masks is not None and len(pixel_heatmaps) > 0:
            flat_heatmaps = np.concatenate([h.flatten() for h in pixel_heatmaps])
            flat_masks = np.concatenate([m.flatten() > 0 for m in pixel_masks]).astype(np.int32)
            if len(np.unique(flat_masks)) > 1 and HAS_SKLEARN:
                pixel_auroc = float(roc_auc_score(flat_masks, flat_heatmaps))

        # 6. Threshold Sensitivity Analysis
        sweep_thresholds = np.linspace(float(np.min(y_s)), float(np.max(y_s)), num=30).tolist()
        sweep_precisions: List[float] = []
        sweep_recalls: List[float] = []
        sweep_f1s: List[float] = []

        for th in sweep_thresholds:
            p_bin = (y_s >= th).astype(np.int32)
            c_tp = np.sum((p_bin == 1) & (y_t == 1))
            c_fp = np.sum((p_bin == 1) & (y_t == 0))
            c_fn = np.sum((p_bin == 0) & (y_t == 1))

            c_p = c_tp / (c_tp + c_fp + 1e-9)
            c_r = c_tp / (c_tp + c_fn + 1e-9)
            c_f = 2 * c_p * c_r / (c_p + c_r + 1e-9)

            sweep_precisions.append(round(float(c_p), 4))
            sweep_recalls.append(round(float(c_r), 4))
            sweep_f1s.append(round(float(c_f), 4))

        sensitivity_data = {
            "thresholds": [round(t, 4) for t in sweep_thresholds],
            "precision": sweep_precisions,
            "recall": sweep_recalls,
            "f1": sweep_f1s,
        }

        return AnomalyEvaluationReport(
            product_id=self.product_id,
            total_images=n_total,
            num_normal=n_norm,
            num_defective=n_def,
            image_auroc=round(img_auroc, 4),
            image_aupr=round(img_aupr, 4),
            pixel_auroc=round(pixel_auroc, 4) if pixel_auroc is not None else None,
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
            false_positive_rate=round(fpr, 4),
            false_negative_rate=round(fnr, 4),
            confusion_matrix=cm,
            mean_latency_ms=round(mean_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            fps=round(fps, 1),
            calibrated_threshold=round(threshold, 4),
            false_positive_samples=fp_samples,
            false_negative_samples=fn_samples,
            threshold_sensitivity=sensitivity_data,
        )
