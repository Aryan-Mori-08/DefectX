"""Unit tests for statistical dual-threshold calibration."""

import numpy as np
import pytest
from src.thresholding import ThresholdCalibrator


def test_threshold_calibration_separates_bimodal_distributions():
    calibrator = ThresholdCalibrator(target_defect_recall=1.0)

    # Distinct distributions: normal in [0.1, 0.3], defects in [0.7, 0.9]
    rng = np.random.RandomState(42)
    normals = rng.uniform(0.1, 0.3, size=50)
    defects = rng.uniform(0.7, 0.9, size=50)

    report = calibrator.calibrate_image_threshold(
        normal_scores=normals,
        defect_scores=defects,
        method="target_recall",
    )

    # Optimal threshold must lie between the two distributions
    assert 0.30 <= report.recommended_image_threshold <= 0.70
    assert report.recall_at_threshold == 1.0
    assert report.fpr_at_threshold == 0.0


def test_youden_j_calibration():
    calibrator = ThresholdCalibrator()
    rng = np.random.RandomState(42)
    normals = rng.normal(loc=0.2, scale=0.05, size=100)
    defects = rng.normal(loc=0.8, scale=0.05, size=100)

    report = calibrator.calibrate_image_threshold(
        normal_scores=normals,
        defect_scores=defects,
        method="youden_j",
    )

    assert 0.35 <= report.recommended_image_threshold <= 0.65
    assert report.best_metric_value > 0.95
