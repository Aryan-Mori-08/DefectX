"""Unit tests for optical capture quality gate."""

import numpy as np
import pytest
from src.capture_quality import CaptureQualityGate


def test_sharp_image_passes_quality_gate():
    gate = CaptureQualityGate(min_laplacian_variance=50.0)
    # Generate high-contrast checkerboard image with sharp edges
    h, w = 256, 256
    checker = (np.indices((h, w)).sum(axis=0) % 16 < 8).astype(np.float32) * 200.0 + 30.0
    img = np.stack([checker] * 3, axis=-1)

    report = gate.evaluate(img)
    assert report.is_valid_capture is True
    assert len(report.rejection_reasons) == 0
    assert report.metrics["laplacian_variance"] > 50.0


def test_blurred_image_is_rejected():
    gate = CaptureQualityGate(min_laplacian_variance=100.0)
    # Uniform smooth image has zero Laplacian variance
    flat_img = np.ones((256, 256, 3), dtype=np.float32) * 128.0

    report = gate.evaluate(flat_img)
    assert report.is_valid_capture is False
    assert any("blurred" in reason.lower() for reason in report.rejection_reasons)


def test_overexposed_image_is_rejected():
    gate = CaptureQualityGate(max_mean_intensity=230.0, max_overexposed_pixel_pct=0.1)
    blown_img = np.ones((256, 256, 3), dtype=np.float32) * 254.0

    report = gate.evaluate(blown_img)
    assert report.is_valid_capture is False
    assert any("overexposed" in reason.lower() for reason in report.rejection_reasons)


def test_underexposed_image_is_rejected():
    gate = CaptureQualityGate(min_mean_intensity=30.0)
    dark_img = np.ones((256, 256, 3), dtype=np.float32) * 10.0

    report = gate.evaluate(dark_img)
    assert report.is_valid_capture is False
    assert any("underexposed" in reason.lower() for reason in report.rejection_reasons)
