"""Unit tests for heatmap synthesis and connected component localization."""

import numpy as np
import pytest
from src.localization import DefectLocalizer


def test_heatmap_upsampling_and_normalization():
    localizer = DefectLocalizer(gaussian_sigma=2.0)
    # 4x4 patch grid -> 64x64 target resolution
    patch_scores = np.random.uniform(0.1, 0.9, size=16).astype(np.float32)

    heatmap = localizer.generate_heatmap(
        patch_scores=patch_scores,
        grid_shape=(4, 4),
        target_size=(64, 64),
    )

    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_connected_component_extraction_finds_injected_defect():
    localizer = DefectLocalizer(min_defect_area_px=10)
    # Background zeros, with a 20x20 defect block in center
    h, w = 100, 100
    heatmap = np.zeros((h, w), dtype=np.float32)
    heatmap[30:50, 40:60] = 0.85  # 20x20 = 400 pixels

    base_rgb = np.ones((h, w, 3), dtype=np.uint8) * 128

    artifacts = localizer.extract_regions(
        heatmap=heatmap,
        pixel_threshold=0.50,
        base_rgb=base_rgb,
    )

    assert artifacts.num_defects >= 1
    defect = artifacts.defect_regions[0]
    assert defect.area_pixels >= 100
    assert defect.max_anomaly_score >= 0.80
    assert defect.severity_grade > 0.0
    assert artifacts.total_defect_area >= 100
