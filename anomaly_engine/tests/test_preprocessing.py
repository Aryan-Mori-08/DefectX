"""Unit tests for industrial image preprocessing pipeline."""

import numpy as np
import pytest
from src.preprocessing import ImagePreprocessor


def test_preprocessing_output_dimensions():
    preprocessor = ImagePreprocessor(target_size=(256, 256), apply_clahe=False, apply_white_balance=False)
    dummy_bgr = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)

    tensor_or_arr, visual_rgb = preprocessor.process(dummy_bgr, is_bgr=True, return_torch=False)

    assert visual_rgb.shape == (256, 256, 3)
    assert tensor_or_arr.shape == (1, 3, 256, 256)
    assert tensor_or_arr.dtype == np.float32


def test_white_balance_equalizes_channel_means():
    preprocessor = ImagePreprocessor(target_size=(128, 128), apply_clahe=False, apply_white_balance=True)
    # Image with severe red tint
    tinted = np.zeros((100, 100, 3), dtype=np.uint8)
    tinted[:, :, 0] = 220  # Red
    tinted[:, :, 1] = 80   # Green
    tinted[:, :, 2] = 80   # Blue

    balanced = preprocessor.apply_gray_world_white_balance(tinted)
    mean_r = np.mean(balanced[:, :, 0])
    mean_g = np.mean(balanced[:, :, 1])
    mean_b = np.mean(balanced[:, :, 2])

    # Channels should converge closer towards common mean
    assert abs(mean_r - mean_g) < abs(220 - 80)
    assert abs(mean_r - mean_b) < abs(220 - 80)
