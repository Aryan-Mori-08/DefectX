"""Deterministic preprocessing pipeline for industrial optical inspection.

Applies white-balance constancy, exposure normalization, CLAHE enhancement,
and standard ImageNet normalization for deep feature extraction backbones.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np

try:
    import cv2  # type: ignore
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import torch  # type: ignore
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ImagePreprocessor:
    """Preprocesses camera captures into standardized neural network inputs.

    Parameters:
        target_size: (H, W) spatial dimensions for network input.
        apply_clahe: Whether to apply contrast-limited adaptive histogram equalization.
        clahe_clip_limit: Threshold for contrast limiting in CLAHE.
        clahe_grid_size: Tile grid dimensions for local CLAHE neighborhoods.
        apply_white_balance: Whether to apply Gray-World color constancy.
        mean: Channel-wise mean for normalization (ImageNet default).
        std: Channel-wise standard deviation for normalization (ImageNet default).
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        apply_white_balance: bool = True,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_grid_size = (int(clahe_grid_size[0]), int(clahe_grid_size[1]))
        self.apply_white_balance = apply_white_balance
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

        if HAS_OPENCV and self.apply_clahe:
            self._clahe_op = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size,
            )
        else:
            self._clahe_op = None

    def ensure_rgb(self, image: np.ndarray, is_bgr: bool = True) -> np.ndarray:
        """Converts raw camera buffer to RGB format."""
        if image.ndim == 2:
            return np.stack([image] * 3, axis=-1)
        if image.shape[2] == 4:
            # Drop alpha channel
            image = image[:, :, :3]
        if is_bgr and HAS_OPENCV:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif is_bgr:
            return image[:, :, ::-1].copy()
        return image.copy()

    def apply_gray_world_white_balance(self, rgb_image: np.ndarray) -> np.ndarray:
        """Applies Gray-World chromatic adaptation assumption."""
        float_img = rgb_image.astype(np.float32)
        mean_r = np.mean(float_img[:, :, 0]) + 1e-6
        mean_g = np.mean(float_img[:, :, 1]) + 1e-6
        mean_b = np.mean(float_img[:, :, 2]) + 1e-6
        gray_val = (mean_r + mean_g + mean_b) / 3.0

        scale_r = gray_val / mean_r
        scale_g = gray_val / mean_g
        scale_b = gray_val / mean_b

        balanced = float_img.copy()
        balanced[:, :, 0] = np.clip(float_img[:, :, 0] * scale_r, 0, 255)
        balanced[:, :, 1] = np.clip(float_img[:, :, 1] * scale_g, 0, 255)
        balanced[:, :, 2] = np.clip(float_img[:, :, 2] * scale_b, 0, 255)
        return balanced.astype(np.uint8)

    def apply_clahe_enhancement(self, rgb_image: np.ndarray) -> np.ndarray:
        """Enhances local contrast by equalizing the luminance channel in LAB space."""
        if not HAS_OPENCV or self._clahe_op is None:
            return rgb_image

        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self._clahe_op.apply(l_channel)
        merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
        return enhanced_rgb

    def resize_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """Resizes image to target dimensions."""
        h, w = self.target_size
        if HAS_OPENCV:
            return cv2.resize(rgb_image, (w, h), interpolation=cv2.INTER_AREA)
        else:
            # Fallback simple bilinear interpolation
            from scipy.ndimage import zoom  # type: ignore
            scale_y = h / rgb_image.shape[0]
            scale_x = w / rgb_image.shape[1]
            return zoom(rgb_image, (scale_y, scale_x, 1), order=1).astype(np.uint8)

    def normalize(self, rgb_image: np.ndarray) -> np.ndarray:
        """Normalizes an uint8 [0, 255] RGB image using ImageNet statistics."""
        img_float = rgb_image.astype(np.float32) / 255.0
        normalized = (img_float - self.mean) / (self.std + 1e-7)
        return normalized

    def process(
        self,
        image: np.ndarray,
        is_bgr: bool = True,
        return_torch: bool = True,
        device: Optional[str] = None,
    ) -> Tuple[Union[np.ndarray, "torch.Tensor"], np.ndarray]:
        """Runs the complete preprocessing pipeline.

        Args:
            image: Raw image array from camera or disk.
            is_bgr: Flag indicating if raw input is OpenCV BGR format.
            return_torch: If True and PyTorch is available, returns a (1, 3, H, W) Tensor.
            device: Target torch device (e.g. 'cuda', 'cpu').

        Returns:
            Tuple of:
                - normalized_tensor: (1, 3, H, W) normalized tensor or (1, 3, H, W) numpy array.
                - visual_rgb: Preprocessed (H, W, 3) uint8 RGB image for heatmap overlay blending.
        """
        rgb = self.ensure_rgb(image, is_bgr=is_bgr)

        if self.apply_white_balance:
            rgb = self.apply_gray_world_white_balance(rgb)

        if self.apply_clahe:
            rgb = self.apply_clahe_enhancement(rgb)

        resized_rgb = self.resize_image(rgb)
        normalized = self.normalize(resized_rgb)  # shape (H, W, 3)

        # Transpose from (H, W, C) to (C, H, W)
        chw = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
        batch_chw = np.expand_dims(chw, axis=0)  # shape (1, 3, H, W)

        if return_torch and HAS_TORCH:
            tensor = torch.from_numpy(batch_chw)
            if device:
                tensor = tensor.to(device)
            return tensor, resized_rgb

        return batch_chw, resized_rgb
