"""Deep feature extraction module for PatchCore anomaly detection.

Extracts localized multi-scale patch embeddings and global image descriptors
using WideResNet-50-2 or DINOv2 backbones.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    import torchvision.models as models  # type: ignore
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FeatureExtractor:
    """Extracts localized patch embeddings and global descriptors.

    Supports:
        - 'wide_resnet50_2': Default industrial standard. Concatenates intermediate
          layers (layer2 + layer3) with local neighborhood pooling.
        - 'dinov2_vits14' / 'dinov2_vitb14': Self-supervised ViT patch embeddings for
          complex or stochastic surface textures.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        extracted_layers: Sequence[str] = ("layer2", "layer3"),
        pooling_kernel_size: int = 3,
        pooling_stride: int = 1,
        pooling_padding: int = 1,
        device: str = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu",
        half_precision: bool = False,
    ) -> None:
        self.backbone_name = backbone_name.lower()
        self.extracted_layers = list(extracted_layers)
        self.device = device
        self.half_precision = half_precision
        self.patch_dim: int = 1536  # Default for WideResNet-50-2 (512 + 1024)

        self._model: Optional[nn.Module] = None
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._intermediate_features: Dict[str, torch.Tensor] = {}

        if HAS_TORCH:
            self._init_pytorch_backbone(
                kernel_size=pooling_kernel_size,
                stride=pooling_stride,
                padding=pooling_padding,
            )

    def _init_pytorch_backbone(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        """Initializes and registers hooks on the chosen neural backbone."""
        if not HAS_TORCH:
            return

        self._local_pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            count_include_pad=False,
        )

        if "dinov2" in self.backbone_name:
            # Load DINOv2 via torch.hub
            try:
                self._model = torch.hub.load(
                    "facebookresearch/dinov2",
                    self.backbone_name,
                    pretrained=True,
                )
                self.patch_dim = self._model.embed_dim  # e.g., 384 for vits14, 768 for vitb14
            except Exception:
                # Fallback to standard WideResNet if offline/hub fails
                self.backbone_name = "wide_resnet50_2"
                self._load_wide_resnet()
        else:
            self._load_wide_resnet()

        if self._model is not None:
            self._model.to(self.device)
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False

            if self.half_precision and self.device != "cpu":
                self._model.half()

    def _load_wide_resnet(self) -> None:
        """Loads torchvision WideResNet-50-2 and attaches feature capture forward hooks."""
        try:
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
            self._model = models.wide_resnet50_2(weights=weights)
        except AttributeError:
            self._model = models.wide_resnet50_2(pretrained=True)

        self._hooks.clear()
        self._intermediate_features.clear()

        def _get_hook(name: str):
            def hook(module, input, output):
                self._intermediate_features[name] = output
            return hook

        for name, module in self._model.named_children():
            if name in self.extracted_layers:
                h = module.register_forward_hook(_get_hook(name))
                self._hooks.append(h)

        # WideResNet50-2 layer2 is 512-dim, layer3 is 1024-dim -> total 1536
        self.patch_dim = 1536

    @torch.no_grad()
    def extract_features(
        self,
        x: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Extracts spatial patch embeddings and a global image representation.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Tuple of:
                - patch_features: Tensor of shape (B * H_p * W_p, D).
                - global_feature: Tensor of shape (B, D) L2-normalized image descriptor.
                - spatial_shape: (H_p, W_p) patch grid dimensions.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for deep feature extraction.")

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        x = x.to(self.device)
        if self.half_precision and self.device != "cpu":
            x = x.half()

        if "dinov2" in self.backbone_name:
            return self._extract_dinov2(x)
        else:
            return self._extract_wide_resnet(x)

    def _extract_wide_resnet(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Processes intermediate Conv features through spatial alignment and pooling."""
        self._intermediate_features.clear()
        _ = self._model(x)

        # Retrieve intermediate activation tensors
        f_layer2 = self._intermediate_features["layer2"]  # (B, 512, H/8, W/8)
        f_layer3 = self._intermediate_features["layer3"]  # (B, 1024, H/16, W/16)

        b, c2, h2, w2 = f_layer2.shape

        # Bilinear align layer3 to layer2 spatial resolution
        f_layer3_upsampled = F.interpolate(
            f_layer3,
            size=(h2, w2),
            mode="bilinear",
            align_corners=False,
        )

        # Concatenate features along channel dimension -> (B, 1536, h2, w2)
        combined_features = torch.cat([f_layer2, f_layer3_upsampled], dim=1)

        # Inject local neighborhood spatial context
        pooled_features = self._local_pool(combined_features)  # (B, 1536, h2, w2)

        # Global feature for product classification via global average pooling
        global_feature = F.adaptive_avg_pool2d(combined_features, (1, 1)).flatten(1)
        global_feature = F.normalize(global_feature, p=2, dim=1)

        # Reshape patches into (B * h2 * w2, Channels)
        # Permute (B, C, H, W) -> (B, H, W, C) -> (B * H * W, C)
        patch_features = pooled_features.permute(0, 2, 3, 1).reshape(-1, pooled_features.shape[1])

        return patch_features, global_feature, (h2, w2)

    def _extract_dinov2(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Extracts patch tokens from self-supervised DINOv2 Vision Transformer."""
        features_dict = self._model.forward_features(x)
        patch_tokens = features_dict["x_norm_patchtokens"]  # (B, N_patches, D)
        cls_token = features_dict["x_norm_clstoken"]        # (B, D)

        b, n_patches, dim = patch_tokens.shape
        grid_dim = int(np.sqrt(n_patches))
        spatial_shape = (grid_dim, grid_dim)

        patch_features = patch_tokens.reshape(-1, dim)
        global_feature = F.normalize(cls_token, p=2, dim=1)

        return patch_features, global_feature, spatial_shape

    def close(self) -> None:
        """Removes forward hooks from the model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._intermediate_features.clear()
