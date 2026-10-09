"""Main PatchCore Anomaly Detection Engine for industrial quality inspection.

Integrates capture-quality gating, preprocessing, zero-shot product routing,
deep feature extraction, nearest-neighbor memory bank distance scoring,
smooth heatmap synthesis, and defect localization.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .capture_quality import CaptureQualityGate, CaptureQualityReport
from .feature_extractor import FeatureExtractor
from .localization import DefectLocalizer, DefectRegion, LocalizationArtifacts
from .memory_bank import ProductMemoryBank
from .preprocessing import ImagePreprocessor
from .product_classifier import ProductClassifier, ProductRouteDecision


@dataclass
class AnomalyInferenceResult:
    """Consolidated inspection outcome for an industrial optical capture."""

    decision: str                      # "PASS" | "REJECT" | "REJECT_CAPTURE"
    image_anomaly_score: float
    image_threshold: float
    pixel_threshold: float
    is_defect: bool
    product_id: str
    product_route: ProductRouteDecision
    quality_report: CaptureQualityReport
    num_defects: int
    total_defect_area_px: int
    defect_regions: List[DefectRegion]
    raw_heatmap: Optional[np.ndarray] = None
    binary_mask: Optional[np.ndarray] = None
    heatmap_overlay: Optional[np.ndarray] = None
    latency_breakdown_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "image_anomaly_score": round(self.image_anomaly_score, 4),
            "image_threshold": round(self.image_threshold, 4),
            "pixel_threshold": round(self.pixel_threshold, 4),
            "is_defect": self.is_defect,
            "product_id": self.product_id,
            "product_route": {
                "predicted_product": self.product_route.predicted_product,
                "confidence": round(self.product_route.confidence, 4),
                "is_confident": self.product_route.is_confident,
            },
            "quality_report": {
                "is_valid": self.quality_report.is_valid_capture,
                "metrics": self.quality_report.metrics,
                "violations": self.quality_report.rejection_reasons,
            },
            "num_defects": self.num_defects,
            "total_defect_area_px": self.total_defect_area_px,
            "defect_regions": [r.to_dict() for r in self.defect_regions],
            "latency_ms": {k: round(v, 2) for k, v in self.latency_breakdown_ms.items()},
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


class PatchCoreEngine:
    """Industrial unsupervised anomaly detection engine.

    Orchestrates the complete inspection pipeline from raw camera frame to
    PASS/REJECT decision, heatmap overlay, and localized defect metrics.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        artifacts_dir: Union[str, Path] = "artifacts",
    ) -> None:
        self.config = config
        self.artifacts_dir = Path(artifacts_dir)

        # 1. Capture Quality Gate
        q_cfg = config.get("quality_gate", {})
        self.quality_gate = CaptureQualityGate(
            min_laplacian_variance=q_cfg.get("min_laplacian_variance", 100.0),
            min_mean_intensity=q_cfg.get("min_mean_intensity", 25.0),
            max_mean_intensity=q_cfg.get("max_mean_intensity", 235.0),
            min_contrast_rms=q_cfg.get("min_contrast_rms", 20.0),
            max_overexposed_pixel_pct=q_cfg.get("max_overexposed_pixel_pct", 0.05),
            max_underexposed_pixel_pct=q_cfg.get("max_underexposed_pixel_pct", 0.05),
        )

        # 2. Preprocessor
        p_cfg = config.get("preprocessing", {})
        self.preprocessor = ImagePreprocessor(
            target_size=tuple(p_cfg.get("target_size", [256, 256])),
            apply_clahe=p_cfg.get("apply_clahe", True),
            clahe_clip_limit=p_cfg.get("clahe_clip_limit", 2.0),
            clahe_grid_size=tuple(p_cfg.get("clahe_grid_size", [8, 8])),
            apply_white_balance=p_cfg.get("apply_white_balance", True),
            mean=p_cfg.get("normalization", {}).get("mean", [0.485, 0.456, 0.406]),
            std=p_cfg.get("normalization", {}).get("std", [0.229, 0.224, 0.225]),
        )

        # 3. Deep Feature Extractor
        m_cfg = config.get("model", {})
        sys_cfg = config.get("system", {})
        self.feature_extractor = FeatureExtractor(
            backbone_name=m_cfg.get("backbone_name", "wide_resnet50_2"),
            extracted_layers=m_cfg.get("extracted_layers", ["layer2", "layer3"]),
            pooling_kernel_size=m_cfg.get("pooling_kernel_size", 3),
            pooling_stride=m_cfg.get("pooling_stride", 1),
            pooling_padding=m_cfg.get("pooling_padding", 1),
            device=sys_cfg.get("device", "cuda"),
            half_precision=sys_cfg.get("precision", "fp32") == "fp16",
        )

        # 4. Product Router
        r_cfg = config.get("product_router", {})
        self.product_classifier = ProductClassifier(
            min_confidence=r_cfg.get("min_confidence_threshold", 0.75),
            min_margin=r_cfg.get("min_margin_threshold", 0.10),
            registry_path=self.artifacts_dir / "models/product_prototypes.json",
        )

        # 5. Defect Localizer
        inf_cfg = config.get("inference", {})
        self.localizer = DefectLocalizer(
            gaussian_sigma=inf_cfg.get("gaussian_sigma", 4.0),
            min_defect_area_px=inf_cfg.get("min_defect_area_px", 12),
        )

        # Cache of loaded product memory banks
        self._memory_banks: Dict[str, ProductMemoryBank] = {}

    def get_memory_bank(self, product_id: str) -> ProductMemoryBank:
        """Retrieves or loads the active memory bank for a given product."""
        if product_id not in self._memory_banks:
            bank = ProductMemoryBank(
                product_id=product_id,
                artifacts_dir=self.artifacts_dir / "memory_banks",
            )
            bank.load()  # Loads active version
            self._memory_banks[product_id] = bank
        return self._memory_banks[product_id]

    def inspect(
        self,
        image: np.ndarray,
        product_id_override: Optional[str] = None,
        image_threshold_override: Optional[float] = None,
        pixel_threshold_override: Optional[float] = None,
        is_bgr: bool = True,
    ) -> AnomalyInferenceResult:
        """Performs full end-to-end anomaly inspection on an optical frame.

        Args:
            image: Raw camera image frame (H, W, 3).
            product_id_override: Optional explicit product ID.
            image_threshold_override: Runtime threshold adjustment for PASS/REJECT.
            pixel_threshold_override: Runtime threshold adjustment for heatmap mask.
            is_bgr: True if image is in OpenCV BGR format.

        Returns:
            AnomalyInferenceResult containing decision, scores, and defect localization.
        """
        t_start = time.perf_counter()
        latencies: Dict[str, float] = {}

        # ---------------------------------------------------------
        # Step 1: Capture Quality Assessment
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        quality_report = self.quality_gate.evaluate(image)
        latencies["quality_gate"] = (time.perf_counter() - t0) * 1000.0

        if not quality_report.is_valid_capture:
            total_lat = (time.perf_counter() - t_start) * 1000.0
            return AnomalyInferenceResult(
                decision="REJECT_CAPTURE",
                image_anomaly_score=1.0,
                image_threshold=0.5,
                pixel_threshold=0.5,
                is_defect=True,
                product_id=product_id_override or "UNKNOWN",
                product_route=ProductRouteDecision(
                    predicted_product="UNKNOWN",
                    confidence=0.0,
                    margin=0.0,
                    is_confident=False,
                    is_manual_override=False,
                    candidate_scores={},
                ),
                quality_report=quality_report,
                num_defects=0,
                total_defect_area_px=0,
                defect_regions=[],
                latency_breakdown_ms=latencies,
                total_latency_ms=total_lat,
            )

        # ---------------------------------------------------------
        # Step 2: Deterministic Preprocessing
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        input_tensor, visual_rgb = self.preprocessor.process(
            image,
            is_bgr=is_bgr,
            return_torch=True,
        )
        latencies["preprocessing"] = (time.perf_counter() - t0) * 1000.0

        # ---------------------------------------------------------
        # Step 3: Deep Feature Extraction
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        patch_feats, global_feat, grid_shape = self.feature_extractor.extract_features(input_tensor)
        latencies["feature_extraction"] = (time.perf_counter() - t0) * 1000.0

        # ---------------------------------------------------------
        # Step 4: Product Identification Router
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        global_np = global_feat.detach().cpu().numpy()
        route_decision = self.product_classifier.route(
            global_np,
            manual_override=product_id_override,
        )
        latencies["product_routing"] = (time.perf_counter() - t0) * 1000.0

        target_product = route_decision.predicted_product
        if target_product == "UNKNOWN_PRODUCT" and not product_id_override:
            total_lat = (time.perf_counter() - t_start) * 1000.0
            return AnomalyInferenceResult(
                decision="UNKNOWN_PRODUCT",
                image_anomaly_score=0.0,
                image_threshold=0.5,
                pixel_threshold=0.5,
                is_defect=False,
                product_id="UNKNOWN_PRODUCT",
                product_route=route_decision,
                quality_report=quality_report,
                num_defects=0,
                total_defect_area_px=0,
                defect_regions=[],
                latency_breakdown_ms=latencies,
                total_latency_ms=total_lat,
            )

        # ---------------------------------------------------------
        # Step 5: Memory Bank Nearest Neighbor Search
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        bank = self.get_memory_bank(target_product)
        k_neighbors = self.config.get("inference", {}).get("k_neighbors", 1)
        distances, _ = bank.query_nearest_neighbors(patch_feats, k=k_neighbors)
        # distances is (N_patches, k), take nearest distance
        patch_distances = distances[:, 0]
        latencies["memory_bank_query"] = (time.perf_counter() - t0) * 1000.0

        # ---------------------------------------------------------
        # Step 6: Image-Level Anomaly Score Aggregation
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        top_k = self.config.get("inference", {}).get("top_k_image_patches", 5)
        # Sort patch distances in descending order
        sorted_dists = np.sort(patch_distances)[::-1]
        top_k_val = min(top_k, len(sorted_dists))
        # Top-k robust mean distance
        image_anomaly_score = float(np.mean(sorted_dists[:top_k_val]))
        latencies["score_aggregation"] = (time.perf_counter() - t0) * 1000.0

        # Determine thresholds
        img_threshold = (
            image_threshold_override
            if image_threshold_override is not None
            else (bank.manifest.image_threshold if bank.manifest else 0.45)
        )
        pix_threshold = (
            pixel_threshold_override
            if pixel_threshold_override is not None
            else (bank.manifest.pixel_threshold if bank.manifest else 0.30)
        )

        # ---------------------------------------------------------
        # Step 7: Heatmap Synthesis and Defect Localization
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        target_size = self.preprocessor.target_size
        heatmap = self.localizer.generate_heatmap(
            patch_scores=patch_distances,
            grid_shape=grid_shape,
            target_size=target_size,
        )

        artifacts = self.localizer.extract_regions(
            heatmap=heatmap,
            pixel_threshold=pix_threshold,
            base_rgb=visual_rgb,
        )
        latencies["localization"] = (time.perf_counter() - t0) * 1000.0

        # ---------------------------------------------------------
        # Step 8: Final Decision
        # ---------------------------------------------------------
        is_defect = (image_anomaly_score >= img_threshold) or (artifacts.num_defects > 0)
        decision = "REJECT" if is_defect else "PASS"

        total_latency = (time.perf_counter() - t_start) * 1000.0

        return AnomalyInferenceResult(
            decision=decision,
            image_anomaly_score=image_anomaly_score,
            image_threshold=img_threshold,
            pixel_threshold=pix_threshold,
            is_defect=is_defect,
            product_id=target_product,
            product_route=route_decision,
            quality_report=quality_report,
            num_defects=artifacts.num_defects,
            total_defect_area_px=artifacts.total_defect_area,
            defect_regions=artifacts.defect_regions,
            raw_heatmap=artifacts.raw_heatmap,
            binary_mask=artifacts.binary_mask,
            heatmap_overlay=artifacts.heatmap_overlay,
            latency_breakdown_ms=latencies,
            total_latency_ms=total_latency,
        )
