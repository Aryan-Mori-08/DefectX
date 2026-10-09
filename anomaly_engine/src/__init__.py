"""PatchCore Industrial Anomaly Detection Engine.

Product-agnostic, unsupervised anomaly detection for real-time industrial quality inspection.
"""

from .capture_quality import CaptureQualityGate, CaptureQualityReport
from .evaluation import AnomalyEvaluationReport, AnomalyEvaluator
from .feature_extractor import FeatureExtractor
from .feedback import FeedbackManager, FeedbackRecord
from .localization import DefectLocalizer, DefectRegion, LocalizationArtifacts
from .memory_bank import GreedyCoresetSampler, MemoryBankManifest, ProductMemoryBank
from .patchcore_engine import AnomalyInferenceResult, PatchCoreEngine
from .preprocessing import ImagePreprocessor
from .product_classifier import ProductClassifier, ProductRouteDecision
from .thresholding import ThresholdCalibrationReport, ThresholdCalibrator, ThresholdConfig

__all__ = [
    "CaptureQualityGate",
    "CaptureQualityReport",
    "ImagePreprocessor",
    "FeatureExtractor",
    "GreedyCoresetSampler",
    "MemoryBankManifest",
    "ProductMemoryBank",
    "ProductClassifier",
    "ProductRouteDecision",
    "DefectLocalizer",
    "DefectRegion",
    "LocalizationArtifacts",
    "ThresholdCalibrator",
    "ThresholdConfig",
    "ThresholdCalibrationReport",
    "FeedbackManager",
    "FeedbackRecord",
    "AnomalyInferenceResult",
    "PatchCoreEngine",
    "AnomalyEvaluator",
    "AnomalyEvaluationReport",
]
