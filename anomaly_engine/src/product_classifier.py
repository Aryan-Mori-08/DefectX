"""Product identification router using cosine similarity against prototype embeddings.

Routes incoming camera captures to the correct product profile memory bank
before anomaly scoring, with confidence guards and manual override support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class ProductRouteDecision:
    """Outcome of product classification routing."""

    predicted_product: str
    confidence: float
    margin: float
    is_confident: bool
    is_manual_override: bool
    candidate_scores: Dict[str, float]

    def summary(self) -> str:
        override_str = " (MANUAL OVERRIDE)" if self.is_manual_override else ""
        return (
            f"ProductRoute: '{self.predicted_product}' "
            f"[Confidence: {self.confidence:.3f}, Margin: {self.margin:.3f}]{override_str}"
        )


class ProductClassifier:
    """Zero-shot prototype router matching global image descriptors to enrolled products.

    Parameters:
        min_confidence: Cosine similarity cutoff for automated acceptance.
        min_margin: Minimum similarity gap required between rank-1 and rank-2 candidates.
        registry_path: Path to JSON file persisting product prototypes.
        unknown_label: Label assigned when confidence thresholds are violated.
    """

    def __init__(
        self,
        min_confidence: float = 0.75,
        min_margin: float = 0.10,
        registry_path: Union[str, Path] = "artifacts/models/product_prototypes.json",
        unknown_label: str = "UNKNOWN_PRODUCT",
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.registry_path = Path(registry_path)
        self.unknown_label = unknown_label

        # Dictionary of product_id -> L2-normalized 1D prototype vector
        self.prototypes: Dict[str, np.ndarray] = {}
        self.load_registry()

    def load_registry(self) -> None:
        """Loads stored product prototypes from disk if available."""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.prototypes = {
                    pid: np.array(vec, dtype=np.float32)
                    for pid, vec in data.items()
                }

    def save_registry(self) -> None:
        """Persists registered product prototypes to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            pid: vec.tolist()
            for pid, vec in self.prototypes.items()
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def register_product(
        self,
        product_id: str,
        prototype_vector: np.ndarray,
        overwrite: bool = True,
    ) -> None:
        """Registers or updates a product's centroid prototype vector."""
        vec = np.asarray(prototype_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        else:
            raise ValueError(f"Cannot register zero-norm prototype vector for '{product_id}'")

        if not overwrite and product_id in self.prototypes:
            # Running exponential moving average update
            old_vec = self.prototypes[product_id]
            updated = 0.8 * old_vec + 0.2 * vec
            self.prototypes[product_id] = updated / np.linalg.norm(updated)
        else:
            self.prototypes[product_id] = vec

        self.save_registry()

    def route(
        self,
        query_descriptor: np.ndarray,
        manual_override: Optional[str] = None,
    ) -> ProductRouteDecision:
        """Routes a query image descriptor to the best matching product memory bank.

        Args:
            query_descriptor: 1D or 2D L2-normalized feature vector from backbone.
            manual_override: Optional explicit product ID specified by line operator or PLC.

        Returns:
            ProductRouteDecision containing predicted product and confidence metrics.
        """
        # 1. Manual Operator Override
        if manual_override:
            return ProductRouteDecision(
                predicted_product=manual_override,
                confidence=1.0,
                margin=1.0,
                is_confident=True,
                is_manual_override=True,
                candidate_scores={manual_override: 1.0},
            )

        if not self.prototypes:
            return ProductRouteDecision(
                predicted_product=self.unknown_label,
                confidence=0.0,
                margin=0.0,
                is_confident=False,
                is_manual_override=False,
                candidate_scores={},
            )

        # Normalize query vector
        q = np.asarray(query_descriptor, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-8:
            q = q / q_norm

        # 2. Compute Cosine Similarity against all enrolled prototypes
        scores: Dict[str, float] = {}
        for pid, proto in self.prototypes.items():
            sim = float(np.dot(q, proto))
            scores[pid] = round(sim, 4)

        # Rank candidates
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top1_prod, top1_score = ranked[0]
        top2_score = ranked[1][1] if len(ranked) > 1 else -1.0

        margin = top1_score - top2_score if len(ranked) > 1 else top1_score

        is_confident = (
            top1_score >= self.min_confidence
            and (len(ranked) == 1 or margin >= self.min_margin)
        )

        predicted = top1_prod if is_confident else self.unknown_label

        return ProductRouteDecision(
            predicted_product=predicted,
            confidence=top1_score,
            margin=margin,
            is_confident=is_confident,
            is_manual_override=False,
            candidate_scores=scores,
        )
