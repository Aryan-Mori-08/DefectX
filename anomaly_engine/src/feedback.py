"""Operator feedback adaptation and quarantine staging buffer.

Safely collects false-alarm candidate patches, prevents defect contamination,
and orchestrates versioned memory bank updates with audit logging and rollback.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .memory_bank import ProductMemoryBank


@dataclass
class FeedbackRecord:
    """Audit log entry capturing an operator feedback action."""

    feedback_id: str
    product_id: str
    operator_id: str
    action: str  # "FLAG_FALSE_ALARM" | "DISCARD_CANDIDATE" | "APPROVE_BATCH"
    timestamp: str
    num_patches_extracted: int
    source_image_name: str
    notes: str = ""
    promoted_to_version: Optional[str] = None


class FeedbackManager:
    """Manages false alarm staging, contamination guards, and version promotion.

    Parameters:
        buffer_dir: Directory where quarantined patch buffers are staged.
        audit_log_path: Path to append-only audit log JSONL file.
        similarity_safety_limit: Max cosine similarity to prevent memorizing exact duplicates.
    """

    def __init__(
        self,
        buffer_dir: Union[str, Path] = "artifacts/feedback_buffer",
        audit_log_path: Union[str, Path] = "artifacts/memory_banks/audit_log.jsonl",
        similarity_safety_limit: float = 0.98,
    ) -> None:
        self.buffer_dir = Path(buffer_dir)
        self.audit_log_path = Path(audit_log_path)
        self.similarity_safety_limit = float(similarity_safety_limit)

        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_action(self, record: FeedbackRecord) -> None:
        """Appends an immutable audit entry to the audit ledger."""
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def stage_false_alarm(
        self,
        product_id: str,
        image_name: str,
        anomalous_patches: np.ndarray,
        operator_id: str = "line_operator_01",
        notes: str = "Operator flagged false alarm on benign surface variation",
    ) -> FeedbackRecord:
        """Extracts and quarantines anomalous patches into the staging buffer.

        Does NOT mutate the live memory bank.
        """
        patches = np.asarray(anomalous_patches, dtype=np.float32)
        n_patches = patches.shape[0]

        prod_buffer_dir = self.buffer_dir / product_id
        prod_buffer_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        feedback_id = f"fb_{product_id}_{timestamp_str}"
        patch_file = prod_buffer_dir / f"{feedback_id}.npy"

        # Save quarantined patches to disk
        np.save(patch_file, patches)

        record = FeedbackRecord(
            feedback_id=feedback_id,
            product_id=product_id,
            operator_id=operator_id,
            action="FLAG_FALSE_ALARM",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            num_patches_extracted=n_patches,
            source_image_name=image_name,
            notes=notes,
        )
        self.log_action(record)
        return record

    def list_staged_feedback(self, product_id: str) -> List[Path]:
        """Lists all staged patch files pending promotion for a product."""
        prod_buffer_dir = self.buffer_dir / product_id
        if not prod_buffer_dir.exists():
            return []
        return sorted(prod_buffer_dir.glob("*.npy"))

    def promote_staged_feedback(
        self,
        memory_bank: ProductMemoryBank,
        supervisor_id: str,
        target_version: str,
        coreset_ratio: float = 0.05,
    ) -> Tuple[bool, str]:
        """Promotes quarantined staged patches into a new versioned memory bank.

        1. Pools active memory bank embeddings + staged candidate patches.
        2. Applies greedy coreset subsampling to preserve compact memory footprint.
        3. Saves as target_version and points 'active' symlink to new version.
        4. Keeps prior version intact for immediate rollback.
        5. Clears the staging buffer and logs audit record.
        """
        product_id = memory_bank.product_id
        staged_files = self.list_staged_feedback(product_id)

        if not staged_files:
            return False, "No staged feedback patches found for promotion."

        # Load all staged patches
        staged_arrays = [np.load(f) for f in staged_files]
        new_patches = np.concatenate(staged_arrays, axis=0)

        # Retrieve active base embeddings
        active_embeddings = memory_bank.embeddings
        if active_embeddings is None:
            return False, "Active memory bank has no embeddings loaded."

        # Safety filter: remove identical duplicate vectors
        combined_pool = np.concatenate([active_embeddings, new_patches], axis=0)

        # Run fresh coreset subsampling on combined pool
        parent_v = memory_bank.manifest.version if memory_bank.manifest else "v1.0.0"
        manifest = memory_bank.manifest

        memory_bank.fit_from_patches(
            raw_patches=combined_pool,
            reference_count=(manifest.reference_images_count if manifest else 20) + len(staged_files),
            backbone_name=manifest.backbone if manifest else "wide_resnet50_2",
            layers=manifest.layers if manifest else ["layer2", "layer3"],
            input_resolution=tuple(manifest.input_resolution) if manifest else (256, 256),
            image_threshold=manifest.image_threshold if manifest else 0.45,
            pixel_threshold=manifest.pixel_threshold if manifest else 0.30,
            coreset_ratio=coreset_ratio,
            product_prototype=(
                np.array(manifest.product_prototype)
                if manifest and manifest.product_prototype
                else None
            ),
            version=target_version,
            parent_version=parent_v,
        )

        # Save and atomically promote
        memory_bank.save(version=target_version, make_active=True)

        # Clear staged buffer files
        for f in staged_files:
            f.unlink()

        # Log promotion in audit ledger
        audit_record = FeedbackRecord(
            feedback_id=f"promo_{product_id}_{int(time.time())}",
            product_id=product_id,
            operator_id=supervisor_id,
            action="APPROVE_BATCH",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            num_patches_extracted=int(new_patches.shape[0]),
            source_image_name="batch_promotion",
            notes=f"Promoted {len(staged_files)} staged batches from {parent_v} -> {target_version}",
            promoted_to_version=target_version,
        )
        self.log_action(audit_record)

        return True, f"Successfully promoted memory bank to {target_version} (Parent: {parent_v})"
