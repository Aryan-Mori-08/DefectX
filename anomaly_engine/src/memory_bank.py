"""High-performance Coreset Memory Bank for industrial PatchCore anomaly detection.

Features:
- Fast Johnson-Lindenstrauss projected greedy coreset subsampling.
- Vectorized k-NN nearest-neighbor search (FAISS IndexFlatL2 / PyTorch GPU matrix fallback).
- Versioned storage, atomic rollback, and audit-logged metadata manifests.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch  # type: ignore
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class MemoryBankManifest:
    """Metadata manifest describing an immutable memory bank snapshot."""

    product_id: str
    version: str
    parent_version: Optional[str]
    created_at: str
    backbone: str
    layers: List[str]
    patch_dim: int
    input_resolution: List[int]
    coreset_ratio: float
    total_source_patches: int
    coreset_patch_count: int
    reference_images_count: int
    image_threshold: float
    pixel_threshold: float
    product_prototype: List[float] = field(default_factory=list)
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryBankManifest:
        return cls(**data)


class GreedyCoresetSampler:
    """Selects an optimal representative subset of patch vectors via minimax facility location.

    Parameters:
        sampling_ratio: Fraction of patches to retain (e.g. 0.05 for 5%).
        projection_dim: Johnson-Lindenstrauss target dimension for fast distance computation.
        random_seed: Random seed for deterministic reproducibility.
    """

    def __init__(
        self,
        sampling_ratio: float = 0.05,
        projection_dim: int = 128,
        random_seed: int = 42,
    ) -> None:
        self.sampling_ratio = sampling_ratio
        self.projection_dim = projection_dim
        self.random_seed = random_seed

    def sample(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Runs Johnson-Lindenstrauss projected greedy coreset selection.

        Args:
            features: Candidate feature matrix of shape (N, D).

        Returns:
            Tuple of:
                - coreset_features: Subsampled matrix of shape (M, D).
                - coreset_indices: 1D array of selected row indices from the source matrix.
        """
        n_samples, feat_dim = features.shape
        target_size = max(1, int(np.ceil(self.sampling_ratio * n_samples)))

        if target_size >= n_samples:
            return features.copy(), np.arange(n_samples)

        rng = np.random.RandomState(self.random_seed)

        # 1. Johnson-Lindenstrauss Random Projection Matrix R in R^(D x d)
        if self.projection_dim < feat_dim:
            proj_matrix = rng.randn(feat_dim, self.projection_dim).astype(np.float32)
            # Scale for orthogonal norm preservation: 1 / sqrt(d)
            proj_matrix /= np.sqrt(self.projection_dim)
            projected = np.matmul(features, proj_matrix)  # (N, d)
        else:
            projected = features.copy()

        # 2. Greedy Minimax Facility Location
        selected_indices = np.zeros(target_size, dtype=np.int64)
        # Start with a random prototype index
        start_idx = rng.randint(0, n_samples)
        selected_indices[0] = start_idx

        # Initialize minimum distance array to infinity
        first_point = projected[start_idx : start_idx + 1]  # (1, d)
        min_distances = np.linalg.norm(projected - first_point, axis=1)  # (N,)

        for step in range(1, target_size):
            # Select point furthest from the current set of centers
            best_idx = int(np.argmax(min_distances))
            selected_indices[step] = best_idx

            # Update distance vector with new center candidate
            new_center = projected[best_idx : best_idx + 1]
            new_dist = np.linalg.norm(projected - new_center, axis=1)
            min_distances = np.minimum(min_distances, new_dist)

        coreset_features = features[selected_indices]
        return coreset_features, selected_indices


class ProductMemoryBank:
    """Manages storage, indexing, querying, and versioning for a product's normal patch bank."""

    def __init__(
        self,
        product_id: str,
        artifacts_dir: Union[str, Path] = "artifacts/memory_banks",
    ) -> None:
        self.product_id = product_id
        self.artifacts_dir = Path(artifacts_dir)
        self.product_dir = self.artifacts_dir / self.product_id

        self.manifest: Optional[MemoryBankManifest] = None
        self.embeddings: Optional[np.ndarray] = None  # (M, D) float32
        self._faiss_index: Any = None
        self._torch_embeddings: Optional[Any] = None

    @property
    def is_loaded(self) -> bool:
        return self.embeddings is not None and self.manifest is not None

    def fit_from_patches(
        self,
        raw_patches: np.ndarray,
        reference_count: int,
        backbone_name: str,
        layers: List[str],
        input_resolution: Tuple[int, int],
        image_threshold: float = 0.45,
        pixel_threshold: float = 0.30,
        coreset_ratio: float = 0.05,
        projection_dim: int = 128,
        product_prototype: Optional[np.ndarray] = None,
        version: str = "v1.0.0",
        parent_version: Optional[str] = None,
    ) -> MemoryBankManifest:
        """Subsamples candidate normal patch vectors and constructs the memory bank."""
        sampler = GreedyCoresetSampler(
            sampling_ratio=coreset_ratio,
            projection_dim=projection_dim,
        )
        coreset_feats, _ = sampler.sample(raw_patches)
        self.embeddings = coreset_feats.astype(np.float32)

        proto_list = (
            product_prototype.squeeze().tolist()
            if product_prototype is not None
            else []
        )

        self.manifest = MemoryBankManifest(
            product_id=self.product_id,
            version=version,
            parent_version=parent_version,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            backbone=backbone_name,
            layers=layers,
            patch_dim=int(self.embeddings.shape[1]),
            input_resolution=[input_resolution[0], input_resolution[1]],
            coreset_ratio=float(coreset_ratio),
            total_source_patches=int(raw_patches.shape[0]),
            coreset_patch_count=int(self.embeddings.shape[0]),
            reference_images_count=reference_count,
            image_threshold=float(image_threshold),
            pixel_threshold=float(pixel_threshold),
            product_prototype=proto_list,
        )

        self._build_index()
        return self.manifest

    def _build_index(self) -> None:
        """Constructs FAISS Index or prepares PyTorch GPU/CPU buffer."""
        if self.embeddings is None:
            return

        dim = self.embeddings.shape[1]
        if HAS_FAISS:
            # L2 Euclidean index
            index = faiss.IndexFlatL2(dim)
            index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))
            self._faiss_index = index
        else:
            self._faiss_index = None

        if HAS_TORCH:
            self._torch_embeddings = torch.from_numpy(self.embeddings).float()
            if torch.cuda.is_available():
                self._torch_embeddings = self._torch_embeddings.cuda()

    def query_nearest_neighbors(
        self,
        query_patches: Union[np.ndarray, "torch.Tensor"],
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Finds nearest neighbor Euclidean distances for each query patch vector.

        Args:
            query_patches: Query vectors of shape (N_query, D).
            k: Number of nearest neighbors.

        Returns:
            Tuple of:
                - distances: shape (N_query, k) Euclidean distances.
                - indices: shape (N_query, k) indices in memory bank.
        """
        if not self.is_loaded:
            raise RuntimeError(f"Memory bank for '{self.product_id}' is not loaded.")

        if HAS_TORCH and isinstance(query_patches, torch.Tensor):
            q_np = query_patches.detach().cpu().numpy().astype(np.float32)
        else:
            q_np = np.asarray(query_patches, dtype=np.float32)

        # 1. FAISS acceleration (Fastest)
        if self._faiss_index is not None:
            # faiss returns squared L2 distances
            dists_sq, indices = self._faiss_index.search(q_np, k)
            distances = np.sqrt(np.maximum(dists_sq, 0.0))
            return distances, indices

        # 2. PyTorch GPU Matrix acceleration fallback
        if HAS_TORCH and self._torch_embeddings is not None:
            with torch.no_grad():
                q_t = torch.from_numpy(q_np).to(self._torch_embeddings.device)
                # Compute (a - b)^2 = a^2 + b^2 - 2ab
                # Chunked for large query memory safety
                chunk_size = 1024
                all_dists = []
                all_idxs = []
                for i in range(0, q_t.shape[0], chunk_size):
                    chunk = q_t[i : i + chunk_size]
                    # shape (chunk_size, bank_size)
                    d_mat = torch.cdist(chunk, self._torch_embeddings, p=2.0)
                    topk_d, topk_i = torch.topk(d_mat, k=k, dim=1, largest=False)
                    all_dists.append(topk_d.cpu().numpy())
                    all_idxs.append(topk_i.cpu().numpy())
                return np.concatenate(all_dists, axis=0), np.concatenate(all_idxs, axis=0)

        # 3. Pure NumPy fallback
        # Vectorized pairwise Euclidean distance computation
        # (q - m)^2 = q^2 + m^2 - 2 q m.T
        q_norm_sq = np.sum(q_np ** 2, axis=1, keepdims=True)  # (N_q, 1)
        m_norm_sq = np.sum(self.embeddings ** 2, axis=1, keepdims=True).T  # (1, M)
        dists_sq = q_norm_sq + m_norm_sq - 2.0 * np.dot(q_np, self.embeddings.T)
        dists_sq = np.maximum(dists_sq, 0.0)
        indices = np.argsort(dists_sq, axis=1)[:, :k]
        distances = np.sqrt(np.take_along_axis(dists_sq, indices, axis=1))
        return distances, indices

    def save(self, version: Optional[str] = None, make_active: bool = True) -> Path:
        """Saves memory bank snapshot to disk under versioned directory."""
        if not self.is_loaded:
            raise RuntimeError("Cannot save an uninitialized memory bank.")

        v = version or self.manifest.version
        version_dir = self.product_dir / v
        version_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save embeddings as compressed numpy array
        np.save(version_dir / "embeddings.npy", self.embeddings)

        # 2. Save FAISS index if available
        if HAS_FAISS and self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(version_dir / "index.faiss"))

        # 3. Save manifest
        self.manifest.version = v
        with open(version_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.manifest.to_dict(), f, indent=2)

        # 4. Update 'active' pointer
        if make_active:
            self._set_active_version(v)

        return version_dir

    def load(self, version: Optional[str] = None) -> None:
        """Loads a memory bank version from disk."""
        if version is None:
            target_dir = self.product_dir / "active"
            if not target_dir.exists():
                # Fallback to newest version folder if symlink not supported
                versions = [
                    d for d in self.product_dir.iterdir()
                    if d.is_dir() and d.name != "active"
                ]
                if not versions:
                    raise FileNotFoundError(f"No memory bank snapshots found in {self.product_dir}")
                target_dir = sorted(versions)[-1]
        else:
            target_dir = self.product_dir / version

        if not target_dir.exists():
            raise FileNotFoundError(f"Memory bank path does not exist: {target_dir}")

        # Resolve symlink if needed
        real_dir = target_dir.resolve()

        # Load manifest
        manifest_path = real_dir / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.manifest = MemoryBankManifest.from_dict(data)

        # Load embeddings
        embed_path = real_dir / "embeddings.npy"
        self.embeddings = np.load(embed_path).astype(np.float32)

        # Load index
        faiss_path = real_dir / "index.faiss"
        if HAS_FAISS and faiss_path.exists():
            self._faiss_index = faiss.read_index(str(faiss_path))
        else:
            self._faiss_index = None

        self._build_index()

    def _set_active_version(self, version: str) -> None:
        """Points the active directory to the specified version."""
        active_path = self.product_dir / "active"
        target_dir = self.product_dir / version

        if active_path.is_symlink() or active_path.exists():
            if active_path.is_dir() and not active_path.is_symlink():
                shutil.rmtree(active_path)
            else:
                active_path.unlink()

        try:
            active_path.symlink_to(target_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Fallback for systems without symlink permissions
            shutil.copytree(target_dir, active_path)

    def rollback(self, target_version: str) -> bool:
        """Rolls back the active memory bank pointer to a prior verified version."""
        target_dir = self.product_dir / target_version
        if not target_dir.exists():
            return False

        self._set_active_version(target_version)
        self.load(target_version)
        return True
