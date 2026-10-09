"""Unit tests for greedy coreset subsampling and memory bank querying."""

import numpy as np
import pytest
from src.memory_bank import GreedyCoresetSampler, ProductMemoryBank


def test_coreset_sampler_ratio():
    sampler = GreedyCoresetSampler(sampling_ratio=0.10, projection_dim=64, random_seed=42)
    features = np.random.randn(500, 256).astype(np.float32)

    coreset, indices = sampler.sample(features)
    assert coreset.shape == (50, 256)
    assert len(indices) == 50
    assert len(np.unique(indices)) == 50


def test_memory_bank_nearest_neighbor_accuracy(tmp_path):
    bank = ProductMemoryBank(product_id="test_unit_prod", artifacts_dir=tmp_path)
    # 100 features in 64 dims
    rng = np.random.RandomState(42)
    raw_patches = rng.randn(100, 64).astype(np.float32)

    bank.fit_from_patches(
        raw_patches=raw_patches,
        reference_count=5,
        backbone_name="mock_backbone",
        layers=["layer2"],
        input_resolution=(128, 128),
        coreset_ratio=0.20,
        projection_dim=32,
    )

    # Query with exact vector present in bank
    sample_vec = bank.embeddings[0:1]
    dists, idxs = bank.query_nearest_neighbors(sample_vec, k=1)

    assert dists.shape == (1, 1)
    # Distance to identical point in memory bank should be near 0
    assert dists[0, 0] < 1e-4
    assert idxs[0, 0] == 0


def test_memory_bank_save_load_roundtrip(tmp_path):
    bank = ProductMemoryBank(product_id="test_product_xyz", artifacts_dir=tmp_path)
    raw_patches = np.random.randn(50, 32).astype(np.float32)

    bank.fit_from_patches(
        raw_patches=raw_patches,
        reference_count=3,
        backbone_name="mock_backbone",
        layers=["layer2"],
        input_resolution=(256, 256),
        coreset_ratio=0.5,
        version="v1.0.0",
    )
    saved_path = bank.save(version="v1.0.0", make_active=True)
    assert (saved_path / "embeddings.npy").exists()
    assert (saved_path / "manifest.json").exists()

    # Create fresh bank instance and load
    loaded_bank = ProductMemoryBank(product_id="test_product_xyz", artifacts_dir=tmp_path)
    loaded_bank.load()
    assert loaded_bank.is_loaded
    assert loaded_bank.manifest.version == "v1.0.0"
    assert loaded_bank.embeddings.shape == bank.embeddings.shape
    np.testing.assert_allclose(loaded_bank.embeddings, bank.embeddings)
