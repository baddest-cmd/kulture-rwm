"""
Unit tests for Phase 4: Serving & Ingestion Bridge (TidalCandidateBridge).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.serving.tidal_bridge import TidalCandidateBridge, TidalCandidatePool


@pytest.fixture
def bridge():
    return TidalCandidateBridge(pool_size=1000, latent_dim=64, seed=42)


def test_synthetic_pool_schema_and_shapes(bridge):
    """Verify synthetic pool matches TIDAL PySpark schema and dimensions."""
    pool = bridge.generate_synthetic_pool(pool_size=1000)

    assert isinstance(pool, TidalCandidatePool)
    assert len(pool) == 1000
    assert pool.track_ids.shape == (1000,)
    assert pool.artist_ids.shape == (1000,)
    assert pool.subgenre_ids.shape == (1000,)
    assert pool.popularity_scores.shape == (1000,)
    assert pool.embeddings.shape == (1000, 64)
    assert pool.is_known_artist.shape == (1000,)
    assert len(pool.subgenre_labels) == 1000

    # Embeddings must be strictly unit-norm on S^(D-1)
    norms = torch.norm(pool.embeddings, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Popularity scores in [0, 1]
    assert (pool.popularity_scores >= 0.0).all()
    assert (pool.popularity_scores <= 1.0).all()


def test_parse_dataframe(bridge):
    """Verify ingestion of external DataFrame conforming to TIDAL PySpark schema."""
    M = 200
    df = pd.DataFrame(
        {
            "track_id": np.arange(1, M + 1),
            "artist_id": np.random.randint(1, 50, size=M),
            "cluster": np.random.randint(0, 5, size=M),
            "popularity": np.random.uniform(0.1, 0.9, size=M),
            "is_mainstream": np.random.choice([True, False], size=M),
        }
    )

    pool = bridge.parse_dataframe(df)
    assert len(pool) == M
    assert pool.track_ids.shape == (M,)
    assert pool.embeddings.shape == (M, 64)
    norms = torch.norm(pool.embeddings, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_heuristic_baseline_filtering(bridge):
    """Verify heuristic rule filters: 20% known artist cap and max 2 per cluster."""
    pool = bridge.generate_synthetic_pool(pool_size=1000)
    user_pref = torch.randn(64)
    user_pref = user_pref / torch.norm(user_pref)

    slate_vecs, slate_genres, slate_indices = bridge.select_heuristic_slate(
        pool=pool,
        user_pref=user_pref,
        slate_size=10,
        max_artist_cap=0.2,
        max_per_cluster=2,
    )

    assert slate_vecs.shape == (10, 64)
    assert slate_genres.shape == (10,)
    assert slate_indices.shape == (10,)

    # Check max 2 tracks per subgenre cluster
    unique_genres, counts = torch.unique(slate_genres, return_counts=True)
    assert (counts <= 2).all(), f"Cluster counts exceeded limit: {counts}"

    # Check 20% known artist cap: at most 2 out of 10 tracks from known artists
    known_count = pool.is_known_artist[slate_indices].sum().item()
    assert known_count <= 2, f"Known artist count exceeded cap: {known_count}"


def test_rwm_slate_selection(bridge):
    """Verify RWM action-to-candidate slate selection produces valid unique tracks."""
    pool = bridge.generate_synthetic_pool(pool_size=1000)
    planned_action = torch.randn(10, 64)
    planned_action = planned_action / torch.norm(planned_action, dim=-1, keepdim=True)

    slate_vecs, slate_genres, slate_indices = bridge.select_rwm_slate(
        pool=pool,
        planned_action=planned_action,
        slate_size=10,
    )

    assert slate_vecs.shape == (10, 64)
    assert slate_genres.shape == (10,)
    assert slate_indices.shape == (10,)

    # Track indices must be unique within the slate
    assert len(torch.unique(slate_indices)) == 10
