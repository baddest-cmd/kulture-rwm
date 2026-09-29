"""
TIDAL Candidate Pool Ingestion Bridge and Serving Adapter.

Adapts candidate pools (top-6,000 candidate track slates emitted by SASRec
in tidal-algorithmic-mixes) for closed-loop evaluation and inference with
the Action-Conditioned Recurrent World Model (kulture-rwm).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.sasrec_backbone import project_to_hypersphere


@dataclass
class TidalCandidatePool:
    """Represents a candidate track pool conforming to TIDAL's PySpark schema."""

    track_ids: torch.Tensor          # Shape: (M,), dtype=torch.long
    artist_ids: torch.Tensor         # Shape: (M,), dtype=torch.long
    subgenre_ids: torch.Tensor       # Shape: (M,), dtype=torch.long
    popularity_scores: torch.Tensor  # Shape: (M,), dtype=torch.float32 in [0, 1]
    embeddings: torch.Tensor         # Shape: (M, D), unit vectors on S^(D-1)
    is_known_artist: torch.Tensor    # Shape: (M,), dtype=torch.bool
    subgenre_labels: List[str]

    def __len__(self) -> int:
        return self.track_ids.shape[0]

    def to(self, device: torch.device) -> TidalCandidatePool:
        return TidalCandidatePool(
            track_ids=self.track_ids.to(device),
            artist_ids=self.artist_ids.to(device),
            subgenre_ids=self.subgenre_ids.to(device),
            popularity_scores=self.popularity_scores.to(device),
            embeddings=self.embeddings.to(device),
            is_known_artist=self.is_known_artist.to(device),
            subgenre_labels=self.subgenre_labels,
        )


class TidalCandidateBridge:
    """
    Ingestion adapter that parses or synthesises top-6,000 candidate track pools
    conforming to TIDAL's open-source PySpark Daily Discovery transformations.
    """

    SUBGENRE_NAMES = [
        "Amapiano",
        "Gqom",
        "Pop",
        "HipHop",
        "Maskandi",
        "Lekompo",
        "Bacardi",
    ]

    def __init__(
        self,
        pool_size: int = 6000,
        latent_dim: int = 64,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = 42,
    ) -> None:
        self.pool_size = pool_size
        self.latent_dim = latent_dim
        self.device = device
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def parse_dataframe(self, df: pd.DataFrame) -> TidalCandidatePool:
        """
        Parses a pandas DataFrame matching TIDAL's PySpark candidate schema:
        Required columns: track_id, artist_id, cluster / subgenre_id.
        Optional columns: popularity / popularity_score, embedding, is_known_artist / is_mainstream.
        """
        required_cols = {"track_id", "artist_id"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame missing required TIDAL columns: {required_cols - set(df.columns)}")

        num_tracks = len(df)
        track_ids = torch.tensor(df["track_id"].values, dtype=torch.long)
        artist_ids = torch.tensor(df["artist_id"].values, dtype=torch.long)

        # Subgenre / cluster ID
        if "subgenre_id" in df.columns:
            subgenre_ids = torch.tensor(df["subgenre_id"].values, dtype=torch.long)
        elif "cluster" in df.columns:
            subgenre_ids = torch.tensor(df["cluster"].values, dtype=torch.long)
        else:
            subgenre_ids = torch.tensor(self.rng.integers(0, len(self.SUBGENRE_NAMES), size=num_tracks), dtype=torch.long)

        # Popularity score
        if "popularity" in df.columns:
            pops = torch.tensor(df["popularity"].values, dtype=torch.float32)
        elif "popularity_score" in df.columns:
            pops = torch.tensor(df["popularity_score"].values, dtype=torch.float32)
        else:
            pops = torch.tensor(self.rng.beta(2.0, 5.0, size=num_tracks), dtype=torch.float32)

        # Hyperspherical embeddings
        if "embedding" in df.columns:
            raw_emb = torch.tensor(np.stack(df["embedding"].values), dtype=torch.float32)
            embeddings = project_to_hypersphere(raw_emb)
        else:
            raw = torch.randn(num_tracks, self.latent_dim)
            embeddings = project_to_hypersphere(raw)

        # Known artist cap flag (20% cap rule)
        if "is_known_artist" in df.columns:
            is_known = torch.tensor(df["is_known_artist"].values, dtype=torch.bool)
        elif "is_mainstream" in df.columns:
            is_known = torch.tensor(df["is_mainstream"].values, dtype=torch.bool)
        else:
            is_known = pops > torch.quantile(pops, 0.80)

        subgenre_labels = [
            self.SUBGENRE_NAMES[c.item() % len(self.SUBGENRE_NAMES)]
            for c in subgenre_ids
        ]

        return TidalCandidatePool(
            track_ids=track_ids.to(self.device),
            artist_ids=artist_ids.to(self.device),
            subgenre_ids=subgenre_ids.to(self.device),
            popularity_scores=pops.to(self.device),
            embeddings=embeddings.to(self.device),
            is_known_artist=is_known.to(self.device),
            subgenre_labels=subgenre_labels,
        )

    def parse_parquet(self, parquet_path: Union[str, Path]) -> TidalCandidatePool:
        """Parses a Parquet file emitted by TIDAL's PySpark transformations."""
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet candidate file not found at: {path}")
        df = pd.read_parquet(path)
        return self.parse_dataframe(df)

    def generate_synthetic_pool(self, pool_size: Optional[int] = None) -> TidalCandidatePool:
        """
        High-throughput synthetic generator emitting a candidate pool conforming to
        TIDAL's exact PySpark schema (6,000 candidate track slates).
        """
        size = pool_size if pool_size is not None else self.pool_size
        track_ids = torch.arange(1, size + 1, dtype=torch.long)

        # Power-law / Zipf artist distribution
        num_artists = max(50, size // 5)
        artist_weights = 1.0 / (np.arange(1, num_artists + 1) ** 0.85)
        artist_weights /= artist_weights.sum()
        np_artist_ids = self.rng.choice(np.arange(1, num_artists + 1), size=size, p=artist_weights)
        artist_ids = torch.tensor(np_artist_ids, dtype=torch.long)

        # Subgenre prototype clusters
        num_subgenres = len(self.SUBGENRE_NAMES)
        np_subgenres = self.rng.integers(0, num_subgenres, size=size)
        subgenre_ids = torch.tensor(np_subgenres, dtype=torch.long)

        # Popularity scores in [0, 1] following beta distribution
        np_pops = self.rng.beta(2.0, 5.0, size=size).astype(np.float32)
        popularity_scores = torch.tensor(np_pops, dtype=torch.float32)

        # Flag top 20% known / mainstream artists
        unique_artists, counts = np.unique(np_artist_ids, return_counts=True)
        top_cutoff = np.percentile(counts, 80)
        mainstream_artists = set(unique_artists[counts >= top_cutoff])
        is_known_artist = torch.tensor(
            [a in mainstream_artists for a in np_artist_ids],
            dtype=torch.bool,
        )

        # Unit hypersphere embeddings S^(D-1) with epsilon inside sqrt
        raw_embeddings = torch.randn(size, self.latent_dim, generator=torch.manual_seed(self.seed or 42))
        embeddings = project_to_hypersphere(raw_embeddings)

        subgenre_labels = [
            self.SUBGENRE_NAMES[c % num_subgenres] for c in np_subgenres
        ]

        return TidalCandidatePool(
            track_ids=track_ids.to(self.device),
            artist_ids=artist_ids.to(self.device),
            subgenre_ids=subgenre_ids.to(self.device),
            popularity_scores=popularity_scores.to(self.device),
            embeddings=embeddings.to(self.device),
            is_known_artist=is_known_artist.to(self.device),
            subgenre_labels=subgenre_labels,
        )

    def select_heuristic_slate(
        self,
        pool: TidalCandidatePool,
        user_pref: torch.Tensor,
        slate_size: int = 10,
        max_artist_cap: float = 0.2,
        max_per_cluster: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes TIDAL Production Baseline heuristic filtering:
        - Rank candidates by dot product (cosine similarity) with user preference.
        - Enforce 20% known artist cap (at most 2 known artists per slate of 10).
        - Enforce max 2 tracks per subgenre cluster.
        
        Returns
        -------
        slate_embeddings : torch.Tensor of shape (slate_size, D)
        slate_subgenre_ids : torch.Tensor of shape (slate_size,)
        slate_indices : torch.Tensor of shape (slate_size,)
        """
        # User pref on device
        u = user_pref.to(pool.embeddings.device)
        if u.dim() > 1:
            u = u.squeeze()

        # Compute cosine similarity
        sims = torch.matmul(pool.embeddings, u)  # (M,)
        # Baseline combines similarity + popularity bias
        ranking_scores = sims + 0.3 * pool.popularity_scores
        sorted_indices = torch.argsort(ranking_scores, descending=True)

        selected_indices: List[int] = []
        cluster_counts: Dict[int, int] = {}
        known_artist_count = 0
        max_known = int(slate_size * max_artist_cap)

        for idx in sorted_indices.tolist():
            if len(selected_indices) >= slate_size:
                break

            cluster = pool.subgenre_ids[idx].item()
            is_known = bool(pool.is_known_artist[idx].item())

            # Rule 1: Subgenre cluster limit
            if cluster_counts.get(cluster, 0) >= max_per_cluster:
                continue

            # Rule 2: 20% Known artist cap
            if is_known and known_artist_count >= max_known:
                continue

            selected_indices.append(idx)
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            if is_known:
                known_artist_count += 1

        # Fallback if filters were too strict to fill slate_size
        if len(selected_indices) < slate_size:
            for idx in sorted_indices.tolist():
                if len(selected_indices) >= slate_size:
                    break
                if idx not in selected_indices:
                    selected_indices.append(idx)

        idx_tensor = torch.tensor(selected_indices, dtype=torch.long, device=pool.embeddings.device)
        return pool.embeddings[idx_tensor], pool.subgenre_ids[idx_tensor], idx_tensor

    def select_rwm_slate(
        self,
        pool: TidalCandidatePool,
        planned_action: torch.Tensor,
        slate_size: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects candidate tracks matching the planned action from CEM planner.
        If planned_action is shaped (slate_size, D), selects top candidate for each slot.
        If planned_action is shaped (D,) or (slate_size * D,), pools and ranks candidates.

        Returns
        -------
        slate_embeddings : torch.Tensor of shape (slate_size, D)
        slate_subgenre_ids : torch.Tensor of shape (slate_size,)
        slate_indices : torch.Tensor of shape (slate_size,)
        """
        device = pool.embeddings.device
        action = planned_action.to(device)

        if action.dim() == 2 and action.shape[0] == slate_size and action.shape[1] == self.latent_dim:
            # Slot-wise nearest-neighbour matching with uniqueness
            selected_indices: List[int] = []
            used_set = set()
            for k in range(slate_size):
                target = action[k]
                sims = torch.matmul(pool.embeddings, target)
                sorted_idx = torch.argsort(sims, descending=True).tolist()
                for c_idx in sorted_idx:
                    if c_idx not in used_set:
                        selected_indices.append(c_idx)
                        used_set.add(c_idx)
                        break
            idx_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
        else:
            # Flattened or single-vector action: match top-K closest candidates
            flat_target = action.view(-1)
            if flat_target.shape[0] >= self.latent_dim:
                target_vec = project_to_hypersphere(flat_target[:self.latent_dim].unsqueeze(0)).squeeze(0)
            else:
                target_vec = project_to_hypersphere(torch.randn(self.latent_dim, device=device))

            sims = torch.matmul(pool.embeddings, target_vec)
            top_k_indices = torch.topk(sims, k=slate_size).indices
            idx_tensor = top_k_indices

        return pool.embeddings[idx_tensor], pool.subgenre_ids[idx_tensor], idx_tensor


def select_rwm_slate(
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Selects candidate tracks matching the planned action or CEM planner.

    Can be invoked in two ways:
    1. Benchmark evaluation style:
       ``slate, track_ids = select_rwm_slate(planner, candidate_pool, obs)``
    2. Bridge/Pool style:
       ``slate, track_ids = select_rwm_slate(candidate_pool, planned_action, slate_size=10)``

    Returns
    -------
    slate : torch.Tensor of shape (slate_size, D)
    track_ids : torch.Tensor of shape (slate_size,)
    """
    from src.models.rssm_dynamics import LatentState

    # Signature 1: select_rwm_slate(planner, candidate_pool, obs)
    if len(args) >= 2 and hasattr(args[0], "plan"):
        planner = args[0]
        pool = args[1]
        obs = args[2] if len(args) > 2 else kwargs.get("obs", None)
        slate_size = kwargs.get("slate_size", getattr(planner, "slate_size", 10) or 10)

        device = getattr(planner, "device", pool.embeddings.device)
        rssm = getattr(planner, "rssm", None)

        if rssm is not None:
            batch_size = 1
            h = torch.zeros(batch_size, rssm.recurrent_dim, device=device)
            if isinstance(obs, np.ndarray):
                obs_t = torch.from_numpy(obs).float().to(device)
            elif isinstance(obs, torch.Tensor):
                obs_t = obs.float().to(device)
            else:
                obs_t = torch.randn(rssm.stochastic_dim, device=device)

            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            if obs_t.shape[-1] == rssm.stochastic_dim:
                z = project_to_hypersphere(obs_t)
            else:
                z = project_to_hypersphere(torch.randn(batch_size, rssm.stochastic_dim, device=device))

            mu, std = rssm._parameterise_distribution(rssm.prior_mlp(h))
            init_state = LatentState(h=h, z=z, prior_mu=mu, prior_std=std)
            planned_action = planner.plan(init_state)
        else:
            planned_action = torch.randn(slate_size, pool.embeddings.shape[-1], device=device)

        bridge = TidalCandidateBridge(latent_dim=pool.embeddings.shape[-1], device=pool.embeddings.device)
        slate_embeddings, _, idx_tensor = bridge.select_rwm_slate(
            pool=pool,
            planned_action=planned_action,
            slate_size=slate_size,
        )
        track_ids = pool.track_ids[idx_tensor]
        return slate_embeddings, track_ids

    # Signature 2: select_rwm_slate(candidate_pool, planned_action, ...)
    elif len(args) >= 2 and isinstance(args[0], TidalCandidatePool):
        pool = args[0]
        planned_action = args[1]
        slate_size = kwargs.get("slate_size", 10)
        return_details = kwargs.pop("return_details", False)
        bridge = TidalCandidateBridge(latent_dim=pool.embeddings.shape[-1], device=pool.embeddings.device)
        slate_embeddings, subgenre_ids, idx_tensor = bridge.select_rwm_slate(
            pool=pool,
            planned_action=planned_action,
            slate_size=slate_size,
            **kwargs,
        )
        if return_details:
            return slate_embeddings, subgenre_ids, idx_tensor
        return slate_embeddings, pool.track_ids[idx_tensor]

    raise ValueError(f"Invalid arguments for select_rwm_slate: {args}, {kwargs}")


def select_heuristic_slate(
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Executes TIDAL Production Baseline heuristic filtering.

    Can be invoked in two ways:
    1. Direct call:
       ``slate, track_ids = select_heuristic_slate(candidate_pool, user_pref, slate_size=10)``
    2. Bridge wrapper:
       ``select_heuristic_slate(bridge, candidate_pool, user_pref, ...)``

    Returns
    -------
    slate : torch.Tensor of shape (slate_size, D)
    track_ids : torch.Tensor of shape (slate_size,)
    """
    if len(args) >= 1 and isinstance(args[0], TidalCandidateBridge):
        return args[0].select_heuristic_slate(*args[1:], **kwargs)

    elif len(args) >= 2 and isinstance(args[0], TidalCandidatePool):
        pool = args[0]
        user_pref = args[1]
        if isinstance(user_pref, np.ndarray):
            user_pref = torch.from_numpy(user_pref).float()
        return_details = kwargs.pop("return_details", False)
        bridge = TidalCandidateBridge(latent_dim=pool.embeddings.shape[-1], device=pool.embeddings.device)
        slate_vecs, slate_genres, slate_indices = bridge.select_heuristic_slate(
            pool=pool, user_pref=user_pref, **kwargs
        )
        if return_details:
            return slate_vecs, slate_genres, slate_indices
        return slate_vecs, pool.track_ids[slate_indices]

    raise ValueError(f"Invalid arguments for select_heuristic_slate: {args}, {kwargs}")


__all__ = [
    "TidalCandidatePool",
    "TidalCandidateBridge",
    "select_rwm_slate",
    "select_heuristic_slate",
]

