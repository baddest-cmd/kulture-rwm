"""
PyTorch Dataset and DataLoader for multi-step recommendation trajectories.

Batches sequence actions, user feedback, and Context Tax signals for
training the SASRec sequence backbone and RSSM transition dynamics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceTrajectoryDataset(Dataset):
    """
    Trajectory dataset containing action-observation sequences for offline training.
    """

    def __init__(self, trajectories: List[Dict[str, np.ndarray]]) -> None:
        self.trajectories = trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.trajectories[idx]
        return {
            "user_id": torch.from_numpy(item["user_id"]).squeeze(),
            "actions": torch.from_numpy(item["actions"]).long(),
            "streams": torch.from_numpy(item["streams"]).float(),
            "context_taxes": torch.from_numpy(item["context_taxes"]).float(),
        }


def trajectory_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Vectorised collate function ensuring uniform tensor dimensions across batch.
    """
    user_ids = torch.stack([b["user_id"] for b in batch], dim=0)
    actions = torch.stack([b["actions"] for b in batch], dim=0)
    streams = torch.stack([b["streams"] for b in batch], dim=0)
    context_taxes = torch.stack([b["context_taxes"] for b in batch], dim=0)

    return {
        "user_ids": user_ids,
        "actions": actions,           # Shape: (Batch, TrajectoryLen, SlateSize)
        "streams": streams,           # Shape: (Batch, TrajectoryLen, SlateSize)
        "context_taxes": context_taxes, # Shape: (Batch, TrajectoryLen, 1)
    }


def create_trajectory_dataloader(
    trajectories: List[Dict[str, np.ndarray]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Creates a configured DataLoader for sequence trajectories."""
    dataset = SequenceTrajectoryDataset(trajectories)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=trajectory_collate_fn,
        num_workers=num_workers,
    )
