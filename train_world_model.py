"""
RSSM World Model Training Engine for kulture-rwm.

Fits the Action-Conditioned Recurrent State Space Model parameters Theta = {theta, phi}
on trajectory rollouts generated from TidalKultureGymEnv and TIDAL interaction logs.

Loss formulations:
1. Reconstruction MSE Loss (L_recon)
2. Geodesic KL Divergence Loss (L_kl) with balanced prior/posterior transitions
3. Prototype Simplex Repulsion Loss (L_proto) enforcing regular simplex geometry on S^(D-1)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Local module imports
from src.environment.tidal_gym_env import TidalKultureGymEnv
from src.serving.tidal_bridge import TidalCandidateBridge
from src.models.sasrec_backbone import project_to_hypersphere
from src.models.rssm_dynamics import RecurrentStateSpaceModel, LatentState
from src.models.predictors import PrototypeSimplexLoss


class ObservationDecoder(nn.Module):
    """
    Decodes the joint latent state (h_t, z_t) into predicted environment feedback.
    Reconstructs the 11-D observation vector: 10 slate stream indicators + 1 context tax.
    """

    def __init__(
        self,
        recurrent_dim: int = 128,
        stochastic_dim: int = 64,
        obs_dim: int = 11,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(recurrent_dim + stochastic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        features = torch.cat([h, z], dim=-1)
        return self.net(features)


class TrajectoryDataset(Dataset):
    """PyTorch Dataset holding interaction trajectories for RSSM training."""

    def __init__(
        self,
        actions: torch.Tensor,         # (N, T, ActionDim)
        observations: torch.Tensor,    # (N, T, ObsDim)
        track_embeddings: torch.Tensor # (N, T, SlateSize, TrackDim)
    ) -> None:
        self.actions = actions
        self.observations = observations
        self.track_embeddings = track_embeddings

    def __len__(self) -> int:
        return self.actions.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actions[idx], self.observations[idx], self.track_embeddings[idx]


def generate_rollout_dataset(
    num_episodes: int = 64,
    trajectory_length: int = 20,
    slate_size: int = 10,
    latent_dim: int = 64,
    obs_dim: int = 11,
    pool_size: int = 6000,
    device: torch.device = torch.device("cpu"),
) -> TrajectoryDataset:
    """
    Generates multi-step trajectory rollouts using TidalKultureGymEnv
    and candidate pools from TidalCandidateBridge.
    """
    env = TidalKultureGymEnv()
    bridge = TidalCandidateBridge(pool_size=pool_size, latent_dim=latent_dim, device=device)
    pool = bridge.generate_synthetic_pool()

    all_actions: List[torch.Tensor] = []
    all_observations: List[torch.Tensor] = []
    all_tracks: List[torch.Tensor] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        ep_actions: List[torch.Tensor] = []
        ep_obs: List[torch.Tensor] = []
        ep_tracks: List[torch.Tensor] = []

        for t in range(trajectory_length):
            # Sample random candidate slate on S^(D-1)
            slate_vecs, slate_genres, _ = bridge.select_heuristic_slate(
                pool=pool,
                user_pref=torch.tensor(obs, dtype=torch.float32, device=device),
                slate_size=slate_size,
            )

            # Action representation (mean pooled slate vector of dimension 64)
            action_repr = project_to_hypersphere(slate_vecs.mean(dim=0, keepdim=True)).squeeze(0)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(
                action=slate_vecs,
                subgenre_ids=slate_genres,
            )

            # Construct 11-D observation vector: 10 fatigued utilities + 1 context tax
            fatigued_utils = info["fatigued_utilities"]
            tau_c = torch.tensor([info["tau_c"]], dtype=torch.float32, device=device)
            obs_vector = torch.cat([fatigued_utils, tau_c], dim=-1)

            ep_actions.append(action_repr)
            ep_obs.append(obs_vector)
            ep_tracks.append(slate_vecs)

            obs = next_obs
            if terminated or truncated:
                # Pad remaining trajectory steps if terminated early
                while len(ep_actions) < trajectory_length:
                    ep_actions.append(ep_actions[-1])
                    ep_obs.append(ep_obs[-1])
                    ep_tracks.append(ep_tracks[-1])
                break

        all_actions.append(torch.stack(ep_actions, dim=0))
        all_observations.append(torch.stack(ep_obs, dim=0))
        all_tracks.append(torch.stack(ep_tracks, dim=0))

    actions_tensor = torch.stack(all_actions, dim=0)          # (N, T, ActionDim)
    obs_tensor = torch.stack(all_observations, dim=0)          # (N, T, ObsDim)
    tracks_tensor = torch.stack(all_tracks, dim=0)             # (N, T, K, D)

    return TrajectoryDataset(actions_tensor, obs_tensor, tracks_tensor)


def train_world_model(
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    beta_kl: float = 1.0,
    lambda_proto: float = 0.1,
    action_dim: int = 64,
    recurrent_dim: int = 128,
    stochastic_dim: int = 64,
    obs_dim: int = 11,
    hidden_dim: int = 128,
    num_subgenres: int = 7,
    use_wandb: bool = False,
    project: str = "kulture-rwm",
    entity: Optional[str] = None,
    save_path: Optional[str] = "checkpoints/rssm_world_model.pt",
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    print("=" * 80)
    print(" " * 20 + "kulture-rwm RSSM World Model Training")
    print("=" * 80)
    print(f"Configurations:")
    print(f"  Training Epochs:        {epochs}")
    print(f"  Batch Size:             {batch_size}")
    print(f"  Initial Learning Rate:  {learning_rate}")
    print(f"  Beta KL Weight:         {beta_kl}")
    print(f"  Lambda Proto Weight:    {lambda_proto}")
    print(f"  Action Dimension:       {action_dim}")
    print(f"  Recurrent Dimension:    {recurrent_dim}")
    print(f"  Stochastic Dimension:   {stochastic_dim}")
    print(f"  Observation Dimension:  {obs_dim}")
    print(f"  Compute Device:         {device}")
    print("-" * 80)

    # Instantiate Model Architectures
    rssm = RecurrentStateSpaceModel(
        action_dim=action_dim,
        recurrent_dim=recurrent_dim,
        stochastic_dim=stochastic_dim,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    decoder = ObservationDecoder(
        recurrent_dim=recurrent_dim,
        stochastic_dim=stochastic_dim,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    proto_loss_fn = PrototypeSimplexLoss(
        num_prototypes=num_subgenres,
        dim=stochastic_dim,
        lambda_sep=0.1,
    ).to(device)

    # Optimizer & Cosine Annealing Learning Rate Scheduler
    trainable_params = list(rssm.parameters()) + list(decoder.parameters()) + list(proto_loss_fn.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Generate Training Trajectories
    print("Generating training rollouts from TidalKultureGymEnv...")
    dataset = generate_rollout_dataset(
        num_episodes=64,
        trajectory_length=15,
        slate_size=10,
        latent_dim=stochastic_dim,
        obs_dim=obs_dim,
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset generated: {len(dataset)} trajectories.")

    # Initialize Weights & Biases if enabled
    if use_wandb:
        try:
            import wandb

            wandb.init(
                project=project,
                entity=entity,
                name="rssm-world-model-training",
                config={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "beta_kl": beta_kl,
                    "lambda_proto": lambda_proto,
                    "action_dim": action_dim,
                    "recurrent_dim": recurrent_dim,
                    "stochastic_dim": stochastic_dim,
                    "obs_dim": obs_dim,
                },
            )
        except Exception as e:
            print(f"Warning: W&B initialization failed: {e}")
            use_wandb = False

    # Training Loop
    rssm.train()
    decoder.train()
    proto_loss_fn.train()

    final_metrics: Dict[str, float] = {}

    print("\nStarting Training Execution...")
    for epoch in range(1, epochs + 1):
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_proto = 0.0
        epoch_total = 0.0
        batch_count = 0

        for b_actions, b_obs, b_tracks in dataloader:
            b_actions = b_actions.to(device)
            b_obs = b_obs.to(device)
            b_tracks = b_tracks.to(device)
            B, T, _ = b_actions.shape

            optimizer.zero_grad()

            state = rssm.initial_state(batch_size=B, device=device)
            loss_recon_traj = torch.tensor(0.0, device=device)
            loss_kl_traj = torch.tensor(0.0, device=device)

            for t in range(T):
                a_t = b_actions[:, t, :]
                o_t = b_obs[:, t, :]

                # Posterior inference observe_step: q_phi(s_t | s_<t, a_<t, o_t)
                state = rssm.observe_step(state, a_t, o_t)

                # Reconstruction: p(o_t | h_t, z_t)
                pred_o = decoder(state.h, state.z)
                loss_recon_traj = loss_recon_traj + F.mse_loss(pred_o, o_t)

                # Geodesic KL divergence: D_KL(q_phi || p_theta)
                kl_val, _ = rssm.compute_kl_divergence(
                    state.post_mu, state.post_std, state.prior_mu, state.prior_std
                )
                loss_kl_traj = loss_kl_traj + kl_val

            loss_recon = loss_recon_traj / T
            loss_kl = loss_kl_traj / T

            # Prototype Simplex Repulsion Loss over candidate track embeddings
            flat_tracks = b_tracks.view(-1, stochastic_dim)
            loss_proto, _, _ = proto_loss_fn(flat_tracks)

            # Multi-term unified objective
            loss_total = loss_recon + beta_kl * loss_kl + lambda_proto * loss_proto

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_recon += loss_recon.item()
            epoch_kl += loss_kl.item()
            epoch_proto += loss_proto.item()
            epoch_total += loss_total.item()
            batch_count += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        mean_recon = epoch_recon / max(1, batch_count)
        mean_kl = epoch_kl / max(1, batch_count)
        mean_proto = epoch_proto / max(1, batch_count)
        mean_total = epoch_total / max(1, batch_count)

        final_metrics = {
            "train/loss_total": mean_total,
            "train/loss_recon": mean_recon,
            "train/loss_kl": mean_kl,
            "train/loss_proto": mean_proto,
            "train/learning_rate": current_lr,
        }

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Total Loss: {mean_total:7.4f} | "
            f"Recon: {mean_recon:7.4f} | "
            f"KL: {mean_kl:7.4f} | "
            f"Proto: {mean_proto:7.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if use_wandb:
            import wandb
            wandb.log(final_metrics, step=epoch)

    if use_wandb:
        import wandb
        wandb.finish()

    # Save trained checkpoint if specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "rssm_state_dict": rssm.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "proto_loss_state_dict": proto_loss_fn.state_dict(),
                "final_metrics": final_metrics,
            },
            save_path,
        )
        print(f"\nModel checkpoint successfully saved to: {save_path}")

    print("=" * 80)
    print("RSSM Training Completed Successfully.")
    print("=" * 80)
    return final_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="RSSM World Model Training Engine")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--beta_kl", type=float, default=1.0, help="KL divergence weighting beta (default: 1.0)")
    parser.add_argument("--lambda_proto", type=float, default=0.1, help="Prototype simplex weight (default: 0.1)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--project", type=str, default="kulture-rwm", help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--save_path", type=str, default="checkpoints/rssm_world_model.pt", help="Checkpoint save path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu")
    train_world_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        beta_kl=args.beta_kl,
        lambda_proto=args.lambda_proto,
        use_wandb=args.use_wandb,
        project=args.project,
        entity=args.entity,
        save_path=args.save_path,
        device=device,
    )
