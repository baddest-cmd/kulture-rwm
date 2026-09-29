import argparse
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from src.models.rssm_dynamics import RecurrentStateSpaceModel
from src.models.predictors import PrototypeSimplexLoss
from src.environment.tidal_gym_env import TidalKultureGymEnv

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialise W&B tracking run if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name="rssm-world-model-training",
            config={
                "architecture": "Action-Conditioned Hyperspherical RSSM",
                "manifold": "S^(D-1) Unit Hypersphere",
                "latent_dim": 64,
                "stochastic_dim": 64,
                "deterministic_dim": 256,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lambda_proto": 0.1,
                "beta_kl": 1.0,
            }
        )

    # Instantiate model components
    rssm = RecurrentStateSpaceModel(
        state_dim=64,
        action_dim=64,
        stochastic_dim=64,
        deterministic_dim=256
    ).to(device)
    
    proto_criterion = PrototypeSimplexLoss(num_prototypes=10, embed_dim=64).to(device)
    recon_criterion = nn.MSELoss()
    
    optimizer = AdamW(rssm.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Starting training for {args.epochs} epochs on device: {device}")

    rssm.train()
    for epoch in range(1, args.epochs + 1):
        # Simulated training batch forward pass
        dummy_state = torch.randn(args.batch_size, 64, device=device)
        dummy_action = torch.randn(args.batch_size, 64, device=device)
        dummy_target = torch.randn(args.batch_size, 11, device=device)
        
        optimizer.zero_grad()
        
        # RSSM Forward Transition
        prior_dist, posterior_dist, h_next = rssm.imagine_step(dummy_state, dummy_action)
        obs_pred = rssm.decode_observation(h_next, posterior_dist.sample())
        
        # Loss Formulations
        loss_recon = recon_criterion(obs_pred, dummy_target)
        loss_kl = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist).mean()
        loss_proto = proto_criterion(dummy_state)
        
        total_loss = loss_recon + 1.0 * loss_kl + 0.1 * loss_proto
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Log metrics to console
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {total_loss.item():.4f} (Recon: {loss_recon.item():.4f}, KL: {loss_kl.item():.4f}, Proto: {loss_proto.item():.4f})")

        # Log metrics to W&B
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss_total": total_loss.item(),
                "train/loss_recon": loss_recon.item(),
                "train/loss_kl": loss_kl.item(),
                "train/loss_proto": loss_proto.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
            })

    # Save trained model weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(rssm.state_dict(), "checkpoints/rssm_world_model.pt")
    print("Model checkpoint saved to checkpoints/rssm_world_model.pt")

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train kulture-rwm RSSM World Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="kulture-rwm", help="W&B project name")
    args = parser.parse_args()
    train(args)
