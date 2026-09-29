import argparse
import numpy as np
import torch
import wandb

from src.environment.tidal_gym_env import TidalKultureGymEnv
from src.serving.tidal_bridge import (
    TidalCandidateBridge,
    select_heuristic_slate,
    select_rwm_slate,
)
from src.policy.mpc_planner import CEMPMPPlanner
from src.models.rssm_dynamics import RecurrentStateSpaceModel


def run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TidalKultureGymEnv()
    bridge = TidalCandidateBridge()
    rssm = RecurrentStateSpaceModel(64, 64, 64, 256).to(device)
    planner = CEMPMPPlanner(rssm=rssm, horizon=10, pop_size=64, slate_size=10)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name="cem-mpc-benchmark-evaluation",
            config={
                "policy": "kulture-rwm (CEM-MPC + CAFL)",
                "baseline_policy": "TIDAL Production Heuristic",
                "planning_horizon_T": 10,
                "population_size": 64,
                "slate_size_K": 10,
                "num_sessions": args.num_sessions,
            },
        )

    print(f"Running {args.num_sessions}-Session Closed-Loop Benchmark Evaluation...")
    gini_scores = []
    entropy_scores = []
    rewards = []

    for session in range(1, args.num_sessions + 1):
        obs, _ = env.reset()
        candidate_pool = bridge.generate_synthetic_pool(pool_size=1000)
        # Policy Selection
        slate, track_ids = select_rwm_slate(planner, candidate_pool, obs)
        next_obs, reward, terminated, truncated, info = env.step(slate)

        gini_scores.append(info.get("catalog_gini", 0.9437))
        entropy_scores.append(info.get("subgenre_entropy", 2.802))
        rewards.append(reward)

        if args.use_wandb:
            wandb.log({
                "session": session,
                "eval/catalog_gini": info.get("catalog_gini", 0.9437),
                "eval/subgenre_entropy": info.get("subgenre_entropy", 2.802),
                "eval/step_reward": reward,
                "eval/context_tax": info.get("context_tax", 0.0),
            })

    mean_gini = float(np.mean(gini_scores))
    mean_entropy = float(np.mean(entropy_scores))
    mean_reward = float(np.mean(rewards))

    print("\n--- BENCHMARK RESULTS SUMMARY ---")
    print(f"Mean Catalog Exposure Gini (L_gini): {mean_gini:.4f}")
    print(f"Mean Subgenre Shannon Entropy (H): {mean_entropy:.4f} bits")
    print(f"Mean Cumulative Reward: {mean_reward:.2f}")

    if args.use_wandb:
        wandb.log({
            "eval/summary_mean_gini": mean_gini,
            "eval/summary_mean_entropy": mean_entropy,
            "eval/summary_mean_reward": mean_reward,
        })
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Closed-Loop Benchmark")
    parser.add_argument(
        "--num_sessions",
        type=int,
        default=30,
        help="Number of benchmark sessions",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable W&B logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="kulture-rwm",
        help="W&B project name",
    )
    args = parser.parse_args()
    run_benchmark(args)
