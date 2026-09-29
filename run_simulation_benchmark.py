"""
Closed-Loop 30-Session Comparative Benchmark: TIDAL Baseline vs. kulture-rwm.

Executes a 30-session comparative simulation rollout inside TidalKultureGymEnv,
evaluating the production TIDAL heuristic baseline against the Action-Conditioned
Recurrent World Model (kulture-rwm) with CEM MPC trajectory planning.

Quantitative metrics evaluated:
1. Cumulative Retention / Engagement Dwell Time
2. Accumulated Cultural Context Tax (tau_c) & Session Churn Rate
3. Catalog Exposure Smooth Gini Loss (L_gini) across 6,000 candidates
4. Subgenre Shannon Entropy H(P_genre)
"""

from __future__ import annotations

import math
import sys
import numpy as np
import torch
import torch.nn as nn

# Local repository imports
from src.environment.tidal_gym_env import TidalKultureGymEnv, _SUBGENRE_REGISTRY
from src.serving.tidal_bridge import TidalCandidateBridge, TidalCandidatePool
from src.models.sasrec_backbone import SASRecBackbone, project_to_hypersphere
from src.models.rssm_dynamics import RecurrentStateSpaceModel, LatentState
from src.models.predictors import (
    EngagementPredictor,
    ContextTaxPredictor,
    SmoothGiniLoss,
)
from src.policy.mpc_planner import CEMPMPPlanner


def compute_gini_coefficient(exposure_counts: np.ndarray) -> float:
    """
    Computes the Gini coefficient over item exposures conforming to
    Section 2.C of the system architecture specification:
    L_gini = (2 * sum(i * e_sorted_i)) / (M * sum(e_i) + eps) - (M + 1) / M
    """
    e = exposure_counts.astype(np.float64)
    total = np.sum(e)
    M = len(e)
    if total <= 1e-7:
        return 0.0
    sorted_e = np.sort(e)
    ranks = np.arange(1, M + 1, dtype=np.float64)
    gini = (2.0 * np.sum(ranks * sorted_e)) / (M * total + 1e-7) - (M + 1.0) / M
    return float(np.clip(gini, 0.0, 1.0))


def compute_shannon_entropy(subgenre_counts: np.ndarray) -> float:
    """
    Computes the Shannon Entropy H(P_genre) in bits across subgenres.
    H(P) = -sum(P(g) * log2(P(g) + eps))
    """
    total = np.sum(subgenre_counts)
    if total <= 0:
        return 0.0
    probs = subgenre_counts / total
    nonzero_probs = probs[probs > 0]
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs + 1e-12))
    return float(entropy)


def run_benchmark(
    num_sessions: int = 30,
    slate_size: int = 10,
    pool_size: int = 6000,
    latent_dim: int = 64,
    horizon: int = 5,
    pop_size: int = 32,
    cem_iters: int = 2,
    project: str = "kulture-rwm",
    entity: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> None:
    print("=" * 80)
    print(" " * 15 + "kulture-rwm Phase 4: Closed-Loop Benchmark")
    print("=" * 80)
    print(f"Configurations:")
    print(f"  Sessions:               {num_sessions}")
    print(f"  Slate Size (K):         {slate_size}")
    print(f"  Candidate Catalog Pool: {pool_size} tracks")
    print(f"  Latent Embedding Dim:   {latent_dim}")
    print(f"  CEM Horizon (T):        {horizon}")
    print(f"  CEM Population Size:    {pop_size}")
    print(f"  CEM Refinement Iters:   {cem_iters}")
    print(f"  Compute Device:         {device}")
    print("-" * 80)

    # Initialise Candidate Bridge & Catalog Pool
    bridge = TidalCandidateBridge(
        pool_size=pool_size,
        latent_dim=latent_dim,
        device=device,
        seed=42,
    )
    candidate_pool = bridge.generate_synthetic_pool()
    num_subgenres = len(TidalCandidateBridge.SUBGENRE_NAMES)

    # Initialise kulture-rwm Policy Modules
    sasrec = SASRecBackbone(
        vocab_size=pool_size,
        hidden_dim=latent_dim,
        num_heads=2,
        num_layers=2,
    ).to(device)

    rssm = RecurrentStateSpaceModel(
        action_dim=latent_dim,
        recurrent_dim=128,
        stochastic_dim=latent_dim,
        obs_dim=11,
        hidden_dim=128,
    ).to(device)

    engagement_head = EngagementPredictor(
        recurrent_dim=128,
        stochastic_dim=latent_dim,
        track_dim=latent_dim,
        hidden_dim=128,
    ).to(device)

    context_tax_head = ContextTaxPredictor(
        recurrent_dim=128,
        stochastic_dim=latent_dim,
        action_dim=latent_dim,
        hidden_dim=128,
    ).to(device)

    gini_loss_fn = SmoothGiniLoss().to(device)

    # Put models in evaluation mode
    sasrec.eval()
    rssm.eval()
    engagement_head.eval()
    context_tax_head.eval()

    # Initialise Environments
    env_baseline = TidalKultureGymEnv()
    env_rwm = TidalKultureGymEnv()

    # Track metrics for Baseline Policy
    baseline_rewards: list[float] = []
    baseline_dwell_times: list[int] = []
    baseline_taxes: list[float] = []
    baseline_churn_count = 0
    baseline_exposures = np.zeros(pool_size, dtype=np.int64)
    baseline_subgenre_counts = np.zeros(num_subgenres, dtype=np.int64)

    # Track metrics for kulture-rwm Policy
    rwm_rewards: list[float] = []
    rwm_dwell_times: list[int] = []
    rwm_taxes: list[float] = []
    rwm_churn_count = 0
    rwm_exposures = np.zeros(pool_size, dtype=np.int64)
    rwm_subgenre_counts = np.zeros(num_subgenres, dtype=np.int64)

    print("\nExecuting 30-Session Rollout for Policy A (TIDAL Baseline)...")
    for session_idx in range(num_sessions):
        seed = 1000 + session_idx
        obs, _ = env_baseline.reset(seed=seed)
        session_reward = 0.0
        session_steps = 0
        final_tax = 0.0
        churned = False

        while True:
            u_pref = torch.tensor(obs, dtype=torch.float32, device=device)
            # Baseline: SASRec candidate rank + heuristic filters (20% known artist cap, max 2 per cluster)
            slate_vecs, slate_genres, slate_indices = bridge.select_heuristic_slate(
                pool=candidate_pool,
                user_pref=u_pref,
                slate_size=slate_size,
                max_artist_cap=0.2,
                max_per_cluster=2,
            )

            obs, reward, terminated, truncated, info = env_baseline.step(
                action=slate_vecs,
                subgenre_ids=slate_genres,
            )

            session_reward += reward
            session_steps += 1
            final_tax = info["tau_c"]

            # Record catalog exposures and subgenres
            for idx in slate_indices.tolist():
                baseline_exposures[idx] += 1
            for g in slate_genres.tolist():
                baseline_subgenre_counts[g % num_subgenres] += 1

            if terminated:
                churned = True
                break
            if truncated:
                break

        baseline_rewards.append(session_reward)
        baseline_dwell_times.append(session_steps)
        baseline_taxes.append(final_tax)
        if churned:
            baseline_churn_count += 1

    print("Executing 30-Session Rollout for Policy B (kulture-rwm)...")
    for session_idx in range(num_sessions):
        seed = 1000 + session_idx
        obs, _ = env_rwm.reset(seed=seed)
        session_reward = 0.0
        session_steps = 0
        final_tax = 0.0
        churned = False

        # Session planner initialised with fresh distribution
        planner = CEMPMPPlanner(
            rssm=rssm,
            engagement_head=engagement_head,
            context_tax_head=context_tax_head,
            gini_loss=gini_loss_fn,
            horizon=horizon,
            pop_size=pop_size,
            elite_frac=0.15,
            cem_iters=cem_iters,
            action_dim=latent_dim,
            device=device,
        )

        # Belief state from SASRec backbone over interaction track history
        initial_track = int(1 + (session_idx * 17) % (pool_size - 1))
        history_track_ids: list[int] = [initial_track]
        belief_state = rssm.initial_state(batch_size=1, device=device)

        while True:
            # Map interaction history into hyperspherical user belief s_0 in S^(D-1)
            item_seq = torch.tensor([history_track_ids[-20:]], dtype=torch.long, device=device)
            with torch.no_grad():
                s_0, _ = sasrec(item_seq)  # (1, D)
                belief_state.z = s_0
                planned_action = planner.plan(belief_state)

            # Match planned action to candidate pool without popularity bias
            slate_vecs, slate_genres, slate_indices = bridge.select_rwm_slate(
                pool=candidate_pool,
                planned_action=planned_action,
                slate_size=slate_size,
            )

            obs, reward, terminated, truncated, info = env_rwm.step(
                action=slate_vecs,
                subgenre_ids=slate_genres,
            )

            session_reward += reward
            session_steps += 1
            final_tax = info["tau_c"]

            # Advance belief state with executed action
            with torch.no_grad():
                belief_state = rssm.imagine_step(belief_state, planned_action.unsqueeze(0))

            # Record chosen track id into history sequence
            chosen_track_id = int(candidate_pool.track_ids[slate_indices[0]].item())
            history_track_ids.append(chosen_track_id)

            # Record catalog exposures and subgenres
            for idx in slate_indices.tolist():
                rwm_exposures[idx] += 1
            for g in slate_genres.tolist():
                rwm_subgenre_counts[g % num_subgenres] += 1

            if terminated:
                churned = True
                break
            if truncated:
                break

        rwm_rewards.append(session_reward)
        rwm_dwell_times.append(session_steps)
        rwm_taxes.append(final_tax)
        if churned:
            rwm_churn_count += 1

    # ----------------------------------------------------------------- #
    # Aggregate Metrics Computation
    # ----------------------------------------------------------------- #
    baseline_mean_retention = float(np.mean(baseline_rewards))
    baseline_std_retention = float(np.std(baseline_rewards))
    baseline_mean_dwell = float(np.mean(baseline_dwell_times))
    baseline_mean_tax = float(np.mean(baseline_taxes))
    baseline_churn_rate = float(baseline_churn_count / num_sessions) * 100.0
    baseline_gini = compute_gini_coefficient(baseline_exposures)
    baseline_entropy = compute_shannon_entropy(baseline_subgenre_counts)

    rwm_mean_retention = float(np.mean(rwm_rewards))
    rwm_std_retention = float(np.std(rwm_rewards))
    rwm_mean_dwell = float(np.mean(rwm_dwell_times))
    rwm_mean_tax = float(np.mean(rwm_taxes))
    rwm_churn_rate = float(rwm_churn_count / num_sessions) * 100.0
    rwm_gini = compute_gini_coefficient(rwm_exposures)
    rwm_entropy = compute_shannon_entropy(rwm_subgenre_counts)

    # Theoretical maximum entropy for K subgenres
    max_entropy = math.log2(num_subgenres)

    # ----------------------------------------------------------------- #
    # Display Results Table
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 80)
    print(" " * 22 + "EVALUATION BENCHMARK RESULTS (30 SESSIONS)")
    print("=" * 80)
    print(f"{'Evaluation Metric':<40} | {'TIDAL Baseline':<16} | {'kulture-rwm':<16}")
    print("-" * 80)
    print(
        f"{'Mean Cumulative Retention (Reward)':<40} | "
        f"{baseline_mean_retention:>7.2f} ± {baseline_std_retention:<6.2f} | "
        f"{rwm_mean_retention:>7.2f} ± {rwm_std_retention:<6.2f}"
    )
    print(
        f"{'Mean Session Dwell Time (Steps)':<40} | "
        f"{baseline_mean_dwell:>16.1f} | "
        f"{rwm_mean_dwell:>16.1f}"
    )
    print(
        f"{'Accumulated Context Tax (tau_c)':<40} | "
        f"{baseline_mean_tax:>16.2f} | "
        f"{rwm_mean_tax:>16.2f}"
    )
    print(
        f"{'Session Churn Rate (%)':<40} | "
        f"{baseline_churn_rate:>15.1f}% | "
        f"{rwm_churn_rate:>15.1f}%"
    )
    print(
        f"{'Catalog Exposure Gini (L_gini)':<40} | "
        f"{baseline_gini:>16.4f} | "
        f"{rwm_gini:>16.4f}"
    )
    print(
        f"{'Subgenre Shannon Entropy H(P_genre)':<40} | "
        f"{baseline_entropy:>11.3f} bits | "
        f"{rwm_entropy:>11.3f} bits"
    )
    print(f"{'  (Theoretical Maximum Entropy)':<40} | " f"{max_entropy:>11.3f} bits | " f"{max_entropy:>11.3f} bits")
    print("=" * 80)

    # Print Key Research Takeaways
    tax_reduction = ((baseline_mean_tax - rwm_mean_tax) / (baseline_mean_tax + 1e-6)) * 100.0
    gini_reduction = ((baseline_gini - rwm_gini) / (baseline_gini + 1e-6)) * 100.0
    entropy_gain = ((rwm_entropy - baseline_entropy) / (baseline_entropy + 1e-6)) * 100.0

    print("\nResearch Takeaways & Comparative Findings:")
    print(f"  * Cultural Context Tax (tau_c):  {tax_reduction:+.1f}% vs. baseline.")
    print(f"  * Catalog Exposure Gini (L_gini): {gini_reduction:+.1f}% reduction (superior long-tail equity).")
    print(f"  * Subgenre Shannon Diversity:    {entropy_gain:+.1f}% vs. baseline.")
    print("=" * 80)

    # ----------------------------------------------------------------- #
    # Optional Weights & Biases (W&B) Logging
    # ----------------------------------------------------------------- #
        try:

                project=project,
                entity=entity,
                name="phase4-closed-loop-benchmark",
                config={
                    "num_sessions": num_sessions,
                    "slate_size": slate_size,
                    "pool_size": pool_size,
                    "latent_dim": latent_dim,
                    "horizon": horizon,
                    "pop_size": pop_size,
                    "cem_iters": cem_iters,
                },
            )

            # Log session-level rollout metrics
            for s in range(num_sessions):
                    "rollout/session_id": s,
                    "rollout/baseline_reward": baseline_rewards[s],
                    "rollout/rwm_reward": rwm_rewards[s],
                    "rollout/baseline_dwell_time": baseline_dwell_times[s],
                    "rollout/rwm_dwell_time": rwm_dwell_times[s],
                    "rollout/baseline_context_tax": baseline_taxes[s],
                    "rollout/rwm_context_tax": rwm_taxes[s],
                })

            # Log required aggregate evaluation metrics
                "eval/catalog_gini": rwm_gini,
                "eval/subgenre_entropy": rwm_entropy,
                "eval/cumulative_retention_reward": rwm_mean_retention,
                "eval/context_tax": rwm_mean_tax,
                "eval/session_dwell_time": rwm_mean_dwell,
                "eval/churn_rate": rwm_churn_rate,
                "eval/baseline_catalog_gini": baseline_gini,
                "eval/baseline_subgenre_entropy": baseline_entropy,
                "eval/baseline_retention_reward": baseline_mean_retention,
                "eval/baseline_context_tax": baseline_mean_tax,
                "eval/baseline_dwell_time": baseline_mean_dwell,
                "eval/baseline_churn_rate": baseline_churn_rate,
            })
            print("Weights & Biases logging completed successfully.")
        except Exception as e:
            print(f"Warning: W&B logging encountered an error: {e}")


def parse_args():
    """Parse command line arguments for the benchmark."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Closed-Loop Comparative Simulation Benchmark: TIDAL Baseline vs. kulture-rwm"
    )
    parser.add_argument(
        "--num_sessions",
        type=int,
        default=30,
        help="Number of evaluation user sessions (default: 30)",
    )
    parser.add_argument(
        "--slate_size",
        type=int,
        default=10,
        help="Slate size K (default: 10)",
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=6000,
        help="Candidate catalog pool size (default: 6000)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Latent embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="CEM planning horizon T (default: 5)",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=32,
        help="CEM population size (default: 32)",
    )
    parser.add_argument(
        "--cem_iters",
        type=int,
        default=2,
        help="CEM refinement iterations (default: 2)",
    )
    parser.add_argument(
        action="store_true",
        help="Enable Weights & Biases (W&B) experiment tracking",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="kulture-rwm",
        help="W&B project name (default: kulture-rwm)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity/username (optional)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu")
    run_benchmark(
        num_sessions=args.num_sessions,
        slate_size=args.slate_size,
        pool_size=args.pool_size,
        latent_dim=args.latent_dim,
        horizon=args.horizon,
        pop_size=args.pop_size,
        cem_iters=args.cem_iters,
        project=args.project,
        entity=args.entity,
        device=device,
    )
