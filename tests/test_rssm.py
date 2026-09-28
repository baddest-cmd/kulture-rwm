r"""
Phase 2 unit-tests for the Core Representation & RSSM Dynamics Engine.

Test suite
----------
``test_sasrec_forward_shapes``
    Verifies that SASRecBackbone produces correct output shapes for batch
    rollouts: initial belief \(\mathbf{s}_0\) of shape ``(B, D)`` and full
    sequence states of shape ``(B, T, D)``.

``test_sasrec_unit_norm``
    Asserts that every output vector from SASRec lives on the unit
    hypersphere: \(\|\mathbf{s}\|_2 = 1.0 \pm 10^{-5}\).

``test_rssm_observe_shapes``
    Verifies LatentState shapes after ``observe_step`` with real
    observations (posterior + prior).

``test_rssm_imagine_rollout_shapes``
    Verifies shapes across a T-step imagination rollout via
    ``rollout_imagination``.

``test_rssm_unit_norm_across_rollout``
    Asserts that all stochastic states \(\mathbf{z}_t\) remain on
    \(\mathbb{S}^{D-1}\) throughout an imagined trajectory.

``test_kl_divergence_nonneg``
    Verifies that balanced KL divergence is non-negative and finite.

``test_engagement_predictor_shapes``
    Checks engagement predictor output shape ``(B, K)``.

``test_context_tax_predictor_shapes``
    Checks context-tax predictor output shape ``(B, 1)``.

``test_prototype_simplex_loss``
    Verifies prototype simplex loss produces finite scalar outputs and
    that normalised prototypes live on \(\mathbb{S}^{D-1}\).

``test_gini_o_m_log_m_memory``
    Memory efficiency test: the O(M log M) Gini implementation must not
    allocate O(M^2) intermediate tensors for M = 10,000 items.

``test_gini_perfect_equality``
    Gini of a uniform exposure vector must be approximately 0.

``test_gini_maximum_inequality``
    Gini of a one-hot exposure vector must approach 1 as M grows.
"""

import pytest
import torch
import numpy as np
import tracemalloc

from src.models.sasrec_backbone import SASRecBackbone, project_to_hypersphere
from src.models.rssm_dynamics import RecurrentStateSpaceModel, LatentState
from src.models.predictors import (
    EngagementPredictor,
    ContextTaxPredictor,
    PrototypeSimplexLoss,
    SmoothGiniLoss,
    differentiable_smooth_gini,
)


# ===================================================================== #
# Constants
# ===================================================================== #
BATCH = 4
SEQ_LEN = 20
VOCAB = 500
D = 64          # latent / stochastic dim
H = 128         # recurrent dim
K = 10          # slate size
HORIZON = 10    # imagination rollout length
OBS_DIM = 11    # observation dim for RSSM
EPS = 1e-5      # tolerance for unit-norm checks


# ===================================================================== #
# Fixtures
# ===================================================================== #
@pytest.fixture
def sasrec():
    return SASRecBackbone(
        vocab_size=VOCAB,
        hidden_dim=D,
        num_heads=4,
        num_layers=2,
        max_seq_len=SEQ_LEN,
    )


@pytest.fixture
def rssm():
    return RecurrentStateSpaceModel(
        action_dim=D,
        recurrent_dim=H,
        stochastic_dim=D,
        obs_dim=OBS_DIM,
        hidden_dim=H,
    )


@pytest.fixture
def engagement_head():
    return EngagementPredictor(
        recurrent_dim=H,
        stochastic_dim=D,
        track_dim=D,
        hidden_dim=H,
    )


@pytest.fixture
def context_tax_head():
    return ContextTaxPredictor(
        recurrent_dim=H,
        stochastic_dim=D,
        action_dim=D,
        hidden_dim=H,
    )


@pytest.fixture
def proto_loss():
    return PrototypeSimplexLoss(num_prototypes=4, dim=D)


@pytest.fixture
def gini_loss():
    return SmoothGiniLoss()


# ===================================================================== #
# SASRec Backbone Tests
# ===================================================================== #
class TestSASRecBackbone:
    """Tests for the SASRec Transformer sequence encoder."""

    def test_forward_shapes(self, sasrec):
        """Forward pass must produce (B, D) belief and (B, T, D) states."""
        item_seq = torch.randint(1, VOCAB + 1, (BATCH, SEQ_LEN))
        s_0, seq_states = sasrec(item_seq)

        assert s_0.shape == (BATCH, D), (
            f"Expected s_0 shape {(BATCH, D)}, got {s_0.shape}"
        )
        assert seq_states.shape == (BATCH, SEQ_LEN, D), (
            f"Expected seq_states shape {(BATCH, SEQ_LEN, D)}, "
            f"got {seq_states.shape}"
        )

    def test_unit_norm(self, sasrec):
        """All SASRec outputs must live on S^(D-1)."""
        item_seq = torch.randint(1, VOCAB + 1, (BATCH, SEQ_LEN))
        s_0, seq_states = sasrec(item_seq)

        # Check s_0 norms
        s0_norms = torch.norm(s_0, dim=-1)
        assert torch.allclose(
            s0_norms, torch.ones(BATCH), atol=EPS
        ), f"s_0 norms deviate from 1.0: {s0_norms}"

        # Check all sequence state norms
        seq_norms = torch.norm(seq_states, dim=-1)  # (B, T)
        assert torch.allclose(
            seq_norms, torch.ones(BATCH, SEQ_LEN), atol=EPS
        ), f"Sequence state norms deviate from 1.0"

    def test_shorter_sequence(self, sasrec):
        """SASRec must handle sequences shorter than max_seq_len."""
        short_len = 5
        item_seq = torch.randint(1, VOCAB + 1, (BATCH, short_len))
        s_0, seq_states = sasrec(item_seq)

        assert s_0.shape == (BATCH, D)
        assert seq_states.shape == (BATCH, short_len, D)


# ===================================================================== #
# RSSM Dynamics Tests
# ===================================================================== #
class TestRSSMDynamics:
    """Tests for the Action-Conditioned Recurrent State Space Model."""

    def test_initial_state_shapes(self, rssm):
        """Initial state must produce correct shapes."""
        state = rssm.initial_state(BATCH, device=torch.device("cpu"))

        assert state.h.shape == (BATCH, H)
        assert state.z.shape == (BATCH, D)
        assert state.prior_mu.shape == (BATCH, D)
        assert state.prior_std.shape == (BATCH, D)

    def test_observe_step_shapes(self, rssm):
        """observe_step must produce both posterior and prior parameters."""
        state = rssm.initial_state(BATCH, device=torch.device("cpu"))
        action = torch.randn(BATCH, D)
        obs = torch.randn(BATCH, OBS_DIM)

        next_state = rssm.observe_step(state, action, obs)

        assert next_state.h.shape == (BATCH, H)
        assert next_state.z.shape == (BATCH, D)
        assert next_state.prior_mu.shape == (BATCH, D)
        assert next_state.prior_std.shape == (BATCH, D)
        assert next_state.post_mu is not None
        assert next_state.post_mu.shape == (BATCH, D)
        assert next_state.post_std is not None
        assert next_state.post_std.shape == (BATCH, D)

    def test_imagine_rollout_shapes(self, rssm):
        """Imagination rollout must produce HORIZON states with correct shapes."""
        state = rssm.initial_state(BATCH, device=torch.device("cpu"))
        actions = torch.randn(BATCH, HORIZON, D)

        trajectory = rssm.rollout_imagination(state, actions)

        assert len(trajectory) == HORIZON
        for t, s in enumerate(trajectory):
            assert s.h.shape == (BATCH, H), f"Step {t}: h shape mismatch"
            assert s.z.shape == (BATCH, D), f"Step {t}: z shape mismatch"
            # Imagination uses only prior (no posterior)
            assert s.post_mu is None, f"Step {t}: imagination should not have posterior"

    def test_unit_norm_across_rollout(self, rssm):
        r"""All z_t must remain on S^(D-1) throughout imagination."""
        state = rssm.initial_state(BATCH, device=torch.device("cpu"))
        actions = torch.randn(BATCH, HORIZON, D)

        trajectory = rssm.rollout_imagination(state, actions)

        for t, s in enumerate(trajectory):
            z_norms = torch.norm(s.z, dim=-1)
            assert torch.allclose(
                z_norms, torch.ones(BATCH), atol=EPS
            ), f"Step {t}: z_t norms deviate from 1.0: {z_norms}"

    def test_kl_divergence_nonneg(self, rssm):
        """Balanced KL divergence must be non-negative and finite."""
        state = rssm.initial_state(BATCH, device=torch.device("cpu"))
        action = torch.randn(BATCH, D)
        obs = torch.randn(BATCH, OBS_DIM)

        next_state = rssm.observe_step(state, action, obs)

        balanced_kl, raw_kl = rssm.compute_kl_divergence(
            post_mu=next_state.post_mu,
            post_std=next_state.post_std,
            prior_mu=next_state.prior_mu,
            prior_std=next_state.prior_std,
        )

        assert torch.isfinite(balanced_kl), f"Balanced KL is not finite: {balanced_kl}"
        assert torch.isfinite(raw_kl), f"Raw KL is not finite: {raw_kl}"
        assert raw_kl >= 0.0, f"Raw KL is negative: {raw_kl}"

    def test_observe_then_imagine_consistency(self, rssm):
        """After observe_step, continuing with imagine_step must not crash."""
        state = rssm.initial_state(BATCH, device=torch.device("cpu"))
        action = torch.randn(BATCH, D)
        obs = torch.randn(BATCH, OBS_DIM)

        # One observation step
        observed = rssm.observe_step(state, action, obs)

        # Continue with imagination
        future_actions = torch.randn(BATCH, 5, D)
        trajectory = rssm.rollout_imagination(observed, future_actions)

        assert len(trajectory) == 5
        for s in trajectory:
            z_norms = torch.norm(s.z, dim=-1)
            assert torch.allclose(z_norms, torch.ones(BATCH), atol=EPS)


# ===================================================================== #
# Predictor Head Tests
# ===================================================================== #
class TestPredictorHeads:
    """Tests for the multi-task predictor heads."""

    def test_engagement_predictor_shapes(self, engagement_head):
        """Engagement predictor output must be (B, K) probabilities in [0, 1]."""
        h = torch.randn(BATCH, H)
        z = torch.randn(BATCH, D)
        tracks = torch.randn(BATCH, K, D)

        probs = engagement_head(h, z, tracks)

        assert probs.shape == (BATCH, K), (
            f"Expected shape {(BATCH, K)}, got {probs.shape}"
        )
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0), (
            "Engagement probabilities must be in [0, 1]"
        )

    def test_context_tax_predictor_shapes(self, context_tax_head):
        r"""Context Tax predictor output must be (B, 1) bounded in [-1, 1]."""
        h = torch.randn(BATCH, H)
        z = torch.randn(BATCH, D)
        action_repr = torch.randn(BATCH, D)

        tau = context_tax_head(h, z, action_repr)

        assert tau.shape == (BATCH, 1), (
            f"Expected shape {(BATCH, 1)}, got {tau.shape}"
        )
        assert torch.all(tau >= -1.0) and torch.all(tau <= 1.0), (
            "Context Tax must be bounded in [-1, 1] (Tanh output)"
        )


# ===================================================================== #
# Prototype Simplex Loss Tests
# ===================================================================== #
class TestPrototypeSimplexLoss:
    """Tests for the regular simplex prototype spanning loss."""

    def test_loss_finite(self, proto_loss):
        """Prototype loss must produce finite scalar outputs."""
        tracks = project_to_hypersphere(torch.randn(100, D))
        total, attraction, separation = proto_loss(tracks)

        assert torch.isfinite(total), f"Total loss not finite: {total}"
        assert torch.isfinite(attraction), f"Attraction loss not finite"
        assert torch.isfinite(separation), f"Separation loss not finite"
        assert total >= 0.0, f"Total loss should be non-negative"

    def test_prototypes_on_sphere(self, proto_loss):
        r"""Normalised prototypes must live on S^(D-1)."""
        protos = proto_loss.get_normalised_prototypes()
        norms = torch.norm(protos, dim=-1)

        assert torch.allclose(
            norms,
            torch.ones(proto_loss.num_prototypes),
            atol=EPS,
        ), f"Prototype norms deviate from 1.0: {norms}"


# ===================================================================== #
# Gini Loss Tests
# ===================================================================== #
class TestGiniLoss:
    """Tests for the O(M log M) differentiable smooth Gini loss."""

    def test_perfect_equality(self):
        """Uniform exposure must produce Gini ≈ 0."""
        M = 1000
        exposures = torch.ones(M)
        gini = differentiable_smooth_gini(exposures)

        assert torch.isfinite(gini)
        assert gini.item() < 0.01, (
            f"Gini for uniform exposures should be ≈ 0, got {gini.item():.4f}"
        )

    def test_maximum_inequality(self):
        """One-hot exposure must produce Gini close to 1 for large M."""
        M = 10000
        exposures = torch.zeros(M)
        exposures[0] = 1.0
        gini = differentiable_smooth_gini(exposures)

        assert torch.isfinite(gini)
        assert gini.item() > 0.95, (
            f"Gini for one-hot exposures should approach 1.0, got {gini.item():.4f}"
        )

    def test_batched_gini(self):
        """Gini must handle batched inputs (B, M)."""
        B, M = 8, 500
        exposures = torch.rand(B, M)
        gini = differentiable_smooth_gini(exposures)

        assert gini.shape == (B,), f"Expected shape ({B},), got {gini.shape}"
        assert torch.all(torch.isfinite(gini))

    def test_gini_differentiable(self):
        """Gini loss must produce finite gradients via autograd."""
        M = 100
        exposures = torch.rand(M, requires_grad=True)
        gini = differentiable_smooth_gini(exposures)
        gini.backward()

        assert exposures.grad is not None
        assert torch.all(torch.isfinite(exposures.grad)), (
            "Gini gradient contains NaN/Inf"
        )

    def test_o_m_log_m_memory(self):
        r"""Gini for M = 10,000 must NOT allocate O(M^2) memory.

        The O(M^2) pairwise reference approach (``for-the-kulture``
        ``metrics.py``) would allocate M^2 * 4 bytes = ~381 MB for
        M = 10,000 float32 entries.  We assert peak allocation stays
        well below that threshold.
        """
        M = 10_000

        # Pre-allocate the input tensor BEFORE we start tracking
        exposures = torch.rand(M)

        # Track only the Gini computation itself
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        _ = differentiable_smooth_gini(exposures)

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare peak memory between snapshots
        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        peak_increase_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)

        # O(M^2) for float32: 10000^2 * 4 ≈ 381 MB
        # O(M log M) should be well under 10 MB
        max_allowed_bytes = 10 * 1024 * 1024  # 10 MB
        assert peak_increase_bytes < max_allowed_bytes, (
            f"Gini allocated {peak_increase_bytes / (1024**2):.1f} MB, "
            f"exceeding O(M log M) budget of "
            f"{max_allowed_bytes / (1024**2):.0f} MB. "
            f"Likely constructing O(M^2) pairwise matrices."
        )


# ===================================================================== #
# SmoothGiniLoss Module Wrapper Tests
# ===================================================================== #
class TestSmoothGiniLossModule:
    """Tests for the nn.Module wrapper."""

    def test_forward_returns_scalar(self, gini_loss):
        """SmoothGiniLoss.forward must return a scalar."""
        exposures = torch.rand(200)
        loss = gini_loss(exposures)

        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss)

    def test_forward_batched(self, gini_loss):
        """SmoothGiniLoss must handle batched inputs and return a scalar mean."""
        exposures = torch.rand(8, 200)
        loss = gini_loss(exposures)

        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss)
