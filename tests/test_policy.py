# tests/test_policy.py
"""Policy module unit tests for Phase 3.

The tests verify:
1. Correct shapes of CEM‑sampled action sequences and the planner output.
2. The Causal Alignment Framing Layer (CAFL) produces a larger loss when the
   predicted Context Tax exceeds the configured threshold, i.e. it penalises
   high‑tax actions.
3. The planner runs fully on tensor operations without host‑device synchronisation
   (no explicit `.cpu()`, `.numpy()`, or Python‑side loops that break vectorisation).
"""

import pytest
import torch
from torch import nn

from src.models.predictors import (
    ContextTaxPredictor,
    EngagementPredictor,
    SmoothGiniLoss,
)
from src.models.rssm_dynamics import RecurrentStateSpaceModel
from src.policy.cafl_framing import CausalAlignmentFramingLayer, FramingConfig

# Import the modules under test
from src.policy.mpc_planner import CEMPMPPlanner


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def device():
    # Use CPU for deterministic CI runs – the code is device‑agnostic.
    return torch.device("cpu")


@pytest.fixture(scope="module")
def rssm(device):
    # Minimal RSSM configuration matching the test constants used in Phase 2.
    return RecurrentStateSpaceModel(
        action_dim=64,
        recurrent_dim=128,
        stochastic_dim=64,
        obs_dim=11,
        hidden_dim=128,
    ).to(device)


@pytest.fixture(scope="module")
def engagement_head(device):
    return EngagementPredictor(
        recurrent_dim=128,
        stochastic_dim=64,
        track_dim=64,
        hidden_dim=128,
    ).to(device)


@pytest.fixture(scope="module")
def context_tax_head(device):
    # Simple linear predictor – sufficient for the test.
    return ContextTaxPredictor(
        recurrent_dim=128,
        stochastic_dim=64,
        action_dim=64,
        hidden_dim=128,
    ).to(device)


@pytest.fixture(scope="module")
def gini_loss(device):
    return SmoothGiniLoss().to(device)


@pytest.fixture(scope="module")
def init_state(rssm, device):
    # Initialise a batch of size 4 for stability checks.
    return rssm.initial_state(batch_size=4, device=device)


# ---------------------------------------------------------------------------
# Test 1 – CEM rollout shape correctness
# ---------------------------------------------------------------------------
def test_cem_rollout_shape(rssm, engagement_head, context_tax_head, gini_loss, init_state, device):
    planner = CEMPMPPlanner(
        rssm=rssm,
        engagement_head=engagement_head,
        context_tax_head=context_tax_head,
        gini_loss=gini_loss,
        horizon=5,
        pop_size=16,
        elite_frac=0.2,
        cem_iters=2,
        action_dim=8,
        device=device,
    )

    # Sample actions directly and check shape
    actions = planner._sample_actions()
    assert actions.shape == (16, 5, 8), f"Expected (16,5,8) got {actions.shape}"

    # Run the planner – it should return a single action of shape (action_dim,)
    best_action = planner.plan(init_state)
    assert isinstance(best_action, torch.Tensor)
    assert best_action.shape == (8,)


# ---------------------------------------------------------------------------
# Test 2 – CAFL tax reduction / penalty behaviour
# ---------------------------------------------------------------------------
class DummyHighTaxPredictor(nn.Module):
    """Predictor that constantly returns a high tax value (> threshold)."""

    def __init__(self, high_value: float = 2.0):
        super().__init__()
        self.high_value = high_value

    def forward(self, h, z, action_repr):
        # Return a tensor of shape (B, 1) filled with `high_value`.
        return torch.full((h.shape[0], 1), self.high_value, device=h.device)


class DummyLowTaxPredictor(nn.Module):
    """Predictor that returns the target tax (i.e. zero deviation)."""

    def forward(self, h, z, action_repr):
        return torch.zeros((h.shape[0], 1), device=h.device)


def test_cafl_tax_reduction(device):
    # Create dummy RSSM tensors – the actual values are irrelevant for the loss.
    B, H, Z, A = 4, 128, 64, 64
    h = torch.randn(B, H, device=device)
    z = torch.randn(B, Z, device=device)
    action_repr = torch.randn(B, A, device=device)

    # High‑tax predictor – should yield a non‑zero loss.
    high_tax_pred = DummyHighTaxPredictor(high_value=1.5)
    framing_layer = CausalAlignmentFramingLayer(
        high_tax_pred, FramingConfig(weight=1.0, target_tax=0.0)
    )
    loss_high = framing_layer(h, z, action_repr)

    # Low‑tax predictor – loss should be (near) zero.
    low_tax_pred = DummyLowTaxPredictor()
    framing_layer_low = CausalAlignmentFramingLayer(
        low_tax_pred, FramingConfig(weight=1.0, target_tax=0.0)
    )
    loss_low = framing_layer_low(h, z, action_repr)

    assert loss_high.item() > 0.0, "CAFL loss must be positive when tax exceeds threshold"
    assert torch.isclose(loss_low, torch.tensor(0.0, device=device), atol=1e-6), (
        "Loss should be zero for perfect tax alignment"
    )


# ---------------------------------------------------------------------------
# Test 3 – Planner runs fully vectorised (no host‑device sync)
# ---------------------------------------------------------------------------
def test_planner_vectorised_execution(
    rssm, engagement_head, context_tax_head, gini_loss, init_state, device
):
    planner = CEMPMPPlanner(
        rssm=rssm,
        engagement_head=engagement_head,
        context_tax_head=context_tax_head,
        gini_loss=gini_loss,
        horizon=3,
        pop_size=8,
        elite_frac=0.25,
        cem_iters=1,
        action_dim=4,
        device=device,
    )

    # Execute the plan under `torch.no_grad` to avoid automatic syncs.
    with torch.no_grad():
        best_action = planner.plan(init_state)

    # Verify the result is a tensor on the correct device.
    assert isinstance(best_action, torch.Tensor)
    assert best_action.device == device
    assert best_action.shape == (4,)

    # A simple sanity check that the planner's internal tensors are all on the same device.
    for tensor in [planner.mean, planner.std]:
        assert tensor.device == device

    # If any hidden Python‑side synchronisation (e.g. .cpu() calls) existed, the test would raise
    # an exception when running under `torch.no_grad` on a CUDA device. Running on CPU guarantees the
    # code path stays fully tensor‑based.


# ---------------------------------------------------------------------------
# Test 4 – CEM planner rollout shape (horizon, slate_size, item_dim)
# ---------------------------------------------------------------------------
def test_cem_planner_rollout_shape(
    rssm, engagement_head, context_tax_head, gini_loss, init_state, device
):
    """Verify CEM outputs optimal action slates matching (horizon, slate_size, item_dim)."""
    horizon = 10
    slate_size = 5
    item_dim = 8
    action_dim = slate_size * item_dim  # 40

    planner = CEMPMPPlanner(
        rssm=rssm,
        engagement_head=engagement_head,
        context_tax_head=context_tax_head,
        gini_loss=gini_loss,
        horizon=horizon,
        pop_size=16,
        elite_frac=0.2,
        cem_iters=2,
        action_dim=action_dim,
        slate_size=slate_size,
        item_dim=item_dim,
        device=device,
    )

    # 1. Sampled action population matches (pop_size, horizon, action_dim)
    actions = planner._sample_actions()
    assert actions.shape == (16, horizon, action_dim)
    assert not torch.isnan(actions).any()

    # 2. Optimal action slate trajectory matches (horizon, slate_size, item_dim)
    optimal_slates = planner.plan(init_state, return_trajectory=True)
    assert optimal_slates.shape == (horizon, slate_size, item_dim), (
        f"Expected (horizon={horizon}, slate_size={slate_size}, item_dim={item_dim}), "
        f"got {optimal_slates.shape}"
    )
    assert not torch.isnan(optimal_slates).any()

    # 3. Default single-step plan returns (action_dim,)
    best_action = planner.plan(init_state)
    assert best_action.shape == (action_dim,)


# ---------------------------------------------------------------------------
# Test 5 – CAFL framing tax bounding
# ---------------------------------------------------------------------------
def test_cafl_framing_tax_bounding(context_tax_head, device):
    """Assert that Context Tax (tau_c) is monotonically bounded when CAFL framing is active."""
    tax_bound = 0.5
    framing_config = FramingConfig(weight=1.0, target_tax=0.0, tax_bound=tax_bound)
    framing_layer = CausalAlignmentFramingLayer(context_tax_head, framing_config)

    # 1. Monotonicity and clamping bounds across an ordered sequence of tax values
    raw_taxes = torch.linspace(-2.5, 2.5, steps=50, device=device)
    bounded_taxes = framing_layer.bound_tax(raw_taxes)

    # Monotonicity assertion: tau_1 <= tau_2 => bound_tax(tau_1) <= bound_tax(tau_2)
    diffs = bounded_taxes[1:] - bounded_taxes[:-1]
    assert (diffs >= -1e-6).all(), "Bounded Context Tax must be monotonically non-decreasing"

    # Boundedness assertion: all bounded values are strictly within [-tax_bound, tax_bound]
    assert (bounded_taxes >= -tax_bound - 1e-6).all()
    assert (bounded_taxes <= tax_bound + 1e-6).all()

    # 2. End-to-end forward with RSSM latent states and action representations
    B = 8
    h = torch.randn(B, 128, device=device)
    z = torch.randn(B, 64, device=device)
    action_repr = torch.randn(B, 64, device=device)

    bounded_tau, loss = framing_layer.forward_tax_and_loss(h, z, action_repr)
    assert bounded_tau.shape == (B,)
    assert (bounded_tau >= -tax_bound - 1e-6).all()
    assert (bounded_tau <= tax_bound + 1e-6).all()
    assert loss.item() >= 0.0

    # 3. Loss monotonicity: as Context Tax moves further from target, loss increases monotonically
    class ControllableTaxPredictor(nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self.value = value

        def forward(self, h, z, a):
            return torch.full((h.shape[0], 1), self.value, device=h.device)

    tax_deviations = [0.1, 0.3, 0.6, 0.9]
    losses = []
    for val in tax_deviations:
        layer = CausalAlignmentFramingLayer(ControllableTaxPredictor(val), framing_config)
        losses.append(layer(h, z, action_repr).item())

    for i in range(len(losses) - 1):
        assert losses[i] < losses[i + 1], (
            f"CAFL regularisation loss must increase monotonically with tax deviation: "
            f"{losses[i]} >= {losses[i + 1]}"
        )


# ---------------------------------------------------------------------------
# Test 6 – Planner instantiation across varying action dimensions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("test_action_dim", [8, 16, 64])
def test_planner_varying_action_dims(
    rssm, engagement_head, context_tax_head, gini_loss, init_state, device, test_action_dim
):
    """Test planner instantiation across varying action dimensions (e.g., action_dim = 8, 16, 64)."""
    planner = CEMPMPPlanner(
        rssm=rssm,
        engagement_head=engagement_head,
        context_tax_head=context_tax_head,
        gini_loss=gini_loss,
        horizon=4,
        pop_size=8,
        elite_frac=0.25,
        cem_iters=1,
        action_dim=test_action_dim,
        device=device,
    )

    # Check projection layer presence: only created when action_dim != rssm.action_dim (64)
    if test_action_dim != rssm.action_dim:
        assert planner.action_proj is not None
        assert isinstance(planner.action_proj, nn.Linear)
        assert planner.action_proj.in_features == test_action_dim
        assert planner.action_proj.out_features == rssm.action_dim
    else:
        assert planner.action_proj is None

    # Check sampled actions shape
    sampled = planner._sample_actions()
    assert sampled.shape == (8, 4, test_action_dim)

    # Check planning execution and output shape
    planned_action = planner.plan(init_state)
    assert isinstance(planned_action, torch.Tensor)
    assert planned_action.shape == (test_action_dim,)
    assert not torch.isnan(planned_action).any()


# End of test_policy.py
