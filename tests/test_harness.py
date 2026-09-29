"""
Phase 1 unit-tests for the synthetic environment harness.

Test suite
----------
``test_random_rollout_norm``
    Performs a 10-step rollout with random slates and asserts that the
    agent's preference state remains on the unit hypersphere
    :math:`\\|\\mathbf{s}_t\\|_2 = 1.0 \\pm 10^{-5}` after every step.

``test_context_tax_monotonicity``
    Property test: when the input slate completely omits the agent's
    favourite sub-genre prototype IDs for T = 5 consecutive steps, the
    context tax :math:`\\tau_c` must increase monotonically.
"""

import numpy as np
import pytest
import torch

from src.environment.tidal_gym_env import TidalKultureGymEnv
from src.environment.user_agent import SyntheticUserAgent


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #
@pytest.fixture
def env():
    """Standard environment with default config."""
    return TidalKultureGymEnv()


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _random_slate(K: int, D: int) -> torch.Tensor:
    """Generate a random unit-normalised slate with eps inside sqrt."""
    eps = 1e-7
    vecs = torch.randn(K, D)
    norms = torch.sqrt(torch.sum(vecs**2, dim=-1, keepdim=True) + eps)
    return vecs / norms


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #
def test_random_rollout_norm(env):
    """Preference state must stay on the unit sphere across a rollout."""
    obs, _ = env.reset()
    D = env.latent_dim
    K = env.K

    # Initial observation should already be unit-norm
    assert np.isclose(np.linalg.norm(obs), 1.0, atol=1e-5), "Initial state not on unit sphere"

    for step_idx in range(10):
        slate = _random_slate(K, D)
        obs, reward, terminated, truncated, info = env.step(slate)

        # Unit-norm invariance after each step
        assert np.isclose(np.linalg.norm(obs), 1.0, atol=1e-5), (
            f"Step {step_idx}: state deviates from unit sphere"
        )

        # Reward must be a finite scalar
        assert np.isfinite(reward), f"Step {step_idx}: non-finite reward"

        # Diagnostic tensors should remain as torch.Tensor (not numpy)
        assert isinstance(info["raw_utilities"], torch.Tensor), (
            "raw_utilities should be a torch.Tensor on device"
        )

        if terminated or truncated:
            break

    # At least one step was executed
    assert step_idx >= 0


def test_context_tax_monotonicity():
    """tau_c must increase monotonically when favourites are absent.

    We construct a ``SyntheticUserAgent`` whose favourite sub-genre IDs
    are ``{0, 3}`` (Amapiano, HipHop).  Then we feed T = 5 slates
    whose sub-genre IDs are exclusively ``{1, 2, 4}`` (Gqom, Pop,
    Maskandi) – none of which are favourites.

    Assertion: ``tau_c`` strictly increases after every step.
    """
    D = 64
    K = 10
    T = 5

    agent = SyntheticUserAgent(
        name="test_monotonicity",
        region="test",
        favourite_subgenre_ids={0, 3},  # Amapiano, HipHop
        latent_dim=D,
        fatigue_rate=0.5,
        tau_increment=1.0,
    )

    # Sub-genre IDs that deliberately exclude favourites {0, 3}
    non_fav_ids = torch.tensor([1, 2, 4, 1, 2, 4, 1, 2, 4, 1], dtype=torch.long)
    assert non_fav_ids.shape == (K,)

    previous_tau = agent.get_context_tax()

    for t in range(T):
        slate = _random_slate(K, D)
        _, _, info = agent.step(slate, non_fav_ids)
        current_tau = info["tau_c"]

        assert current_tau > previous_tau, (
            f"Step {t}: tau_c did not increase (prev={previous_tau:.4f}, curr={current_tau:.4f})"
        )
        previous_tau = current_tau

    # Final tau_c should equal T * tau_increment (since we started at 0)
    assert np.isclose(agent.get_context_tax(), T * 1.0, atol=1e-6), (
        f"Final tau_c should be {T * 1.0}, got {agent.get_context_tax()}"
    )
