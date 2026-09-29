"""
Gymnasium environment that wraps the synthetic user agent defined in
``src/environment/user_agent.py``.

The environment models a single user session: at each step the caller
provides a slate of K track embeddings (``action``) and the environment
returns the next preference state, an engagement reward, and termination
flags when the context-tax exceeds a threshold or the session length
limit is reached.

Performance notes
-----------------
* All vector operations are fully vectorised with PyTorch.
* Intermediate diagnostic tensors from ``SyntheticUserAgent.step()`` stay
  on the active compute device; only the final **observation** is cast to
  a NumPy array (required by the Gymnasium API).
* Sub-genre metadata is passed as integer prototype IDs (``torch.long``)
  to avoid CPU-GPU synchronisation barriers.
"""

import gymnasium as gym
from gymnasium import spaces
import torch
import yaml
import pathlib
from typing import Tuple, Dict, Any, Optional

from .user_agent import SyntheticUserAgent


# --------------------------------------------------------------------- #
# Subgenre registry – maps human-readable names to integer prototype IDs
# --------------------------------------------------------------------- #
_SUBGENRE_REGISTRY: Dict[str, int] = {
    "Amapiano": 0,
    "Gqom": 1,
    "Pop": 2,
    "HipHop": 3,
    "Maskandi": 4,
    "Lekompo": 5,
    "Bacardi": 6,
}

_SUBGENRE_CYCLE_IDS = torch.tensor(
    [_SUBGENRE_REGISTRY[n] for n in ["Amapiano", "Gqom", "Pop", "HipHop", "Maskandi"]],
    dtype=torch.long,
)


class TidalKultureGymEnv(gym.Env):
    """Gymnasium-compatible environment for the tidal-kulture RWM.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML configuration file under ``configs/``.  If omitted
        the default ``configs/harness_config.yaml`` is loaded.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path: Optional[str] = None):
        # ------------------------------------------------------------- #
        # Load configuration
        # ------------------------------------------------------------- #
        if config_path is None:
            cfg_path = (
                pathlib.Path(__file__).parents[2] / "configs" / "harness_config.yaml"
            )
        else:
            cfg_path = pathlib.Path(config_path)

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        env_cfg = cfg.get("environment", {})
        self.max_session_steps: int = int(env_cfg.get("max_session_steps", 30))
        self.candidate_pool_size: int = int(
            env_cfg.get("candidate_pool_size", 6000)
        )
        self.tau_threshold: float = float(env_cfg.get("tau_threshold", 10.0))
        self.latent_dim: int = int(env_cfg.get("latent_dim", 64))
        self.fatigue_rate: float = float(env_cfg.get("fatigue_rate", 0.5))
        self.tau_increment: float = float(env_cfg.get("tau_increment", 1.0))

        # ------------------------------------------------------------- #
        # Action / observation spaces
        # ------------------------------------------------------------- #
        self.K: int = 10  # slate size as required by the spec
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.latent_dim,), dtype=float
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.K, self.latent_dim), dtype=float
        )

        # ------------------------------------------------------------- #
        # Favourite sub-genre prototype IDs for the default cohort
        # ------------------------------------------------------------- #
        self._favourite_ids = {
            _SUBGENRE_REGISTRY["Amapiano"],
            _SUBGENRE_REGISTRY["HipHop"],
        }
        self._favourite_names = ["Amapiano", "HipHop"]

        # ------------------------------------------------------------- #
        # Initialise the synthetic user agent
        # ------------------------------------------------------------- #
        self.agent = self._make_agent()
        self.current_step: int = 0

    # ----------------------------------------------------------------- #
    def _make_agent(self) -> SyntheticUserAgent:
        """Construct a fresh ``SyntheticUserAgent``."""
        return SyntheticUserAgent(
            name="cohort_1",
            region="global",
            favourite_subgenre_ids=self._favourite_ids,
            favourite_subgenre_names=self._favourite_names,
            latent_dim=self.latent_dim,
            fatigue_rate=self.fatigue_rate,
            tau_increment=self.tau_increment,
        )

    # ----------------------------------------------------------------- #
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple:
        """Reset the environment and return ``(observation, info)``.

        Follows the Gymnasium API.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.agent = self._make_agent()

        obs = self.agent.get_state().numpy()
        info: Dict[str, Any] = {"step": self.current_step}
        return obs, info

    # ----------------------------------------------------------------- #
    def step(
        self,
        action: torch.Tensor,
        subgenre_ids: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Execute one environment step.

        Parameters
        ----------
        action : torch.Tensor, shape ``(K, D)``
            Slate of track embeddings.  The caller must ensure the vectors
            are normalised; the environment will re-normalise just in case.
        subgenre_ids : torch.Tensor, optional, shape ``(K,)``
            Integer sub-genre prototype IDs for each track in the slate.
            If omitted, default cyclic prototype IDs are used.

        Returns
        -------
        observation : np.ndarray, shape ``(D,)``
            Updated preference state (cast to NumPy for Gym API compliance).
        reward : float
            Summed fatigue-modulated utilities for the slate.
        terminated : bool
            ``True`` when context-tax exceeds ``tau_threshold``.
        truncated : bool
            ``True`` when the maximum step count is reached.
        info : dict
            Diagnostic data from the agent (tensors stay on device).
        """
        assert isinstance(action, torch.Tensor), "Action must be a torch.Tensor"
        assert action.shape == (self.K, self.latent_dim), (
            f"Expected action shape {(self.K, self.latent_dim)} "
            f"but got {action.shape}"
        )

        # Sub-genre integer IDs from candidate pool metadata or fallback cycle
        if subgenre_ids is None:
            subgenre_ids = _SUBGENRE_CYCLE_IDS[
                torch.arange(self.K) % len(_SUBGENRE_CYCLE_IDS)
            ]
        else:
            assert isinstance(subgenre_ids, torch.Tensor), "subgenre_ids must be a torch.Tensor"
            assert subgenre_ids.shape == (self.K,), f"Expected shape ({self.K},), got {subgenre_ids.shape}"

        # Agent processes the slate – returns tensors on device
        new_state, reward, info = self.agent.step(action, subgenre_ids)

        self.current_step += 1
        terminated = info["tau_c"] > self.tau_threshold
        truncated = self.current_step >= self.max_session_steps
        info["context_tax"] = float(info.get("tau_c", 0.0))

        # Only convert the observation to NumPy (Gym API requirement)
        observation = new_state.numpy()
        return observation, reward, terminated, truncated, info

    # ----------------------------------------------------------------- #
    def render(self):
        raise NotImplementedError(
            "Render not implemented for this lightweight env"
        )

    def close(self):
        pass
