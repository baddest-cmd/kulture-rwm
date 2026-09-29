r"""
Synthetic user agent for tidal-kulture-rwm.

The agent maintains a preference vector \( \mathbf{u} \) on the unit hypersphere
\( \mathbb{S}^{D-1} \).  At each step it receives a slate of candidate track
embeddings together with their **integer** subgenre prototype IDs.  The marginal
utility for each candidate is the cosine similarity between the preference
vector and the candidate embedding.  When consecutive tracks belong to the same
sub-genre prototype the utility decays exponentially (fatigue).  The agent also
tracks a *Context Tax* \( \tau_c \) that grows whenever the slate does not
contain any of the agent's favourite local sub-genres.

Performance & autodiff notes
-----------------------------
* **No Python loops in tensor paths** -- subgenre fatigue masks are computed
  via pure ``torch`` tensor comparisons on whichever device the data resides.
* **Epsilon inside the sqrt** in ``_project`` guarantees smooth gradient flow
  through the L2 norm even when the input is near the origin.
* **Diagnostic tensors stay on device** -- ``info`` dict values are returned as
  live ``torch.Tensor`` objects; conversion to NumPy is deferred to the caller.
"""

from typing import Any

import torch


class SyntheticUserAgent:
    """Synthetic population cohort on the unit hypersphere.

    Parameters
    ----------
    name : str
        Human-readable identifier for the cohort.
    region : str
        Geographic region (e.g. ``"SouthAfrica"``) used for context-tax
        calculations.
    favourite_subgenre_ids : Set[int]
        Integer prototype IDs for sub-genres the user strongly prefers.
        Omission from a slate increments the context tax.
    favourite_subgenre_names : List[str]
        Human-readable names (kept for logging only).
    latent_dim : int, default 64
        Dimensionality of the hyperspherical preference vector.
    fatigue_rate : float, default 0.5
        Exponential decay factor applied when consecutive tracks share the
        same sub-genre prototype.
    tau_increment : float, default 1.0
        Increment added to the context tax when a slate lacks any favourite
        sub-genre.
    eps : float, default 1e-7
        Small constant placed **inside** the square-root norm for numerical
        stability and smooth autodiff gradients.
    """

    def __init__(
        self,
        name: str,
        region: str,
        favourite_subgenre_ids: set[int],
        favourite_subgenre_names: list[str] | None = None,
        latent_dim: int = 64,
        fatigue_rate: float = 0.5,
        tau_increment: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        self.name = name
        self.region = region
        self.favourite_subgenre_ids = favourite_subgenre_ids
        self.favourite_subgenre_names = favourite_subgenre_names or []
        self.latent_dim = latent_dim
        self.fatigue_rate = fatigue_rate
        self.tau_increment = tau_increment
        self.eps = eps

        # Pre-compute a long tensor of favourite ids for vectorised membership
        self._fav_ids_tensor: torch.Tensor = torch.tensor(
            sorted(favourite_subgenre_ids), dtype=torch.long
        )

        # Initialise preference state on the unit hypersphere
        raw = torch.randn(latent_dim)
        self.u: torch.Tensor = self._project(raw)  # (D,)

        # Context tax accumulator
        self.tau_c: float = 0.0

        # Last observed sub-genre prototype id (None on first step)
        self._last_subgenre_id: int | None = None

    # ------------------------------------------------------------------ #
    # Helper: safe L2 projection with epsilon **inside** the sqrt
    # ------------------------------------------------------------------ #
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` onto the unit hypersphere with stable gradients.

        Placing ``eps`` inside the square root guarantees that
        :math:`\\nabla_{\\mathbf{x}} \\|\\mathbf{x}\\|_2` is finite even
        when :math:`\\mathbf{x} \\approx \\mathbf{0}`.
        """
        norm = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm

    # ------------------------------------------------------------------ #
    # Core interaction – fully vectorised over the slate
    # ------------------------------------------------------------------ #
    def step(
        self,
        slate_vectors: torch.Tensor,  # (K, D)
        slate_subgenre_ids: torch.Tensor,  # (K,) long tensor on same device
    ) -> tuple[torch.Tensor, float, dict[str, Any]]:
        """Process a recommended slate and update internal state.

        Parameters
        ----------
        slate_vectors : torch.Tensor, shape ``(K, D)``
            Track embeddings for the current slate.
        slate_subgenre_ids : torch.Tensor, shape ``(K,)``
            Integer sub-genre prototype IDs for each track in the slate.

        Returns
        -------
        new_pref : torch.Tensor, shape ``(D,)``
            Updated preference vector (unit-norm, on same device as input).
        reward : float
            Sum of fatigue-modulated utilities for the slate.
        info : dict
            Diagnostic tensors **kept on device** (no `.cpu().numpy()` here).
        """
        K, D = slate_vectors.shape
        assert D == self.latent_dim, f"Slate dimension {D} mismatches agent dim {self.latent_dim}"
        assert slate_subgenre_ids.shape == (K,), "Subgenre id tensor must have shape (K,)"

        device = slate_vectors.device

        # Normalise slate vectors onto the hypersphere (safe)
        slate_vecs = self._project(slate_vectors)  # (K, D)

        # Move preference vector to the same device if needed
        if self.u.device != device:
            self.u = self.u.to(device)

        # Cosine similarity (dot product – both unit-norm)
        utilities = torch.matmul(slate_vecs, self.u)  # (K,)

        # ---- Fatigue mask: pure tensor comparison, zero Python loops ----
        if self._last_subgenre_id is None:
            fatigue_mask = torch.zeros(K, dtype=torch.float32, device=device)
        else:
            fatigue_mask = (slate_subgenre_ids == self._last_subgenre_id).float()

        # Exponential decay for repeated sub-genre
        decay = torch.exp(-self.fatigue_rate * fatigue_mask)
        fatigued_util = utilities * decay

        # Reward = sum of fatigued utilities (scalar)
        reward: float = fatigued_util.sum().item()

        # ---- Preference update: small LERP + re-project ----
        best_idx = torch.argmax(fatigued_util)
        best_vec = slate_vecs[best_idx]
        lr = 0.05
        self.u = self._project((1.0 - lr) * self.u + lr * best_vec)

        # ---- Context Tax: vectorised favourite-id membership check ----
        fav_ids = self._fav_ids_tensor.to(device)
        # Check if *any* track in the slate has a favourite sub-genre id.
        # Uses broadcasting: (K, 1) == (1, |F|) -> (K, |F|) -> any()
        slate_ids_col = slate_subgenre_ids.unsqueeze(1)  # (K, 1)
        fav_ids_row = fav_ids.unsqueeze(0)  # (1, |F|)
        has_fav = (slate_ids_col == fav_ids_row).any().item()

        if not has_fav:
            self.tau_c += self.tau_increment
        else:
            self.tau_c = max(0.0, self.tau_c - self.tau_increment * 0.2)

        # Update last observed sub-genre prototype id
        self._last_subgenre_id = slate_subgenre_ids[best_idx].item()

        # ---- Diagnostics: tensors stay on device ----
        info: dict[str, Any] = {
            "raw_utilities": utilities,  # (K,) tensor on device
            "fatigue_mask": fatigue_mask,  # (K,) tensor on device
            "fatigued_utilities": fatigued_util,  # (K,) tensor on device
            "selected_subgenre_id": self._last_subgenre_id,
            "tau_c": self.tau_c,
        }

        return self.u.detach().clone(), reward, info

    # ------------------------------------------------------------------ #
    def get_state(self) -> torch.Tensor:
        """Return a detached copy of the current preference vector."""
        return self.u.detach().clone()

    def get_context_tax(self) -> float:
        """Return the current context-tax accumulator value."""
        return self.tau_c
