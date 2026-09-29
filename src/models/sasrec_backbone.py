"""
SASRec Transformer sequence backbone with unit hypersphere projection.

Maps historical user interaction sequences x_{1:t} to initial belief state
s_0 in S^(D-1) adhering to numerical stability constraints.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def project_to_hypersphere(tensor: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Projects arbitrary embeddings onto the unit hypersphere S^(D-1).

    Numerical Constraint:
        epsilon is strictly evaluated inside the square root to guarantee
        non-zero denominators and bounded Lipschitz gradients around the origin.
    """
    norm = torch.sqrt(torch.sum(tensor**2, dim=-1, keepdim=True) + eps)
    return tensor / norm


class SelfAttentionBlock(nn.Module):
    """Causal Transformer self-attention block with LayerNorm and point-wise FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN Transformer architecture
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x


class SASRecBackbone(nn.Module):
    """
    Self-Attentive Sequential Recommender mapping interaction history to S^(D-1).
    """

    def __init__(
        self,
        vocab_size: int = 6000,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.eps = eps

        # Item embedding (padding index 0 reserved)
        self.item_embedding = nn.Embedding(vocab_size + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # Final projection head to S^(D-1)
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generates upper-triangular causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(
        self,
        item_seq: torch.Tensor,
        return_all_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through causal Transformer backbone.

        Args:
            item_seq: Tensor of track IDs, shape (Batch, SeqLen).
            return_all_states: If True, returns representations for all sequence steps.

        Returns:
            s_0: Final sequence representation projected onto S^(D-1), shape (Batch, D).
            seq_states: Full sequence states on S^(D-1), shape (Batch, SeqLen, D).
        """
        batch_size, seq_len = item_seq.shape
        seq_len = min(seq_len, self.max_seq_len)
        item_seq = item_seq[:, -seq_len:]

        positions = (
            torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        )
        x = self.item_embedding(item_seq) + self.pos_embedding(positions)
        x = self.dropout(self.layer_norm(x))

        causal_mask = self._generate_causal_mask(seq_len, item_seq.device)

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        # Project representations onto hypersphere S^(D-1)
        projected_all = project_to_hypersphere(self.head(x), eps=self.eps)

        # Extract final valid state per sequence
        s_0 = projected_all[:, -1, :]

        return s_0, projected_all
