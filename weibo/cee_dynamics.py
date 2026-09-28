import math
from typing import Optional

import torch
import torch.nn as nn


def transform_cee_nll_to_score(nll_scores: torch.Tensor, score_mode: str = "log_likelihood") -> torch.Tensor:
    """Convert autoregressive NLL to the reported CEE score.

    The dynamics head is still pretrained by minimizing Gaussian NLL. For the
    paper-facing CDF score, method 2 reports conditional log-likelihood, so a
    higher value means the trajectory is easier to predict under the frozen
    dynamics model.
    """
    mode = str(score_mode or "log_likelihood").lower()
    if mode in {"log_likelihood", "logp", "log_likelihood_score"}:
        return -nll_scores
    if mode in {"nll", "negative_log_likelihood", "raw_nll"}:
        return nll_scores
    raise ValueError(f"Unsupported cee_score_mode: {score_mode}")


class CEEDynamicsHead(nn.Module):
    """Frozen autoregressive dynamics model mu_theta(x_t) -> x_{t+1}."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_t)

    def freeze(self) -> "CEEDynamicsHead":
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        return self

    def compute_cee(
        self,
        states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sigma: float = 1.0,
        include_constant: bool = True,
    ) -> torch.Tensor:
        """
        Compute per-sample Gaussian negative log likelihood over valid transitions.

        states: [B, T, D] or [T, D]
        attention_mask: optional [B, T] or [T], True/1 for valid timesteps
        returns: [B] raw NLL scores; lower means easier to predict.
        """
        squeeze_batch = False
        if states.dim() == 2:
            states = states.unsqueeze(0)
            squeeze_batch = True
        if states.dim() != 3:
            raise ValueError(f"states must be [B,T,D] or [T,D], got shape={tuple(states.shape)}")

        batch_size, seq_len, dim = states.shape
        if dim != self.input_dim:
            raise ValueError(f"state dim mismatch: expected {self.input_dim}, got {dim}")
        if seq_len < 2:
            out = states.new_zeros(batch_size)
            return out.squeeze(0) if squeeze_batch else out

        if attention_mask is None:
            transition_mask = torch.ones(batch_size, seq_len - 1, device=states.device, dtype=torch.bool)
        else:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            transition_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()

        x_t = states[:, :-1, :]
        x_next = states[:, 1:, :]
        pred_next = self.forward(x_t)

        sq_error = (x_next - pred_next).pow(2).sum(dim=-1)
        sigma_value = float(sigma)
        if sigma_value <= 0:
            raise ValueError("sigma must be positive")

        nll = sq_error / (2.0 * sigma_value * sigma_value)
        if include_constant:
            nll = nll + 0.5 * dim * math.log(2.0 * math.pi * sigma_value * sigma_value)

        weights = transition_mask.to(nll.dtype)
        counts = weights.sum(dim=1)
        scores = (nll * weights).sum(dim=1) / counts.clamp_min(1.0)
        scores = torch.where(counts > 0, scores, torch.zeros_like(scores))

        return scores.squeeze(0) if squeeze_batch else scores


def load_cee_dynamics_head(
    checkpoint_path: str,
    input_dim: int,
    hidden_dim: int,
    map_location="cpu",
) -> CEEDynamicsHead:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    ckpt_input_dim = checkpoint.get("input_dim") if isinstance(checkpoint, dict) else None
    ckpt_hidden_dim = checkpoint.get("hidden_dim") if isinstance(checkpoint, dict) else None
    if ckpt_input_dim is not None and int(ckpt_input_dim) != int(input_dim):
        raise ValueError(f"CEE checkpoint input_dim={ckpt_input_dim}, expected {input_dim}")
    if ckpt_hidden_dim is not None and int(ckpt_hidden_dim) != int(hidden_dim):
        raise ValueError(f"CEE checkpoint hidden_dim={ckpt_hidden_dim}, expected {hidden_dim}")

    head = CEEDynamicsHead(input_dim=input_dim, hidden_dim=hidden_dim)
    head.load_state_dict(state_dict, strict=True)
    return head.freeze()
