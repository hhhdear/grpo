"""Small, auditable GRPO reference equations for numerical cross-checks."""
from __future__ import annotations

import torch

from .dapo import asymmetric_clipped_surrogate, token_level_policy_loss


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-8) -> torch.Tensor:
    if rewards.ndim != 1 or group_size < 2 or rewards.numel() % group_size:
        raise ValueError("rewards must be a flat tensor of complete groups")
    grouped = rewards.reshape(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    return ((grouped - mean) / (std + eps)).reshape_as(rewards)


def grpo_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor,
              rewards: torch.Tensor, mask: torch.Tensor, group_size: int,
              epsilon_low: float = 0.2, epsilon_high: float = 0.2) -> torch.Tensor:
    advantages = group_advantages(rewards, group_size).unsqueeze(1).expand_as(log_probs)
    surrogate, _ = asymmetric_clipped_surrogate(
        log_probs, old_log_probs, advantages, epsilon_low, epsilon_high
    )
    return token_level_policy_loss(surrogate, mask)

