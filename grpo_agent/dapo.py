"""Pure PyTorch implementations of the four DAPO mechanisms.

These functions deliberately do not depend on TRL internals.  A version-pinned
trainer adapter can call them, while unit tests can verify their mathematics on
CPU.  They are not evidence that a model-training experiment has been run.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


def asymmetric_clipped_surrogate(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DAPO Clip-Higher surrogate, before token masking/reduction."""
    if epsilon_low < 0 or epsilon_high < 0:
        raise ValueError("clip epsilons must be non-negative")
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = ratio.clamp(1.0 - epsilon_low, 1.0 + epsilon_high)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    stats = {
        "clip_ratio_low": float((ratio < 1.0 - epsilon_low).float().mean().detach()),
        "clip_ratio_high": float((ratio > 1.0 + epsilon_high).float().mean().detach()),
        "clip_ratio": float((ratio.ne(clipped)).float().mean().detach()),
    }
    return surrogate, stats


def token_level_policy_loss(surrogate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Token-level loss: normalize globally by valid completion tokens."""
    if surrogate.shape != mask.shape:
        raise ValueError("surrogate and mask must have identical shapes")
    valid = mask.to(dtype=surrogate.dtype)
    denominator = valid.sum()
    if denominator.item() <= 0:
        raise ValueError("at least one completion token is required")
    return -(surrogate * valid).sum() / denominator


@dataclass(frozen=True)
class SamplingResult:
    kept_indices: tuple[int, ...]
    group_std: tuple[float, ...]
    frac_reward_zero_std: float
    frac_zero_var_filtered: float


def dynamic_sampling(rewards: Sequence[float], group_size: int, tolerance: float = 1e-8) -> SamplingResult:
    """Remove zero-variance prompt groups, with unambiguous group statistics."""
    if group_size < 2 or len(rewards) % group_size:
        raise ValueError("rewards must contain complete groups of size >= 2")
    groups = [rewards[i:i + group_size] for i in range(0, len(rewards), group_size)]
    stds = [float(torch.tensor(group, dtype=torch.float64).std(unbiased=False)) for group in groups]
    useful = [i for i, std in enumerate(stds) if math.isfinite(std) and std > tolerance]
    kept = tuple(i * group_size + j for i in useful for j in range(group_size))
    zero_fraction = 1.0 - len(useful) / len(groups)
    filtered_fraction = 1.0 - len(kept) / len(rewards)
    return SamplingResult(kept, tuple(stds), zero_fraction, filtered_fraction)


def overlong_reward_penalty(length: int, max_length: int, buffer: int, penalty: float) -> float:
    """DAPO soft overlong punishment: zero, linear buffer, then full penalty."""
    if length < 0 or max_length < 1 or buffer < 1 or penalty < 0:
        raise ValueError("invalid overlong shaping arguments")
    safe = max(0, max_length - buffer)
    if length <= safe:
        return 0.0
    if length >= max_length:
        return -float(penalty)
    return -float(penalty) * (length - safe) / buffer

