from __future__ import annotations

import torch
from trl import GRPOTrainer


class ZeroVarianceFilteringGRPOTrainer(GRPOTrainer):
    """DAPO Dynamic Sampling component: drop whole groups whose rewards have zero variance.

    A zero-variance group yields no GRPO advantage, so it contributes nothing but gradient
    noise while still consuming the token budget that normalizes the loss. TRL 1.15 has no
    built-in support, so the groups are removed right after `_generate_and_score_completions`
    returns and the removed fraction is reported as `frac_zero_var_filtered`.

    The filter never empties a batch: if every group would be dropped, the original batch is
    returned untouched and the ratio still records 1.0, because a batch with no gradient
    signal is a fact about the run, not something to hide by crashing.
    """

    FILTERABLE_KEYS = ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "advantages",
                       "old_per_token_logps", "ref_per_token_logps", "importance_sampling_ratio", "tool_mask")

    def __init__(self, *args, filter_zero_variance_groups: bool = True, zero_variance_eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_zero_variance_groups = filter_zero_variance_groups
        self.zero_variance_eps = zero_variance_eps
        self.filtered_groups = 0
        self.total_groups = 0

    def zero_variance_group_fraction(self) -> float:
        return self.filtered_groups / self.total_groups if self.total_groups else 0.0

    def _record(self, mode: str, fraction: float) -> None:
        metrics = getattr(self, "_metrics", None)
        if metrics is None:
            return
        metrics.setdefault(mode, {}).setdefault("frac_zero_var_filtered", []).append(float(fraction))

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        mode = "train" if self.model.training else "eval"
        advantages = output.get("advantages")
        generations = int(getattr(self, "num_generations", 0) or getattr(self.args, "num_generations", 0))
        if not self.filter_zero_variance_groups or advantages is None or generations < 1:
            self._record(mode, 0.0)
            return output
        total = int(advantages.shape[0])
        if total % generations != 0:
            self._record(mode, 0.0)
            return output
        groups = total // generations
        keep = [bool(torch.max(torch.abs(advantages[index * generations:(index + 1) * generations])) > self.zero_variance_eps)
                for index in range(groups)]
        kept = sum(keep)
        dropped = groups - kept
        self.filtered_groups += dropped
        self.total_groups += groups
        self._record(mode, dropped / groups if groups else 0.0)
        if dropped == 0 or kept == 0:
            return output
        rows = torch.tensor(keep, dtype=torch.bool, device=advantages.device)
        rows = rows.repeat_interleave(generations).nonzero().squeeze(-1)
        for key in self.FILTERABLE_KEYS:
            value = output.get(key)
            if isinstance(value, torch.Tensor) and value.shape[0] == total:
                output[key] = value.index_select(0, rows)
        completion_mask = output.get("completion_mask")
        if isinstance(completion_mask, torch.Tensor):
            # Renormalize the token budget so the loss matches the surviving samples.
            output["num_items_in_batch"] = completion_mask.sum()
        return output
