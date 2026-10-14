import unittest

import torch

from grpo_agent.dapo import (asymmetric_clipped_surrogate, dynamic_sampling,
                             overlong_reward_penalty, token_level_policy_loss)
from grpo_agent.reference_grpo import group_advantages, grpo_loss


class DapoMathTests(unittest.TestCase):
    def test_clip_higher_is_asymmetric(self):
        ratios = torch.tensor([0.5, 1.25, 1.4])
        new = ratios.log()
        surrogate, stats = asymmetric_clipped_surrogate(new, torch.zeros(3), torch.ones(3), .2, .3)
        self.assertTrue(torch.allclose(surrogate, torch.tensor([.5, 1.25, 1.3])))
        self.assertAlmostEqual(stats["clip_ratio_high"], 1 / 3)

    def test_token_loss_uses_global_valid_token_denominator(self):
        values = torch.tensor([[1., 3., 99.], [2., 99., 99.]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        self.assertAlmostEqual(token_level_policy_loss(values, mask).item(), -2.0)

    def test_dynamic_sampling_metrics_and_all_invalid_terminate(self):
        result = dynamic_sampling([1, 1, 0, 2], 2)
        self.assertEqual(result.kept_indices, (2, 3))
        self.assertEqual(result.frac_reward_zero_std, .5)
        self.assertEqual(result.frac_zero_var_filtered, .5)
        empty = dynamic_sampling([1, 1, 2, 2], 2)
        self.assertEqual(empty.kept_indices, ())

    def test_overlong_reward_boundaries(self):
        self.assertEqual(overlong_reward_penalty(80, 100, 20, 1), 0)
        self.assertEqual(overlong_reward_penalty(90, 100, 20, 1), -.5)
        self.assertEqual(overlong_reward_penalty(100, 100, 20, 1), -1)
        self.assertEqual(overlong_reward_penalty(110, 100, 20, 1), -1)

    def test_reference_advantages_and_gradient(self):
        rewards = torch.tensor([1., 3., 2., 2.])
        advantages = group_advantages(rewards, 2)
        self.assertTrue(torch.allclose(advantages[:2], torch.tensor([-1., 1.])))
        self.assertTrue(torch.allclose(advantages[2:], torch.zeros(2)))
        logits = torch.zeros((4, 2), requires_grad=True)
        loss = grpo_loss(logits, torch.zeros_like(logits), rewards, torch.ones_like(logits), 2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(logits.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()

