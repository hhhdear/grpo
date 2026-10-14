from __future__ import annotations

import json
import tempfile
import unittest

from grpo_agent.environment import OrderEnvironment
from grpo_agent.metrics import aggregate_metrics
from grpo_agent.parser import parse_action
from grpo_agent.policies import oracle_policy
from grpo_agent.rewards import calculate_reward
from grpo_agent.rollout import run_rollout
from grpo_agent.task_generator import generate_task, generate_tasks, read_jsonl, write_jsonl

WEIGHTS = {"task_success": 2.0, "rule_compliance": 0.5, "verification": 0.5,
           "invalid_action": -1.0, "redundant_calls": -0.2, "execution_cost": -0.05}


class EnvironmentTests(unittest.TestCase):
    def test_valid_sequence_changes_and_verifies(self):
        task = generate_task(10, 0).to_dict()
        env = OrderEnvironment(task)
        order = env.execute("query_order", {"order_id": task["order_id"]})
        env.execute("query_stock", {"product_id": order.data["product_id"]})
        self.assertTrue(env.execute("modify_order", {"order_id": task["order_id"], "quantity": task["requested_quantity"]}).ok)
        env.execute("verify_order", {"order_id": task["order_id"]})
        self.assertTrue(env.verify_task()["task_success"])

    def test_shipped_and_stock_rules(self):
        for index, code in ((1, "forbidden_status"), (2, "insufficient_stock")):
            task = generate_task(20 + index, index).to_dict()
            env = OrderEnvironment(task)
            result = env.execute("modify_order", {"order_id": task["order_id"], "quantity": task["requested_quantity"]})
            self.assertFalse(result.ok)
            self.assertEqual(result.code, code)

    def test_argument_validation_and_temporary_failure(self):
        task = generate_task(30, 4).to_dict()
        env = OrderEnvironment(task)
        self.assertEqual(env.execute("query_order", {"order_id": task["order_id"]}).code, "temporary_failure")
        self.assertTrue(env.execute("query_order", {"order_id": task["order_id"]}).ok)
        self.assertEqual(env.execute("modify_order", {"order_id": task["order_id"], "quantity": True}).code, "invalid_arguments")


class PipelineTests(unittest.TestCase):
    def test_parser_is_strict(self):
        action = parse_action('{"tool":"query_order","arguments":{"order_id":"x"}}')
        self.assertEqual(action.tool, "query_order")
        with self.assertRaises(ValueError):
            parse_action("I will call a tool")

    def test_generator_reproducible_and_jsonl(self):
        first, second = generate_tasks(8, 44), generate_tasks(8, 44)
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/tasks.jsonl"
            write_jsonl(first, path)
            self.assertEqual(len(read_jsonl(path)), 8)

    def test_oracle_handles_all_scenarios(self):
        results = [run_rollout(t.to_dict(), oracle_policy, WEIGHTS) for t in generate_tasks(12, 100)]
        self.assertTrue(all(r.verification["task_success"] for r in results))
        rows = [{"scenario": generate_tasks(12, 100)[i].scenario, **r.to_dict()} for i, r in enumerate(results)]
        self.assertEqual(aggregate_metrics(rows)["task_success_rate"], 1.0)

    def test_claimed_success_does_not_earn_success_reward(self):
        task = generate_task(5, 0).to_dict()
        result = run_rollout(task, lambda messages, env: '{"final":"success"}', WEIGHTS)
        self.assertFalse(result.verification["task_success"])
        self.assertEqual(result.rewards["task_success"], 0.0)

    def test_unsupported_claim_without_evidence_does_not_earn_refusal_reward(self):
        task = generate_task(6, 1).to_dict()
        result = run_rollout(task, lambda messages, env: '{"final":"I refuse"}', WEIGHTS)
        self.assertFalse(result.verification["task_success"])
        self.assertEqual(result.rewards["task_success"], 0.0)


if __name__ == "__main__":
    unittest.main()
