from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable

SCENARIOS = ("normal", "shipped", "insufficient_stock", "invalid_order", "transient_failure", "combined")


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    scenario: str
    order_id: str
    product_id: str
    initial_order_status: str
    initial_quantity: int
    available_stock: int
    requested_quantity: int
    expected_outcome: str
    seed: int
    transient_failures: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def generate_task(seed: int, index: int = 0) -> Task:
    rng = random.Random(seed)
    scenario = SCENARIOS[index % len(SCENARIOS)]
    actual_order_id = f"ORD-{seed % 1000000:06d}"
    requested_order_id = actual_order_id
    product_id = f"SKU-{rng.randint(100, 999)}"
    initial = rng.randint(1, 4)
    requested = rng.randint(initial + 1, initial + 5)
    stock = requested + rng.randint(0, 5)
    status = "pending"
    expected = "modified"
    failures: dict[str, int] = {}
    if scenario == "shipped":
        status, expected = "shipped", "rejected"
    elif scenario == "insufficient_stock":
        stock, expected = max(0, requested - 1), "rejected"
    elif scenario == "invalid_order":
        requested_order_id, expected = f"BAD-{seed}", "rejected"
    elif scenario == "transient_failure":
        failures = {"query_order": 1}
    elif scenario == "combined":
        status, stock, expected = "shipped", max(0, requested - 2), "rejected"
    variants = [
        "Please change order {oid} to quantity {qty}. Check everything needed and verify the result.",
        "For {oid}, set the item count to {qty} only if policy and stock allow it; confirm the actual outcome.",
        "Handle order {oid}: I want {qty} units. Use the available order tools and verify any change.",
    ]
    prompt = variants[rng.randrange(len(variants))].format(oid=requested_order_id, qty=requested)
    return Task(f"task-{seed}-{index}", prompt, scenario, requested_order_id, product_id, status,
                initial, stock, requested, expected, seed, failures)


def generate_tasks(count: int, seed: int) -> list[Task]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    return [generate_task(seed + i * 104729, i) for i in range(count)]


def write_jsonl(tasks: Iterable[Task], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
