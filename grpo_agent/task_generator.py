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


def generate_task(seed: int, index: int = 0, wide: bool = False) -> Task:
    """Build one task. `wide` widens the parameter space so that thousands of tasks can
    still be pairwise disjoint on their parameter combination; the default keeps the
    original narrow ranges that the unit tests pin down.
    """
    rng = random.Random(seed)
    scenario = SCENARIOS[index % len(SCENARIOS)]
    actual_order_id = f"ORD-{seed % 1000000:06d}"
    requested_order_id = actual_order_id
    product_id = f"SKU-{rng.randint(100, 999)}"
    if wide:
        initial = rng.randint(1, 12)
        requested = initial + rng.randint(1, 12)
        surplus = rng.randint(0, 12)
        shortage = rng.randint(1, 6)
    else:
        initial = rng.randint(1, 4)
        requested = rng.randint(initial + 1, initial + 5)
        surplus = rng.randint(0, 5)
        shortage = 1
    stock = requested + surplus
    status = "pending"
    expected = "modified"
    failures: dict[str, int] = {}
    if scenario == "shipped":
        status, expected = "shipped", "rejected"
    elif scenario == "insufficient_stock":
        stock, expected = max(0, requested - shortage), "rejected"
    elif scenario == "invalid_order":
        requested_order_id, expected = f"BAD-{seed}", "rejected"
    elif scenario == "transient_failure":
        failures = {"query_order": 1}
    elif scenario == "combined":
        status, stock, expected = "shipped", max(0, requested - shortage), "rejected"
    variants = [
        "Please change order {oid} to quantity {qty}. Check everything needed and verify the result.",
        "For {oid}, set the item count to {qty} only if policy and stock allow it; confirm the actual outcome.",
        "Handle order {oid}: I want {qty} units. Use the available order tools and verify any change.",
    ]
    prompt = variants[rng.randrange(len(variants))].format(oid=requested_order_id, qty=requested)
    return Task(f"task-{seed}-{index}", prompt, scenario, requested_order_id, product_id, status,
                initial, stock, requested, expected, seed, failures)


def generate_tasks(count: int, seed: int, wide: bool = False) -> list[Task]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    return [generate_task(seed + i * 104729, i, wide=wide) for i in range(count)]


def combination_signature(task: Task) -> tuple:
    """The parameter combination a task instantiates.

    Splitting train and test on this signature is what keeps the test set from being a
    re-run of training combinations; it is the programmatic split the design promises.
    """
    return (task.scenario, task.initial_order_status, task.initial_quantity,
            task.requested_quantity - task.initial_quantity,
            task.available_stock - task.requested_quantity,
            int(bool(task.transient_failures)))


def write_jsonl(tasks: Iterable[Task], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
