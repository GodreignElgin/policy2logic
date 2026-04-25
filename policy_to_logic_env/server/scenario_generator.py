"""
Scenario Generator for the Policy-to-Logic RL Environment.

Generates diverse test scenarios using four strategies:
  1. Random      (~30%) — Uniform sampling from variable spaces
  2. Boundary    (~20%) — Edge values for numeric fields
  3. Pairwise    (~30%) — Systematic variable combinations
  4. Adversarial (~20%) — Cases designed to expose common mistakes

All scenarios are deterministically seeded for reproducibility.
"""

import random
from itertools import product
from typing import Optional

from .policies import TaskConfig, get_task
from .ground_truth import evaluate_ground_truth


def generate_scenarios(
    task_name: str,
    count: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Generate test scenarios for a task.

    Args:
        task_name: Task to generate scenarios for
        count: Number of scenarios (defaults to task config)
        seed: Random seed for reproducibility

    Returns:
        List of scenario dicts, each with field values and 'expected_decision'
    """
    task = get_task(task_name)
    if count is None:
        count = task.scenario_count

    rng = random.Random(seed)

    # Allocate by strategy
    n_boundary = max(1, int(count * 0.20))
    n_adversarial = max(1, int(count * 0.20))
    n_pairwise = max(1, int(count * 0.30))
    n_random = count - n_boundary - n_adversarial - n_pairwise

    scenarios = []

    # 1. Boundary scenarios
    scenarios.extend(_generate_boundary(task, n_boundary, rng))

    # 2. Pairwise scenarios
    scenarios.extend(_generate_pairwise(task, n_pairwise, rng))

    # 3. Adversarial scenarios
    scenarios.extend(_generate_adversarial(task, n_adversarial, rng))

    # 4. Random scenarios (fill remainder)
    scenarios.extend(_generate_random(task, n_random, rng))

    # Deduplicate
    seen = set()
    unique_scenarios = []
    for s in scenarios:
        key = tuple(sorted((k, v) for k, v in s.items() if k != "expected_decision"))
        if key not in seen:
            seen.add(key)
            unique_scenarios.append(s)

    # Fill back up if dedup removed too many
    while len(unique_scenarios) < count:
        extra = _generate_random(task, 1, rng)
        for s in extra:
            key = tuple(sorted((k, v) for k, v in s.items() if k != "expected_decision"))
            if key not in seen:
                seen.add(key)
                unique_scenarios.append(s)

    # Trim to exact count
    unique_scenarios = unique_scenarios[:count]

    # Add expected decisions
    for s in unique_scenarios:
        if "expected_decision" not in s:
            s["expected_decision"] = evaluate_ground_truth(task_name, s)

    # Shuffle for variety
    rng.shuffle(unique_scenarios)

    return unique_scenarios


def _generate_random(task: TaskConfig, count: int, rng: random.Random) -> list[dict]:
    """Generate random scenarios by sampling uniformly from variable spaces."""
    scenarios = []
    for _ in range(count):
        scenario = {}
        for field, values in task.variables.items():
            scenario[field] = rng.choice(values)
        scenario["expected_decision"] = evaluate_ground_truth(task.name, scenario)
        scenarios.append(scenario)
    return scenarios


def _generate_boundary(task: TaskConfig, count: int, rng: random.Random) -> list[dict]:
    """Generate boundary scenarios focusing on edge values."""
    scenarios = []
    boundary_values = _get_boundary_values(task)

    for _ in range(count):
        scenario = {}
        # Pick at least one boundary field
        boundary_field = rng.choice(list(boundary_values.keys()))

        for field, values in task.variables.items():
            if field == boundary_field:
                scenario[field] = rng.choice(boundary_values[field])
            else:
                scenario[field] = rng.choice(values)

        scenario["expected_decision"] = evaluate_ground_truth(task.name, scenario)
        scenarios.append(scenario)

    return scenarios


def _get_boundary_values(task: TaskConfig) -> dict[str, list]:
    """Extract boundary values for numeric fields based on hidden params."""
    boundaries = {}

    for field, values in task.variables.items():
        if all(isinstance(v, (int, float)) for v in values):
            # Get boundary points from hidden params
            boundary_points = set()
            for param_name, param_val in task.hidden_params.items():
                if isinstance(param_val, (int, float)):
                    boundary_points.add(param_val)
                    boundary_points.add(param_val - 1)
                    boundary_points.add(param_val + 1)

            # Filter to valid values in the variable range
            min_val = min(values)
            max_val = max(values)
            valid_boundaries = [
                v for v in boundary_points
                if min_val <= v <= max_val
            ]

            # Add min and max
            valid_boundaries.extend([min_val, max_val])
            boundaries[field] = list(set(valid_boundaries))

    return boundaries


def _generate_pairwise(task: TaskConfig, count: int, rng: random.Random) -> list[dict]:
    """Generate pairwise combinatorial scenarios."""
    scenarios = []
    fields = list(task.variables.keys())

    # For each pair of fields, generate systematic combinations
    all_combos = []
    if len(fields) >= 2:
        for i in range(len(fields)):
            for j in range(i + 1, len(fields)):
                f1, f2 = fields[i], fields[j]
                v1_sample = _sample_representative(task.variables[f1], rng, max_n=4)
                v2_sample = _sample_representative(task.variables[f2], rng, max_n=4)

                for val1, val2 in product(v1_sample, v2_sample):
                    combo = {}
                    combo[f1] = val1
                    combo[f2] = val2
                    # Fill remaining fields randomly
                    for f in fields:
                        if f not in combo:
                            combo[f] = rng.choice(task.variables[f])
                    combo["expected_decision"] = evaluate_ground_truth(task.name, combo)
                    all_combos.append(combo)

    rng.shuffle(all_combos)
    scenarios = all_combos[:count]

    # Fill if needed
    while len(scenarios) < count:
        scenarios.extend(_generate_random(task, count - len(scenarios), rng))

    return scenarios[:count]


def _sample_representative(values: list, rng: random.Random, max_n: int = 4) -> list:
    """Get representative values from a list (min, max, middle, random)."""
    if len(values) <= max_n:
        return values

    representatives = set()

    # Include min and max for numeric values
    if all(isinstance(v, (int, float)) for v in values):
        sorted_vals = sorted(values)
        representatives.add(sorted_vals[0])
        representatives.add(sorted_vals[-1])
        representatives.add(sorted_vals[len(sorted_vals) // 2])

    # Fill remaining with random
    remaining = [v for v in values if v not in representatives]
    while len(representatives) < max_n and remaining:
        val = rng.choice(remaining)
        representatives.add(val)
        remaining.remove(val)

    return list(representatives)


def _generate_adversarial(task: TaskConfig, count: int, rng: random.Random) -> list[dict]:
    """
    Generate adversarial scenarios designed to expose common mistakes.

    Strategies:
    - Scenarios where the decision depends on a single field change
    - Scenarios near decision boundaries
    - Scenarios testing default vs. explicit rules
    """
    scenarios = []

    if task.name == "data_access":
        # Test boundary: time exactly at 9 and 18
        adversarial_cases = [
            {"time": 9, "data_type": "sensitive"},   # should be ALLOW (start of hours)
            {"time": 18, "data_type": "sensitive"},   # should be DENY (end of hours, >= 18)
            {"time": 8, "data_type": "sensitive"},    # should be DENY (just before hours)
            {"time": 17, "data_type": "sensitive"},   # should be ALLOW (just before end)
            {"time": 0, "data_type": "public"},       # should be ALLOW (public always)
            {"time": 23, "data_type": "internal"},    # should be DENY (internal = sensitive)
            {"time": 12, "data_type": "internal"},    # should be ALLOW (internal during hours)
        ]

    elif task.name == "resource_access":
        adversarial_cases = [
            {"role": "junior", "time": 8, "document_type": "confidential"},     # DENY
            {"role": "junior", "time": 7, "document_type": "internal"},         # DENY
            {"role": "junior", "time": 17, "document_type": "internal"},        # DENY (17 = outside)
            {"role": "junior", "time": 16, "document_type": "internal"},        # ALLOW (16 < 17)
            {"role": "contractor", "time": 12, "document_type": "internal"},    # DENY
            {"role": "senior", "time": 2, "document_type": "confidential"},     # ALLOW
            {"role": "junior", "time": 12, "document_type": "public"},          # ALLOW
            {"role": "contractor", "time": 12, "document_type": "public"},      # ALLOW
        ]

    elif task.name == "transaction_approval":
        adversarial_cases = [
            {"amount": 5000, "transfer_type": "domestic", "time": 12, "initiator_role": "employee"},    # APPROVE (at limit, not above)
            {"amount": 5001, "transfer_type": "domestic", "time": 12, "initiator_role": "employee"},    # REQUIRE_APPROVAL
            {"amount": 5001, "transfer_type": "domestic", "time": 12, "initiator_role": "manager"},     # APPROVE (manager exempt)
            {"amount": 10000, "transfer_type": "domestic", "time": 20, "initiator_role": "employee"},   # HOLD (high-value, non-business)
            {"amount": 10000, "transfer_type": "domestic", "time": 12, "initiator_role": "employee"},   # REQUIRE_APPROVAL (high-value but business hours)
            {"amount": 100, "transfer_type": "international", "time": 12, "initiator_role": "employee"},  # COMPLIANCE_REVIEW
            {"amount": 50000, "transfer_type": "international", "time": 3, "initiator_role": "manager"},  # COMPLIANCE_REVIEW (intl trumps all)
            {"amount": 9999, "transfer_type": "domestic", "time": 20, "initiator_role": "employee"},    # REQUIRE_APPROVAL (not high-value for HOLD)
            {"amount": 10000, "transfer_type": "domestic", "time": 9, "initiator_role": "employee"},    # REQUIRE_APPROVAL (business hours)
            {"amount": 10000, "transfer_type": "domestic", "time": 17, "initiator_role": "employee"},   # HOLD (17 = non-business)
        ]
    else:
        adversarial_cases = []

    # Add expected decisions
    for case in adversarial_cases:
        case["expected_decision"] = evaluate_ground_truth(task.name, case)

    rng.shuffle(adversarial_cases)
    scenarios = adversarial_cases[:count]

    # Fill if needed
    while len(scenarios) < count:
        scenarios.extend(_generate_random(get_task(task.name), count - len(scenarios), rng))

    return scenarios[:count]
