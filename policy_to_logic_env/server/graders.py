"""
Task Graders for the Policy-to-Logic RL Environment.

Each grader produces a deterministic score in [0.0, 1.0] by:
  1. Generating a fixed set of scenarios (seeded)
  2. Executing the agent's rules against each scenario
  3. Comparing results to ground truth
  4. Computing accuracy as the final score

Graders are used both during episodes (for feedback) and at the
end of episodes (for final scoring).
"""

from .dsl_engine import execute_rules, validate_rules
from .ground_truth import evaluate_ground_truth
from .scenario_generator import generate_scenarios


def grade_task(
    task_name: str,
    rules_data: dict,
    scenarios: list[dict] | None = None,
    seed: int = 42,
) -> tuple[float, dict]:
    """
    Grade an agent's rules against a task.

    Args:
        task_name: Task to grade
        rules_data: The agent's rule set (validated DSL)
        scenarios: Pre-generated scenarios (if None, generates fresh ones)
        seed: Random seed for scenario generation

    Returns:
        (score, details) where:
          - score is in [0.0, 1.0]
          - details contains per-scenario results
    """
    # Validate rules first
    is_valid, errors = validate_rules(rules_data)
    if not is_valid:
        return 0.0, {
            "error": "Invalid rules",
            "validation_errors": errors,
            "passed": 0,
            "failed": 0,
            "total": 0,
        }

    # Generate scenarios if not provided
    if scenarios is None:
        scenarios = generate_scenarios(task_name, seed=seed)

    passed = 0
    failed = 0
    failures = []

    for scenario in scenarios:
        expected = scenario.get("expected_decision")
        if expected is None:
            expected = evaluate_ground_truth(task_name, scenario)

        # Execute agent's rules
        actual = execute_rules(rules_data, scenario)

        if actual.upper() == expected.upper():
            passed += 1
        else:
            failed += 1
            if len(failures) < 5:  # Limit failure details for readability
                failures.append({
                    "scenario": {k: v for k, v in scenario.items() if k != "expected_decision"},
                    "expected": expected,
                    "got": actual,
                })

    total = passed + failed
    score = passed / total if total > 0 else 0.0

    details = {
        "passed": passed,
        "failed": failed,
        "total": total,
        "score": round(score, 4),
        "sample_failures": failures,
    }

    return score, details


def quick_grade(
    task_name: str,
    rules_data: dict,
    scenarios: list[dict],
) -> float:
    """
    Fast grading — returns just the accuracy score.
    Used during step processing for efficiency.
    """
    is_valid, _ = validate_rules(rules_data)
    if not is_valid:
        return 0.0

    correct = 0
    total = len(scenarios)

    for scenario in scenarios:
        expected = scenario.get("expected_decision")
        if expected is None:
            expected = evaluate_ground_truth(task_name, scenario)

        actual = execute_rules(rules_data, scenario)
        if actual.upper() == expected.upper():
            correct += 1

    return correct / total if total > 0 else 0.0
