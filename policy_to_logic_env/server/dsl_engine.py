"""
DSL Engine — Parser and executor for the Policy-to-Logic rule format.

The DSL is an intentionally constrained JSON-based rule language:
- Rules are a list of IF-THEN conditions
- All conditions within a rule are AND-ed
- Rules are evaluated top-to-bottom, first match wins
- A default decision applies when no rules match

Supported operators: >, <, >=, <=, ==, !=
Supported value types: int, float, str

Example DSL:
{
    "rules": [
        {
            "if": [
                {"field": "time", "op": ">", "value": 18},
                {"field": "data_type", "op": "==", "value": "sensitive"}
            ],
            "then": "DENY"
        }
    ],
    "default": "ALLOW"
}
"""

import json
from typing import Any


# ─── Supported comparison operators ───────────────────────────────
OPERATORS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

VALID_OPS = set(OPERATORS.keys())


# ─── Validation ───────────────────────────────────────────────────
def validate_rules(rules_data: dict) -> tuple[bool, list[str]]:
    """
    Validate a rule set for structural correctness.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    if not isinstance(rules_data, dict):
        return False, ["Rules must be a JSON object (dict)"]

    if "rules" not in rules_data:
        errors.append("Missing 'rules' key")
    elif not isinstance(rules_data["rules"], list):
        errors.append("'rules' must be a list")
    else:
        for i, rule in enumerate(rules_data["rules"]):
            rule_errors = _validate_single_rule(rule, i)
            errors.extend(rule_errors)

    if "default" not in rules_data:
        errors.append("Missing 'default' key (e.g., 'ALLOW' or 'DENY')")
    elif not isinstance(rules_data["default"], str):
        errors.append("'default' must be a string")

    return len(errors) == 0, errors


def _validate_single_rule(rule: dict, index: int) -> list[str]:
    """Validate a single rule entry."""
    errors = []
    prefix = f"Rule {index}"

    if not isinstance(rule, dict):
        return [f"{prefix}: must be an object"]

    if "if" not in rule:
        errors.append(f"{prefix}: missing 'if' key")
    elif not isinstance(rule["if"], list):
        errors.append(f"{prefix}: 'if' must be a list of conditions")
    else:
        for j, cond in enumerate(rule["if"]):
            cond_errors = _validate_condition(cond, f"{prefix}, condition {j}")
            errors.extend(cond_errors)

    if "then" not in rule:
        errors.append(f"{prefix}: missing 'then' key")
    elif not isinstance(rule["then"], str):
        errors.append(f"{prefix}: 'then' must be a string")

    return errors


def _validate_condition(cond: dict, prefix: str) -> list[str]:
    """Validate a single condition within a rule."""
    errors = []

    if not isinstance(cond, dict):
        return [f"{prefix}: condition must be an object"]

    for key in ("field", "op", "value"):
        if key not in cond:
            errors.append(f"{prefix}: missing '{key}'")

    if "field" in cond and not isinstance(cond["field"], str):
        errors.append(f"{prefix}: 'field' must be a string")

    if "op" in cond and cond["op"] not in VALID_OPS:
        errors.append(f"{prefix}: invalid operator '{cond['op']}'. Must be one of: {VALID_OPS}")

    return errors


# ─── Execution ────────────────────────────────────────────────────
def execute_rules(rules_data: dict, scenario: dict) -> str:
    """
    Execute a rule set against a scenario.

    Rules are evaluated top-to-bottom. The first rule where ALL
    conditions match produces the decision. If no rules match,
    the default decision is returned.

    Args:
        rules_data: Validated rule set with 'rules' and 'default' keys
        scenario: Dict of field→value pairs representing a test case

    Returns:
        Decision string (e.g., "ALLOW", "DENY", "REVIEW")
    """
    for rule in rules_data.get("rules", []):
        if _evaluate_rule(rule, scenario):
            return rule["then"]

    return rules_data.get("default", "ALLOW")


def _evaluate_rule(rule: dict, scenario: dict) -> bool:
    """
    Evaluate whether ALL conditions in a rule match the scenario.
    All conditions are AND-ed.
    """
    conditions = rule.get("if", [])
    if not conditions:
        return False

    for cond in conditions:
        if not _evaluate_condition(cond, scenario):
            return False

    return True


def _evaluate_condition(cond: dict, scenario: dict) -> bool:
    """Evaluate a single condition against a scenario."""
    field = cond.get("field", "")
    op = cond.get("op", "")
    value = cond.get("value")

    if field not in scenario:
        return False

    scenario_value = scenario[field]
    op_func = OPERATORS.get(op)

    if op_func is None:
        return False

    try:
        # Type coercion: try to match types for comparison
        if isinstance(value, (int, float)) and isinstance(scenario_value, str):
            try:
                scenario_value = type(value)(scenario_value)
            except (ValueError, TypeError):
                return False
        elif isinstance(scenario_value, (int, float)) and isinstance(value, str):
            try:
                value = type(scenario_value)(value)
            except (ValueError, TypeError):
                return False

        return op_func(scenario_value, value)
    except (TypeError, ValueError):
        return False


# ─── Parsing ──────────────────────────────────────────────────────
def parse_rules(content_str: str) -> tuple[dict | None, list[str]]:
    """
    Parse a JSON string into a validated rule set.

    Returns:
        (rules_data or None, list_of_errors)
    """
    try:
        data = json.loads(content_str)
    except json.JSONDecodeError as e:
        return None, [f"Invalid JSON: {e}"]

    is_valid, errors = validate_rules(data)
    if not is_valid:
        return None, errors

    return data, []
