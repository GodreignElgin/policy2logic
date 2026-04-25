"""
Local test script — verifies the environment works without needing an LLM API.

Run: uv run python test_local.py
"""

import json
import sys

# Test imports
print("=" * 60)
print("Testing Policy-to-Logic RL Environment")
print("=" * 60)

# ── Test 1: DSL Engine ────────────────────────────────────────────
print("\n[1/5] Testing DSL Engine...")
from policy_to_logic_env.server.dsl_engine import parse_rules, execute_rules, validate_rules

rules_json = json.dumps({
    "rules": [
        {
            "if": [
                {"field": "time", "op": ">=", "value": 18},
                {"field": "data_type", "op": "==", "value": "sensitive"}
            ],
            "then": "DENY"
        },
        {
            "if": [
                {"field": "time", "op": "<", "value": 9},
                {"field": "data_type", "op": "==", "value": "sensitive"}
            ],
            "then": "DENY"
        },
        {
            "if": [
                {"field": "time", "op": ">=", "value": 18},
                {"field": "data_type", "op": "==", "value": "internal"}
            ],
            "then": "DENY"
        },
        {
            "if": [
                {"field": "time", "op": "<", "value": 9},
                {"field": "data_type", "op": "==", "value": "internal"}
            ],
            "then": "DENY"
        }
    ],
    "default": "ALLOW"
})

rules_data, errors = parse_rules(rules_json)
assert rules_data is not None, f"Parse failed: {errors}"
assert len(errors) == 0

# Test execution
result = execute_rules(rules_data, {"time": 20, "data_type": "sensitive"})
assert result == "DENY", f"Expected DENY, got {result}"

result = execute_rules(rules_data, {"time": 12, "data_type": "sensitive"})
assert result == "ALLOW", f"Expected ALLOW, got {result}"

result = execute_rules(rules_data, {"time": 22, "data_type": "public"})
assert result == "ALLOW", f"Expected ALLOW, got {result}"

print("   ✅ DSL Engine working correctly")


# ── Test 2: Scenario Generator ────────────────────────────────────
print("\n[2/5] Testing Scenario Generator...")
from policy_to_logic_env.server.scenario_generator import generate_scenarios

for task_name in ["data_access", "resource_access", "transaction_approval"]:
    scenarios = generate_scenarios(task_name)
    assert len(scenarios) > 0, f"No scenarios for {task_name}"
    assert all("expected_decision" in s for s in scenarios), f"Missing expected_decision in {task_name}"
    print(f"   ✅ {task_name}: {len(scenarios)} scenarios generated")


# ── Test 3: Ground Truth ─────────────────────────────────────────
print("\n[3/5] Testing Ground Truth Engine...")
from policy_to_logic_env.server.ground_truth import evaluate_ground_truth, answer_clarification

# Test data_access
assert evaluate_ground_truth("data_access", {"time": 20, "data_type": "sensitive"}) == "DENY"
assert evaluate_ground_truth("data_access", {"time": 12, "data_type": "sensitive"}) == "ALLOW"
assert evaluate_ground_truth("data_access", {"time": 3, "data_type": "public"}) == "ALLOW"

# Test resource_access
assert evaluate_ground_truth("resource_access", {"role": "senior", "time": 3, "document_type": "confidential"}) == "ALLOW"
assert evaluate_ground_truth("resource_access", {"role": "contractor", "time": 12, "document_type": "internal"}) == "DENY"
assert evaluate_ground_truth("resource_access", {"role": "junior", "time": 12, "document_type": "internal"}) == "ALLOW"

# Test transaction_approval
assert evaluate_ground_truth("transaction_approval", {"amount": 100, "transfer_type": "international", "time": 12, "initiator_role": "employee"}) == "COMPLIANCE_REVIEW"
assert evaluate_ground_truth("transaction_approval", {"amount": 10000, "transfer_type": "domestic", "time": 20, "initiator_role": "employee"}) == "HOLD"
assert evaluate_ground_truth("transaction_approval", {"amount": 6000, "transfer_type": "domestic", "time": 12, "initiator_role": "manager"}) == "APPROVE"

# Test clarification oracle
answer = answer_clarification("transaction_approval", "What is the standard limit?")
assert "5,000" in answer

print("   ✅ Ground Truth and Oracle working correctly")


# ── Test 4: Graders ───────────────────────────────────────────────
print("\n[4/5] Testing Graders...")
from policy_to_logic_env.server.graders import grade_task

# Grade a perfect ruleset for data_access
perfect_rules = {
    "rules": [
        {
            "if": [
                {"field": "time", "op": ">=", "value": 18},
                {"field": "data_type", "op": "==", "value": "sensitive"}
            ],
            "then": "DENY"
        },
        {
            "if": [
                {"field": "time", "op": "<", "value": 9},
                {"field": "data_type", "op": "==", "value": "sensitive"}
            ],
            "then": "DENY"
        },
        {
            "if": [
                {"field": "time", "op": ">=", "value": 18},
                {"field": "data_type", "op": "==", "value": "internal"}
            ],
            "then": "DENY"
        },
        {
            "if": [
                {"field": "time", "op": "<", "value": 9},
                {"field": "data_type", "op": "==", "value": "internal"}
            ],
            "then": "DENY"
        }
    ],
    "default": "ALLOW"
}

score, details = grade_task("data_access", perfect_rules)
print(f"   Perfect rules score: {score:.2%} ({details['passed']}/{details['total']})")
assert score >= 0.9, f"Perfect rules should score >=0.9, got {score}"

# Grade an empty ruleset
empty_rules = {"rules": [], "default": "ALLOW"}
score_empty, details_empty = grade_task("data_access", empty_rules)
print(f"   Empty rules score: {score_empty:.2%} ({details_empty['passed']}/{details_empty['total']})")

print("   ✅ Graders working correctly")


# ── Test 5: Full Environment Loop ─────────────────────────────────
print("\n[5/5] Testing Full Environment Loop...")
from policy_to_logic_env.server.environment import PolicyToLogicEnvironment
from policy_to_logic_env.models import PolicyToLogicAction

env = PolicyToLogicEnvironment()

# Reset
result = env.reset(task_name="data_access")
assert not result.done
assert result.observation.task_name == "data_access"
assert result.observation.step_number == 0
print(f"   Reset OK. Policy: {result.observation.policy_text[:60]}...")

# Step 1: Ask clarification
result = env.step(PolicyToLogicAction(
    action_type="ask_clarification",
    content=json.dumps({"question": "What are working hours?"})
))
assert not result.done
assert result.observation.clarification_response is not None
print(f"   Step 1 (clarify): answer='{result.observation.clarification_response[:60]}...', reward={result.reward:.2f}")

# Step 2: Propose rules
result = env.step(PolicyToLogicAction(
    action_type="propose_rules",
    content=json.dumps(perfect_rules)
))
print(f"   Step 2 (propose): accuracy={result.observation.current_accuracy:.2%}, reward={result.reward:.2f}, done={result.done}")

# Check state
state = env.state()
print(f"   State: episode={state.episode_id}, steps={state.step_count}, questions={state.questions_asked}")

print("   ✅ Full environment loop working correctly")


# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED! Environment is working correctly.")
print("=" * 60)
print("\nNext steps:")
print("  1. Start server:  uv run python main.py")
print("  2. Test API:      curl -X POST http://localhost:7860/reset -H 'Content-Type: application/json' -d '{}'")
print("  3. Run inference:  HF_TOKEN=xxx uv run python inference.py")
