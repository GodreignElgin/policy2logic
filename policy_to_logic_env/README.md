# 🧠 Policy-to-Logic RL Environment

> An OpenEnv-compliant environment where AI agents learn to convert natural language access control policies into executable logic rules through iterative interaction and feedback.

## 🎯 What This Environment Does

This environment simulates a **real-world compliance task**: translating human-readable access control policies into machine-executable decision rules. The agent must:

1. **Read** a natural language policy
2. **Ask clarifications** when terms are ambiguous (e.g., "What are business hours?")
3. **Propose rules** in a structured DSL format
4. **Refine rules** based on test scenario feedback
5. **Achieve ≥90% accuracy** to complete the episode

This tests structured reasoning, information-seeking behavior, and iterative improvement — all critical capabilities for real-world AI systems.

## 🏗 Environment Design

### Action Space

| Action | Description |
|--------|-------------|
| `ask_clarification` | Ask a question about unclear policy terms |
| `propose_rules` | Submit a complete rule set in JSON DSL format |
| `refine_rules` | Improve previously proposed rules based on feedback |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `policy_text` | string | The natural language policy to convert |
| `task_name` | string | Current task identifier |
| `step_number` | int | Current step (1-indexed) |
| `max_steps` | int | Episode step budget |
| `clarification_response` | string? | Answer to a clarification question |
| `test_results` | dict? | `{passed, failed, total, sample_failures}` |
| `current_accuracy` | float | Current rule accuracy (0.0–1.0) |
| `feedback` | string? | Human-readable feedback |
| `dsl_format` | string | DSL format specification |

### Reward Structure (Dense, Multi-Component)

| Component | Weight | Signal |
|-----------|--------|--------|
| Accuracy | 0.50 | % of test scenarios passed |
| Improvement | 0.20 | Delta from previous accuracy |
| Efficiency | 0.15 | Penalty for excess steps |
| Clarification Quality | 0.15 | Useful questions → bonus, wasteful → penalty |

### DSL Format

Rules are expressed in a constrained JSON format:

```json
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
```

- **Operators**: `>`, `<`, `>=`, `<=`, `==`, `!=`
- **Logic**: All conditions within a rule are AND-ed
- **Evaluation**: Top-to-bottom, first match wins
- **Default**: Applied when no rules match

## 📋 Tasks (Easy → Medium → Hard)

### Task 1: `data_access` (Easy)
**Policy**: "Employees must not access sensitive data after working hours."
- 2 variables: `time`, `data_type`
- 2 decisions: ALLOW, DENY
- 30 test scenarios
- 5 steps max
- **No ambiguity** — all information is in the policy text

### Task 2: `resource_access` (Medium)
**Policy**: Role-based document access with multiple employee types and document classifications.
- 3 variables: `role`, `time`, `document_type`
- 2 decisions: ALLOW, DENY
- 50 test scenarios
- 7 steps max
- **One hidden parameter** — business hours definition requires clarification

### Task 3: `transaction_approval` (Hard)
**Policy**: Financial transaction approval with limits, international rules, and role exemptions.
- 4 variables: `amount`, `transfer_type`, `time`, `initiator_role`
- 4 decisions: APPROVE, REQUIRE_APPROVAL, COMPLIANCE_REVIEW, HOLD
- 80 test scenarios
- 7 steps max
- **Multiple hidden parameters** — thresholds and definitions require clarification

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- [UV](https://docs.astral.sh/uv/) for package management

### Install

```bash
cd OpenenvHack
uv sync
```

### Run the Environment Server

```bash
uv run uvicorn policy_to_logic_env.server.app:app --host 0.0.0.0 --port 7860
```

### Run Inference Script

In a separate terminal (with environment variables set):

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

uv run python inference.py
```

### Docker

```bash
cd policy_to_logic_env
docker build -t policy-to-logic-env .
docker run -p 7860:7860 policy-to-logic-env
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start a new episode (optional: `{"task_name": "..."}`) |
| POST | `/step` | Take an action `{"action_type": "...", "content": "..."}` |
| GET | `/state` | Get current episode state |
| GET | `/health` | Health check |
| GET | `/tasks` | List available tasks |

## 📈 Baseline Scores

| Task | Baseline Score | Difficulty |
|------|---------------|------------|
| `data_access` | ~0.70–0.85 | Easy |
| `resource_access` | ~0.50–0.70 | Medium |
| `transaction_approval` | ~0.30–0.50 | Hard |

## 🧪 Why This Environment Matters

- **Real-world utility**: Policy-to-logic translation is a genuine enterprise need
- **Verifiable outcomes**: Rules are tested against deterministic scenarios
- **Rich learning signal**: Dense, multi-component reward (not sparse binary)
- **Novel domain**: Underexplored in RL/LLM training
- **Structured reasoning**: Tests when to ask, when to act, and how to refine

## 📄 License

MIT
