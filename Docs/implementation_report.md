# Policy-to-Logic RL Environment — Complete Implementation Report

> **Purpose**: Exhaustive description of everything implemented, with exact logic, edge cases, and formulas. Intended for AI-assisted gap analysis against the original plan.

---

## 1. Project Architecture

```
OpenenvHack/
├── main.py                          # Entry point: uvicorn on port 7860
├── Dockerfile                       # Docker SDK deployment for HF Spaces
├── inference.py                     # LLM agent loop (Qwen2.5-72B via OpenAI API)
├── pyproject.toml                   # UV project: pydantic, fastapi, uvicorn, openai, huggingface-hub
├── test_hf_spaces.py                # Remote endpoint tests against HF Spaces
├── test_all.py                      # Local test runner (starts server, runs tests, stops)
├── test_local.py / test_endpoints.py # Additional test scripts
├── policy_to_logic_env/
│   ├── __init__.py                  # Package exports: models + client
│   ├── models.py                    # Pydantic models: Action, Observation, State, StepResult
│   ├── client.py                    # HTTP client wrapper for the environment
│   ├── openenv.yaml                 # OpenEnv specification file
│   └── server/
│       ├── app.py                   # FastAPI app with 6 endpoints
│       ├── environment.py           # Core environment: reset(), step(), state()
│       ├── policies.py              # 3 task definitions with clarification maps
│       ├── ground_truth.py          # Programmatic ground truth + clarification oracle
│       ├── scenario_generator.py    # 4-strategy scenario generation (seeded)
│       ├── graders.py               # Rule grading against scenarios
│       ├── dsl_engine.py            # JSON DSL parser, validator, executor
│       ├── rewards.py               # 4-component reward system
│       └── requirements.txt         # Server deps: openenv-core, pydantic, fastapi, uvicorn, requests
```

**Deployment**: Docker on HF Spaces at `https://godreign-policy2logic.hf.space`, port 7860.

---

## 2. HTTP API (app.py)

Single FastAPI app with CORS `allow_origins=["*"]`. One global `PolicyToLogicEnvironment()` instance (single-session).

| Endpoint | Method | Request Body | Response | Purpose |
|---|---|---|---|---|
| `/` | GET | — | `{name, version, status, endpoints, docs, redoc}` | Root probe / API info |
| `/health` | GET | — | `{status: "ok", environment: "policy_to_logic"}` | Health check |
| `/tasks` | GET | — | `{tasks: {name: {difficulty, max_steps, scenario_count, valid_decisions, variables}}}` | List all 3 tasks |
| `/reset` | POST | `{task_name: str \| null}` | `StepResult` (observation + reward=0 + done=false) | Start new episode |
| `/step` | POST | `{action_type: str, content: str}` | `StepResult` (observation + reward + done) | Take an action |
| `/state` | GET | — | `PolicyToLogicState` (full episode metadata) | Get current state |

If `task_name` is null or invalid in `/reset`, defaults to `"data_access"`.

---

## 3. Data Models (models.py)

### Action
```
action_type: Literal["ask_clarification", "propose_rules", "refine_rules"]
content: str  # JSON string payload
```

### Observation (returned in every StepResult)
```
policy_text: str          # The natural language policy (always present)
task_name: str
step_number: int          # 0 on reset, 1+ on steps
max_steps: int
clarification_response: str | None    # Oracle answer if ask_clarification
test_results: dict | None             # {passed, failed, total, score, sample_failures}
current_accuracy: float               # 0.0-1.0
available_actions: list[str]          # What the agent can do next
feedback: str | None                  # Human-readable feedback
dsl_format: str                       # DSL syntax instructions (always present)
```

### State
```
episode_id: str
step_count: int
task_name: str
current_rules: list | None
accuracy_history: list[float]
questions_asked: int
questions_log: list[str]
done: bool
total_reward: float
```

### StepResult
```
observation: Observation
reward: float    # 0.0-1.0 per step
done: bool
info: dict       # Contains reward_breakdown, episode_score, errors, etc.
```

---

## 4. Episode Lifecycle (environment.py)

### reset(task_name)
1. Load task config from registry (defaults to `"data_access"`)
2. Generate scenarios via `generate_scenarios(task_name)` with `seed=42`
3. Initialize state: `step_count=0`, `accuracy=0`, `done=false`
4. Return observation with policy text, DSL format, available decisions/variables

### step(action)
1. Guard: if `state is None` or `done == True` → error result
2. Increment `step_count`
3. Dispatch by `action_type`:
   - `"ask_clarification"` → `_handle_clarification()`
   - `"propose_rules"` → `_handle_propose()`
   - `"refine_rules"` → `_handle_refine()`

### Termination Conditions
Episode ends (`done=True`) when **either**:
- `accuracy >= 0.9` (success)
- `step_count >= max_steps` (budget exhausted)

### Clarification Handling
1. Parse content as JSON to extract `question`, or use raw content as the question
2. Call `answer_clarification(task_name, question)` → deterministic oracle answer
3. Usefulness check: `is_useful = "I can provide information" not in answer`
4. Compute reward (accuracy stays unchanged, clarification component applies)
5. `refine_rules` is only available after at least one `propose_rules`

### Rule Proposal/Refinement Handling
1. Parse JSON content via `parse_rules()` → validates DSL structure
2. If invalid: penalty reward, feedback with parse errors
3. If valid: grade rules against stored scenarios → accuracy
4. Compute reward using accuracy delta
5. Feedback includes: accuracy, improvement direction, passed/total, sample failure
6. If `accuracy >= 0.9`: feedback says "Target accuracy reached! Episode complete."
7. On episode end: compute `episode_score` and include in info

---

## 5. The Three Tasks (policies.py)

### Task 1: `data_access` (Easy)

| Property | Value |
|---|---|
| Difficulty | easy |
| Max Steps | 5 |
| Scenario Count | 30 |
| Variables | `time` (0-23), `data_type` (sensitive, public, internal) |
| Valid Decisions | ALLOW, DENY |
| Hidden Params | `work_start=9`, `work_end=18` |

**Policy Text** (what the agent sees):
> Employees must not access sensitive data after working hours. Working hours are from 9 AM to 6 PM (9:00 to 18:00). Public data can be accessed at any time. Internal data follows the same rules as sensitive data.

---

### Task 2: `resource_access` (Medium)

| Property | Value |
|---|---|
| Difficulty | medium |
| Max Steps | 7 |
| Scenario Count | 50 |
| Variables | `role` (junior, senior, contractor), `time` (0-23), `document_type` (public, internal, confidential) |
| Valid Decisions | ALLOW, DENY |
| Hidden Params | `business_start=8`, `business_end=17` |

**Policy Text**:
> Junior employees cannot access confidential documents outside business hours. Senior employees have unrestricted access to all document types. Contractors can only access public documents, regardless of time. During business hours, junior employees may access public and internal documents.

**Intentional Ambiguity**: The policy says juniors "cannot access confidential documents outside business hours" — implying they CAN during business hours. But the ground truth DENIES confidential for juniors at ALL times. This is a deliberate trap the agent must discover through testing.

---

### Task 3: `transaction_approval` (Hard)

| Property | Value |
|---|---|
| Difficulty | hard |
| Max Steps | 7 |
| Scenario Count | 80 |
| Variables | `amount` (100..50000, 12 values), `transfer_type` (domestic, international), `time` (0-23), `initiator_role` (employee, manager, system) |
| Valid Decisions | APPROVE, REQUIRE_APPROVAL, COMPLIANCE_REVIEW, HOLD |
| Hidden Params | `standard_limit=5000`, `high_value_threshold=10000`, `business_start=9`, `business_end=17` |

**Policy Text**:
> Transactions exceeding the standard limit require manager approval. International transfers always need compliance review regardless of amount. High-value domestic transactions during non-business hours are automatically held for review. Routine domestic transactions within limits are auto-approved. Manager-initiated transactions are exempt from the standard limit.

---

## 6. Ground Truth Logic (ground_truth.py)

### Task 1: `_ground_truth_data_access`

```python
if data_type == "public":           → ALLOW
if 9 <= time < 18:                  → ALLOW   # sensitive or internal
else:                               → DENY
```

**Complete Decision Table**:

| data_type | time | Decision | Why |
|---|---|---|---|
| public | any (0-23) | ALLOW | Public is always accessible |
| sensitive | 0-8 | DENY | Before working hours |
| sensitive | 9-17 | ALLOW | During working hours |
| sensitive | 18-23 | DENY | After working hours (18 is OUTSIDE) |
| internal | 0-8 | DENY | Same rules as sensitive |
| internal | 9-17 | ALLOW | Same rules as sensitive |
| internal | 18-23 | DENY | Same rules as sensitive |

> [!IMPORTANT]
> **Critical boundary**: `time=18` → DENY. The interval is half-open: `[9, 18)`. Hour 18 is the first after-hours hour. Hour 17 is the last working hour.

---

### Task 2: `_ground_truth_resource_access`

```python
if role == "senior":                                    → ALLOW
if role == "contractor":
    if doc_type == "public":                            → ALLOW
    else:                                               → DENY
# Junior employee:
is_business_hours = (8 <= time < 17)
if doc_type == "public":                                → ALLOW
if is_business_hours and doc_type == "internal":        → ALLOW
else:                                                   → DENY
```

**Complete Decision Table for Junior Employees**:

| document_type | time | Decision | Why |
|---|---|---|---|
| public | any (0-23) | ALLOW | Public always allowed for all roles |
| internal | 0-7 | DENY | Before business hours |
| internal | 8-16 | ALLOW | During business hours |
| internal | 17-23 | DENY | After business hours (17 is OUTSIDE) |
| confidential | any (0-23) | **DENY** | **Always denied for juniors** |

**Senior**: ALLOW for everything, always.
**Contractor**: ALLOW only for `public`, DENY for `internal` and `confidential`, at all times.

> [!IMPORTANT]
> **Critical boundary**: `time=17` → outside business hours. Interval: `[8, 17)`. Hour 16 is the last business hour.
>
> **Critical trap**: `confidential` is ALWAYS denied for juniors, even during business hours. The policy text misleadingly implies otherwise.

---

### Task 3: `_ground_truth_transaction_approval`

Rules evaluated in strict priority order (first match wins):

```python
# Rule 1: International → COMPLIANCE_REVIEW (always, regardless of everything)
if transfer_type == "international":                    → COMPLIANCE_REVIEW

# Rule 2: High-value domestic outside business hours → HOLD
if amount >= 10000 and not (9 <= time < 17):            → HOLD

# Rule 3: Above standard limit, not manager → REQUIRE_APPROVAL
if amount > 5000 and initiator_role != "manager":       → REQUIRE_APPROVAL

# Rule 4: Everything else → APPROVE
else:                                                   → APPROVE
```

**Critical Edge Cases**:

| amount | transfer_type | time | initiator_role | Decision | Why |
|---|---|---|---|---|---|
| 5000 | domestic | 12 | employee | **APPROVE** | At limit, not above (> 5000 fails) |
| 5001 | domestic | 12 | employee | REQUIRE_APPROVAL | Above limit, not manager |
| 5001 | domestic | 12 | manager | **APPROVE** | Manager exempt from limit |
| 10000 | domestic | 20 | employee | **HOLD** | High-value + non-business hours |
| 10000 | domestic | 12 | employee | REQUIRE_APPROVAL | High-value but business hours (Rule 2 skipped, Rule 3 matches) |
| 10000 | domestic | 17 | employee | **HOLD** | 17 is non-business hours |
| 10000 | domestic | 20 | **manager** | **HOLD** | Managers NOT exempt from HOLD rule |
| 100 | international | 12 | employee | COMPLIANCE_REVIEW | International always |
| 50000 | international | 3 | manager | COMPLIANCE_REVIEW | International trumps everything |
| 9999 | domestic | 20 | employee | REQUIRE_APPROVAL | NOT high-value (< 10000), but above limit |
| 100 | domestic | 3 | employee | APPROVE | Within limit |
| 100 | domestic | 3 | system | APPROVE | System = employee |

> [!IMPORTANT]
> **Standard limit comparison**: `amount > 5000` (strict greater than). $5,000 exactly = APPROVE.
>
> **High-value comparison**: `amount >= 10000` (greater than or equal). $10,000 exactly = high-value.
>
> **Manager exemption scope**: Only exempts from Rule 3 (standard limit). Managers are still subject to Rule 1 (international) and Rule 2 (high-value HOLD).
>
> **Business hours**: `[9, 17)`. Hour 17 is non-business.

---

## 7. Clarification Oracle (ground_truth.py)

### Matching Algorithm

```
Input:  question string (free text from agent)
Output: best matching answer from task's clarification_map

Algorithm:
1. Lowercase the question
2. For each keyword in clarification_map:
   a. Split keyword into parts by spaces
   b. Check if ALL parts appear as substrings in the question
   c. Score = (number_of_parts, total_keyword_length)
   d. Highest score wins
3. If no match: return generic fallback (contains "I can provide information")
```

**Key property**: "junior confidential" matches when BOTH "junior" AND "confidential" appear anywhere in the question (order-independent). This 2-part keyword beats any 1-part keyword like "junior" alone.

### Usefulness Detection

In `environment.py`, line 203:
```python
is_useful = "I can provide information" not in answer
```
Any answer that matches a keyword entry is "useful". Only the generic fallback is "not useful".

### 3-Tier Progressive Revelation Design

Each task's `clarification_map` has three levels:

| Tier | Keyword Type | Answer Quality | Training Purpose |
|---|---|---|---|
| Level 1 | Single short words | Partial truths, technically correct but incomplete/misleading | Agent builds initial (wrong) rules |
| Level 2 | Common phrases | More detail, boundary still ambiguous | Agent narrows down the problem |
| Level 3 | Compound/multi-word | Precise, ground-truth-aligned, corrects Level 1 | Agent fixes rules after failures |

**Example — resource_access contradiction**:
- Agent asks "What can junior employees access?" → matches `"junior"` (Level 1) → *"...but not confidential documents outside business hours"* (implies CAN during hours)
- Agent proposes rules allowing junior+confidential during hours → **fails**
- Agent asks "Can junior employees access confidential documents?" → matches `"junior confidential"` (Level 3, 2 parts > 1 part) → *"CANNOT access confidential at ANY time"*
- Agent refines rules → **passes**

### Clarification Map Entry Counts

| Task | Level 1 | Level 2 | Level 3 | Total |
|---|---|---|---|---|
| data_access | 5 | 3 | 6 | 14 |
| resource_access | 7 | 3 | 8 | 18 |
| transaction_approval | 9 | 7 | 10 | 26 |

---

## 8. DSL Engine (dsl_engine.py)

### DSL Format

```json
{
    "rules": [
        {
            "if": [
                {"field": "<name>", "op": "<operator>", "value": <value>}
            ],
            "then": "<DECISION>"
        }
    ],
    "default": "<DEFAULT_DECISION>"
}
```

### Supported Operators
`>`, `<`, `>=`, `<=`, `==`, `!=`

### Validation (`validate_rules`)
Checks:
- Root is a dict
- Has `"rules"` key (must be list)
- Has `"default"` key (must be string)
- Each rule has `"if"` (list) and `"then"` (string)
- Each condition has `"field"` (string), `"op"` (valid operator), `"value"`

Returns `(is_valid: bool, errors: list[str])`.

### Execution (`execute_rules`)
1. Iterate rules top-to-bottom
2. For each rule, evaluate ALL conditions (AND logic)
3. First rule where all conditions match → return its `"then"` decision
4. If no rules match → return `"default"`

### Type Coercion
If scenario has `time=9` (int) and rule has `"value": "9"` (str), coerces the string to int. Works both directions. If coercion fails, condition evaluates to `False`.

### Parsing (`parse_rules`)
1. `json.loads()` the content string
2. `validate_rules()` on the parsed dict
3. Returns `(rules_data, [])` on success or `(None, errors)` on failure

---

## 9. Scenario Generator (scenario_generator.py)

### Strategy Allocation

| Strategy | Share | Purpose |
|---|---|---|
| Boundary | ~20% | Edge values near hidden param thresholds |
| Pairwise | ~30% | Systematic variable combinations |
| Adversarial | ~20% | Hand-crafted traps for common mistakes |
| Random | remainder | Uniform sampling from variable space |

All seeded with `seed=42` for reproducibility. Scenarios are deduplicated by field values.

### Boundary Strategy
Extracts numeric hidden params, generates scenarios at `param ± 1` and at variable min/max.

### Pairwise Strategy
For each pair of variables, samples up to 4 representative values (min, max, middle, random), generates cross-product combinations.

### Adversarial Strategy
**Hand-crafted per task** — these are the exact scenarios:

#### data_access adversarial:
| time | data_type | Expected | Tests |
|---|---|---|---|
| 9 | sensitive | ALLOW | Start boundary |
| 18 | sensitive | DENY | End boundary (exclusive) |
| 8 | sensitive | DENY | Just before start |
| 17 | sensitive | ALLOW | Just before end |
| 0 | public | ALLOW | Public at midnight |
| 23 | internal | DENY | Internal late night |
| 12 | internal | ALLOW | Internal during hours |

#### resource_access adversarial:
| role | time | document_type | Expected | Tests |
|---|---|---|---|---|
| junior | 8 | confidential | DENY | Confidential at business start |
| junior | 7 | internal | DENY | Internal before hours |
| junior | 17 | internal | DENY | Internal at boundary (17=outside) |
| junior | 16 | internal | ALLOW | Internal just before boundary |
| contractor | 12 | internal | DENY | Contractor restricted |
| senior | 2 | confidential | ALLOW | Senior unrestricted |
| junior | 12 | public | ALLOW | Junior public during hours |
| contractor | 12 | public | ALLOW | Contractor public |

#### transaction_approval adversarial:
| amount | transfer | time | role | Expected | Tests |
|---|---|---|---|---|---|
| 5000 | domestic | 12 | employee | APPROVE | At limit (not above) |
| 5001 | domestic | 12 | employee | REQ_APPROVAL | Just above limit |
| 5001 | domestic | 12 | manager | APPROVE | Manager exempt |
| 10000 | domestic | 20 | employee | HOLD | High-value non-business |
| 10000 | domestic | 12 | employee | REQ_APPROVAL | High-value business hours |
| 100 | international | 12 | employee | COMPLIANCE | International small |
| 50000 | international | 3 | manager | COMPLIANCE | International manager |
| 9999 | domestic | 20 | employee | REQ_APPROVAL | Below high-value threshold |
| 10000 | domestic | 9 | employee | REQ_APPROVAL | High-value at business start |
| 10000 | domestic | 17 | employee | HOLD | 17=non-business |

---

## 10. Grading (graders.py)

### `grade_task(task_name, rules_data, scenarios)`
1. Validate rules → if invalid, return `score=0.0`
2. For each scenario: execute agent's rules, compare to `expected_decision`
3. Comparison: `actual.upper() == expected.upper()` (case-insensitive)
4. `score = passed / total`
5. Returns up to 5 `sample_failures` with scenario details, expected, got

### `quick_grade(task_name, rules_data, scenarios)`
Same logic, returns only the float score. Used during step processing.

---

## 11. Reward System (rewards.py)

### Per-Step Reward: `compute_reward()`

4 components, clamped to `[0.0, 1.0]`:

| Component | Weight | Formula |
|---|---|---|
| **Accuracy** | 0.50 | `current_accuracy × 0.50` |
| **Improvement** | 0.20 | `min(delta × 2.0, 1.0) × 0.20` if delta > 0; `max(delta × 1.5, -0.5) × 0.20` if delta < 0; `0` if unchanged |
| **Efficiency** | 0.15 | `max(-0.02 × step_number [+ 0.05 × steps_saved if acc≥0.9], -0.15) × 0.15` |
| **Clarification** | 0.15 | See below |

**Clarification component details**:
- `ask_clarification` + useful + questions ≤ 3: `+0.3 × 0.15 = +0.045`
- `ask_clarification` + useful + questions > 3: `+0.1 × 0.15 = +0.015` (diminishing)
- `ask_clarification` + not useful: `-0.05 × 0.15 = -0.0075`
- `propose_rules/refine_rules` + invalid DSL: `-0.1 × 0.15 = -0.015`
- `propose_rules/refine_rules` + valid DSL: `0`

### Episode Score: `compute_episode_score()`

Used for final grading, `[0.0, 1.0]`:

```
score = final_accuracy × 0.80
     + max(0, 1 - steps/max_steps) × 0.10
     + question_bonus × 0.10

question_bonus = 1.0 if questions ≤ 2
               = 0.5 if questions ≤ 4
               = 0.0 if questions > 4
```

---

## 12. Inference Agent (inference.py)

### Configuration
- Model: `Qwen/Qwen2.5-72B-Instruct` (via `HF_TOKEN`)
- API: `https://router.huggingface.co/v1` (OpenAI-compatible)
- Temperature: 0.3, Max tokens: 1024
- Env URL: `http://localhost:7860` (configurable via `ENV_BASE_URL`)

### Agent Loop
```
for each task in [data_access, resource_access, transaction_approval]:
    result = env.reset(task)
    for step in 1..max_steps:
        if result.done: break
        action_type, content = get_agent_action(llm, observation, step, history)
        result = env.step(action)
        history.append(summary)
```

### Prompt Design
- **System prompt**: Describes available actions, DSL format, strategy guidelines
- **User prompt**: Built per-step with policy text, feedback, clarification answers, test results, sample failures, DSL format, action history (last 3)
- LLM response parsed as JSON: `{"action_type": "...", "content": "..."}`
- Handles markdown code blocks (`\`\`\`json ... \`\`\``)
- Fallback: if unparseable, tries extracting `"rules"`, otherwise submits empty rules

### Output Format
```
[START] task=<name> env=policy_to_logic model=<model>
[STEP]  step=<n> action=<summary> reward=<float> done=<bool> error=<msg|null>
[END]   success=<bool> steps=<n> score=<float> rewards=<r1,r2,...>
```

---

## 13. Client Library (client.py)

HTTP client using `requests.Session()`:
- `reset(task_name)` → POST `/reset` → `PolicyToLogicStepResult`
- `step(action)` → POST `/step` → `PolicyToLogicStepResult`
- `state()` → GET `/state` → `PolicyToLogicState`
- `health()` → GET `/health` → dict
- `list_tasks()` → GET `/tasks` → dict
- Context manager support (`with PolicyToLogicEnv() as env:`)

---

## 14. Deployment

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY policy_to_logic_env/server/requirements.txt → pip install
COPY policy_to_logic_env/, main.py, inference.py
EXPOSE 7860
HEALTHCHECK: curl -f http://localhost:7860/health
CMD: python -m uvicorn policy_to_logic_env.server.app:app --host 0.0.0.0 --port 7860
```

### HF Spaces Config (README.md)
```yaml
sdk: docker
app_port: 7860
```

Live at: `https://godreign-policy2logic.hf.space`

---

## 15. Known Design Decisions & Limitations

1. **Single-session**: One global environment instance. Concurrent clients will interfere. Suitable for sequential benchmarking, not parallel RL training.

2. **Deterministic scenarios**: `seed=42` always produces the same scenarios. Agent is graded on the same set every episode. Prevents overfitting variance but could lead to memorization.

3. **Stateful server**: The environment holds state in memory. Server restart loses episode state. No persistence layer.

4. **Clarification is keyword-based**: The oracle is not an LLM — it's a deterministic keyword matcher. Agent questions that don't contain any keyword get the generic fallback (penalized as "not useful").

5. **Progressive revelation by design**: Level 1 clarification answers are intentionally misleading partial truths. This is NOT a bug — it's the core RL training signal. Agents that trust Level 1 answers will fail and must learn to ask better (Level 3) questions.

6. **No `refine_rules` before `propose_rules`**: The environment returns a feedback message if the agent tries to refine before proposing. Not an error, just 0 reward + feedback.

7. **Case-insensitive grading**: `actual.upper() == expected.upper()`. Agent can output "allow" or "Allow" or "ALLOW".

8. **DSL type coercion**: Integer-string mismatches are auto-coerced. `"9"` and `9` compare equally.
