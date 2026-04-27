---
title: Policy2Logic
emoji: 🏆
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Meta pytorch hugging face hackathon
---

# Policy-to-Logic RL Environment

> A verifiable reinforcement learning environment where an agent learns to convert natural language organizational policies into executable, testable rules — through interaction, clarification, and reward-guided optimization.

---

## Problem Statement

### The Scenario

In every company, policies, guardrails, ethical guidelines, and compliance rules are some of the most important things that govern how people work. They define the boundaries for:

- Developers
- Data scientists
- Decision makers
- General users

...and everyone else in the organization.

These are always written in natural language. And natural language is inherently prone to ambiguity, vagueness, loopholes, and inconsistencies. On top of that, these policies change over time — new regulations, new business contexts, new edge cases.

This makes it extremely difficult to ensure that every rule is correctly captured, every edge case is handled, and nothing is lost in translation when policies get updated. One ambiguous clause can create a legal vulnerability. One missed condition can open a compliance gap.

**This is the exact gap we are trying to solve.**

---

### Why This Is Worth Solving

**Is this a real problem in companies?** Yes, very much so. Policy drift, ambiguous compliance rules, and inconsistent enforcement are well-documented problems across industries — from finance and healthcare to SaaS and enterprise software.

**Why does AI fit here?** Because it's too much of a hard, repetitive, and error-prone task for humans to do perfectly at scale. When a policy has 20 conditions across 5 roles and 3 time windows, humans miss things. AI doesn't get tired, can ask the right questions, and can explain its reasoning. That's exactly where it fits.

**How much would this help?** Improper policy formation leads to legal issues, financial loss, and reputational damage. If this problem is solved well, it can save companies enormous amounts of time and money — and make compliance something that's actually verifiable, not just assumed.

**Can this be integrated into existing company systems?** Yes, easily. Any system that uses access control, compliance rules, or decision logic is a candidate.

**How does RL and LLM fine-tuning help?** Creating a specialized model that can think through policy ambiguity, ask the right clarifying questions, and produce structured executable logic — in the right context — is an enormously valuable capability. Wrapped with a proper application environment, this is a startup-grade product.

---

### The Ideal End State

A company feeds their updated policy document into an application powered by our fine-tuned model. The model asks targeted questions to eliminate ambiguity. Once clarified, it produces a clean, structured, verifiable ruleset that accounts for all previous and new rules. A human reviews and iterates if needed. The final rules are integrated everywhere compliance is required.

That's the vision. This project is the training and verification framework that makes that vision buildable.

---

## How the Idea Is Structured

The core insight is this: converting a policy into executable logic is not a one-shot task. It's a **sequential decision-making process**. You read the policy, you realize something is unclear, you ask a question, you get an answer, you propose rules, some fail, you refine. That loop — ask, propose, test, refine — is what a good policy analyst actually does. We formalize that loop as a reinforcement learning problem.

Here's how each part is structured:

**The Agent** reads the policy and decides what to do next. It has three actions available at every step:
- `ask_clarification` — ask a targeted question about something ambiguous in the policy
- `propose_rules` — submit a set of executable rules for evaluation
- `refine_rules` — update the existing rules based on what failed

**The Environment** is the testbed. It holds the policy, generates test scenarios, evaluates the agent's rules against a programmatic ground truth, and returns a reward. It's essentially a unit testing framework for policy logic — objective, deterministic, and automatic.

**The DSL (Domain Specific Language)** is the structured format the agent writes rules in. Intentionally minimal — JSON conditions with numeric comparisons, equality checks, and AND logic. Priority-ordered, first-match-wins. Simple enough to be tractable, expressive enough to cover real access control policies. We can extend this as the system matures.

**The Reward System** has multiple components so the agent can't exploit a single signal:
- *Accuracy reward* — how many scenarios does the ruleset get right vs ground truth
- *Improvement reward* — did accuracy go up this step compared to last
- *Efficiency penalty* — using too many steps when fewer would do costs reward
- *Clarification scoring* — asking a useful question gives a small positive signal; asking something unnecessary gives a penalty

**The Clarification Oracle** simulates the policy author. When the agent asks a question, the oracle responds with a structured answer. It's deliberately tiered — early questions get partial information, pushing the agent to reason with incomplete context rather than just dumping everything upfront.

**The Scenario Generator** creates test cases for every policy using four strategies: random sampling for general coverage, boundary cases to catch edge conditions, pairwise combinations to test variable interactions, and adversarial cases to stress-test the rules. 50–100 scenarios per policy, with a train/test split.

**The Training Loop** uses reward-guided trajectory accumulation. The agent runs episodes, and high-reward interaction sequences are stored in a trajectory bank. Subsequent episodes inject the best past trajectories as few-shot examples — so the agent's context gets progressively better without any weight updates. It's in-context policy optimization driven entirely by environment reward signal.

All of this is intentionally simple in v1. The DSL can be extended with OR logic, nested conditions, and temporal rules. The oracle can be made semantic. The training loop can be upgraded to GRPO-based fine-tuning with TRL. The foundation is designed to be extended — we just built it right first.

---

## Deliverables

| Deliverable | Link |
|---|---|
| **Live Environment** | [godreign-policy2logic.hf.space](https://godreign-policy2logic.hf.space) |
| **Training Notebook** | [Open in Colab](https://colab.research.google.com/github/GodreignElgin/policy2logic/blob/main/training/colab_training.ipynb) |
| **Experiment Tracking** | [Weights & Biases Dashboard](https://wandb.ai/godreignelgin-sri-krishna-college-of-technology-org/policy-to-logic-rl) |
| **Video Walkthrough** | [YouTube](https://youtu.be/bQliR2nl2S8) |
| **Write-up** | [Medium Blog](https://medium.com/@godreignelgin/policy-to-logic-rl-environment-61da6176ff9c) |

---

## Results

The agent was trained across three tasks of increasing difficulty using the reward-guided trajectory loop. Key findings:

- `data_access` and `transaction_approval` hit >98% accuracy — these are within the zero-shot capability of Qwen2.5-72B, which validates that the environment and DSL are correctly designed
- `resource_access` shows genuine learning behavior — multi-role, time-conditional policies create real ambiguity that requires clarification and iteration to resolve
- The trajectory bank demonstrates cold-start bootstrapping — episode 1 context is zero-shot, and accumulated trajectories progressively anchor subsequent episodes

### Reward Curve
![Reward Curve](training/plots/reward_curve.png)

*Average reward per step — removes step-count accumulation artifacts.*

### Accuracy Curve
![Accuracy Curve](training/plots/accuracy_curve.png)

### Per-Task Summary
![Improvement Chart](training/plots/improvement_chart.png)

---

## Environment API

The environment is live and publicly accessible. Any agent can interact with it over HTTP.

**Base URL:** `https://godreign-policy2logic.hf.space`

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/tasks` | GET | List available tasks |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit an action |
| `/state` | GET | Get current episode state |

### Quick Start

```python
import requests, json

base = "https://godreign-policy2logic.hf.space"

# Start an episode
obs = requests.post(f"{base}/reset", json={"task_name": "data_access"}).json()
print(obs["observation"]["policy_text"])

# Propose rules
result = requests.post(f"{base}/step", json={
    "action_type": "propose_rules",
    "content": json.dumps({
        "rules": [
            {"if": [{"field": "time", "op": ">=", "value": 9},
                    {"field": "time", "op": "<", "value": 18}], "then": "ALLOW"}
        ],
        "default": "DENY"
    })
}).json()

print(f"Accuracy: {result['observation']['current_accuracy']}")
print(f"Reward: {result['reward']}")
```

---

## Architecture

```
Policy → Agent → (Ask / Propose / Refine)
       → Environment → (Scenarios + Evaluation)
       → Reward → Trajectory Bank → Improved Agent
```

### Tasks (Increasing Difficulty)

| Task | Difficulty | Key Challenge |
|---|---|---|
| `data_access` | Easy | Time + data type conditions |
| `resource_access` | Medium | Multi-role, ambiguous access patterns |
| `transaction_approval` | Hard | 4-outcome decision with priority ordering |
| `transaction_approval_hard` | Very Hard | Vague thresholds requiring clarification |

### DSL Format

Rules are written as ordered JSON conditions. First match wins. Default applies if nothing matches.

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

Supported operators: `>`, `<`, `>=`, `<=`, `==`, `!=`

---

## Repository Structure

```
├── policy_to_logic_env/
│   ├── server/
│   │   ├── app.py                 # FastAPI endpoints
│   │   ├── environment.py         # RL environment (reset/step/state)
│   │   ├── policies.py            # Task definitions
│   │   ├── ground_truth.py        # Ground truth + clarification oracle
│   │   ├── scenario_generator.py  # 4-strategy scenario generation
│   │   ├── dsl_engine.py          # JSON DSL parser and executor
│   │   ├── rewards.py             # Multi-component reward system
│   │   └── graders.py             # Rule evaluation
│   ├── models.py                  # Pydantic data models
│   ├── client.py                  # HTTP client library
│   └── openenv.yaml               # OpenEnv specification
├── training/
│   ├── trajectory_optimizer.py    # Training loop
│   ├── colab_training.ipynb       # Colab notebook
│   └── plots/                     # Committed training evidence
├── main.py
├── Dockerfile
└── README.md
```

---

## OpenEnv Compliance

- Gym-style `reset()` / `step()` / `state()` interface
- Valid `openenv.yaml` at `policy_to_logic_env/openenv.yaml`
- Pydantic v2 models for all inputs and outputs
- HTTP API for remote agent interaction

---

## Honest Assessment

This is a training and evaluation framework, not a finished product. A few things worth being clear about:

The training loop uses trajectory accumulation, not gradient-based optimization. This is intentional — it's a legitimate form of in-context policy improvement that works within hackathon constraints. The natural upgrade path is GRPO-based fine-tuning with TRL, which the environment is already structured to support.

The clarification oracle is keyword-based, not semantic. It works for the current task set and is the right thing to build first. A semantic oracle using embedding-based matching is the obvious next step.

Qwen2.5-72B solves the simpler tasks zero-shot. This is actually a validation result — it confirms the environment and DSL are correctly designed. The interesting learning behavior lives in the harder, more ambiguous tasks.

The server handles one session at a time. Parallel training runs require either multiple instances or a session management layer — straightforward to add, not in scope for v1.
