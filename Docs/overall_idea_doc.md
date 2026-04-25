# 🧠 Policy-to-Logic RL Environment (Final Handoff Document)

## OpenEnv Hackathon – Idea, Evolution, and Final Design (Implementation-Ready)

---

# 1. 🚀 What We Are Building (Final, Locked Statement)

> We are building a **verifiable RL environment** for policy-to-logic reasoning, where an agent interacts with a controlled system to **iteratively resolve policies into executable rules**, and we demonstrate **early-stage learning behavior**.

---

# 2. 🧠 What This Project Is (and Is NOT)

## ✅ This IS:

* an **environment + benchmark**
* a **training loop demonstration**
* a **verification framework**
* a **structured reasoning testbed**

---

## ❌ This is NOT:

* a production system
* a full policy understanding engine
* a complete ambiguity solver
* a fully trained RL model

---

# 3. 🔄 Ideation Journey (Critical Context)

## Phase 1: Initial Idea (Rejected)

> “Train an RL model to convert policies into rules”

### Problem:

* circular (ground truth already defines solution)
* better suited for supervised learning
* not sequential → weak RL justification

---

## Phase 2: Add Clarification Loop

> “Agent asks questions + generates rules”

### Improvement:

* introduces interaction

### Problem:

* clarification oracle shifts human effort, doesn’t remove it
* still weak RL justification
* unclear episode structure

---

## Phase 3: Final Pivot (LOCKED)

> “Environment where agent learns to act under uncertainty through interaction and feedback”

### Key Shift:

* NOT learning mapping
* learning **decision strategy over time**

---

## Final Framing:

> The agent learns **when to ask, when to act, and how to refine**, using environment feedback.

---

# 4. 🧩 Final System Architecture

```text
Policy → Agent → (Ask / Propose / Refine)
        → Environment → (Scenarios + Evaluation)
        → Reward → Trainer → Updated Agent
```

---

# 5. 🎯 Core Components (Implementation-Level)

---

## 5.1 Agent (LLM)

### Responsibilities:

* read policy
* decide action:

  * ask_clarification
  * propose_rules
  * refine_rules

---

### Output Types:

#### 1. Clarification

```json
{"type": "clarification", "question": "..."}
```

#### 2. Rule Proposal

```json
{"type": "propose", "rules": [...]}
```

#### 3. Rule Refinement

```json
{"type": "refine", "delta": [...]}
```

---

## 5.2 Environment

### Responsibilities:

* maintain state
* respond to actions
* generate scenarios
* evaluate rules
* compute reward

---

### Core Functions:

```python
reset() → policy + initial state
step(action) → observation, reward, done
```

---

## 5.3 DSL (LOCKED – DO NOT EXPAND)

### Supported:

* numeric comparisons (`>`, `<`)
* equality (`==`)
* logical AND

---

### Example:

```json
{
  "if": [
    {"field": "time", "op": ">", "value": 18},
    {"field": "data_type", "op": "==", "value": "sensitive"}
  ],
  "then": "DENY"
}
```

---

### NOT ALLOWED:

* OR
* nested logic
* exceptions
* priorities

---

# 6. 🔁 Episode Structure (FINAL)

---

## Step Flow:

```text
1. Policy given
2. Agent chooses action:
   - ask_clarification
   - propose_rules
   - refine_rules
3. Environment responds
4. Reward assigned
5. Repeat
```

---

## Termination Conditions:

* rules reach threshold (e.g., ≥90% accuracy)
  OR
* max steps reached (5–7 steps)

---

# 7. 🎯 Scenario Generator (Coverage Strategy)

---

## Target: 50–100 scenarios per policy

---

## Composition:

| Type               | Purpose              |
| ------------------ | -------------------- |
| Random (~30%)      | general coverage     |
| Boundary (~20%)    | edge conditions      |
| Pairwise (~30%)    | variable interaction |
| Adversarial (~20%) | robustness           |

---

## Important:

* NOT exhaustive coverage
* structured sampling

---

# 8. 🧠 Ground Truth Engine

---

## Role:

* define correct outcomes programmatically

---

## Example:

```python
def ground_truth(s):
    if s.time > 18 and s.data_type == "sensitive":
        return "DENY"
    return "ALLOW"
```

---

## Clarification Oracle

* deterministic mapping
* used only for:

  * training signal
  * evaluation consistency

---

## Important Positioning:

> Oracle defines evaluation, not solution.

---

# 9. 🎯 Reward System (Implementation-Ready)

---

## IMPORTANT NOTE (from critique):

Reward weights are **hyperparameters**, not theoretically fixed.

---

## Strategy:

We will:

* start with heuristic values
* optionally run small sweeps

---

## Components:

### 1. Final Reward (Primary Signal)

* accuracy over scenarios (0–1)

---

### 2. Step-Level Signals (Dense Reward)

#### Clarification:

* useful → +α
* unnecessary → -α

---

#### Refinement:

* improvement → +β
* degradation → -β

---

### 3. Efficiency Penalty:

* more steps → penalty

---

## Key Clarification Definition:

A clarification is **useful if**:

> It resolves a variable that appears in failing scenarios and leads to measurable improvement in subsequent rule accuracy.

---

👉 This is **retrospective attribution**, explicitly acknowledged.

---

# 10. 📊 Baseline (LOCK THIS)

We use:

## ✅ Zero-shot LLM baseline

* same prompt
* no RL
* single-step rule generation

---

## Optional Secondary Baseline:

* random rule generator (sanity check)

---

## Why:

* realistic comparison
* directly shows RL value

---

# 11. 🧠 RL Algorithm (LOCKED DECISION)

---

## Approach:

👉 **Use TRL (GRPO or PPO-style) with small-scale training**

---

## Practical Constraint:

We are NOT aiming for:

* full convergence

We ARE aiming for:

* early learning signal

---

## Alternative (if needed):

* treat LLM as black-box
* optimize via sampling + selection

---

## Important Positioning:

> “We demonstrate learning trends, not full optimization.”

---

# 12. ⚠️ Known Challenges + Solutions

---

## 1. Circular Ground Truth

✔ Accept and reframe as evaluation mechanism

---

## 2. RL Justification

✔ solved via sequential interaction

---

## 3. Clarification Oracle

✔ deterministic + acknowledged limitation

---

## 4. DSL Complexity

✔ aggressively constrained

---

## 5. Scenario Coverage

✔ structured sampling strategy

---

## 6. Reward Design

✔ treated as tunable hyperparameters

---

## 7. Training Feasibility

✔ limited training + early signal

---

# 13. 📊 What We Will Show (Demo Plan)

---

## 1. Baseline Output

* poor rule quality

---

## 2. Trained Agent Output

* improved accuracy
* better decisions

---

## 3. Metrics

* reward curve
* accuracy improvement
* fewer violations

---

## 4. Behavior

* better refinement
* smarter clarification

---

# 14. 🎯 Final Value Proposition

---

## What this contributes:

* a **verifiable RL environment**
* a **structured policy reasoning benchmark**
* a **framework for training and evaluation**

---

## Why it matters:

* policy-to-logic is underexplored
* evaluation is usually subjective
* this introduces measurable learning

---

# 15. 🚀 Implementation Priority (DO THIS NEXT)

---

## MUST BUILD FIRST:

1. DSL parser + executor
2. Scenario generator
3. Ground truth function
4. Environment (reset/step)

---

## THEN:

5. reward function
6. baseline evaluation
7. minimal RL loop

---

# 16. 🔚 Final Statement

> This project is not about solving policy reasoning completely.
> It is about **creating a structured, verifiable environment where such reasoning can be trained and evaluated.**

---
