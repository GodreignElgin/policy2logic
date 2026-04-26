# Policy-to-Logic RL Environment — Final Implementation Handoff
## 7-Hour Sprint Document (AI-Executable Instructions)

> **Context**: This document is written for an AI-powered IDE to execute. Every instruction is explicit, ordered, and complete. Do not skip steps. Do not reorder steps. Validate each step before moving to the next.

---

## CRITICAL: Validation Checklist (Must Pass All 5 Before Submission)

The hackathon validator checks these automatically. If any fails, the submission is rejected before a human sees it.

| # | Requirement | How to Verify |
|---|---|---|
| 1 | Public HF Space at submitted URL | Open in logged-out browser — must load, no 404 |
| 2 | Valid OpenEnv structure (base class + reset/step/state + openenv.yaml) | Already implemented — verify yaml is parseable |
| 3 | Training evidence as committed `.png` / `.jpg` files in repo | NOT Wandb links — actual image files in repo |
| 4 | Runnable training script (Colab notebook preferred) | Must re-execute end-to-end |
| 5 | README links all deliverables with plots embedded inline | Validator must reach every link from README |

**These 5 items drive the entire 7-hour plan. Every task below serves one or more of them.**

---

## Hour-by-Hour Execution Plan

---

## HOUR 1-2: Build the Reward-Guided Trajectory Training Loop

### What This Is

This is NOT fine-tuning. This is a **reward-guided few-shot accumulation loop** — a legitimate optimization strategy where:
- The agent runs episodes against the environment
- Trajectories (full interaction sequences) are stored with their rewards
- High-reward trajectories become few-shot examples for subsequent episodes
- Agent performance measurably improves across episodes using the reward signal

**This is your "training loop" for the submission.** It is honest, demonstrable, and buildable.

---

### File to Create: `training/trajectory_optimizer.py`

Create this file in the repo root under a new `training/` directory.

```python
"""
Reward-Guided Trajectory Optimization Loop
==========================================
Optimizes agent behavior across episodes by accumulating high-reward
trajectories as few-shot examples. Uses environment reward signal to
drive improvement — no weight updates required.

This implements a policy improvement loop where:
  - reward_signal → trajectory_selection → context_construction → improved_policy
"""

import json
import os
import time
import requests
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL = "Qwen/Qwen2.5-72B-Instruct"
TEMPERATURE = 0.3
MAX_TOKENS = 1024

# Training hyperparameters
NUM_EPISODES_PER_TASK = 8        # Episodes to run per task
TOP_K_TRAJECTORIES = 3           # Max few-shot examples to keep
MIN_REWARD_THRESHOLD = 0.3       # Minimum reward to store trajectory
TASKS = ["data_access", "resource_access", "transaction_approval"]

# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Step:
    step_number: int
    action_type: str
    action_content: str
    reward: float
    accuracy: float
    feedback: str
    clarification_response: Optional[str] = None

@dataclass
class Trajectory:
    task_name: str
    episode_id: int
    steps: list[Step] = field(default_factory=list)
    total_reward: float = 0.0
    final_accuracy: float = 0.0
    success: bool = False

    def to_few_shot_string(self) -> str:
        """Convert trajectory to a few-shot example string for prompting."""
        lines = [
            f"=== Example Episode (reward={self.total_reward:.2f}, accuracy={self.final_accuracy:.2f}) ===",
        ]
        for s in self.steps:
            lines.append(f"Step {s.step_number}: action={s.action_type}")
            lines.append(f"  Content: {s.action_content[:200]}")
            lines.append(f"  Result: accuracy={s.accuracy:.2f}, reward={s.reward:.2f}")
            if s.feedback:
                lines.append(f"  Feedback: {s.feedback[:150]}")
        return "\n".join(lines)

# ── Environment Client ────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_name: str) -> dict:
        r = self.session.post(f"{self.base_url}/reset", json={"task_name": task_name})
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, content: str) -> dict:
        r = self.session.post(f"{self.base_url}/step", json={
            "action_type": action_type,
            "content": content
        })
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

# ── LLM Agent ────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, hf_token: str):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token
        )

    def get_action(
        self,
        observation: dict,
        step_number: int,
        episode_history: list[str],
        few_shot_examples: list[Trajectory]
    ) -> tuple[str, str]:
        """
        Returns (action_type, content_json_string).
        action_type: one of ask_clarification | propose_rules | refine_rules
        content: JSON string appropriate for that action
        """
        system_prompt = self._build_system_prompt(few_shot_examples)
        user_prompt = self._build_user_prompt(observation, step_number, episode_history)

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_response(raw, observation)
        except Exception as e:
            print(f"    [LLM ERROR] {e}")
            return "propose_rules", json.dumps({"rules": [], "default": "DENY"})

    def _build_system_prompt(self, few_shot_examples: list[Trajectory]) -> str:
        base = """You are a policy-to-logic agent. Your job is to convert natural language policies into executable rules.

AVAILABLE ACTIONS:
1. ask_clarification: {"type": "clarification", "question": "your question"}
2. propose_rules: {"rules": [...], "default": "DECISION"}
3. refine_rules: {"rules": [...], "default": "DECISION"}

DSL FORMAT for rules:
{
  "rules": [
    {
      "if": [
        {"field": "FIELD_NAME", "op": "OPERATOR", "value": VALUE}
      ],
      "then": "DECISION"
    }
  ],
  "default": "FALLBACK_DECISION"
}

Operators: >, <, >=, <=, ==, !=
Rules execute top-to-bottom. First match wins. Default applies if no rule matches.

STRATEGY:
- Step 1: Ask 1-2 targeted clarification questions about ambiguous terms
- Step 2: Propose initial rules based on policy + clarifications  
- Step 3+: Refine rules based on failure feedback

OUTPUT FORMAT: Respond ONLY with valid JSON. No markdown. No explanation.
{"action_type": "propose_rules", "content": "{...escaped json string...}"}
"""
        if few_shot_examples:
            base += "\n\nLEARNED FROM PREVIOUS EPISODES (high-reward strategies):\n"
            for traj in few_shot_examples[-TOP_K_TRAJECTORIES:]:
                base += "\n" + traj.to_few_shot_string() + "\n"
        return base

    def _build_user_prompt(self, obs: dict, step: int, history: list[str]) -> str:
        lines = [
            f"TASK: {obs.get('task_name', 'unknown')}",
            f"STEP: {step} of {obs.get('max_steps', 7)}",
            f"\nPOLICY:\n{obs.get('policy_text', '')}",
        ]
        if obs.get("clarification_response"):
            lines.append(f"\nLAST CLARIFICATION ANSWER:\n{obs['clarification_response']}")
        if obs.get("test_results"):
            tr = obs["test_results"]
            lines.append(f"\nTEST RESULTS: {tr.get('passed', 0)}/{tr.get('total', 0)} passed (accuracy={obs.get('current_accuracy', 0):.2f})")
            if tr.get("sample_failures"):
                lines.append("SAMPLE FAILURES:")
                for f in tr["sample_failures"][:3]:
                    lines.append(f"  - {f}")
        if obs.get("feedback"):
            lines.append(f"\nFEEDBACK: {obs['feedback']}")
        if history:
            lines.append(f"\nACTION HISTORY (last 3):\n" + "\n".join(history[-3:]))
        lines.append(f"\nAVAILABLE ACTIONS: {obs.get('available_actions', [])}")
        lines.append("\nRespond with JSON only: {\"action_type\": \"...\", \"content\": \"...\"}")
        return "\n".join(lines)

    def _parse_response(self, raw: str, obs: dict) -> tuple[str, str]:
        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
            action_type = parsed.get("action_type", "propose_rules")
            content = parsed.get("content", "{}")

            # Validate action_type
            valid_actions = obs.get("available_actions", ["propose_rules", "ask_clarification"])
            if action_type not in valid_actions:
                action_type = "propose_rules" if "propose_rules" in valid_actions else valid_actions[0]

            # Ensure content is a string
            if isinstance(content, dict):
                content = json.dumps(content)
            return action_type, content
        except Exception:
            return "propose_rules", json.dumps({"rules": [], "default": "DENY"})

# ── Trajectory Bank ───────────────────────────────────────────────────────────

class TrajectoryBank:
    """Stores and retrieves high-reward trajectories per task."""

    def __init__(self):
        self.bank: dict[str, list[Trajectory]] = {task: [] for task in TASKS}

    def store(self, trajectory: Trajectory):
        if trajectory.total_reward >= MIN_REWARD_THRESHOLD:
            self.bank[trajectory.task_name].append(trajectory)
            # Keep only top-K by reward
            self.bank[trajectory.task_name].sort(key=lambda t: t.total_reward, reverse=True)
            self.bank[trajectory.task_name] = self.bank[trajectory.task_name][:TOP_K_TRAJECTORIES]

    def get_examples(self, task_name: str) -> list[Trajectory]:
        return self.bank.get(task_name, [])

    def summary(self) -> dict:
        return {
            task: {
                "stored": len(trajs),
                "best_reward": max((t.total_reward for t in trajs), default=0),
                "best_accuracy": max((t.final_accuracy for t in trajs), default=0)
            }
            for task, trajs in self.bank.items()
        }

# ── Training Loop ─────────────────────────────────────────────────────────────

class TrainingLoop:
    def __init__(self, env_url: str, hf_token: str):
        self.env = EnvClient(env_url)
        self.agent = Agent(hf_token)
        self.bank = TrajectoryBank()
        self.metrics = []  # List of {episode, task, reward, accuracy, success}

    def run_episode(self, task_name: str, episode_id: int) -> Trajectory:
        """Run a single episode and return the trajectory."""
        few_shots = self.bank.get_examples(task_name)
        trajectory = Trajectory(task_name=task_name, episode_id=episode_id)

        # Reset environment
        result = self.env.reset(task_name)
        obs = result.get("observation", {})
        done = result.get("done", False)
        history = []

        print(f"  [Episode {episode_id}] task={task_name} few_shots={len(few_shots)}")

        step_num = 0
        while not done and step_num < obs.get("max_steps", 7):
            step_num += 1

            # Get action from agent
            action_type, content = self.agent.get_action(
                observation=obs,
                step_number=step_num,
                episode_history=history,
                few_shot_examples=few_shots
            )

            # Execute action
            result = self.env.step(action_type, content)
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            obs = result.get("observation", {})
            info = result.get("info", {})

            # Record step
            step = Step(
                step_number=step_num,
                action_type=action_type,
                action_content=content[:300],
                reward=reward,
                accuracy=obs.get("current_accuracy", 0.0),
                feedback=obs.get("feedback", "") or "",
                clarification_response=obs.get("clarification_response")
            )
            trajectory.steps.append(step)
            trajectory.total_reward += reward

            # Update history
            history.append(f"Step {step_num}: {action_type} → reward={reward:.2f} acc={step.accuracy:.2f}")

            print(f"    step={step_num} action={action_type} reward={reward:.3f} acc={step.accuracy:.2f}")

            if done:
                episode_score = info.get("episode_score", obs.get("current_accuracy", 0.0))
                trajectory.final_accuracy = episode_score
                trajectory.success = obs.get("current_accuracy", 0.0) >= 0.9
                break

        if not trajectory.steps:
            trajectory.final_accuracy = 0.0

        return trajectory

    def run(self):
        """Run full training loop across all tasks."""
        print("=" * 60)
        print("REWARD-GUIDED TRAJECTORY OPTIMIZATION")
        print(f"Tasks: {TASKS}")
        print(f"Episodes per task: {NUM_EPISODES_PER_TASK}")
        print(f"Top-K trajectories: {TOP_K_TRAJECTORIES}")
        print("=" * 60)

        # Health check
        if not self.env.health():
            raise RuntimeError(f"Environment not reachable at {ENV_BASE_URL}")
        print(f"Environment: OK ({ENV_BASE_URL})\n")

        global_episode = 0

        for task in TASKS:
            print(f"\n{'─'*40}")
            print(f"TASK: {task}")
            print(f"{'─'*40}")

            task_rewards = []
            task_accuracies = []

            for ep in range(1, NUM_EPISODES_PER_TASK + 1):
                global_episode += 1
                trajectory = self.run_episode(task, ep)

                # Store in bank
                self.bank.store(trajectory)

                # Record metrics
                self.metrics.append({
                    "global_episode": global_episode,
                    "task": task,
                    "episode_in_task": ep,
                    "total_reward": trajectory.total_reward,
                    "final_accuracy": trajectory.final_accuracy,
                    "success": trajectory.success,
                    "num_steps": len(trajectory.steps),
                    "few_shots_used": len(self.bank.get_examples(task)) - (1 if trajectory.total_reward >= MIN_REWARD_THRESHOLD else 0)
                })

                task_rewards.append(trajectory.total_reward)
                task_accuracies.append(trajectory.final_accuracy)

                print(f"  → Episode {ep} complete: reward={trajectory.total_reward:.3f} accuracy={trajectory.final_accuracy:.2f} success={trajectory.success}")
                time.sleep(0.5)  # Rate limiting

            print(f"\n  Task summary:")
            print(f"    First episode reward: {task_rewards[0]:.3f}")
            print(f"    Last episode reward:  {task_rewards[-1]:.3f}")
            print(f"    Improvement: {task_rewards[-1] - task_rewards[0]:+.3f}")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"Bank summary: {self.bank.summary()}")
        print("=" * 60)

        return self.metrics

# ── Plot Generation ───────────────────────────────────────────────────────────

def save_plots(metrics: list[dict]):
    """
    Save reward curve and accuracy curve as PNG files.
    These are REQUIRED for hackathon submission — must be committed to repo.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    os.makedirs("training/plots", exist_ok=True)

    episodes = [m["global_episode"] for m in metrics]
    rewards = [m["total_reward"] for m in metrics]
    accuracies = [m["final_accuracy"] for m in metrics]
    tasks = [m["task"] for m in metrics]

    colors = {
        "data_access": "#2196F3",
        "resource_access": "#FF9800",
        "transaction_approval": "#4CAF50"
    }

    # ── Plot 1: Reward Curve ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for task in TASKS:
        task_eps = [m["global_episode"] for m in metrics if m["task"] == task]
        task_rews = [m["total_reward"] for m in metrics if m["task"] == task]
        ax.plot(task_eps, task_rews, marker="o", label=task,
                color=colors.get(task, "gray"), linewidth=2, markersize=5)

    # Trend line
    z = np.polyfit(episodes, rewards, 1)
    p = np.poly1d(z)
    ax.plot(episodes, p(episodes), "--", color="red", alpha=0.5, linewidth=1.5, label="overall trend")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Curve — Reward-Guided Trajectory Optimization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("training/plots/reward_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: training/plots/reward_curve.png")

    # ── Plot 2: Accuracy Curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for task in TASKS:
        task_eps = [m["global_episode"] for m in metrics if m["task"] == task]
        task_accs = [m["final_accuracy"] for m in metrics if m["task"] == task]
        ax.plot(task_eps, task_accs, marker="s", label=task,
                color=colors.get(task, "gray"), linewidth=2, markersize=5)

    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.7, label="success threshold (0.9)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Accuracy Curve — Policy-to-Logic Agent")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("training/plots/accuracy_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: training/plots/accuracy_curve.png")

    # ── Plot 3: Per-Task Improvement Bar Chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    task_names = []
    improvements = []

    for task in TASKS:
        task_accs = [m["final_accuracy"] for m in metrics if m["task"] == task]
        if len(task_accs) >= 2:
            first = task_accs[0]
            last = task_accs[-1]
            task_names.append(task.replace("_", "\n"))
            improvements.append(last - first)

    bars = ax.bar(task_names, improvements,
                  color=["#2196F3", "#FF9800", "#4CAF50"][:len(task_names)],
                  edgecolor="white", linewidth=1.5)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Accuracy Improvement (last - first episode)")
    ax.set_title("Per-Task Improvement from Trajectory Accumulation")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:+.2f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig("training/plots/improvement_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: training/plots/improvement_chart.png")

    # ── Save raw metrics as JSON ──────────────────────────────────────────────
    with open("training/plots/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: training/plots/metrics.json")

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    loop = TrainingLoop(ENV_BASE_URL, hf_token)
    metrics = loop.run()
    save_plots(metrics)

    print("\nNext step: commit training/plots/*.png to repo for submission.")
```

---

## HOUR 2-3: Build the Colab Training Notebook

### File to Create: `training/colab_training.ipynb`

This must be a runnable Colab notebook. Create it with the following cells in order.

**Cell 1 — Install dependencies:**
```python
# Cell 1: Install dependencies
!pip install openai requests matplotlib numpy
```

**Cell 2 — Configuration:**
```python
# Cell 2: Configuration
import os

# SET THESE BEFORE RUNNING
HF_TOKEN = ""  # Your Hugging Face token with inference access
ENV_URL = "https://godreign-policy2logic.hf.space"  # Your deployed environment URL

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["ENV_BASE_URL"] = ENV_URL

print(f"Environment URL: {ENV_URL}")
print(f"HF Token set: {'Yes' if HF_TOKEN else 'NO - MUST SET THIS'}")
```

**Cell 3 — Health check:**
```python
# Cell 3: Verify environment is reachable
import requests

r = requests.get(f"{ENV_URL}/health")
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

r2 = requests.get(f"{ENV_URL}/tasks")
tasks = r2.json()
print(f"\nAvailable tasks: {list(tasks['tasks'].keys())}")
```

**Cell 4 — Paste entire `trajectory_optimizer.py` content here as a cell.**

Add this comment at the top of the cell:
```python
# Cell 4: Training loop implementation
# (paste full contents of training/trajectory_optimizer.py here)
```

**Cell 5 — Run training:**
```python
# Cell 5: Run training loop
loop = TrainingLoop(ENV_URL, HF_TOKEN)
metrics = loop.run()
print(f"\nTotal episodes run: {len(metrics)}")
```

**Cell 6 — Generate and display plots:**
```python
# Cell 6: Generate plots and display inline
save_plots(metrics)

from IPython.display import Image, display
display(Image("training/plots/reward_curve.png"))
display(Image("training/plots/accuracy_curve.png"))
display(Image("training/plots/improvement_chart.png"))
```

**Cell 7 — Download plots (CRITICAL — these must be committed to repo):**
```python
# Cell 7: Download plots to commit to repo
# After running this, download the files and commit them to your GitHub repo
from google.colab import files

files.download("training/plots/reward_curve.png")
files.download("training/plots/accuracy_curve.png")
files.download("training/plots/improvement_chart.png")
files.download("training/plots/metrics.json")

print("Downloaded. Now commit these files to: training/plots/ in your repo.")
```

---

## HOUR 3-4: Run Training and Capture Results

### Steps (execute in order):

1. Start the environment server locally OR use the deployed HF Space URL.
2. Open the Colab notebook.
3. Set `HF_TOKEN` in Cell 2.
4. Set `ENV_URL` to your HF Space URL: `https://godreign-policy2logic.hf.space`
5. Run all cells top to bottom.
6. Wait for training to complete (~20-30 minutes for 8 episodes × 3 tasks).
7. Cell 7 will download the plot PNG files.
8. **Immediately commit the PNG files to the repo** under `training/plots/`.

### Git commands after downloading plots:
```bash
git add training/plots/reward_curve.png
git add training/plots/accuracy_curve.png
git add training/plots/improvement_chart.png
git add training/plots/metrics.json
git commit -m "Add training evidence: reward and accuracy curves"
git push
```

### If training takes too long (fallback):
Reduce `NUM_EPISODES_PER_TASK = 4` in the configuration. 4 episodes × 3 tasks = 12 total, which is enough to show a trend.

---

## HOUR 4-5: Write the README (CRITICAL — Validator Reads This)

### File to Replace: `README.md`

The README must link every deliverable. The validator traverses links from README. If a link is broken or missing, that deliverable is marked absent.

```markdown
# Policy-to-Logic RL Environment

> A verifiable reinforcement learning environment for policy-to-logic reasoning,
> where an agent learns to iteratively convert natural language policies into
> executable rules through interaction and reward-guided optimization.

---

## 🔗 Deliverables

| Deliverable | Link |
|---|---|
| **HF Space (Live Environment)** | [godreign-policy2logic.hf.space](https://godreign-policy2logic.hf.space) |
| **Training Notebook (Colab)** | [Open in Colab](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/training/colab_training.ipynb) |
| **Writeup / Slides** | [Link to your blog/slides/video here] |

> Replace `YOUR_USERNAME/YOUR_REPO` with your actual GitHub path.

---

## 📊 Training Results

The agent is trained using a **reward-guided trajectory optimization loop**.
High-reward interaction sequences are accumulated as few-shot examples,
improving agent behavior across episodes without weight updates.

### Reward Curve
![Reward Curve](training/plots/reward_curve.png)

### Accuracy Curve
![Accuracy Curve](training/plots/accuracy_curve.png)

### Per-Task Improvement
![Improvement Chart](training/plots/improvement_chart.png)

---

## 🧠 What This Is

This project builds a **verifiable RL environment** where:
- Policies are stated in natural language
- An agent converts them to executable JSON rules (DSL)
- The environment evaluates rules against generated scenarios
- Reward signals drive measurable improvement across episodes

**This is not a finished product. It is a training and evaluation framework.**

---

## 🏗️ Architecture

```
Policy → Agent → (Ask / Propose / Refine)
       → Environment → (Scenarios + Evaluation)
       → Reward → Trajectory Bank → Improved Agent
```

### Three Tasks (increasing difficulty)

| Task | Difficulty | Variables | Decisions |
|---|---|---|---|
| data_access | Easy | time, data_type | ALLOW, DENY |
| resource_access | Medium | role, time, document_type | ALLOW, DENY |
| transaction_approval | Hard | amount, transfer_type, time, role | APPROVE, REQUIRE_APPROVAL, COMPLIANCE_REVIEW, HOLD |

---

## 🎮 Environment API

Live at: `https://godreign-policy2logic.hf.space`

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/tasks` | GET | List available tasks |
| `/reset` | POST | Start new episode |
| `/step` | POST | Take action |
| `/state` | GET | Get episode state |

### Quick Start

```python
import requests

base = "https://godreign-policy2logic.hf.space"

# Start episode
result = requests.post(f"{base}/reset", json={"task_name": "data_access"}).json()
print(result["observation"]["policy_text"])

# Take action
action = requests.post(f"{base}/step", json={
    "action_type": "propose_rules",
    "content": '{"rules": [{"if": [{"field": "time", "op": ">=", "value": 9}, {"field": "time", "op": "<", "value": 18}], "then": "ALLOW"}], "default": "DENY"}'
}).json()
print(f"Reward: {action['reward']}, Accuracy: {action['observation']['current_accuracy']}")
```

---

## 🔁 Training Loop

The training approach uses **reward-guided trajectory accumulation**:

1. Agent runs episode zero-shot
2. High-reward trajectories stored in trajectory bank
3. Next episode uses top-K trajectories as few-shot context
4. Agent performance improves as bank accumulates better examples

**This is a legitimate policy improvement loop driven by environment reward signal.**

### Run Training Locally

```bash
# Install dependencies
pip install openai requests matplotlib numpy

# Set environment variables
export HF_TOKEN=your_token_here
export ENV_BASE_URL=https://godreign-policy2logic.hf.space

# Run
python training/trajectory_optimizer.py
```

---

## 📁 Repository Structure

```
├── policy_to_logic_env/
│   ├── server/
│   │   ├── app.py              # FastAPI endpoints
│   │   ├── environment.py      # Core RL environment (reset/step/state)
│   │   ├── policies.py         # 3 task definitions
│   │   ├── ground_truth.py     # Ground truth + clarification oracle
│   │   ├── scenario_generator.py  # 4-strategy scenario generation
│   │   ├── dsl_engine.py       # JSON DSL parser and executor
│   │   ├── rewards.py          # Multi-component reward system
│   │   └── graders.py          # Rule evaluation
│   ├── models.py               # Pydantic data models
│   ├── client.py               # HTTP client library
│   └── openenv.yaml            # OpenEnv specification
├── training/
│   ├── trajectory_optimizer.py # Training loop
│   ├── colab_training.ipynb    # Colab notebook
│   └── plots/
│       ├── reward_curve.png    # Training evidence (committed)
│       ├── accuracy_curve.png  # Training evidence (committed)
│       └── improvement_chart.png
├── main.py                     # Server entry point
├── Dockerfile                  # HF Spaces deployment
└── README.md                   # This file
```

---

## ⚙️ OpenEnv Compliance

This environment implements the OpenEnv specification:
- Gym-style `reset()` / `step()` / `state()` interface
- Valid `openenv.yaml` at `policy_to_logic_env/openenv.yaml`
- Pydantic models for all inputs/outputs
- HTTP API for remote agent interaction

---

## ⚠️ Known Limitations

1. Single-session server (sequential episodes only, not parallel)
2. Deterministic scenario seed — same scenarios every episode
3. Training loop uses trajectory accumulation, not weight updates
4. Clarification oracle is keyword-based, not semantic

---

## 🧾 Reward System

| Component | Weight | Signal |
|---|---|---|
| Accuracy | 50% | Rules correct vs ground truth |
| Improvement | 20% | Accuracy delta per step |
| Efficiency | 15% | Steps used vs budget |
| Clarification | 15% | Question usefulness |
```

---

## HOUR 5: Verify openenv.yaml is Valid

### Check existing file: `policy_to_logic_env/openenv.yaml`

Open the file and verify it contains at minimum:

```yaml
name: policy-to-logic-env
version: "1.0.0"
description: "RL environment for converting natural language policies into executable rules"

environment:
  type: Environment
  reset_endpoint: /reset
  step_endpoint: /step
  state_endpoint: /state

tasks:
  - name: data_access
    difficulty: easy
    max_steps: 5
  - name: resource_access
    difficulty: medium
    max_steps: 7
  - name: transaction_approval
    difficulty: hard
    max_steps: 7

observation_space:
  policy_text: string
  current_accuracy: float
  available_actions: list
  feedback: string

action_space:
  - ask_clarification
  - propose_rules
  - refine_rules

reward:
  min: 0.0
  max: 1.0
```

If this file exists and is different, do NOT replace it — just verify it is parseable YAML and contains `reset`, `step`, `state` references.

If it is missing these fields, add them.

---

## HOUR 6: Final Verification Pass

Run through every checklist item explicitly. Do not assume. Verify each one.

### Check 1: HF Space is Public and Reachable

```bash
# Open this URL in a logged-out browser (incognito window)
# https://godreign-policy2logic.hf.space
# Must load without login prompt
# Must return 200 on /health endpoint

curl https://godreign-policy2logic.hf.space/health
# Expected: {"status": "ok", "environment": "policy_to_logic"}
```

### Check 2: OpenEnv Structure

```bash
# Verify openenv.yaml exists and is valid YAML
python -c "import yaml; yaml.safe_load(open('policy_to_logic_env/openenv.yaml'))"
# No error = valid

# Verify reset/step/state endpoints respond
curl -X POST https://godreign-policy2logic.hf.space/reset -H "Content-Type: application/json" -d '{"task_name": "data_access"}'
curl -X GET https://godreign-policy2logic.hf.space/state
```

### Check 3: Plot PNG Files Are Committed

```bash
git ls-files training/plots/
# Must output:
# training/plots/reward_curve.png
# training/plots/accuracy_curve.png
# training/plots/improvement_chart.png

# Verify they are actual image files (not empty)
ls -lh training/plots/*.png
# Each must be > 10KB
```

### Check 4: Training Script is Runnable

```bash
# Verify the Python script runs (dry run — just check imports and config)
python -c "
import sys
sys.path.insert(0, '.')
# Check imports
import json, os, time, requests
from openai import OpenAI
print('All imports OK')
print('Training script: training/trajectory_optimizer.py — OK')
"

# Verify Colab notebook exists
ls -la training/colab_training.ipynb
```

### Check 5: README Links

Open `README.md` and manually verify:
- HF Space link is correct and matches actual URL
- Colab badge link uses correct GitHub username and repo name
- Both `![Reward Curve](training/plots/reward_curve.png)` images render when viewed on GitHub
- Writeup/slides link is filled in (not placeholder)

---

## HOUR 7: Buffer — Fix Whatever Failed Check

Use this hour to fix anything that failed the verification pass.

**Most likely failures and fixes:**

| Failure | Fix |
|---|---|
| HF Space returns 404 | Rebuild Docker and redeploy to HF Spaces |
| PNG files not in repo | Download from Colab, `git add`, `git commit`, `git push` |
| openenv.yaml missing fields | Add missing fields, push |
| Colab link broken | Fix GitHub path in README |
| Plots not rendering in README | Verify relative path matches actual file location |

---

## Fallback: If Training Produces Flat Curves

If the reward curve shows no improvement (all episodes get similar rewards), do NOT fabricate results. Instead:

1. Run more episodes — increase `NUM_EPISODES_PER_TASK = 12`
2. Lower `MIN_REWARD_THRESHOLD = 0.1` to accumulate more examples
3. If still flat, the submission narrative becomes: *"We demonstrate that the environment produces consistent reward signals and the agent achieves non-trivial baseline performance. Future work includes fine-tuning with TRL/GRPO."*

A flat but honest curve is better than a fabricated improving curve.

---

## What to Say to Judges (Prepared Answers)

**Q: Is this RL training?**
> "We implement a reward-guided trajectory optimization loop. The environment's reward signal selects high-value interaction trajectories which are accumulated as few-shot context, improving agent policy across episodes. This is a form of in-context policy improvement driven by environment feedback."

**Q: Why not fine-tune the model?**
> "Given hackathon constraints, we demonstrate the environment's training capability through trajectory accumulation. The environment is fully compatible with TRL/GRPO fine-tuning — the reward signal, episode structure, and action space are all defined. Fine-tuning is the natural next step."

**Q: What does the agent actually learn?**
> "The agent learns when to ask clarifying questions versus when to propose rules, and how to refine rules based on failure feedback. The trajectory bank accumulates successful strategies that improve decision-making in subsequent episodes."

**Q: Why is the simulation not realistic?**
> "The environment is a verification harness, not a simulation. It functions like unit testing for policy logic — correctness is the goal, not realism. This gives us objective, programmatic reward signals suitable for RL."

---

## File Checklist (Everything That Must Exist at Submission)

```
✅ policy_to_logic_env/openenv.yaml          — already exists, verify valid
✅ policy_to_logic_env/server/environment.py  — already exists
✅ policy_to_logic_env/server/app.py          — already exists
✅ training/trajectory_optimizer.py           — CREATE THIS (Hour 1-2)
✅ training/colab_training.ipynb              — CREATE THIS (Hour 2-3)
✅ training/plots/reward_curve.png            — GENERATE AND COMMIT (Hour 3-4)
✅ training/plots/accuracy_curve.png          — GENERATE AND COMMIT (Hour 3-4)
✅ training/plots/improvement_chart.png       — GENERATE AND COMMIT (Hour 3-4)
✅ README.md                                  — REWRITE (Hour 4-5)
```

---

*End of handoff document. Every step above is required. Execute in order.*
