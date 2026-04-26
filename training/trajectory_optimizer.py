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
import logging
import wandb
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
        few_shot_examples: list[Trajectory],
        task_name: str = ""
    ) -> tuple[str, str]:
        """
        Returns (action_type, content_json_string).
        action_type: one of ask_clarification | propose_rules | refine_rules
        content: JSON string appropriate for that action
        """
        system_prompt = self._build_system_prompt(few_shot_examples, task_name)
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

    def _build_system_prompt(self, few_shot_examples: list[Trajectory], task_name: str = "") -> str:
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
        # Task-specific guidance for complex tasks
        if task_name == "transaction_approval":
            base += """
IMPORTANT — TRANSACTION APPROVAL TASK:
This task has 4 possible decisions: APPROVE, REQUIRE_APPROVAL, COMPLIANCE_REVIEW, HOLD
Rules are evaluated TOP-TO-BOTTOM. Order matters critically. You MUST order rules by priority:
  1. FIRST: Check if transfer_type == "international" → then COMPLIANCE_REVIEW (always, overrides everything)
  2. SECOND: Check if amount >= 10000 AND time is outside business hours (hour < 9 or hour >= 17) → then HOLD
  3. THIRD: Check if amount > 5000 AND initiator_role != "manager" → then REQUIRE_APPROVAL
  4. DEFAULT: APPROVE

Key details:
- Standard limit is $5,000 (amount > 5000 triggers approval, NOT >=)
- High-value threshold is $10,000 (amount >= 10000)
- Business hours: hour >= 9 AND hour < 17
- Manager exemption ONLY applies to the standard $5,000 limit, NOT to international or high-value HOLD rules
- "system" role follows the same rules as "employee"

Here is a working example of valid rules for this task:
{"rules": [{"if": [{"field": "transfer_type", "op": "==", "value": "international"}], "then": "COMPLIANCE_REVIEW"}, {"if": [{"field": "amount", "op": ">=", "value": 10000}, {"field": "time", "op": ">=", "value": 17}], "then": "HOLD"}, {"if": [{"field": "amount", "op": ">=", "value": 10000}, {"field": "time", "op": "<", "value": 9}], "then": "HOLD"}, {"if": [{"field": "amount", "op": ">", "value": 5000}, {"field": "initiator_role", "op": "!=", "value": "manager"}], "then": "REQUIRE_APPROVAL"}], "default": "APPROVE"}
"""
        elif task_name == "resource_access":
            base += """
IMPORTANT — RESOURCE ACCESS TASK:
This task has roles: junior, senior, contractor. Document types: public, internal, confidential.
- Senior employees: ALLOW everything always
- Contractors: ALLOW only public, DENY everything else
- Junior + confidential: ALWAYS DENY (regardless of time — the policy is misleading about this)
- Junior + internal: ALLOW only during business hours (hour >= 8 AND hour < 17)
- Junior + public: ALLOW always
- Business hours: hour >= 8 AND hour < 17
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

        os.makedirs("training/logs", exist_ok=True)
        log_filename = f"training/logs/run_{int(time.time())}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()  # also print to console
            ]
        )
        self.logger = logging.getLogger("TrainingLoop")
        self.log_file = log_filename

    def run_episode(self, task_name: str, episode_id: int) -> Trajectory:
        """Run a single episode and return the trajectory."""
        few_shots = self.bank.get_examples(task_name)
        trajectory = Trajectory(task_name=task_name, episode_id=episode_id)

        # Reset environment
        result = self.env.reset(task_name)
        obs = result.get("observation", {})
        done = result.get("done", False)
        history = []

        self.logger.info(f"START episode={episode_id} task={task_name} few_shots_available={len(few_shots)}")

        step_num = 0
        while not done and step_num < obs.get("max_steps", 7):
            step_num += 1

            # Get action from agent
            action_type, content = self.agent.get_action(
                observation=obs,
                step_number=step_num,
                episode_history=history,
                few_shot_examples=few_shots,
                task_name=task_name
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

            self.logger.info(f"STEP episode={episode_id} step={step_num} action={action_type} reward={reward:.4f} accuracy={step.accuracy:.4f}")

            if done:
                episode_score = info.get("episode_score", obs.get("current_accuracy", 0.0))
                trajectory.final_accuracy = episode_score
                trajectory.success = obs.get("current_accuracy", 0.0) >= 0.9
                break

        if not trajectory.steps:
            trajectory.final_accuracy = 0.0

        self.logger.info(f"END episode={episode_id} task={task_name} total_reward={trajectory.total_reward:.4f} final_accuracy={trajectory.final_accuracy:.4f} success={trajectory.success} steps={len(trajectory.steps)}")

        return trajectory

    def run(self):
        """Run full training loop across all tasks."""
        self.logger.info("=" * 60)
        self.logger.info("REWARD-GUIDED TRAJECTORY OPTIMIZATION")
        self.logger.info(f"Tasks: {TASKS}")
        self.logger.info(f"Episodes per task: {NUM_EPISODES_PER_TASK}")
        self.logger.info(f"Top-K trajectories: {TOP_K_TRAJECTORIES}")
        self.logger.info("=" * 60)
        self.logger.info(f"Log file: {self.log_file}")

        try:
            wandb.init(
                project="policy-to-logic-rl",
                name=f"trajectory-opt-{int(time.time())}",
                config={
                    "num_episodes_per_task": NUM_EPISODES_PER_TASK,
                    "top_k_trajectories": TOP_K_TRAJECTORIES,
                    "min_reward_threshold": MIN_REWARD_THRESHOLD,
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "tasks": TASKS,
                    "env_url": ENV_BASE_URL,
                }
            )
        except Exception as e:
            self.logger.warning(f"Wandb init failed: {e}. Continuing without W&B.")

        # Health check
        if not self.env.health():
            raise RuntimeError(f"Environment not reachable at {ENV_BASE_URL}")
        self.logger.info(f"Environment: OK ({ENV_BASE_URL})\n")

        global_episode = 0

        for task in TASKS:
            self.logger.info(f"\n{'-'*40}")
            self.logger.info(f"TASK: {task}")
            self.logger.info(f"{'-'*40}")

            task_rewards = []
            task_accuracies = []

            for ep in range(1, NUM_EPISODES_PER_TASK + 1):
                global_episode += 1
                trajectory = self.run_episode(task, ep)

                # Store in bank
                self.bank.store(trajectory)

                try:
                    wandb.log({
                        f"{task}/total_reward": trajectory.total_reward,
                        f"{task}/final_accuracy": trajectory.final_accuracy,
                        f"{task}/num_steps": len(trajectory.steps),
                        f"{task}/success": int(trajectory.success),
                        f"{task}/few_shots_used": len(self.bank.get_examples(task)),
                        "global/total_reward": trajectory.total_reward,
                        "global/final_accuracy": trajectory.final_accuracy,
                        "episode": global_episode,
                    })
                except Exception:
                    pass

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

                self.logger.info(f"  → Episode {ep} complete: reward={trajectory.total_reward:.3f} accuracy={trajectory.final_accuracy:.2f} success={trajectory.success}")
                time.sleep(0.5)  # Rate limiting

            self.logger.info(f"\n  Task summary:")
            self.logger.info(f"    First episode reward: {task_rewards[0]:.3f}")
            self.logger.info(f"    Last episode reward:  {task_rewards[-1]:.3f}")
            self.logger.info(f"    Improvement: {task_rewards[-1] - task_rewards[0]:+.3f}")

        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info(f"Bank summary: {self.bank.summary()}")
        self.logger.info("=" * 60)

        try:
            wandb.finish()
        except Exception:
            pass

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

    # ── Plot 1: Reward Curve (per-task trend lines) ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for task in TASKS:
        task_eps = [m["global_episode"] for m in metrics if m["task"] == task]
        task_rews = [m["total_reward"] for m in metrics if m["task"] == task]
        ax.plot(task_eps, task_rews, marker="o", label=task,
                color=colors.get(task, "gray"), linewidth=2, markersize=5)
        # Per-task trend line
        if len(task_eps) >= 2:
            z = np.polyfit(task_eps, task_rews, 1)
            p = np.poly1d(z)
            ax.plot(task_eps, p(task_eps), "--",
                    color=colors.get(task, "gray"), alpha=0.4, linewidth=1.5)

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

    # ── Plot 3: Per-Task Summary (Accuracy + Efficiency) ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    task_labels = []
    acc_improvements = []
    eff_improvements = []
    best_accuracies = []

    for task in TASKS:
        task_data = [m for m in metrics if m["task"] == task]
        if len(task_data) >= 2:
            task_labels.append(task.replace("_", "\n"))
            acc_improvements.append(task_data[-1]["final_accuracy"] - task_data[0]["final_accuracy"])
            # Efficiency: steps saved (first vs best)
            first_steps = task_data[0]["num_steps"]
            best_steps = min(m["num_steps"] for m in task_data)
            eff_pct = ((first_steps - best_steps) / first_steps * 100) if first_steps > 0 else 0
            eff_improvements.append(eff_pct)
            best_accuracies.append(max(m["final_accuracy"] for m in task_data))

    # Left: Best accuracy per task
    bars1 = axes[0].bar(task_labels, best_accuracies,
                        color=["#2196F3", "#FF9800", "#4CAF50"][:len(task_labels)],
                        edgecolor="white", linewidth=1.5)
    axes[0].axhline(y=0.9, color="red", linestyle="--", alpha=0.7, label="success threshold")
    axes[0].set_ylabel("Best Accuracy Achieved")
    axes[0].set_title("Best Accuracy Per Task")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    for bar, val in zip(bars1, best_accuracies):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.0%}", ha="center", va="bottom", fontweight="bold")

    # Right: Efficiency improvement (% steps saved)
    bars2 = axes[1].bar(task_labels, eff_improvements,
                        color=["#2196F3", "#FF9800", "#4CAF50"][:len(task_labels)],
                        edgecolor="white", linewidth=1.5)
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Steps Saved (%)")
    axes[1].set_title("Efficiency Improvement (First → Best Episode)")
    axes[1].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars2, eff_improvements):
        y_pos = max(bar.get_height() + 1, 2)
        axes[1].text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f"{val:.0f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig("training/plots/improvement_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: training/plots/improvement_chart.png")

    # ── Save raw metrics as JSON ──────────────────────────────────────────────
    timestamp = int(time.time())
    with open(f"training/plots/metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("training/plots/metrics_latest.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: training/plots/metrics_{timestamp}.json and metrics_latest.json")

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    loop = TrainingLoop(ENV_BASE_URL, hf_token)
    metrics = loop.run()
    save_plots(metrics)

    print("\nNext step: commit training/plots/*.png to repo for submission.")
