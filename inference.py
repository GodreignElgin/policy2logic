"""
Inference Script for the Policy-to-Logic RL Environment
=======================================================

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

This script runs the LLM agent against all 3 tasks in the environment
and produces reproducible baseline scores.
"""

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ─── Add parent to path for imports ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy_to_logic_env.client import PolicyToLogicEnv
from policy_to_logic_env.models import PolicyToLogicAction

# ─── Configuration ───────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "policy_to_logic"

TASKS = ["data_access", "resource_access", "transaction_approval"]
TEMPERATURE = 0.3
MAX_TOKENS = 1024


# ─── Logging ─────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Truncate action for readability
    action_short = action[:80].replace("\n", " ") if action else "none"
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─── System Prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an AI agent interacting with a Policy-to-Logic RL environment.

Your goal is to convert natural language access control policies into
executable logic rules using a simple JSON-based DSL.

AVAILABLE ACTIONS:
1. ask_clarification - Ask a question about unclear policy terms
2. propose_rules - Submit a complete rule set
3. refine_rules - Improve your previous rules based on feedback

RESPONSE FORMAT:
You must respond with a valid JSON object:

For asking clarification:
{"action_type": "ask_clarification", "content": "{\\"question\\": \\"What are business hours?\\"}"}

For proposing or refining rules:
{"action_type": "propose_rules", "content": "{\\"rules\\": [{\\"if\\": [{\\"field\\": \\"time\\", \\"op\\": \\">\\", \\"value\\": 18}, {\\"field\\": \\"data_type\\", \\"op\\": \\"==\\", \\"value\\": \\"sensitive\\"}], \\"then\\": \\"DENY\\"}], \\"default\\": \\"ALLOW\\"}"}

DSL FORMAT:
- Rules are a list of IF-THEN objects
- All conditions in a rule are AND-ed
- Rules evaluated top-to-bottom, first match wins
- Operators: >, <, >=, <=, ==, !=

STRATEGY:
- Read the policy carefully
- Ask 1-2 clarification questions for ambiguous terms (thresholds, hours)
- Propose rules covering all cases
- Refine based on test result feedback
- Aim for >90% accuracy to complete the episode

Respond with ONLY the JSON object, nothing else.
""").strip()


# ─── Agent Logic ─────────────────────────────────────────────────
def build_user_prompt(observation: dict, step: int, history: List[str]) -> str:
    """Build the user prompt from the current observation."""
    parts = [f"Step {step} of {observation.get('max_steps', 7)}"]

    parts.append(f"\n--- POLICY ---\n{observation.get('policy_text', '')}")

    if observation.get("feedback"):
        parts.append(f"\n--- FEEDBACK ---\n{observation['feedback']}")

    if observation.get("clarification_response"):
        parts.append(f"\n--- CLARIFICATION ANSWER ---\n{observation['clarification_response']}")

    if observation.get("test_results"):
        tr = observation["test_results"]
        parts.append(
            f"\n--- TEST RESULTS ---\n"
            f"Passed: {tr.get('passed', 0)}/{tr.get('total', 0)}\n"
            f"Accuracy: {observation.get('current_accuracy', 0):.1%}"
        )
        if tr.get("sample_failures"):
            parts.append("Sample failures:")
            for fail in tr["sample_failures"][:3]:
                parts.append(f"  Scenario: {fail.get('scenario', {})}")
                parts.append(f"  Expected: {fail.get('expected', '?')}, Got: {fail.get('got', '?')}")

    parts.append(f"\n--- DSL FORMAT ---\n{observation.get('dsl_format', '')}")
    parts.append(f"\nAvailable actions: {observation.get('available_actions', [])}")
    parts.append(f"Current accuracy: {observation.get('current_accuracy', 0):.1%}")

    if history:
        parts.append(f"\n--- HISTORY ---")
        for h in history[-3:]:
            parts.append(h)

    parts.append("\nRespond with your next action as a JSON object.")

    return "\n".join(parts)


def get_agent_action(
    client: OpenAI, observation: dict, step: int, history: List[str]
) -> tuple[str, str]:
    """
    Query the LLM to get the next action.

    Returns (action_type, content)
    """
    user_prompt = build_user_prompt(observation, step, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Parse the response
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            parsed = json.loads(raw)
            action_type = parsed.get("action_type", "propose_rules")
            content = parsed.get("content", "{}")

            # If content is already a dict, serialize it
            if isinstance(content, dict):
                content = json.dumps(content)

            return action_type, content

        except (json.JSONDecodeError, IndexError):
            # Fallback: try to extract rules from raw text
            if '"rules"' in raw:
                return "propose_rules", raw
            else:
                return "propose_rules", json.dumps({
                    "rules": [],
                    "default": "ALLOW"
                })

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "propose_rules", json.dumps({"rules": [], "default": "ALLOW"})


# ─── Main Loop ───────────────────────────────────────────────────
def run_task(llm_client: OpenAI, env_client: PolicyToLogicEnv, task_name: str) -> float:
    """Run a single task and return the score."""
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        result = env_client.reset(task_name=task_name)
        obs = result.observation.model_dump()
        max_steps = obs.get("max_steps", 7)

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Get agent action
            action_type, content = get_agent_action(llm_client, obs, step, history)

            # Step the environment
            action = PolicyToLogicAction(action_type=action_type, content=content)
            result = env_client.step(action)
            obs = result.observation.model_dump()

            reward = result.reward
            done = result.done
            error = result.info.get("error") if result.info else None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=f"{action_type}({content[:50]}...)" if len(content) > 50 else f"{action_type}({content})",
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {action_type} → accuracy={obs.get('current_accuracy', 0):.1%}, reward={reward:.2f}"
            )

            if done:
                break

        # Compute final score
        final_accuracy = obs.get("current_accuracy", 0.0)
        score = final_accuracy  # Clean score = accuracy
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    """Run inference against all tasks."""
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY not set. Please set one of these environment variables.", flush=True)
        sys.exit(1)

    # Initialize clients
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = PolicyToLogicEnv(base_url=ENV_BASE_URL)

    print(f"[INFO] Running inference with model={MODEL_NAME}", flush=True)
    print(f"[INFO] Environment: {ENV_BASE_URL}", flush=True)
    print(f"[INFO] Tasks: {TASKS}", flush=True)
    print("", flush=True)

    scores = {}

    for task_name in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)

        score = run_task(llm_client, env_client, task_name)
        scores[task_name] = score

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for task, score in scores.items():
        print(f"  {task}: {score:.2f}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  Average: {avg:.2f}", flush=True)

    env_client.close()


if __name__ == "__main__":
    main()
