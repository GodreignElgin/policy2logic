"""
Core Environment for the Policy-to-Logic RL Environment.

Implements the OpenEnv interface:
  - reset(task_name) → StepResult with initial observation
  - step(action) → StepResult with observation, reward, done
  - state() → current episode state

Episode lifecycle:
  1. reset() selects a task, generates scenarios, returns policy text
  2. Agent sends actions (max N steps depending on task difficulty)
  3. Each step() processes action, evaluates rules if proposed, returns feedback
  4. Episode ends when accuracy >= 0.9 OR max steps reached
"""

import json
import uuid
from typing import Optional

from ..models import (
    PolicyToLogicAction,
    PolicyToLogicObservation,
    PolicyToLogicState,
    PolicyToLogicStepResult,
)
from .policies import get_task, TASK_NAMES, TaskConfig
from .scenario_generator import generate_scenarios
from .ground_truth import answer_clarification
from .graders import quick_grade, grade_task
from .rewards import compute_reward, compute_episode_score
from .dsl_engine import parse_rules, validate_rules


# ─── DSL Format Description (shown to agent) ─────────────────────
DSL_FORMAT_DESCRIPTION = """
Rules must be in JSON format with the following structure:
{
    "rules": [
        {
            "if": [
                {"field": "<field_name>", "op": "<operator>", "value": <value>}
            ],
            "then": "<DECISION>"
        }
    ],
    "default": "<DEFAULT_DECISION>"
}

Operators: >, <, >=, <=, ==, !=
All conditions within a rule are AND-ed.
Rules are evaluated top-to-bottom; first matching rule wins.
If no rules match, the default decision is used.
""".strip()


class PolicyToLogicEnvironment:
    """
    The Policy-to-Logic RL environment server.

    Manages episode state and processes agent interactions.
    """

    def __init__(self):
        self._state: Optional[PolicyToLogicState] = None
        self._task: Optional[TaskConfig] = None
        self._scenarios: list[dict] = []
        self._current_rules: Optional[dict] = None
        self._previous_accuracy: float = 0.0
        self._episode_rewards: list[float] = []

    # ── reset() ───────────────────────────────────────────────────
    def reset(self, task_name: Optional[str] = None) -> PolicyToLogicStepResult:
        """
        Initialize a new episode.

        Args:
            task_name: Task to load. If None, defaults to "data_access".

        Returns:
            StepResult with the initial observation.
        """
        if task_name is None:
            task_name = "data_access"

        if task_name not in TASK_NAMES:
            task_name = "data_access"

        self._task = get_task(task_name)
        self._scenarios = generate_scenarios(task_name)
        self._current_rules = None
        self._previous_accuracy = 0.0
        self._episode_rewards = []

        episode_id = str(uuid.uuid4())[:8]

        self._state = PolicyToLogicState(
            episode_id=episode_id,
            step_count=0,
            task_name=task_name,
            current_rules=None,
            accuracy_history=[],
            questions_asked=0,
            questions_log=[],
            done=False,
            total_reward=0.0,
        )

        # Build available decisions info for the observation
        decisions_info = ", ".join(self._task.valid_decisions)
        variables_info = ", ".join(
            f"{field} ({type(values[0]).__name__})"
            for field, values in self._task.variables.items()
        )

        observation = PolicyToLogicObservation(
            policy_text=self._task.policy_text,
            task_name=task_name,
            step_number=0,
            max_steps=self._task.max_steps,
            clarification_response=None,
            test_results=None,
            current_accuracy=0.0,
            available_actions=["ask_clarification", "propose_rules"],
            feedback=(
                f"New episode started. Task: {task_name} ({self._task.difficulty}).\n"
                f"Available decisions: {decisions_info}\n"
                f"Scenario variables: {variables_info}\n"
                f"You have {self._task.max_steps} steps. "
                f"Read the policy and propose rules or ask clarification questions."
            ),
            dsl_format=DSL_FORMAT_DESCRIPTION,
        )

        return PolicyToLogicStepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info={"episode_id": episode_id, "task": task_name},
        )

    # ── step() ────────────────────────────────────────────────────
    def step(self, action: PolicyToLogicAction) -> PolicyToLogicStepResult:
        """
        Process an agent action and return the result.

        Args:
            action: The agent's action (ask_clarification, propose_rules, or refine_rules)

        Returns:
            StepResult with observation, reward, and done flag.
        """
        if self._state is None or self._state.done:
            return self._error_result("Episode not started. Call reset() first.")

        if self._task is None:
            return self._error_result("No task loaded.")

        # Increment step
        self._state.step_count += 1
        step_num = self._state.step_count

        # Process action by type
        if action.action_type == "ask_clarification":
            return self._handle_clarification(action, step_num)
        elif action.action_type == "propose_rules":
            return self._handle_propose(action, step_num)
        elif action.action_type == "refine_rules":
            return self._handle_refine(action, step_num)
        else:
            return self._error_result(f"Unknown action type: {action.action_type}")

    # ── state() ───────────────────────────────────────────────────
    def state(self) -> PolicyToLogicState:
        """Return current episode state."""
        if self._state is None:
            return PolicyToLogicState(
                episode_id="none",
                task_name="none",
                done=True,
            )
        return self._state

    # ── Action Handlers ───────────────────────────────────────────

    def _handle_clarification(
        self, action: PolicyToLogicAction, step_num: int
    ) -> PolicyToLogicStepResult:
        """Handle a clarification question."""
        try:
            content = json.loads(action.content)
            question = content.get("question", action.content)
        except (json.JSONDecodeError, AttributeError):
            question = action.content

        # Get oracle answer
        answer = answer_clarification(self._task.name, question)
        self._state.questions_asked += 1
        self._state.questions_log.append(question)

        # Determine if the question was useful
        # A question is "useful" if the oracle returns a specific answer
        # (not the generic fallback)
        is_useful = "I can provide information" not in answer

        # Compute reward
        reward, reward_breakdown = compute_reward(
            current_accuracy=self._previous_accuracy,
            previous_accuracy=self._previous_accuracy,
            step_number=step_num,
            max_steps=self._task.max_steps,
            action_type="ask_clarification",
            clarification_was_useful=is_useful,
            total_questions_asked=self._state.questions_asked,
        )

        self._episode_rewards.append(reward)
        self._state.total_reward += reward

        # Check if episode should end
        done = step_num >= self._task.max_steps
        self._state.done = done

        # Build available actions
        available = ["ask_clarification", "propose_rules"]
        if self._current_rules is not None:
            available.append("refine_rules")

        observation = PolicyToLogicObservation(
            policy_text=self._task.policy_text,
            task_name=self._task.name,
            step_number=step_num,
            max_steps=self._task.max_steps,
            clarification_response=answer,
            test_results=None,
            current_accuracy=self._previous_accuracy,
            available_actions=available,
            feedback=(
                f"Clarification received. "
                f"{'This information should help refine your rules.' if is_useful else 'Try asking about specific terms in the policy.'} "
                f"Steps remaining: {self._task.max_steps - step_num}"
            ),
            dsl_format=DSL_FORMAT_DESCRIPTION,
        )

        return PolicyToLogicStepResult(
            observation=observation,
            reward=reward,
            done=done,
            info={
                "action_type": "ask_clarification",
                "question": question,
                "answer_useful": is_useful,
                "reward_breakdown": reward_breakdown,
            },
        )

    def _handle_propose(
        self, action: PolicyToLogicAction, step_num: int
    ) -> PolicyToLogicStepResult:
        """Handle a rule proposal."""
        return self._process_rules(action, step_num, is_refinement=False)

    def _handle_refine(
        self, action: PolicyToLogicAction, step_num: int
    ) -> PolicyToLogicStepResult:
        """Handle a rule refinement."""
        if self._current_rules is None:
            return self._make_result(
                step_num=step_num,
                feedback="Cannot refine rules — no rules have been proposed yet. Use 'propose_rules' first.",
                reward=0.0,
                done=step_num >= self._task.max_steps,
            )
        return self._process_rules(action, step_num, is_refinement=True)

    def _process_rules(
        self, action: PolicyToLogicAction, step_num: int, is_refinement: bool
    ) -> PolicyToLogicStepResult:
        """Common processing for propose and refine actions."""
        # Parse the rules
        rules_data, parse_errors = parse_rules(action.content)

        if rules_data is None:
            # Invalid DSL
            reward, reward_breakdown = compute_reward(
                current_accuracy=self._previous_accuracy,
                previous_accuracy=self._previous_accuracy,
                step_number=step_num,
                max_steps=self._task.max_steps,
                action_type=action.action_type,
                rules_were_valid=False,
            )

            self._episode_rewards.append(reward)
            self._state.total_reward += reward
            done = step_num >= self._task.max_steps
            self._state.done = done

            return self._make_result(
                step_num=step_num,
                feedback=f"Invalid rule format. Errors: {'; '.join(parse_errors)}. Please fix and try again.",
                reward=reward,
                done=done,
                info={"validation_errors": parse_errors, "reward_breakdown": reward_breakdown},
            )

        # Update current rules
        self._current_rules = rules_data
        self._state.current_rules = rules_data.get("rules", [])

        # Grade the rules
        accuracy, test_details = grade_task(
            self._task.name, rules_data, self._scenarios
        )

        # Compute reward
        reward, reward_breakdown = compute_reward(
            current_accuracy=accuracy,
            previous_accuracy=self._previous_accuracy,
            step_number=step_num,
            max_steps=self._task.max_steps,
            action_type=action.action_type,
            rules_were_valid=True,
        )

        # Update state
        self._state.accuracy_history.append(accuracy)
        self._episode_rewards.append(reward)
        self._state.total_reward += reward

        prev = self._previous_accuracy
        self._previous_accuracy = accuracy

        # Check termination
        done = accuracy >= 0.9 or step_num >= self._task.max_steps
        self._state.done = done

        # Build feedback
        delta = accuracy - prev
        if delta > 0:
            direction = f"improved by {delta:.1%}"
        elif delta < 0:
            direction = f"decreased by {abs(delta):.1%}"
        else:
            direction = "unchanged"

        action_word = "Refinement" if is_refinement else "Proposal"

        feedback = (
            f"{action_word} evaluated. Accuracy: {accuracy:.1%} ({direction}). "
            f"Passed {test_details['passed']}/{test_details['total']} scenarios."
        )

        if accuracy >= 0.9:
            feedback += " ✅ Target accuracy reached! Episode complete."
        elif test_details.get("sample_failures"):
            # Show a sample failure to guide improvement
            fail = test_details["sample_failures"][0]
            feedback += (
                f" Example failure: scenario {fail['scenario']} "
                f"expected {fail['expected']}, got {fail['got']}."
            )

        if done and accuracy < 0.9:
            feedback += f" Episode ended (max steps reached). Final accuracy: {accuracy:.1%}."

        observation = PolicyToLogicObservation(
            policy_text=self._task.policy_text,
            task_name=self._task.name,
            step_number=step_num,
            max_steps=self._task.max_steps,
            clarification_response=None,
            test_results=test_details,
            current_accuracy=accuracy,
            available_actions=["ask_clarification", "propose_rules", "refine_rules"],
            feedback=feedback,
            dsl_format=DSL_FORMAT_DESCRIPTION,
        )

        info = {
            "action_type": action.action_type,
            "accuracy": accuracy,
            "previous_accuracy": prev,
            "improvement": delta,
            "reward_breakdown": reward_breakdown,
        }

        if done:
            episode_score = compute_episode_score(
                final_accuracy=accuracy,
                total_steps=step_num,
                max_steps=self._task.max_steps,
                total_questions=self._state.questions_asked,
            )
            info["episode_score"] = episode_score
            info["episode_rewards"] = self._episode_rewards

        return PolicyToLogicStepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _make_result(
        self,
        step_num: int,
        feedback: str,
        reward: float = 0.0,
        done: bool = False,
        info: dict = None,
    ) -> PolicyToLogicStepResult:
        """Build a StepResult with current state."""
        available = ["ask_clarification", "propose_rules"]
        if self._current_rules is not None:
            available.append("refine_rules")

        observation = PolicyToLogicObservation(
            policy_text=self._task.policy_text if self._task else "",
            task_name=self._task.name if self._task else "",
            step_number=step_num,
            max_steps=self._task.max_steps if self._task else 0,
            clarification_response=None,
            test_results=None,
            current_accuracy=self._previous_accuracy,
            available_actions=available,
            feedback=feedback,
            dsl_format=DSL_FORMAT_DESCRIPTION,
        )

        return PolicyToLogicStepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info or {},
        )

    def _error_result(self, message: str) -> PolicyToLogicStepResult:
        """Build an error StepResult."""
        return PolicyToLogicStepResult(
            observation=PolicyToLogicObservation(
                policy_text="",
                task_name="",
                step_number=0,
                max_steps=0,
                feedback=f"Error: {message}",
                dsl_format=DSL_FORMAT_DESCRIPTION,
            ),
            reward=0.0,
            done=True,
            info={"error": message},
        )
