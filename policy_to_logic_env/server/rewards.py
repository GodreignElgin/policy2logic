"""
Reward System for the Policy-to-Logic RL Environment.

Multi-component reward providing dense signal across the full trajectory:

  1. Accuracy       (w=0.50) — % of test scenarios the agent's rules handle correctly
  2. Improvement    (w=0.20) — Delta from previous accuracy (bonus for progress)
  3. Efficiency     (w=0.15) — Penalty for excess steps
  4. Clarification  (w=0.15) — Quality of questions asked

Rewards are clamped to [0.0, 1.0] per step.
"""

from typing import Optional


# ─── Reward Weights (tunable hyperparameters) ─────────────────────
W_ACCURACY = 0.50
W_IMPROVEMENT = 0.20
W_EFFICIENCY = 0.15
W_CLARIFICATION = 0.15

# ─── Constants ────────────────────────────────────────────────────
STEP_PENALTY_RATE = 0.02       # penalty per step
MAX_USEFUL_QUESTIONS = 3       # diminishing returns after this
USELESS_QUESTION_PENALTY = 0.05


def compute_reward(
    current_accuracy: float,
    previous_accuracy: float,
    step_number: int,
    max_steps: int,
    action_type: str,
    clarification_was_useful: bool = False,
    total_questions_asked: int = 0,
    rules_were_valid: bool = True,
) -> tuple[float, dict]:
    """
    Compute the reward for a single step.

    Args:
        current_accuracy: Accuracy after this step (0.0-1.0)
        previous_accuracy: Accuracy before this step (0.0-1.0)
        step_number: Current step (1-indexed)
        max_steps: Max steps for the episode
        action_type: "ask_clarification", "propose_rules", or "refine_rules"
        clarification_was_useful: Whether a clarification led to information gain
        total_questions_asked: Total clarification questions asked so far
        rules_were_valid: Whether proposed/refined rules were valid DSL

    Returns:
        (total_reward, component_breakdown)
    """
    components = {}

    # ── 1. Accuracy component ─────────────────────────────────────
    components["accuracy"] = current_accuracy * W_ACCURACY

    # ── 2. Improvement component ──────────────────────────────────
    delta = current_accuracy - previous_accuracy
    if delta > 0:
        # Reward improvement, scaled by magnitude
        improvement_score = min(delta * 2.0, 1.0)  # cap at 1.0
    elif delta < 0:
        # Penalize regression
        improvement_score = max(delta * 1.5, -0.5)  # floor at -0.5
    else:
        improvement_score = 0.0

    components["improvement"] = improvement_score * W_IMPROVEMENT

    # ── 3. Efficiency component ───────────────────────────────────
    # Small penalty per step to encourage efficient solutions
    step_penalty = -STEP_PENALTY_RATE * step_number
    # Bonus for finishing early
    if current_accuracy >= 0.9:
        steps_saved = max_steps - step_number
        step_penalty += 0.05 * steps_saved

    components["efficiency"] = max(step_penalty, -0.15) * W_EFFICIENCY

    # ── 4. Clarification quality component ────────────────────────
    clarification_score = 0.0

    if action_type == "ask_clarification":
        if clarification_was_useful:
            # Useful question, but diminishing returns
            if total_questions_asked <= MAX_USEFUL_QUESTIONS:
                clarification_score = 0.3
            else:
                clarification_score = 0.1  # diminishing
        else:
            # Useless question penalty
            clarification_score = -USELESS_QUESTION_PENALTY
    elif action_type in ("propose_rules", "refine_rules"):
        if not rules_were_valid:
            clarification_score = -0.1  # penalty for invalid DSL

    components["clarification"] = clarification_score * W_CLARIFICATION

    # ── Total reward ──────────────────────────────────────────────
    total = sum(components.values())
    total = max(0.0, min(1.0, total))  # clamp to [0.0, 1.0]

    components["total"] = total
    return total, components


def compute_episode_score(
    final_accuracy: float,
    total_steps: int,
    max_steps: int,
    total_questions: int,
) -> float:
    """
    Compute the final episode score (used for grading).

    This is a clean 0.0-1.0 score based primarily on final accuracy
    with small bonuses for efficiency.

    Args:
        final_accuracy: Final accuracy of the agent's rules
        total_steps: Steps taken in the episode
        max_steps: Maximum allowed steps
        total_questions: Number of clarification questions asked

    Returns:
        Episode score in [0.0, 1.0]
    """
    # Primary: accuracy (80% weight)
    score = final_accuracy * 0.80

    # Efficiency bonus (10% weight)
    efficiency = 1.0 - (total_steps / max_steps)
    score += max(0.0, efficiency) * 0.10

    # Question efficiency bonus (10% weight)
    if total_questions <= 2:
        q_bonus = 1.0
    elif total_questions <= 4:
        q_bonus = 0.5
    else:
        q_bonus = 0.0
    score += q_bonus * 0.10

    return max(0.0, min(1.0, score))
