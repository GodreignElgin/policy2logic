"""
Pydantic models for the Policy-to-Logic RL Environment.

Defines the typed Action, Observation, and State models
used across client and server for type-safe communication.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ─── Action ────────────────────────────────────────────────────────
class PolicyToLogicAction(BaseModel):
    """
    An action the agent can take in the environment.

    action_type:
        - "ask_clarification": Agent asks a question about the policy.
        - "propose_rules": Agent proposes a full rule set in DSL format.
        - "refine_rules": Agent submits a modified/improved rule set.

    content:
        JSON string containing the action payload:
        - For ask_clarification: {"question": "What are business hours?"}
        - For propose_rules:     {"rules": [...], "default": "ALLOW"}
        - For refine_rules:      {"rules": [...], "default": "ALLOW"}
    """

    action_type: Literal["ask_clarification", "propose_rules", "refine_rules"] = Field(
        ...,
        description="The type of action: ask_clarification, propose_rules, or refine_rules",
    )
    content: str = Field(
        ...,
        description="JSON string with the action payload",
    )


# ─── Observation ───────────────────────────────────────────────────
class PolicyToLogicObservation(BaseModel):
    """
    What the agent observes after each step.
    """

    policy_text: str = Field(
        ...,
        description="The natural language policy to convert into logic rules",
    )
    task_name: str = Field(
        ...,
        description="Name of the current task (data_access, resource_access, transaction_approval)",
    )
    step_number: int = Field(
        ...,
        description="Current step in the episode (1-indexed)",
    )
    max_steps: int = Field(
        ...,
        description="Maximum steps allowed in this episode",
    )
    clarification_response: Optional[str] = Field(
        default=None,
        description="Answer to a clarification question, if one was asked",
    )
    test_results: Optional[dict] = Field(
        default=None,
        description="Results of testing proposed rules: {passed, failed, total, details}",
    )
    current_accuracy: float = Field(
        default=0.0,
        description="Current accuracy of the agent's rules (0.0-1.0)",
    )
    available_actions: list[str] = Field(
        default_factory=lambda: ["ask_clarification", "propose_rules", "refine_rules"],
        description="Actions available to the agent in the current state",
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Human-readable feedback about the last action",
    )
    dsl_format: str = Field(
        default="",
        description="Description of the expected DSL rule format",
    )


# ─── State ─────────────────────────────────────────────────────────
class PolicyToLogicState(BaseModel):
    """
    Full episode state (server-side metadata).
    """

    episode_id: str = Field(
        ...,
        description="Unique identifier for the current episode",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken so far",
    )
    task_name: str = Field(
        ...,
        description="Name of the current task",
    )
    current_rules: Optional[list] = Field(
        default=None,
        description="The agent's current rule set (if any)",
    )
    accuracy_history: list[float] = Field(
        default_factory=list,
        description="Accuracy at each step where rules were evaluated",
    )
    questions_asked: int = Field(
        default=0,
        description="Number of clarification questions asked",
    )
    questions_log: list[str] = Field(
        default_factory=list,
        description="Log of all questions asked",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended",
    )
    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward for the episode",
    )


# ─── StepResult ────────────────────────────────────────────────────
class PolicyToLogicStepResult(BaseModel):
    """
    Result returned by step() and reset().
    """

    observation: PolicyToLogicObservation
    reward: float = Field(
        default=0.0,
        description="Reward for this step (0.0-1.0)",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode is complete",
    )
    info: dict = Field(
        default_factory=dict,
        description="Additional metadata about this step",
    )
