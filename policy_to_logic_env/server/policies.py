"""
Policy Definitions for the Policy-to-Logic RL Environment.

Defines 3 tasks of increasing difficulty within the access control domain:
  1. data_access (Easy)     — Single policy, no ambiguity
  2. resource_access (Medium) — Multiple roles, one hidden parameter
  3. transaction_approval (Hard) — Multiple overlapping rules, several hidden params

Each task includes:
  - policy_text: What the agent sees
  - variables: What fields exist in scenarios
  - valid_decisions: What outcomes are possible
  - hidden_params: Ground truth values not visible to the agent
  - clarification_map: Deterministic oracle responses
  - max_steps: Episode length budget
"""

from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    """Configuration for a single task."""
    name: str
    difficulty: str  # "easy", "medium", "hard"
    policy_text: str
    variables: dict[str, list]          # field_name → possible values
    valid_decisions: list[str]
    hidden_params: dict[str, any]
    clarification_map: dict[str, str]   # question_keyword → answer
    max_steps: int
    scenario_count: int


# ─── Task 1: Data Access (Easy) ──────────────────────────────────
DATA_ACCESS = TaskConfig(
    name="data_access",
    difficulty="easy",
    policy_text=(
        "Company Policy: Data Access Control\n"
        "---\n"
        "Employees must not access sensitive data after working hours.\n"
        "Working hours are from 9 AM to 6 PM (9:00 to 18:00).\n"
        "Public data can be accessed at any time.\n"
        "Internal data follows the same rules as sensitive data."
    ),
    variables={
        "time": list(range(0, 24)),       # 0-23 (hour of day)
        "data_type": ["sensitive", "public", "internal"],
    },
    valid_decisions=["ALLOW", "DENY"],
    hidden_params={
        "work_start": 9,
        "work_end": 18,
    },
    clarification_map={
        "working hours": "Working hours are 9:00 AM to 6:00 PM (9 to 18 in 24-hour format).",
        "work hours": "Working hours are 9:00 AM to 6:00 PM (9 to 18 in 24-hour format).",
        "sensitive": "Sensitive data includes personal records, financial data, and proprietary information.",
        "internal": "Internal data follows the same access rules as sensitive data.",
        "public": "Public data has no access restrictions and can be accessed at any time.",
        "after hours": "After hours means any time before 9:00 AM or after 6:00 PM (before 9 or after 18).",
    },
    max_steps=5,
    scenario_count=30,
)


# ─── Task 2: Resource Access (Medium) ────────────────────────────
RESOURCE_ACCESS = TaskConfig(
    name="resource_access",
    difficulty="medium",
    policy_text=(
        "Company Policy: Employee Resource Access\n"
        "---\n"
        "Junior employees cannot access confidential documents outside business hours.\n"
        "Senior employees have unrestricted access to all document types.\n"
        "Contractors can only access public documents, regardless of time.\n"
        "During business hours, junior employees may access public and internal documents."
    ),
    variables={
        "role": ["junior", "senior", "contractor"],
        "time": list(range(0, 24)),
        "document_type": ["public", "internal", "confidential"],
    },
    valid_decisions=["ALLOW", "DENY"],
    hidden_params={
        "business_start": 8,
        "business_end": 17,
    },
    clarification_map={
        "business hours": "Business hours are 8:00 AM to 5:00 PM (8 to 17 in 24-hour format).",
        "work hours": "Business hours are 8:00 AM to 5:00 PM (8 to 17 in 24-hour format).",
        "junior": "Junior employees are entry-level staff. They can access public and internal documents during business hours, but not confidential documents outside business hours.",
        "senior": "Senior employees have unrestricted access to all documents at all times.",
        "contractor": "Contractors can only access public documents. They cannot access internal or confidential documents at any time.",
        "confidential": "Confidential documents include board minutes, salary data, and strategic plans.",
        "internal": "Internal documents include team wikis, project plans, and internal communications.",
        "public": "Public documents include published reports, press releases, and public-facing content.",
        "outside business": "Outside business hours means before 8:00 AM or after 5:00 PM.",
    },
    max_steps=7,
    scenario_count=50,
)


# ─── Task 3: Transaction Approval (Hard) ─────────────────────────
TRANSACTION_APPROVAL = TaskConfig(
    name="transaction_approval",
    difficulty="hard",
    policy_text=(
        "Company Policy: Transaction Approval Workflow\n"
        "---\n"
        "Transactions exceeding the standard limit require manager approval.\n"
        "International transfers always need compliance review regardless of amount.\n"
        "High-value domestic transactions during non-business hours are automatically\n"
        "held for review.\n"
        "Routine domestic transactions within limits are auto-approved.\n"
        "Manager-initiated transactions are exempt from the standard limit."
    ),
    variables={
        "amount": [100, 500, 1000, 2500, 4000, 5000, 5001, 7500, 10000, 15000, 25000, 50000],
        "transfer_type": ["domestic", "international"],
        "time": list(range(0, 24)),
        "initiator_role": ["employee", "manager", "system"],
    },
    valid_decisions=["APPROVE", "REQUIRE_APPROVAL", "COMPLIANCE_REVIEW", "HOLD"],
    hidden_params={
        "standard_limit": 5000,
        "high_value_threshold": 10000,
        "business_start": 9,
        "business_end": 17,
    },
    clarification_map={
        "standard limit": "The standard transaction limit is $5,000. Transactions above this amount require manager approval.",
        "limit": "The standard transaction limit is $5,000.",
        "threshold": "The standard limit is $5,000. The high-value threshold for domestic transactions is $10,000.",
        "high-value": "High-value domestic transactions are those with an amount of $10,000 or more.",
        "high value": "High-value domestic transactions are those with an amount of $10,000 or more.",
        "business hours": "Business hours are 9:00 AM to 5:00 PM (9 to 17 in 24-hour format).",
        "work hours": "Business hours are 9:00 AM to 5:00 PM (9 to 17 in 24-hour format).",
        "international": "All international transfers require compliance review, regardless of amount or time.",
        "manager": "Manager-initiated transactions are exempt from the standard limit requirement. They are auto-approved for domestic transactions unless high-value and outside business hours.",
        "compliance": "Compliance review is required for all international transfers. It is a separate process from manager approval.",
        "routine": "Routine transactions are domestic transactions within the standard limit ($5,000 or less).",
        "exempt": "Manager-initiated transactions are exempt from the standard limit. However, they still follow high-value and international rules.",
        "non-business": "Non-business hours means before 9:00 AM or after 5:00 PM.",
        "system": "System-initiated transactions follow the same rules as employee-initiated ones.",
        "domestic": "Domestic transactions that are within limits are auto-approved. High-value ones outside business hours are held.",
    },
    max_steps=7,
    scenario_count=80,
)


# ─── Task Registry ───────────────────────────────────────────────
TASK_REGISTRY: dict[str, TaskConfig] = {
    "data_access": DATA_ACCESS,
    "resource_access": RESOURCE_ACCESS,
    "transaction_approval": TRANSACTION_APPROVAL,
}

TASK_NAMES = list(TASK_REGISTRY.keys())


def get_task(task_name: str) -> TaskConfig:
    """Get a task configuration by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name}. Available: {TASK_NAMES}"
        )
    return TASK_REGISTRY[task_name]
