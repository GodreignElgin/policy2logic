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
        # ── Level 1: General (single short keyword → partial/ambiguous) ──
        "hours": (
            "Working hours are from 9 AM to 6 PM."
        ),
        "sensitive": (
            "Sensitive data includes personal records, financial data, "
            "and proprietary information."
        ),
        "internal": (
            "Internal data follows the same access rules as sensitive data."
        ),
        "public": (
            "Public data has no access restrictions and can be accessed "
            "at any time."
        ),
        "access": (
            "Access depends on the data type and the current hour. "
            "Public data is unrestricted. Other types have time-based rules."
        ),

        # ── Level 2: Medium specificity (common phrases → more detail) ──
        "working hours": (
            "Working hours are 9:00 AM to 6:00 PM (9 to 18 in 24-hour format). "
            "Sensitive and internal data can only be accessed during this window."
        ),
        "work hours": (
            "Working hours are 9:00 AM to 6:00 PM (9 to 18 in 24-hour format)."
        ),
        "after hours": (
            "After hours means outside the working hours window. "
            "This includes early morning hours and evening hours from 6 PM onward."
        ),

        # ── Level 3: Precise (compound/specific → ground truth aligned) ──
        "working hours boundary": (
            "Working hours use a half-open interval: hour >= 9 AND hour < 18. "
            "Hour 9 is the first working hour. Hour 17 is the last working hour. "
            "Hour 18 (6:00 PM) is NOT within working hours — it is the start of "
            "after-hours."
        ),
        "exactly 18": (
            "Hour 18 (6:00 PM) is considered after hours. The working hours "
            "window ends BEFORE 18, so 18 is outside. Access to sensitive "
            "and internal data is denied at hour 18."
        ),
        "time boundary": (
            "The time boundaries are strict: working hours are "
            "hour >= 9 AND hour < 18. Hour 9 is inside working hours, "
            "hour 18 is outside. The last valid working hour is 17."
        ),
        "sensitive time": (
            "Sensitive data can only be accessed when the hour is >= 9 AND "
            "strictly less than 18. At hour 18, access is denied."
        ),
        "internal time": (
            "Internal data follows the exact same time rules as sensitive "
            "data: allowed when hour >= 9 AND hour < 18."
        ),
        "deny allow": (
            "The decision is ALLOW for public data at any time, or for "
            "sensitive/internal data during hours 9 through 17 (inclusive). "
            "The decision is DENY for sensitive/internal data at hours 0-8 "
            "and 18-23."
        ),
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
        # ── Level 1: General (single keyword → partial/ambiguous truths) ──
        # NOTE: "junior" answer is technically true but intentionally
        # incomplete — it only mentions the "outside business hours"
        # restriction, which can mislead the agent into thinking
        # confidential IS accessible during business hours.
        "junior": (
            "Junior employees are entry-level staff. They can access public "
            "and internal documents during business hours, but not confidential "
            "documents outside business hours."
        ),
        "senior": (
            "Senior employees have unrestricted access to all documents "
            "at all times."
        ),
        "contractor": (
            "Contractors can only access public documents. They cannot access "
            "internal or confidential documents at any time."
        ),
        "confidential": (
            "Confidential documents include board minutes, salary data, and "
            "strategic plans. Access is highly restricted."
        ),
        "internal": (
            "Internal documents include team wikis, project plans, and "
            "internal communications."
        ),
        "public": (
            "Public documents include published reports, press releases, and "
            "public-facing content. No access restrictions."
        ),
        "hours": (
            "Business hours are 8 AM to 5 PM."
        ),

        # ── Level 2: Medium specificity (common phrases → more detail) ──
        "business hours": (
            "Business hours are 8:00 AM to 5:00 PM (8 to 17 in 24-hour format). "
            "Access permissions change based on whether the current hour falls "
            "within this range."
        ),
        "work hours": (
            "Business hours are 8:00 AM to 5:00 PM (8 to 17 in 24-hour format)."
        ),
        "outside business": (
            "Outside business hours includes early morning and evening. "
            "Restrictions are tighter outside this window for junior staff."
        ),

        # ── Level 3: Precise (compound keywords → ground truth aligned) ──
        # These answers reveal the FULL truth, correcting any misleading
        # impressions from Level 1 answers.
        "junior confidential": (
            "Junior employees CANNOT access confidential documents at ANY time, "
            "regardless of whether it is during or outside business hours. "
            "The policy statement about 'outside business hours' is a minimum "
            "restriction — the actual rule is a blanket denial of confidential "
            "access for juniors."
        ),
        "junior internal": (
            "Junior employees can access internal documents ONLY during business "
            "hours (hour >= 8 AND hour < 17). Outside business hours, internal "
            "documents are denied for juniors."
        ),
        "junior public": (
            "Junior employees can access public documents at any time. "
            "Public access has no restrictions for any role."
        ),
        "business hours boundary": (
            "Business hours use a half-open interval: hour >= 8 AND hour < 17. "
            "Hour 8 is the first business hour. Hour 16 is the last business hour. "
            "Hour 17 (5:00 PM) is NOT within business hours."
        ),
        "exactly 17": (
            "Hour 17 (5:00 PM) is considered outside business hours. "
            "The business hours window ends BEFORE 17. Junior employees lose "
            "access to internal documents at hour 17."
        ),
        "time boundary": (
            "Business hours are hour >= 8 AND hour < 17. "
            "Hour 8 is inside, hour 17 is outside. "
            "The last valid business hour is 16."
        ),
        "confidential during": (
            "Confidential documents are NOT accessible to junior employees "
            "during business hours. The policy only explicitly mentions the "
            "'outside business hours' restriction, but the actual rule denies "
            "junior access to confidential at all times."
        ),
        "contractor internal": (
            "Contractors cannot access internal documents. They are restricted "
            "to public documents only, regardless of time."
        ),
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
        # ── Level 1: General (single keyword → partial/vague) ──
        "limit": (
            "The standard transaction limit is $5,000."
        ),
        "international": (
            "All international transfers require compliance review, "
            "regardless of amount or time."
        ),
        "manager": (
            "Manager-initiated transactions are exempt from the standard "
            "limit requirement. They are auto-approved for domestic "
            "transactions unless other rules apply."
        ),
        "domestic": (
            "Domestic transactions follow different rules based on amount "
            "and time of day."
        ),
        "compliance": (
            "Compliance review is required for all international transfers. "
            "It is a separate process from manager approval."
        ),
        "routine": (
            "Routine transactions are domestic transactions within the "
            "standard limit."
        ),
        "system": (
            "System-initiated transactions follow the same rules as "
            "employee-initiated ones."
        ),
        "hours": (
            "Business hours are 9 AM to 5 PM."
        ),
        "exempt": (
            "Manager-initiated transactions are exempt from the standard "
            "limit. However, other rules may still apply."
        ),

        # ── Level 2: Medium specificity ──
        "standard limit": (
            "The standard transaction limit is $5,000. Transactions above "
            "this amount require manager approval unless the initiator is "
            "a manager."
        ),
        "high-value": (
            "High-value domestic transactions are those with an amount of "
            "$10,000 or more."
        ),
        "high value": (
            "High-value domestic transactions are those with an amount of "
            "$10,000 or more."
        ),
        "threshold": (
            "The standard limit is $5,000. The high-value threshold for "
            "domestic transactions is $10,000."
        ),
        "business hours": (
            "Business hours are 9:00 AM to 5:00 PM (9 to 17 in 24-hour format)."
        ),
        "work hours": (
            "Business hours are 9:00 AM to 5:00 PM (9 to 17 in 24-hour format)."
        ),
        "non-business": (
            "Non-business hours means outside the 9 AM to 5 PM window, "
            "including the 5 PM hour itself."
        ),

        # ── Level 3: Precise (compound keywords → ground truth) ──
        "exactly 5000": (
            "A transaction of exactly $5,000 is WITHIN the standard limit and "
            "is auto-approved for domestic transactions. Only amounts STRICTLY "
            "above $5,000 (i.e., $5,001+) trigger the approval requirement. "
            "The comparison is amount > 5000, not amount >= 5000."
        ),
        "exactly 10000": (
            "A domestic transaction of exactly $10,000 IS considered high-value. "
            "The threshold is amount >= 10000. A $9,999 transaction is NOT "
            "high-value."
        ),
        "exactly 17": (
            "Hour 17 (5:00 PM) is considered non-business hours. The business "
            "hours window ends BEFORE 17. A high-value domestic transaction at "
            "hour 17 would be held for review."
        ),
        "business hours boundary": (
            "Business hours use a half-open interval: hour >= 9 AND hour < 17. "
            "Hour 9 is business hours. Hour 16 is the last business hour. "
            "Hour 17 (5:00 PM) is NOT business hours."
        ),
        "time boundary": (
            "Business hours are hour >= 9 AND hour < 17. "
            "Hour 9 is inside, hour 17 is outside."
        ),
        "manager exempt": (
            "Manager-initiated transactions are exempt from the standard $5,000 "
            "limit only. They are NOT exempt from international compliance review "
            "or the high-value domestic HOLD rule. A manager's $10,000 domestic "
            "transaction outside business hours is still HELD."
        ),
        "manager high-value": (
            "Manager exemption only applies to the standard limit ($5,000). "
            "Managers are still subject to: (1) COMPLIANCE_REVIEW for international "
            "transfers, and (2) HOLD for high-value domestic transactions (>= $10,000) "
            "outside business hours. The exemption is narrow."
        ),
        "manager international": (
            "Even manager-initiated international transfers require COMPLIANCE_REVIEW. "
            "The manager exemption does not override the international transfer rule."
        ),
        "domestic hold": (
            "A domestic transaction is HELD when: (1) amount >= $10,000 AND "
            "(2) the hour is outside business hours (hour < 9 or hour >= 17). "
            "Both conditions must be true. During business hours, high-value "
            "domestic transactions get REQUIRE_APPROVAL instead (if not manager-initiated)."
        ),
        "rule priority": (
            "Rules are evaluated in priority order: "
            "(1) International transfers → COMPLIANCE_REVIEW always. "
            "(2) High-value domestic (>= $10,000) outside business hours → HOLD. "
            "(3) Above standard limit (> $5,000) and not manager → REQUIRE_APPROVAL. "
            "(4) Everything else → APPROVE."
        ),
        "international manager": (
            "International transfers ALWAYS go to COMPLIANCE_REVIEW, even for "
            "managers. This is the highest-priority rule."
        ),
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
