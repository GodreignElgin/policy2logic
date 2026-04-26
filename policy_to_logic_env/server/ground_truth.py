"""
Ground Truth Engine for the Policy-to-Logic RL Environment.

Provides:
  1. Programmatic ground truth functions for each task
  2. Clarification oracle — deterministic question → answer mapping

The ground truth is the source of correctness for scenario evaluation.
It is NEVER shown to the agent directly — only used for scoring.
"""

from .policies import get_task, TaskConfig


# ─── Ground Truth Functions ───────────────────────────────────────

def evaluate_ground_truth(task_name: str, scenario: dict) -> str:
    """
    Return the correct decision for a scenario in a given task.

    Args:
        task_name: One of "data_access", "resource_access", "transaction_approval"
        scenario: Dict of field → value pairs

    Returns:
        Correct decision string (e.g., "ALLOW", "DENY", "APPROVE", etc.)
    """
    if task_name == "data_access":
        return _ground_truth_data_access(scenario)
    elif task_name == "resource_access":
        return _ground_truth_resource_access(scenario)
    elif task_name == "transaction_approval":
        return _ground_truth_transaction_approval(scenario)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def _ground_truth_data_access(s: dict) -> str:
    """
    Ground truth for Task 1 (Easy): Data Access Control.

    Rules:
    - Public data: always ALLOW
    - Sensitive/internal data during working hours (9-18): ALLOW
    - Sensitive/internal data outside working hours: DENY
    """
    time = s.get("time", 12)
    data_type = s.get("data_type", "public")

    if data_type == "public":
        return "ALLOW"

    # sensitive or internal
    if 9 <= time < 18:
        return "ALLOW"
    else:
        return "DENY"


def _ground_truth_resource_access(s: dict) -> str:
    """
    Ground truth for Task 2 (Medium): Employee Resource Access.

    Rules (evaluated in priority order):
    1. Senior employees: always ALLOW
    2. Contractors: ALLOW only for public documents
    3. Junior employees during business hours (8-17): ALLOW for public + internal
    4. Junior employees outside business hours: ALLOW only for public
    """
    role = s.get("role", "junior")
    time = s.get("time", 12)
    doc_type = s.get("document_type", "public")

    # Senior: unrestricted
    if role == "senior":
        return "ALLOW"

    # Contractor: public only
    if role == "contractor":
        return "ALLOW" if doc_type == "public" else "DENY"

    # Junior employee
    is_business_hours = 8 <= time < 17

    if doc_type == "public":
        return "ALLOW"

    if is_business_hours:
        # During business hours: public + internal OK, confidential DENY
        if doc_type == "internal":
            return "ALLOW"
        else:  # confidential
            return "DENY"
    else:
        # Outside business hours: only public OK
        return "DENY"


def _ground_truth_transaction_approval(s: dict) -> str:
    """
    Ground truth for Task 3 (Hard): Transaction Approval Workflow.

    Rules (evaluated in priority order):
    1. International transfers → COMPLIANCE_REVIEW (always)
    2. High-value (≥$10k) domestic during non-business hours → HOLD
    3. Amount > $5,000 and not manager-initiated → REQUIRE_APPROVAL
    4. Everything else → APPROVE
    """
    amount = s.get("amount", 0)
    transfer_type = s.get("transfer_type", "domestic")
    time = s.get("time", 12)
    initiator_role = s.get("initiator_role", "employee")

    is_business_hours = 9 <= time < 17

    # Rule 1: International transfers always need compliance review
    if transfer_type == "international":
        return "COMPLIANCE_REVIEW"

    # Rule 2: High-value domestic transactions outside business hours → HOLD
    if amount >= 10000 and not is_business_hours:
        return "HOLD"

    # Rule 3: Above standard limit and not manager → needs approval
    if amount > 5000 and initiator_role != "manager":
        return "REQUIRE_APPROVAL"

    # Rule 4: Everything else is approved
    return "APPROVE"


# ─── Clarification Oracle ────────────────────────────────────────

def answer_clarification(task_name: str, question: str) -> str:
    """
    Deterministic clarification oracle with progressive revelation.

    Uses compound keyword matching to provide layered answers:
      - Vague questions (match short keywords) → partial, potentially
        ambiguous truths that may mislead if taken at face value.
      - Specific questions (match long/compound keywords) → precise,
        ground-truth-aligned answers.

    Compound keywords: if a keyword contains spaces, ALL space-separated
    words must appear anywhere in the question (order-independent).
    More matched keywords = higher priority (more specific answer wins).

    This design supports RL training where agents must learn to:
      1. Detect ambiguity in initial policy text
      2. Ask targeted questions to resolve ambiguity
      3. Recognize when earlier (vague) answers were misleading
      4. Reconcile contradictory signals by drilling deeper

    Args:
        task_name: Current task name
        question: The agent's clarification question (free text)

    Returns:
        Answer string
    """
    task = get_task(task_name)
    question_lower = question.lower().strip()

    best_match = None
    best_match_score = (0, 0)  # (num_parts, total_length)

    for keyword, answer in task.clarification_map.items():
        keyword_lower = keyword.lower()
        keyword_parts = keyword_lower.split()

        # ALL parts of the keyword must appear in the question
        if all(part in question_lower for part in keyword_parts):
            # Score: more keyword parts = more specific = higher priority
            # Tiebreak by total keyword length
            score = (len(keyword_parts), len(keyword_lower))
            if score > best_match_score:
                best_match = answer
                best_match_score = score

    if best_match:
        return best_match

    return (
        "I can provide information about the specific terms and parameters "
        "mentioned in the policy. Try asking about specific aspects like "
        "time boundaries, exact thresholds, role-specific permissions, "
        "or how specific edge cases are handled."
    )
