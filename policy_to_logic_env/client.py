"""
Client for the Policy-to-Logic RL Environment.

Provides a typed Python client for interacting with the environment
server via HTTP requests.
"""

import requests
from typing import Optional

from .models import (
    PolicyToLogicAction,
    PolicyToLogicObservation,
    PolicyToLogicState,
    PolicyToLogicStepResult,
)


class PolicyToLogicEnv:
    """
    HTTP client for the Policy-to-Logic RL environment.

    Usage:
        env = PolicyToLogicEnv(base_url="http://localhost:7860")
        result = env.reset(task_name="data_access")
        result = env.step(PolicyToLogicAction(
            action_type="propose_rules",
            content='{"rules": [...], "default": "ALLOW"}'
        ))
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_name: Optional[str] = None) -> PolicyToLogicStepResult:
        """Reset the environment and start a new episode."""
        payload = {}
        if task_name:
            payload["task_name"] = task_name

        response = self.session.post(
            f"{self.base_url}/reset",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return PolicyToLogicStepResult(**data)

    def step(self, action: PolicyToLogicAction) -> PolicyToLogicStepResult:
        """Take an action in the environment."""
        response = self.session.post(
            f"{self.base_url}/step",
            json={
                "action_type": action.action_type,
                "content": action.content,
            },
        )
        response.raise_for_status()
        data = response.json()
        return PolicyToLogicStepResult(**data)

    def state(self) -> PolicyToLogicState:
        """Get the current episode state."""
        response = self.session.get(f"{self.base_url}/state")
        response.raise_for_status()
        data = response.json()
        return PolicyToLogicState(**data)

    def health(self) -> dict:
        """Check environment health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_tasks(self) -> dict:
        """List available tasks."""
        response = self.session.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
