"""
FastAPI Application for the Policy-to-Logic RL Environment.

Exposes the environment via HTTP endpoints:
  POST /reset    — Start a new episode
  POST /step     — Take an action
  GET  /state    — Get current state
  GET  /health   — Health check
  GET  /tasks    — List available tasks
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .environment import PolicyToLogicEnvironment
from ..models import PolicyToLogicAction


# ─── App Configuration ────────────────────────────────────────────
app = FastAPI(
    title="Policy-to-Logic RL Environment",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to convert "
        "natural language policies into executable logic rules through "
        "iterative interaction and feedback."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Environment Instance ────────────────────────────────────────
env = PolicyToLogicEnvironment()


# ─── Request/Response Models ─────────────────────────────────────
class ResetRequest(BaseModel):
    task_name: Optional[str] = None


class StepRequest(BaseModel):
    action_type: str
    content: str


# ─── Endpoints ────────────────────────────────────────────────────

@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and start a new episode.

    Optionally specify a task_name: data_access, resource_access, or transaction_approval.
    """
    try:
        result = env.reset(task_name=request.task_name)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """
    Take an action in the environment.

    action_type: "ask_clarification", "propose_rules", or "refine_rules"
    content: JSON string with the action payload
    """
    try:
        action = PolicyToLogicAction(
            action_type=request.action_type,
            content=request.content,
        )
        result = env.step(action)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Get the current episode state."""
    try:
        current_state = env.state()
        return current_state.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "environment": "policy_to_logic"}


@app.get("/tasks")
async def list_tasks():
    """List available tasks with descriptions."""
    from .policies import TASK_REGISTRY
    tasks = {}
    for name, config in TASK_REGISTRY.items():
        tasks[name] = {
            "difficulty": config.difficulty,
            "max_steps": config.max_steps,
            "scenario_count": config.scenario_count,
            "valid_decisions": config.valid_decisions,
            "variables": list(config.variables.keys()),
        }
    return {"tasks": tasks}


# ─── Startup ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
