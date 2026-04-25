"""
Quick-start entry point for the Policy-to-Logic RL Environment.

Run the server:
    uv run python main.py

Or equivalently:
    uv run uvicorn policy_to_logic_env.server.app:app --host 0.0.0.0 --port 7860
"""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "policy_to_logic_env.server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
    )
