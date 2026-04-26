"""
Test script for Policy-to-Logic RL Environment API endpoints.

Run the server first (in another terminal):
    uv run python main.py

Then run this script:
    uv run python test_endpoints.py
"""

import requests
import json
from typing import Any

BASE_URL = "http://localhost:7860"

def test_endpoint(method: str, endpoint: str, data: dict = None, description: str = "") -> Any:
    """Test an API endpoint and print results."""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*70}")
    print(f"🧪 {description or endpoint}")
    print(f"{'='*70}")
    print(f"{method} {url}")
    
    try:
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
        
        print(f"Status: {response.status_code}")
        
        try:
            result = response.json()
            print(f"Response:\n{json.dumps(result, indent=2)}")
            return result
        except:
            print(f"Response (text):\n{response.text}")
            return response.text
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error! Is the server running on {BASE_URL}?")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("\n🚀 Policy-to-Logic RL Environment - API Test Suite\n")
    
    # 1. Health check
    test_endpoint("GET", "/health", description="Health Check")
    
    # 2. List available tasks
    test_endpoint("GET", "/tasks", description="List Available Tasks")
    
    # 3. Reset environment (start new episode)
    reset_result = test_endpoint(
        "POST", 
        "/reset", 
        data={"task_name": None},
        description="Reset Environment (Start New Episode)"
    )
    
    # 4. Get current state
    test_endpoint("GET", "/state", description="Get Current State")
    
    # 5. Take a step - ask clarification
    if reset_result:
        step_result = test_endpoint(
            "POST",
            "/step",
            data={
                "action_type": "ask_clarification",
                "content": "What are the business hours?"
            },
            description="Step 1: Ask Clarification"
        )
    
    # 6. Get state after step
    test_endpoint("GET", "/state", description="Get State After Step")
    
    # 7. Take another step - propose rules
    if reset_result:
        test_endpoint(
            "POST",
            "/step",
            data={
                "action_type": "propose_rules",
                "content": {
                    "rules": [
                        {
                            "condition": "user.role == 'admin'",
                            "action": "ALLOW"
                        }
                    ]
                }
            },
            description="Step 2: Propose Rules"
        )
    
    # 8. Get final state
    test_endpoint("GET", "/state", description="Get Final State")
    
    print(f"\n{'='*70}")
    print("✅ Test suite completed!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
