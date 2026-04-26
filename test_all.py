"""
Automated test runner for Policy-to-Logic RL Environment.

This script:
1. Starts the server in the background
2. Waits for it to be healthy
3. Runs all endpoint tests
4. Generates a detailed report
5. Cleans up (stops the server)

Run it:
    uv run python test_all.py
"""

import subprocess
import time
import requests
import json
import sys
from typing import Any, Dict, List
from datetime import datetime

BASE_URL = "http://localhost:7860"
MAX_RETRIES = 30
RETRY_DELAY = 1

class TestRunner:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.server_process = None
        self.passed = 0
        self.failed = 0
        self.start_time = datetime.now()
        
    def log(self, message: str, level: str = "INFO"):
        """Print formatted log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️ ",
            "TEST": "🧪"
        }
        print(f"[{timestamp}] {prefix.get(level, '')} {message}")
    
    def start_server(self) -> bool:
        """Start the FastAPI server in background."""
        self.log("Starting FastAPI server...", "INFO")
        try:
            self.server_process = subprocess.Popen(
                ["uv", "run", "python", "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.log(f"Server process started (PID: {self.server_process.pid})", "SUCCESS")
            return True
        except Exception as e:
            self.log(f"Failed to start server: {e}", "ERROR")
            return False
    
    def wait_for_server(self) -> bool:
        """Wait for server to be ready."""
        self.log(f"Waiting for server to be ready (max {MAX_RETRIES} attempts)...", "INFO")
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=2)
                if response.status_code == 200:
                    self.log("Server is ready!", "SUCCESS")
                    return True
            except requests.exceptions.ConnectionError:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        
        self.log(f"Server failed to start after {MAX_RETRIES} attempts", "ERROR")
        return False
    
    def test_endpoint(self, method: str, endpoint: str, data: dict = None, 
                      description: str = "") -> bool:
        """Test an API endpoint and record result."""
        url = f"{BASE_URL}{endpoint}"
        display_name = description or endpoint
        self.log(f"Testing: {display_name}", "TEST")
        
        try:
            if method == "POST":
                response = requests.post(url, json=data, timeout=5)
            else:
                response = requests.get(url, timeout=5)
            
            success = response.status_code in [200, 201]
            
            result = {
                "test": display_name,
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "success": success,
                "response": None
            }
            
            try:
                result["response"] = response.json()
            except:
                result["response"] = response.text[:200]
            
            self.results.append(result)
            
            if success:
                self.passed += 1
                self.log(f"  Status: {response.status_code} - PASSED", "SUCCESS")
            else:
                self.failed += 1
                self.log(f"  Status: {response.status_code} - FAILED", "ERROR")
            
            return success
            
        except requests.exceptions.Timeout:
            self.failed += 1
            self.log(f"  TIMEOUT", "ERROR")
            self.results.append({
                "test": display_name,
                "endpoint": endpoint,
                "method": method,
                "status_code": None,
                "success": False,
                "response": "Request timeout"
            })
            return False
        except Exception as e:
            self.failed += 1
            self.log(f"  ERROR: {e}", "ERROR")
            self.results.append({
                "test": display_name,
                "endpoint": endpoint,
                "method": method,
                "status_code": None,
                "success": False,
                "response": str(e)
            })
            return False
    
    def run_tests(self) -> bool:
        """Run all endpoint tests."""
        self.log("Starting test suite...", "INFO")
        print("\n" + "="*70)
        
        # Test 1: Health check
        self.test_endpoint("GET", "/health", description="Health Check")
        
        # Test 2: List tasks
        self.test_endpoint("GET", "/tasks", description="List Available Tasks")
        
        # Test 3: Reset environment
        reset_result = self.test_endpoint(
            "POST", 
            "/reset", 
            data={"task_name": None},
            description="Reset Environment (Start Episode)"
        )
        
        # Test 4: Get state
        self.test_endpoint("GET", "/state", description="Get Current State")
        
        # Test 5: Ask clarification
        if reset_result:
            self.test_endpoint(
                "POST",
                "/step",
                data={
                    "action_type": "ask_clarification",
                    "content": "What are the business hours?"
                },
                description="Step: Ask Clarification"
            )
        
        # Test 6: Get state after step
        self.test_endpoint("GET", "/state", description="Get State After Step")
        
        # Test 7: Propose rules
        if reset_result:
            self.test_endpoint(
                "POST",
                "/step",
                data={
                    "action_type": "propose_rules",
                    "content": {
                        "rules": [
                            {"condition": "user.role == 'admin'", "action": "ALLOW"}
                        ]
                    }
                },
                description="Step: Propose Rules"
            )
        
        # Test 8: Final state
        self.test_endpoint("GET", "/state", description="Get Final State")
        
        print("="*70 + "\n")
        return self.failed == 0
    
    def stop_server(self):
        """Stop the server process."""
        if self.server_process:
            self.log("Stopping server...", "INFO")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                self.log("Server stopped", "SUCCESS")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.log("Server force-killed", "WARNING")
    
    def generate_report(self):
        """Generate and print test report."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*70)
        print("📊 TEST REPORT")
        print("="*70)
        print(f"Timestamp: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration:  {duration:.2f}s")
        print(f"Total:     {self.passed + self.failed} tests")
        print(f"Passed:    {self.passed} ✅")
        print(f"Failed:    {self.failed} ❌")
        print(f"Success Rate: {100 * self.passed / (self.passed + self.failed):.1f}%")
        print("="*70)
        
        print("\n📋 DETAILED RESULTS:\n")
        for i, result in enumerate(self.results, 1):
            status_icon = "✅" if result["success"] else "❌"
            print(f"{i}. {status_icon} {result['test']}")
            print(f"   Endpoint: {result['method']} {result['endpoint']}")
            print(f"   Status: {result['status_code']}")
            if isinstance(result['response'], dict):
                print(f"   Response: {json.dumps(result['response'], indent=6)[:200]}...")
            else:
                print(f"   Response: {str(result['response'])[:100]}")
            print()
        
        print("="*70)
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {self.failed} test(s) failed. Check details above.")
        print("="*70 + "\n")
        
        return self.failed == 0
    
    def run(self) -> bool:
        """Run the entire test suite."""
        print("\n🚀 Policy-to-Logic RL Environment - Automated Test Suite\n")
        
        # Start server
        if not self.start_server():
            return False
        
        # Wait for server
        time.sleep(2)  # Give server time to start
        if not self.wait_for_server():
            self.stop_server()
            return False
        
        # Run tests
        all_passed = self.run_tests()
        
        # Stop server
        self.stop_server()
        
        # Generate report
        self.generate_report()
        
        return all_passed

def main():
    runner = TestRunner()
    try:
        success = runner.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        runner.stop_server()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        runner.stop_server()
        sys.exit(1)

if __name__ == "__main__":
    main()
