"""
HF Spaces Test Runner - Policy-to-Logic Environment

Tests all endpoints on the deployed HF Spaces and generates a report.

Run it:
    uv run python test_hf_spaces.py
"""

import requests
import json
import time
from typing import Dict, Any, List
from datetime import datetime
from urllib.parse import urljoin

# HF Spaces URL
HF_SPACE_URL = "https://huggingface.co/spaces/Godreign/Policy2Logic"
# Extract the actual API endpoint
BASE_URL = HF_SPACE_URL

class HFSpacesTestRunner:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.results: List[Dict[str, Any]] = []
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
    
    def test_endpoint(self, method: str, endpoint: str, data: dict = None, 
                      description: str = "", timeout: int = 10) -> bool:
        """Test an HF Spaces endpoint and record result."""
        url = urljoin(self.base_url, endpoint)
        display_name = description or endpoint
        self.log(f"Testing: {display_name}", "TEST")
        
        try:
            if method == "POST":
                response = requests.post(url, json=data, timeout=timeout)
            else:
                response = requests.get(url, timeout=timeout)
            
            success = response.status_code in [200, 201]
            
            result = {
                "test": display_name,
                "endpoint": endpoint,
                "method": method,
                "url": url,
                "status_code": response.status_code,
                "success": success,
                "response": None,
                "error": None
            }
            
            try:
                result["response"] = response.json()
            except:
                result["response"] = response.text[:300]
            
            self.results.append(result)
            
            if success:
                self.passed += 1
                self.log(f"  Status: {response.status_code} - PASSED", "SUCCESS")
            else:
                self.failed += 1
                self.log(f"  Status: {response.status_code} - FAILED", "ERROR")
                result["error"] = response.text[:200]
            
            return success
            
        except requests.exceptions.Timeout:
            self.failed += 1
            self.log(f"  TIMEOUT (>{timeout}s)", "ERROR")
            self.results.append({
                "test": display_name,
                "endpoint": endpoint,
                "method": method,
                "url": url,
                "status_code": None,
                "success": False,
                "response": None,
                "error": f"Request timeout after {timeout}s"
            })
            return False
        except requests.exceptions.ConnectionError as e:
            self.failed += 1
            self.log(f"  CONNECTION ERROR: {str(e)[:100]}", "ERROR")
            self.results.append({
                "test": display_name,
                "endpoint": endpoint,
                "method": method,
                "url": url,
                "status_code": None,
                "success": False,
                "response": None,
                "error": f"Connection error: {str(e)}"
            })
            return False
        except Exception as e:
            self.failed += 1
            self.log(f"  ERROR: {str(e)[:100]}", "ERROR")
            self.results.append({
                "test": display_name,
                "endpoint": endpoint,
                "method": method,
                "url": url,
                "status_code": None,
                "success": False,
                "response": None,
                "error": str(e)
            })
            return False
    
    def run_tests(self) -> bool:
        """Run all endpoint tests."""
        self.log("Starting HF Spaces test suite...", "INFO")
        print("\n" + "="*70)
        
        # Test 1: Root endpoint
        self.test_endpoint("GET", "/", description="Root Endpoint")
        
        # Test 2: Health check
        self.test_endpoint("GET", "/health", description="Health Check")
        
        # Test 3: Root with query params (HF Spaces probe)
        self.test_endpoint("GET", "/?logs=container", description="Root with Logs Query")
        
        # Test 4: List tasks
        self.test_endpoint("GET", "/tasks", description="List Available Tasks")
        
        # Test 5: Reset environment
        reset_result = self.test_endpoint(
            "POST", 
            "/reset", 
            data={"task_name": None},
            description="Reset Environment (Start Episode)"
        )
        
        # Test 6: Get state
        self.test_endpoint("GET", "/state", description="Get Current State")
        
        # Test 7: Ask clarification
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
        
        # Test 8: Final state
        self.test_endpoint("GET", "/state", description="Get Final State After Step")
        
        print("="*70 + "\n")
        return self.failed == 0
    
    def generate_report(self):
        """Generate and print test report."""
        duration = (datetime.now() - self.start_time).total_seconds()
        total = self.passed + self.failed
        success_rate = 100 * self.passed / total if total > 0 else 0
        
        print("\n" + "="*70)
        print("📊 HF SPACES TEST REPORT")
        print("="*70)
        print(f"Space URL:      {self.base_url}")
        print(f"Timestamp:      {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration:       {duration:.2f}s")
        print(f"Total Tests:    {total}")
        print(f"Passed:         {self.passed} ✅")
        print(f"Failed:         {self.failed} ❌")
        print(f"Success Rate:   {success_rate:.1f}%")
        print("="*70)
        
        print("\n📋 DETAILED RESULTS:\n")
        for i, result in enumerate(self.results, 1):
            status_icon = "✅" if result["success"] else "❌"
            print(f"{i}. {status_icon} {result['test']}")
            print(f"   Method:   {result['method']} {result['endpoint']}")
            print(f"   URL:      {result['url']}")
            if result['status_code']:
                print(f"   Status:   {result['status_code']}")
            
            if result['error']:
                print(f"   Error:    {result['error']}")
            elif result['response']:
                response_preview = result['response']
                if isinstance(response_preview, dict):
                    response_preview = json.dumps(response_preview, indent=4)[:200]
                else:
                    response_preview = str(response_preview)[:200]
                print(f"   Response: {response_preview}...")
            print()
        
        print("="*70)
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED - HF SPACES IS RUNNING!")
        else:
            print(f"⚠️  {self.failed} test(s) failed. Check details above.")
        print("="*70 + "\n")
        
        return self.failed == 0
    
    def run(self) -> bool:
        """Run the entire test suite."""
        print("\n🚀 Policy-to-Logic RL Environment - HF Spaces Test Suite\n")
        
        self.log(f"Target: {self.base_url}", "INFO")
        
        # Run tests
        all_passed = self.run_tests()
        
        # Generate report
        self.generate_report()
        
        return all_passed

def main():
    runner = HFSpacesTestRunner(BASE_URL)
    try:
        success = runner.run()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
