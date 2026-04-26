"""
HF Spaces Test Runner - Policy-to-Logic Environment

Tests all endpoints on the deployed HF Spaces and generates a report.

Run it:
    $env:UV_LINK_MODE="copy"
    uv run python test_hf_spaces.py
"""

import requests
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# ─── URL Construction ─────────────────────────────────────────────
# HF Spaces URL → API endpoint
# https://huggingface.co/spaces/{user}/{repo} → https://{user}-{repo}.hf.space
HF_SPACE_WEB_URL = "https://huggingface.co/spaces/Godreign/Policy2Logic"
parts = HF_SPACE_WEB_URL.split('/')
username = parts[-2]   # "Godreign"
repo_name = parts[-1]  # "Policy2Logic"
BASE_URL = f"https://{username.lower()}-{repo_name.lower()}.hf.space"


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

    # ── Connectivity Check ────────────────────────────────────────
    def check_connectivity(self) -> bool:
        """
        Verify we can reach the HF Space before running tests.
        Returns True if the space is reachable and responding.
        """
        print(f"\n🔗 Connectivity Check")
        print(f"   Target URL: {self.base_url}")
        print()

        try:
            # First, check with allow_redirects=False to detect proxy issues
            resp = requests.get(
                self.base_url,
                timeout=15,
                allow_redirects=False,
            )
            print(f"   Direct response:  status={resp.status_code}")

            if resp.is_redirect or resp.is_permanent_redirect:
                redirect_url = resp.headers.get("Location", "unknown")
                print(f"   ⚠️  REDIRECT detected → {redirect_url}")
                print(f"   The space may not be running or the URL format changed.")
                print(f"   Expected API base: {self.base_url}")
                return False

            # Now check with redirects allowed (normal mode)
            resp = requests.get(self.base_url, timeout=15)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if data.get("status") == "running":
                        print(f"   ✅ Space is RUNNING")
                        print(f"   Response: {json.dumps(data, indent=2)[:200]}")
                        return True
                    else:
                        print(f"   ⚠️  Unexpected response: {data}")
                        return True  # Still reachable
                except ValueError:
                    print(f"   ⚠️  Non-JSON response (may be HF loading page)")
                    print(f"   Body preview: {resp.text[:200]}")
                    return False
            else:
                print(f"   ❌ Got status {resp.status_code}")
                print(f"   Body: {resp.text[:200]}")
                return False

        except requests.exceptions.Timeout:
            print(f"   ❌ Connection TIMEOUT (>15s)")
            print(f"   The space may be sleeping. Visit {HF_SPACE_WEB_URL} to wake it.")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"   ❌ Connection FAILED: {str(e)[:150]}")
            return False
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            return False

    # ── Endpoint Testing ──────────────────────────────────────────
    def test_endpoint(self, method: str, endpoint: str, data: dict = None,
                      description: str = "", timeout: int = 15) -> bool:
        """Test an HF Spaces endpoint and record result."""
        url = f"{self.base_url}{endpoint}"
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
            except ValueError:
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

    # ── Test Suite ────────────────────────────────────────────────
    def run_tests(self) -> bool:
        """Run all endpoint tests."""
        self.log("Starting endpoint tests...", "INFO")
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

        # Test 7: Ask clarification (progressive revelation test)
        if reset_result:
            # Level 1 - vague question
            self.test_endpoint(
                "POST",
                "/step",
                data={
                    "action_type": "ask_clarification",
                    "content": "What are the working hours?"
                },
                description="Step: Ask Clarification (Level 1 - vague)"
            )

            # Level 3 - specific compound question
            self.test_endpoint(
                "POST",
                "/step",
                data={
                    "action_type": "ask_clarification",
                    "content": "What happens at the working hours boundary, exactly at hour 18?"
                },
                description="Step: Ask Clarification (Level 3 - precise)"
            )

        # Test 8: Propose rules (valid DSL)
        if reset_result:
            self.test_endpoint(
                "POST",
                "/step",
                data={
                    "action_type": "propose_rules",
                    "content": json.dumps({
                        "rules": [
                            {
                                "if": [
                                    {"field": "data_type", "op": "==", "value": "public"}
                                ],
                                "then": "ALLOW"
                            },
                            {
                                "if": [
                                    {"field": "time", "op": ">=", "value": 9},
                                    {"field": "time", "op": "<", "value": 18}
                                ],
                                "then": "ALLOW"
                            }
                        ],
                        "default": "DENY"
                    })
                },
                description="Step: Propose Rules (Valid DSL)"
            )

        # Test 9: Final state
        self.test_endpoint("GET", "/state", description="Get Final State After Steps")

        print("="*70 + "\n")
        return self.failed == 0

    # ── Report Generation ─────────────────────────────────────────
    def generate_report(self):
        """Generate and print test report."""
        duration = (datetime.now() - self.start_time).total_seconds()
        total = self.passed + self.failed
        success_rate = 100 * self.passed / total if total > 0 else 0

        print("\n" + "="*70)
        print("📊 HF SPACES TEST REPORT")
        print("="*70)
        print(f"Space URL:      {self.base_url}")
        print(f"Web URL:        {HF_SPACE_WEB_URL}")
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
                    response_preview = json.dumps(response_preview, indent=4)[:300]
                else:
                    response_preview = str(response_preview)[:300]
                print(f"   Response: {response_preview}...")
            print()

        print("="*70)
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED - HF SPACES IS RUNNING!")
        else:
            print(f"⚠️  {self.failed} test(s) failed. Check details above.")
        print("="*70 + "\n")

        return self.failed == 0

    # ── Main Runner ───────────────────────────────────────────────
    def run(self) -> bool:
        """Run the entire test suite."""
        print("\n🚀 Policy-to-Logic RL Environment - HF Spaces Test Suite\n")

        self.log(f"Target API:  {self.base_url}", "INFO")
        self.log(f"Source URL:  {HF_SPACE_WEB_URL}", "INFO")

        # Step 1: Connectivity check
        if not self.check_connectivity():
            print("\n❌ Connectivity check FAILED. Cannot proceed with tests.")
            print(f"   Verify the space is running at: {HF_SPACE_WEB_URL}")
            print(f"   Expected API endpoint: {self.base_url}")
            return False

        print()

        # Step 2: Run endpoint tests
        all_passed = self.run_tests()

        # Step 3: Generate report
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
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
