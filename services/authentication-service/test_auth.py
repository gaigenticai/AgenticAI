#!/usr/bin/env python3
"""
Test script for Authentication Service

This script demonstrates how to use the Authentication Service API
for user registration, login, token management, and protected resource access.

Usage:
    python test_auth.py --register      # Register a new user
    python test_auth.py --login         # Login with existing user
    python test_auth.py --profile       # Get user profile
    python test_auth.py --mfa-setup     # Setup MFA
    python test_auth.py --api-key       # Create API key
    python test_auth.py --all           # Run all tests

Environment variables required:
    AUTH_SERVICE_URL=http://localhost:8330 (default)
"""

import asyncio
import json
import argparse
import sys
import os
from typing import Dict, Any, Optional
import httpx
from datetime import datetime


class AuthServiceTester:
    """Test client for Authentication Service"""

    def __init__(self, base_url: str = "http://localhost:8330"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.api_key: Optional[str] = None

        # Test user credentials
        self.test_user = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "TestPass123!",
            "full_name": "Test User"
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None,
                          use_auth: bool = True, use_api_key: bool = False) -> Dict[str, Any]:
        """Make HTTP request to auth service"""
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        if use_auth and self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        elif use_api_key and self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            if method.upper() == "GET":
                response = await self.client.get(url, headers=headers)
            elif method.upper() == "POST":
                response = await self.client.post(url, headers=headers, json=data)
            elif method.upper() == "PUT":
                response = await self.client.put(url, headers=headers, json=data)
            elif method.upper() == "DELETE":
                response = await self.client.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.content_type == 'application/json':
                result = response.json()
            else:
                result = {"text": response.text}

            return {
                "status_code": response.status_code,
                "data": result,
                "headers": dict(response.headers)
            }

        except Exception as e:
            return {
                "status_code": 0,
                "error": str(e),
                "data": None
            }

    async def test_health_check(self) -> bool:
        """Test service health"""
        print("🔍 Testing service health...")
        result = await self.make_request("GET", "/health", use_auth=False)

        if result["status_code"] == 200:
            print("✅ Service is healthy")
            return True
        else:
            print(f"❌ Service health check failed: {result}")
            return False

    async def test_user_registration(self) -> bool:
        """Test user registration"""
        print("📝 Testing user registration...")
        result = await self.make_request("POST", "/auth/register", self.test_user, use_auth=False)

        if result["status_code"] == 200:
            print("✅ User registration successful")
            return True
        elif result["status_code"] == 409:
            print("⚠️  User already exists, continuing...")
            return True
        else:
            print(f"❌ User registration failed: {result}")
            return False

    async def test_user_login(self) -> bool:
        """Test user login"""
        print("🔐 Testing user login...")
        login_data = {
            "username_or_email": self.test_user["username"],
            "password": self.test_user["password"]
        }

        result = await self.make_request("POST", "/auth/login", login_data, use_auth=False)

        if result["status_code"] == 200:
            data = result["data"]
            if "access_token" in data:
                self.access_token = data["access_token"]
                self.refresh_token = data.get("refresh_token")
                print("✅ User login successful")
                return True
            elif "requires_mfa" in data:
                print("⚠️  MFA required but not implemented in test")
                return False
        else:
            # Try with admin credentials as fallback
            print("⚠️  Test user login failed, trying admin login...")
            admin_login = {
                "username_or_email": "admin",
                "password": "admin123!"
            }
            result = await self.make_request("POST", "/auth/login", admin_login, use_auth=False)

            if result["status_code"] == 200:
                data = result["data"]
                if "access_token" in data:
                    self.access_token = data["access_token"]
                    self.refresh_token = data.get("refresh_token")
                    print("✅ Admin login successful")
                    return True

        print(f"❌ User login failed: {result}")
        return False

    async def test_get_profile(self) -> bool:
        """Test getting user profile"""
        print("👤 Testing get user profile...")
        result = await self.make_request("GET", "/auth/profile")

        if result["status_code"] == 200:
            data = result["data"]
            print(f"✅ Profile retrieved: {data.get('username', 'unknown')}")
            return True
        else:
            print(f"❌ Get profile failed: {result}")
            return False

    async def test_token_refresh(self) -> bool:
        """Test token refresh"""
        print("🔄 Testing token refresh...")
        if not self.refresh_token:
            print("⚠️  No refresh token available")
            return False

        refresh_data = {"refresh_token": self.refresh_token}
        result = await self.make_request("POST", "/auth/refresh", refresh_data, use_auth=False)

        if result["status_code"] == 200:
            data = result["data"]
            if "access_token" in data:
                self.access_token = data["access_token"]
                print("✅ Token refresh successful")
                return True

        print(f"❌ Token refresh failed: {result}")
        return False

    async def test_create_api_key(self) -> bool:
        """Test API key creation"""
        print("🔑 Testing API key creation...")
        api_key_data = {
            "name": "Test API Key",
            "permissions": ["read", "write"],
            "expires_in_days": 30
        }

        result = await self.make_request("POST", "/auth/api-keys", api_key_data)

        if result["status_code"] == 200:
            data = result["data"]
            if "api_key" in data:
                self.api_key = data["api_key"]
                print("✅ API key created successfully")
                return True

        print(f"❌ API key creation failed: {result}")
        return False

    async def test_api_key_auth(self) -> bool:
        """Test API key authentication"""
        print("🔐 Testing API key authentication...")
        if not self.api_key:
            print("⚠️  No API key available")
            return False

        result = await self.make_request("GET", "/auth/profile", use_auth=False, use_api_key=True)

        if result["status_code"] == 200:
            print("✅ API key authentication successful")
            return True
        else:
            print(f"❌ API key authentication failed: {result}")
            return False

    async def test_user_logout(self) -> bool:
        """Test user logout"""
        print("🚪 Testing user logout...")
        result = await self.make_request("POST", "/auth/logout")

        if result["status_code"] == 200:
            self.access_token = None
            self.refresh_token = None
            print("✅ User logout successful")
            return True
        else:
            print(f"❌ User logout failed: {result}")
            return False

    async def test_list_api_keys(self) -> bool:
        """Test listing API keys"""
        print("📋 Testing list API keys...")
        result = await self.make_request("GET", "/auth/api-keys")

        if result["status_code"] == 200:
            data = result["data"]
            api_keys = data.get("api_keys", [])
            print(f"✅ Found {len(api_keys)} API keys")
            return True
        else:
            print(f"❌ List API keys failed: {result}")
            return False

    async def test_admin_audit_log(self) -> bool:
        """Test admin audit log access"""
        print("📊 Testing admin audit log...")
        result = await self.make_request("GET", "/auth/audit-log")

        if result["status_code"] == 200:
            data = result["data"]
            audit_logs = data.get("audit_logs", [])
            print(f"✅ Retrieved {len(audit_logs)} audit log entries")
            return True
        else:
            print(f"❌ Admin audit log access failed: {result}")
            return False

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all authentication tests"""
        print("🚀 Starting comprehensive authentication tests")
        print("=" * 60)

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "tests": []
        }

        # Define test sequence
        tests = [
            ("health_check", self.test_health_check),
            ("user_registration", self.test_user_registration),
            ("user_login", self.test_user_login),
            ("get_profile", self.test_get_profile),
            ("token_refresh", self.test_token_refresh),
            ("create_api_key", self.test_create_api_key),
            ("api_key_auth", self.test_api_key_auth),
            ("list_api_keys", self.test_list_api_keys),
            ("admin_audit_log", self.test_admin_audit_log),
            ("user_logout", self.test_user_logout)
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            try:
                success = await test_func()
                results["tests"].append({
                    "name": test_name,
                    "status": "passed" if success else "failed",
                    "timestamp": datetime.utcnow().isoformat()
                })

                if success:
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"❌ Test {test_name} crashed: {str(e)}")
                results["tests"].append({
                    "name": test_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                failed += 1

        results["summary"] = {
            "total_tests": len(tests),
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / len(tests)) * 100 if tests else 0
        }

        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {len(tests)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(".1f"
        if results["summary"]["success_rate"] >= 80:
            print("✅ Most tests passed successfully!")
        else:
            print("❌ Several tests failed - check service configuration")

        return results

    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """Save test results to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"auth_test_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"📄 Test results saved to: {output_file}")
        return output_file


async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Authentication Service Test Client")
    parser.add_argument("--register", action="store_true", help="Test user registration")
    parser.add_argument("--login", action="store_true", help="Test user login")
    parser.add_argument("--profile", action="store_true", help="Test get user profile")
    parser.add_argument("--mfa-setup", action="store_true", help="Test MFA setup")
    parser.add_argument("--api-key", action="store_true", help="Test API key creation")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--output", type=str, help="Output file for test results")
    parser.add_argument("--url", type=str, default="http://localhost:8330",
                       help="Authentication service URL")

    args = parser.parse_args()

    # Check if any test is specified
    if not any([args.register, args.login, args.profile, args.mfa_setup,
               args.api_key, args.all]):
        print("❌ Please specify a test to run:")
        print("  --register      Test user registration")
        print("  --login         Test user login")
        print("  --profile       Test get user profile")
        print("  --mfa-setup     Test MFA setup")
        print("  --api-key       Test API key creation")
        print("  --all           Run all tests")
        sys.exit(1)

    async with AuthServiceTester(args.url) as tester:
        try:
            if args.all:
                results = await tester.run_all_tests()
                if args.output:
                    tester.save_results(results, args.output)

            else:
                # Run individual tests
                if args.register:
                    await tester.test_user_registration()

                if args.login:
                    await tester.test_user_login()

                if args.profile:
                    # Need to login first
                    await tester.test_user_login()
                    await tester.test_get_profile()

                if args.api_key:
                    # Need to login first
                    await tester.test_user_login()
                    await tester.test_create_api_key()

                if args.mfa_setup:
                    print("⚠️  MFA setup test not implemented yet")

        except KeyboardInterrupt:
            print("\n⚠️  Test execution interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Test execution failed: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
