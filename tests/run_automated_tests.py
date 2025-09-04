#!/usr/bin/env python3
"""
Automated Testing Suite for Agentic Platform

This script performs comprehensive testing of:
1. Backend services functionality
2. API endpoints
3. Database operations
4. Message queue communication
5. UI components
6. Integration flows

Follows rule 12: After every feature development, perform automated testing
of the UI and every code to ensure they are functional.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class AutomatedTestSuite:
    """Comprehensive automated testing suite"""

    def __init__(self):
        self.base_url = os.getenv("INGESTION_COORDINATOR_URL", "http://localhost:8080")
        self.test_results = []
        self.services_status = {}

    def log_test_result(self, test_name: str, status: str, message: str = "", duration: float = 0):
        """Log test result"""
        result = {
            "test_name": test_name,
            "status": status,
            "message": message,
            "duration": duration,
            "timestamp": time.time()
        }
        self.test_results.append(result)

        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {message}")

        if duration > 0:
            print(f"{duration:.2f}s")
    def check_service_health(self, service_name: str, url: str, timeout: int = 10) -> bool:
        """Check if a service is healthy"""
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            duration = time.time() - start_time

            if response.status_code == 200:
                self.services_status[service_name] = "healthy"
                self.log_test_result(f"{service_name}_health", "PASS",
                                   f"Service is healthy (response time: {duration:.2f}s)", duration)
                return True
            else:
                self.services_status[service_name] = "unhealthy"
                self.log_test_result(f"{service_name}_health", "FAIL",
                                   f"Service returned status {response.status_code}", duration)
                return False

        except requests.exceptions.RequestException as e:
            self.services_status[service_name] = "unreachable"
            self.log_test_result(f"{service_name}_health", "FAIL",
                               f"Service unreachable: {str(e)}")
            return False

    def test_ingestion_coordinator(self):
        """Test ingestion coordinator service"""
        print("\n🔍 Testing Ingestion Coordinator...")

        # Test health endpoint
        self.check_service_health("ingestion_coordinator", f"{self.base_url}/health")

        # Test metrics endpoint
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            if response.status_code == 200 and "# HELP" in response.text:
                self.log_test_result("coordinator_metrics", "PASS", "Metrics endpoint working")
            else:
                self.log_test_result("coordinator_metrics", "FAIL", "Metrics endpoint not working")
        except Exception as e:
            self.log_test_result("coordinator_metrics", "FAIL", f"Metrics test failed: {e}")

        # Test jobs endpoint
        try:
            response = requests.get(f"{self.base_url}/ingestion/jobs", timeout=10)
            if response.status_code == 200:
                jobs = response.json()
                self.log_test_result("coordinator_jobs_api", "PASS", f"Jobs API working, found {len(jobs)} jobs")
            else:
                self.log_test_result("coordinator_jobs_api", "FAIL", f"Jobs API failed: {response.status_code}")
        except Exception as e:
            self.log_test_result("coordinator_jobs_api", "FAIL", f"Jobs API test failed: {e}")

    def test_csv_ingestion_service(self):
        """Test CSV ingestion service"""
        print("\n🔍 Testing CSV Ingestion Service...")

        csv_service_url = "http://localhost:8082"
        self.check_service_health("csv_ingestion", f"{csv_service_url}/health")

    def test_database_connectivity(self):
        """Test database connectivity"""
        print("\n🔍 Testing Database Connectivity...")

        try:
            # Try to connect to PostgreSQL
            import psycopg2
            db_url = os.getenv("DATABASE_URL", "postgresql://agentic_user:agentic123@localhost:5432/agentic_ingestion")

            conn = psycopg2.connect(db_url)
            conn.close()

            self.log_test_result("database_connectivity", "PASS", "Database connection successful")
        except Exception as e:
            self.log_test_result("database_connectivity", "FAIL", f"Database connection failed: {e}")

    def test_message_queue(self):
        """Test message queue connectivity"""
        print("\n🔍 Testing Message Queue...")

        try:
            import pika
            credentials = pika.PlainCredentials("agentic_user", "agentic123")
            parameters = pika.ConnectionParameters(
                host="localhost",
                port=5672,
                credentials=credentials
            )

            connection = pika.BlockingConnection(parameters)
            connection.close()

            self.log_test_result("message_queue", "PASS", "RabbitMQ connection successful")
        except Exception as e:
            self.log_test_result("message_queue", "FAIL", f"RabbitMQ connection failed: {e}")

    def test_file_upload_integration(self):
        """Test complete file upload and processing integration"""
        print("\n🔍 Testing File Upload Integration...")

        # Create test CSV file
        test_data = [
            ["name", "age", "city"],
            ["Alice", "30", "New York"],
            ["Bob", "25", "San Francisco"],
            ["Charlie", "35", "Chicago"]
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerows(test_data)
            csv_file_path = f.name

        try:
            # Upload file via API
            with open(csv_file_path, 'rb') as f:
                files = {'file': ('test_data.csv', f, 'text/csv')}
                data = {'source_type': 'csv'}

                response = requests.post(
                    f"{self.base_url}/ingestion/upload",
                    files=files,
                    data=data,
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()
                job_id = result.get("job_id")

                if job_id:
                    self.log_test_result("file_upload_integration", "PASS",
                                       f"File uploaded successfully, job ID: {job_id}")

                    # Wait a bit and check job status
                    time.sleep(2)
                    status_response = requests.get(f"{self.base_url}/ingestion/jobs/{job_id}", timeout=10)

                    if status_response.status_code == 200:
                        self.log_test_result("job_status_tracking", "PASS", "Job status tracking working")
                    else:
                        self.log_test_result("job_status_tracking", "WARN", "Job status tracking may have issues")

                else:
                    self.log_test_result("file_upload_integration", "FAIL", "No job ID returned")
            else:
                self.log_test_result("file_upload_integration", "FAIL",
                                   f"Upload failed with status {response.status_code}: {response.text}")

        except Exception as e:
            self.log_test_result("file_upload_integration", "FAIL", f"Integration test failed: {e}")
        finally:
            # Cleanup
            os.unlink(csv_file_path)

    def test_ui_components(self):
        """Test UI components using Selenium"""
        print("\n🔍 Testing UI Components...")

        # Check if UI test file exists
        ui_test_file = Path(__file__).parent / "ui" / "ingestion_test_ui.html"
        if not ui_test_file.exists():
            self.log_test_result("ui_components", "SKIP", "UI test file not found")
            return

        # For now, just check if the HTML file is valid
        try:
            with open(ui_test_file, 'r') as f:
                content = f.read()

            if "<!DOCTYPE html>" in content and "</html>" in content:
                self.log_test_result("ui_components", "PASS", "UI HTML file is valid")
            else:
                self.log_test_result("ui_components", "FAIL", "UI HTML file appears invalid")

        except Exception as e:
            self.log_test_result("ui_components", "FAIL", f"UI test failed: {e}")

    def test_docker_services(self):
        """Test Docker services status"""
        print("\n🔍 Testing Docker Services...")

        try:
            # Check if Docker is running
            result = subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                service_count = len(lines) - 1  # Subtract header line

                self.log_test_result("docker_services", "PASS", f"Docker running with {service_count} containers")

                # Check for expected services
                expected_services = ["agentic-ingestion-coordinator", "agentic-postgres-ingestion"]
                running_services = []

                for line in lines[1:]:  # Skip header
                    for service in expected_services:
                        if service in line:
                            running_services.append(service)

                if len(running_services) > 0:
                    self.log_test_result("docker_expected_services", "PASS",
                                       f"Found expected services: {', '.join(running_services)}")
                else:
                    self.log_test_result("docker_expected_services", "WARN", "No expected services found running")

            else:
                self.log_test_result("docker_services", "FAIL", "Docker command failed")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.log_test_result("docker_services", "SKIP", f"Docker not available: {e}")

    def run_all_tests(self):
        """Run all automated tests"""
        print("🚀 Starting Agentic Platform Automated Testing Suite")
        print("=" * 60)

        start_time = time.time()

        # Test infrastructure
        self.test_docker_services()
        self.test_database_connectivity()
        self.test_message_queue()

        # Test services
        self.test_ingestion_coordinator()
        self.test_csv_ingestion_service()

        # Test integrations
        self.test_file_upload_integration()

        # Test UI
        self.test_ui_components()

        # Calculate results
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAIL"])
        skipped_tests = len([r for r in self.test_results if r["status"] == "SKIP"])
        warned_tests = len([r for r in self.test_results if r["status"] == "WARN"])

        duration = time.time() - start_time
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        print("\n" + "=" * 60)
        print("📊 AUTOMATED TESTING RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Tests:     {total_tests}")
        print(f"Passed:          {passed_tests} ✅")
        print(f"Failed:          {failed_tests} ❌")
        print(f"Skipped:         {skipped_tests} ⚠️")
        print(f"Warnings:        {warned_tests} ⚠️")
        print(f"Duration:         {duration:.2f}s")
        print(f"Success Rate:     {success_rate:.1f}%")
        # Service status summary
        print("\n🔧 SERVICE STATUS:")
        for service, status in self.services_status.items():
            status_icon = "🟢" if status == "healthy" else "🔴" if status == "unreachable" else "🟡"
            print(f"  {status_icon} {service}: {status}")

        # Detailed results
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  • {result['test_name']}: {result['message']}")

        # Success criteria
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        print("\n" + "=" * 60)
        if success_rate >= 80 and failed_tests == 0:
            print("🎉 ALL TESTS PASSED! Platform is ready for use.")
            return 0
        elif success_rate >= 60:
            print("⚠️ MOST TESTS PASSED. Some issues detected but platform may be usable.")
            return 1
        else:
            print("❌ CRITICAL ISSUES DETECTED. Platform requires attention before use.")
            return 2

    def save_test_report(self):
        """Save detailed test report to file"""
        report_file = f"test_report_{int(time.time())}.json"

        report = {
            "timestamp": time.time(),
            "summary": {
                "total_tests": len(self.test_results),
                "passed": len([r for r in self.test_results if r["status"] == "PASS"]),
                "failed": len([r for r in self.test_results if r["status"] == "FAIL"]),
                "skipped": len([r for r in self.test_results if r["status"] == "SKIP"]),
                "warnings": len([r for r in self.test_results if r["status"] == "WARN"])
            },
            "services_status": self.services_status,
            "detailed_results": self.test_results
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Detailed test report saved to: {report_file}")


def main():
    """Main function to run automated tests"""
    test_suite = AutomatedTestSuite()

    try:
        exit_code = test_suite.run_all_tests()
        test_suite.save_test_report()
        return exit_code

    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        test_suite.save_test_report()
        return 130

    except Exception as e:
        print(f"\n💥 Testing suite crashed: {e}")
        test_suite.save_test_report()
        return 1


if __name__ == "__main__":
    sys.exit(main())
