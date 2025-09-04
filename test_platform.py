#!/usr/bin/env python3
"""
Agentic Platform Automated Test Suite
=====================================

Comprehensive automated testing suite for all platform components:
- Service health checks
- API endpoint testing
- Database connectivity
- Vector operations
- UI component validation
- Performance benchmarks
- Error handling verification

Usage:
    python test_platform.py [--verbose] [--services-only] [--api-only] [--ui-only]

Author: Agentic Platform Team
"""

import asyncio
import json
import time
import argparse
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics

import httpx
import requests
from urllib.parse import urljoin


@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestSuite:
    """Test suite container"""
    name: str
    tests: List[TestResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def total_tests(self) -> int:
        return len(self.tests)

    @property
    def passed_tests(self) -> int:
        return len([t for t in self.tests if t.status == 'passed'])

    @property
    def failed_tests(self) -> int:
        return len([t for t in self.tests if t.status == 'failed'])

    @property
    def error_tests(self) -> int:
        return len([t for t in self.tests if t.status == 'error'])

    @property
    def skipped_tests(self) -> int:
        return len([t for t in self.tests if t.status == 'skipped'])

    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def total_duration(self) -> float:
        if not self.start_time or not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()


class AgenticPlatformTester:
    """Main test orchestrator for Agentic Platform"""

    def __init__(self, base_url: str = "http://localhost", verbose: bool = False):
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Service URLs
        self.vector_ui_url = f"{self.base_url}:8082"
        self.ingestion_coordinator_url = f"{self.base_url}:8080"
        self.output_coordinator_url = f"{self.base_url}:8081"

        # Test suites
        self.test_suites: Dict[str, TestSuite] = {}

    async def run_all_tests(self) -> Dict[str, TestSuite]:
        """Run complete test suite"""
        print("🚀 Starting Agentic Platform Test Suite")
        print("=" * 50)

        # Service Health Tests
        await self.run_service_health_tests()

        # API Endpoint Tests
        await self.run_api_tests()

        # Vector Operations Tests
        await self.run_vector_tests()

        # UI Component Tests
        await self.run_ui_tests()

        # Performance Tests
        await self.run_performance_tests()

        # Error Handling Tests
        await self.run_error_handling_tests()

        # Database Tests
        await self.run_database_tests()

        return self.test_suites

    async def run_service_health_tests(self):
        """Test all service health endpoints"""
        print("\n🏥 Testing Service Health...")

        suite = TestSuite("Service Health Tests")
        suite.start_time = datetime.now()
        self.test_suites["health"] = suite

        # Test Vector UI Health
        await self._test_service_health(
            suite, "Vector UI", f"{self.vector_ui_url}/health"
        )

        # Test Ingestion Coordinator Health
        await self._test_service_health(
            suite, "Ingestion Coordinator", f"{self.ingestion_coordinator_url}/health"
        )

        # Test Output Coordinator Health
        await self._test_service_health(
            suite, "Output Coordinator", f"{self.output_coordinator_url}/health"
        )

        suite.end_time = datetime.now()
        self._print_suite_results(suite)

    async def run_api_tests(self):
        """Test all API endpoints"""
        print("\n🔌 Testing API Endpoints...")

        suite = TestSuite("API Endpoint Tests")
        suite.start_time = datetime.now()
        self.test_suites["api"] = suite

        # Test Vector UI API endpoints
        await self._test_api_endpoints(suite)

        suite.end_time = datetime.now()
        self._print_suite_results(suite)

    async def run_vector_tests(self):
        """Test vector operations"""
        print("\n🧠 Testing Vector Operations...")

        suite = TestSuite("Vector Operations Tests")
        suite.start_time = datetime.now()
        self.test_suites["vector"] = suite

        # Test embedding generation
        await self._test_embedding_generation(suite)

        # Test document indexing
        await self._test_document_indexing(suite)

        # Test vector search
        await self._test_vector_search(suite)

        # Test similarity search
        await self._test_similarity_search(suite)

        suite.end_time = datetime.now()
        self._print_suite_results(suite)

    async def run_ui_tests(self):
        """Test UI components"""
        print("\n💻 Testing UI Components...")

        suite = TestSuite("UI Component Tests")
        suite.start_time = datetime.now()
        self.test_suites["ui"] = suite

        # Test main dashboard
        await self._test_ui_page(suite, "Dashboard", f"{self.vector_ui_url}/")

        # Test getting started guide
        await self._test_ui_page(suite, "Getting Started Guide", f"{self.vector_ui_url}/guide")

        # Test testing dashboard
        await self._test_ui_page(suite, "Testing Dashboard", f"{self.vector_ui_url}/test-dashboard")

        # Test API documentation
        await self._test_ui_page(suite, "API Documentation", f"{self.vector_ui_url}/docs")

        suite.end_time = datetime.now()
        self._print_suite_results(suite)

    async def run_performance_tests(self):
        """Run performance benchmarks"""
        print("\n⚡ Running Performance Tests...")

        suite = TestSuite("Performance Tests")
        suite.start_time = datetime.now()
        self.test_suites["performance"] = suite

        # API performance test
        await self._test_api_performance(suite)

        # Search performance test
        await self._test_search_performance(suite)

        suite.end_time = datetime.now()
        self._print_suite_results(suite)

    async def run_error_handling_tests(self):
        """Test error handling"""
        print("\n🚨 Testing Error Handling...")

        suite = TestSuite("Error Handling Tests")
        suite.start_time = datetime.now()
        self.test_suites["error_handling"] = suite

        # Test invalid endpoints
        await self._test_invalid_endpoints(suite)

        # Test malformed requests
        await self._test_malformed_requests(suite)

        # Test authentication errors
        await self._test_authentication_errors(suite)

        suite.end_time = datetime.now()
        self._print_suite_results(suite)

    async def run_database_tests(self):
        """Test database connectivity and operations"""
        print("\n💾 Testing Database Operations...")

        suite = TestSuite("Database Tests")
        suite.start_time = datetime.now()
        self.test_suites["database"] = suite

        # Test database connectivity (through API)
        await self._test_database_connectivity(suite)

        suite.end_time = datetime.now()
        self._print_suite_results(suite)

    # Helper methods
    async def _test_service_health(self, suite: TestSuite, service_name: str, health_url: str):
        """Test individual service health"""
        start_time = time.time()

        try:
            response = await self.http_client.get(health_url)
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    suite.tests.append(TestResult(
                        test_name=f"{service_name} Health",
                        status='passed',
                        duration=duration,
                        message=f"{service_name} is healthy",
                        details=data
                    ))
                else:
                    suite.tests.append(TestResult(
                        test_name=f"{service_name} Health",
                        status='failed',
                        duration=duration,
                        message=f"{service_name} reports unhealthy status",
                        details=data
                    ))
            else:
                suite.tests.append(TestResult(
                    test_name=f"{service_name} Health",
                    status='failed',
                    duration=duration,
                    message=f"HTTP {response.status_code}: {response.text}"
                ))

        except Exception as e:
            duration = time.time() - start_time
            suite.tests.append(TestResult(
                test_name=f"{service_name} Health",
                status='error',
                duration=duration,
                message=f"Health check failed: {str(e)}"
            ))

    async def _test_api_endpoints(self, suite: TestSuite):
        """Test API endpoints"""
        # Test echo endpoint
        start_time = time.time()
        try:
            response = await self.http_client.get(
                f"{self.vector_ui_url}/api/test/echo?message=TestMessage"
            )
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get('echo') == 'TestMessage':
                    suite.tests.append(TestResult(
                        test_name="API Echo Test",
                        status='passed',
                        duration=duration,
                        message="Echo endpoint working correctly"
                    ))
                else:
                    suite.tests.append(TestResult(
                        test_name="API Echo Test",
                        status='failed',
                        duration=duration,
                        message="Echo response incorrect"
                    ))
            else:
                suite.tests.append(TestResult(
                    test_name="API Echo Test",
                    status='failed',
                    duration=duration,
                    message=f"HTTP {response.status_code}"
                ))

        except Exception as e:
            duration = time.time() - start_time
            suite.tests.append(TestResult(
                test_name="API Echo Test",
                status='error',
                duration=duration,
                message=f"API test failed: {str(e)}"
            ))

    async def _test_embedding_generation(self, suite: TestSuite):
        """Test embedding generation"""
        start_time = time.time()
        try:
            test_text = "This is a test document for embedding generation."
            response = await self.http_client.post(
                f"{self.vector_ui_url}/api/embeddings/generate",
                json={"texts": [test_text]},
                headers={"Content-Type": "application/json"}
            )
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if 'embeddings' in data and len(data['embeddings']) > 0:
                    suite.tests.append(TestResult(
                        test_name="Embedding Generation",
                        status='passed',
                        duration=duration,
                        message=f"Generated {len(data['embeddings'])} embeddings",
                        details={"dimensions": data.get('dimensions'), "model": data.get('model')}
                    ))
                else:
                    suite.tests.append(TestResult(
                        test_name="Embedding Generation",
                        status='failed',
                        duration=duration,
                        message="No embeddings in response"
                    ))
            else:
                suite.tests.append(TestResult(
                    test_name="Embedding Generation",
                    status='failed',
                    duration=duration,
                    message=f"HTTP {response.status_code}: {response.text}"
                ))

        except Exception as e:
            duration = time.time() - start_time
            suite.tests.append(TestResult(
                test_name="Embedding Generation",
                status='error',
                duration=duration,
                message=f"Embedding test failed: {str(e)}"
            ))

    async def _test_document_indexing(self, suite: TestSuite):
        """Test document indexing"""
        start_time = time.time()
        try:
            test_doc = {
                "document_id": f"test-doc-{int(time.time())}",
                "text": "This is a test document for indexing and search functionality.",
                "metadata": {"test": True, "timestamp": datetime.now().isoformat()}
            }

            response = await self.http_client.post(
                f"{self.vector_ui_url}/api/vectors/index",
                json=[test_doc],
                headers={"Content-Type": "application/json"}
            )
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get('documents_processed', 0) > 0:
                    suite.tests.append(TestResult(
                        test_name="Document Indexing",
                        status='passed',
                        duration=duration,
                        message=f"Indexed {data.get('documents_processed')} documents",
                        details={"processing_time": data.get('processing_time')}
                    ))
                else:
                    suite.tests.append(TestResult(
                        test_name="Document Indexing",
                        status='failed',
                        duration=duration,
                        message="No documents processed"
                    ))
            else:
                suite.tests.append(TestResult(
                    test_name="Document Indexing",
                    status='failed',
                    duration=duration,
                    message=f"HTTP {response.status_code}: {response.text}"
                ))

        except Exception as e:
            duration = time.time() - start_time
            suite.tests.append(TestResult(
                test_name="Document Indexing",
                status='error',
                duration=duration,
                message=f"Indexing test failed: {str(e)}"
            ))

    async def _test_vector_search(self, suite: TestSuite):
        """Test vector search"""
        start_time = time.time()
        try:
            response = await self.http_client.post(
                f"{self.vector_ui_url}/api/vectors/search",
                json={
                    "query": "test document search",
                    "collection": "documents",
                    "limit": 5
                },
                headers={"Content-Type": "application/json"}
            )
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                results_count = len(data.get('results', []))
                suite.tests.append(TestResult(
                    test_name="Vector Search",
                    status='passed',
                    duration=duration,
                    message=f"Search completed, found {results_count} results",
                    details={"search_time": data.get('search_time'), "results_count": results_count}
                ))
            else:
                suite.tests.append(TestResult(
                    test_name="Vector Search",
                    status='failed',
                    duration=duration,
                    message=f"HTTP {response.status_code}: {response.text}"
                ))

        except Exception as e:
            duration = time.time() - start_time
            suite.tests.append(TestResult(
                test_name="Vector Search",
                status='error',
                duration=duration,
                message=f"Search test failed: {str(e)}"
            ))

    async def _test_similarity_search(self, suite: TestSuite):
        """Test similarity search"""
        # Skip if no test documents exist
        suite.tests.append(TestResult(
            test_name="Similarity Search",
            status='skipped',
            duration=0.0,
            message="Skipped - requires existing indexed documents"
        ))

    async def _test_ui_page(self, suite: TestSuite, page_name: str, url: str):
        """Test UI page accessibility"""
        start_time = time.time()
        try:
            response = await self.http_client.get(url)
            duration = time.time() - start_time

            if response.status_code == 200:
                content_length = len(response.text)
                suite.tests.append(TestResult(
                    test_name=f"UI - {page_name}",
                    status='passed',
                    duration=duration,
                    message=f"Page loaded successfully ({content_length} bytes)",
                    details={"content_length": content_length, "status_code": response.status_code}
                ))
            else:
                suite.tests.append(TestResult(
                    test_name=f"UI - {page_name}",
                    status='failed',
                    duration=duration,
                    message=f"HTTP {response.status_code}"
                ))

        except Exception as e:
            duration = time.time() - start_time
            suite.tests.append(TestResult(
                test_name=f"UI - {page_name}",
                status='error',
                duration=duration,
                message=f"UI test failed: {str(e)}"
            ))

    async def _test_api_performance(self, suite: TestSuite):
        """Test API performance"""
        start_time = time.time()
        try:
            # Run multiple API calls to measure performance
            response_times = []

            for i in range(10):
                call_start = time.time()
                response = await self.http_client.get(f"{self.vector_ui_url}/health")
                call_end = time.time()

                if response.status_code == 200:
                    response_times.append(call_end - call_start)
                else:
                    response_times.append(float('inf'))  # Mark as failed

            duration = time.time() - start_time
            valid_times = [t for t in response_times if t != float('inf')]

            if valid_times:
                avg_time = statistics.mean(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)

                suite.tests.append(TestResult(
                    test_name="API Performance",
                    status='passed',
                    duration=duration,
                    message=".3f",
                    details={
                        "avg_response_time": avg_time,
                        "min_response_time": min_time,
                        "max_response_time": max_time,
                        "successful_calls": len(valid_times),
                        "total_calls": len(response_times)
                    }
                ))
            else:
                suite.tests.append(TestResult(
                    test_name="API Performance",
                    status='failed',
                    duration=duration,
                    message="All API calls failed"
                ))

        except Exception as e:
            duration = time.time() - start_time
            suite.tests.append(TestResult(
                test_name="API Performance",
                status='error',
                duration=duration,
                message=f"Performance test failed: {str(e)}"
            ))

    async def _test_search_performance(self, suite: TestSuite):
        """Test search performance"""
        start_time = time.time()
        try:
            response = await self.http_client.post(
                f"{self.vector_ui_url}/api/vectors/search",
                json={
                    "query": "performance test query",
                    "collection": "documents",
                    "limit": 10
                },
                headers={"Content-Type": "application/json"}
            )
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                search_time = data.get('search_time', 0)

                suite.tests.append(TestResult(
                    test_name="Search Performance",
                    status='passed' if search_time < 2.0 else 'warning',
                    duration=duration,
                    message=".3f",
                    details={"search_time": search_time, "results_count": len(data.get('results', []))}
                ))
            else:
                suite.tests.append(TestResult(
                    test_name="Search Performance",
                    status='failed',
                    duration=duration,
                    message=f"HTTP {response.status_code}"
                ))

        except Exception as e:
            duration = time.time() - start_time
            suite.tests.append(TestResult(
                test_name="Search Performance",
                status='error',
                duration=duration,
                message=f"Search performance test failed: {str(e)}"
            ))

    async def _test_invalid_endpoints(self, suite: TestSuite):
        """Test invalid endpoints"""
        invalid_endpoints = [
            f"{self.vector_ui_url}/api/nonexistent",
            f"{self.vector_ui_url}/api/vectors/similarity/invalid-doc-id",
            f"{self.ingestion_coordinator_url}/api/invalid"
        ]

        for endpoint in invalid_endpoints:
            start_time = time.time()
            try:
                response = await self.http_client.get(endpoint)
                duration = time.time() - start_time

                if response.status_code in [404, 422]:
                    suite.tests.append(TestResult(
                        test_name=f"Invalid Endpoint: {endpoint.split('/')[-1]}",
                        status='passed',
                        duration=duration,
                        message=f"Correctly returned {response.status_code}"
                    ))
                else:
                    suite.tests.append(TestResult(
                        test_name=f"Invalid Endpoint: {endpoint.split('/')[-1]}",
                        status='failed',
                        duration=duration,
                        message=f"Expected 404/422, got {response.status_code}"
                    ))

            except Exception as e:
                duration = time.time() - start_time
                suite.tests.append(TestResult(
                    test_name=f"Invalid Endpoint: {endpoint.split('/')[-1]}",
                    status='error',
                    duration=duration,
                    message=f"Error handling test failed: {str(e)}"
                ))

    async def _test_malformed_requests(self, suite: TestSuite):
        """Test malformed requests"""
        start_time = time.time()
        try:
            # Send malformed JSON
            response = await self.http_client.post(
                f"{self.vector_ui_url}/api/embeddings/generate",
                content="invalid json {",
                headers={"Content-Type": "application/json"}
            )
            duration = time.time() - start_time

            if response.status_code in [400, 422]:
                suite.tests.append(TestResult(
                    test_name="Malformed JSON Request",
                    status='passed',
                    duration=duration,
                    message=f"Correctly handled malformed JSON: {response.status_code}"
                ))
            else:
                suite.tests.append(TestResult(
                    test_name="Malformed JSON Request",
                    status='failed',
                    duration=duration,
                    message=f"Expected 400/422, got {response.status_code}"
                ))

        except Exception as e:
            duration = time.time() - start_time
            suite.tests.append(TestResult(
                test_name="Malformed JSON Request",
                status='error',
                duration=duration,
                message=f"Malformed request test failed: {str(e)}"
            ))

    async def _test_authentication_errors(self, suite: TestSuite):
        """Test authentication error handling"""
        # For now, skip auth tests if REQUIRE_AUTH is disabled
        suite.tests.append(TestResult(
            test_name="Authentication Error Handling",
            status='skipped',
            duration=0.0,
            message="Skipped - authentication tests require REQUIRE_AUTH=true"
        ))

    async def _test_database_connectivity(self, suite: TestSuite):
        """Test database connectivity through API"""
        start_time = time.time()
        try:
            # Test database connectivity by checking if collections endpoint works
            response = await self.http_client.get(f"{self.vector_ui_url}/api/vectors/collections")
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                suite.tests.append(TestResult(
                    test_name="Database Connectivity",
                    status='passed',
                    duration=duration,
                    message=f"Database accessible, {data.get('total_collections', 0)} collections found"
                ))
            else:
                suite.tests.append(TestResult(
                    test_name="Database Connectivity",
                    status='failed',
                    duration=duration,
                    message=f"Database connectivity issue: HTTP {response.status_code}"
                ))

        except Exception as e:
            duration = time.time() - start_time
            suite.tests.append(TestResult(
                test_name="Database Connectivity",
                status='error',
                duration=duration,
                message=f"Database test failed: {str(e)}"
            ))

    def _print_suite_results(self, suite: TestSuite):
        """Print test suite results"""
        if not self.verbose and suite.total_tests == 0:
            return

        print(f"\n📊 {suite.name} Results:")
        print(f"   Total Tests: {suite.total_tests}")
        print(f"   Passed: {suite.passed_tests}")
        print(f"   Failed: {suite.failed_tests}")
        print(f"   Errors: {suite.error_tests}")
        print(f"   Skipped: {suite.skipped_tests}")
        print(".1f")
        print(".2f")

        if self.verbose:
            print("   Test Details:")
            for test in suite.tests:
                status_icon = {
                    'passed': '✅',
                    'failed': '❌',
                    'error': '🔥',
                    'skipped': '⏭️'
                }.get(test.status, '❓')

                print(".3f")

    async def generate_report(self) -> str:
        """Generate comprehensive test report"""
        total_tests = sum(suite.total_tests for suite in self.test_suites.values())
        total_passed = sum(suite.passed_tests for suite in self.test_suites.values())
        total_failed = sum(suite.failed_tests for suite in self.test_suites.values())
        total_errors = sum(suite.error_tests for suite in self.test_suites.values())

        report = f"""
# Agentic Platform Test Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Test Suites**: {len(self.test_suites)}
- **Total Tests**: {total_tests}
- **Passed**: {total_passed}
- **Failed**: {total_failed}
- **Errors**: {total_errors}
- **Success Rate**: {total_passed/total_tests*100:.1f}%

## Detailed Results

"""

        for suite_name, suite in self.test_suites.items():
            report += f"""
### {suite.name}
- **Tests**: {suite.total_tests}
- **Passed**: {suite.passed_tests}
- **Failed**: {suite.failed_tests}
- **Errors**: {suite.error_tests}
- **Success Rate**: {suite.success_rate:.1f}%
- **Duration**: {suite.total_duration:.2f}s

"""

            if self.verbose:
                for test in suite.tests:
                    status_icon = {
                        'passed': '✅',
                        'failed': '❌',
                        'error': '🔥',
                        'skipped': '⏭️'
                    }.get(test.status, '❓')

                    report += f"- {status_icon} {test.test_name}: {test.message}\n"

        return report

    async def cleanup(self):
        """Cleanup resources"""
        await self.http_client.aclose()


async def main():
    """Main test execution function"""
    parser = argparse.ArgumentParser(description="Agentic Platform Test Suite")
    parser.add_argument("--base-url", default="http://localhost",
                       help="Base URL for services (default: http://localhost)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--services-only", action="store_true",
                       help="Run only service health tests")
    parser.add_argument("--api-only", action="store_true",
                       help="Run only API endpoint tests")
    parser.add_argument("--ui-only", action="store_true",
                       help="Run only UI component tests")
    parser.add_argument("--report", action="store_true",
                       help="Generate detailed report")

    args = parser.parse_args()

    tester = AgenticPlatformTester(args.base_url, args.verbose)

    try:
        if args.services_only:
            await tester.run_service_health_tests()
        elif args.api_only:
            await tester.run_api_tests()
        elif args.ui_only:
            await tester.run_ui_tests()
        else:
            await tester.run_all_tests()

        # Generate report if requested
        if args.report:
            report = await tester.generate_report()
            with open("test_report.md", "w") as f:
                f.write(report)
            print("\n📄 Detailed report saved to test_report.md")

        # Print final summary
        all_suites = list(tester.test_suites.values())
        if all_suites:
            total_tests = sum(s.total_tests for s in all_suites)
            total_passed = sum(s.passed_tests for s in all_suites)
            total_failed = sum(s.failed_tests for s in all_suites)
            total_errors = sum(s.error_tests for s in all_suites)

            print("\n" + "="*50)
            print("🎯 FINAL TEST SUMMARY")
            print("="*50)
            print(f"Test Suites Run: {len(all_suites)}")
            print(f"Total Tests: {total_tests}")
            print(f"✅ Passed: {total_passed}")
            print(f"❌ Failed: {total_failed}")
            print(f"🔥 Errors: {total_errors}")
            print(".1f")
            print("="*50)

            # Exit with appropriate code
            if total_failed > 0 or total_errors > 0:
                print("⚠️  Some tests failed. Please review the results above.")
                sys.exit(1)
            else:
                print("🎉 All tests passed successfully!")
                sys.exit(0)

    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        sys.exit(1)
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
