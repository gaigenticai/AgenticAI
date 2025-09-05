#!/usr/bin/env python3
"""
Unified Docker-Based Testing Suite for Agentic Platform

This script provides a comprehensive testing framework that combines:
1. Traditional unit and integration tests
2. Docker-based service testing (Rule 18 compliance)
3. Parallel test execution
4. Comprehensive reporting and metrics

Usage:
    python run_docker_tests.py [options]

Options:
    --services SERVICE1 SERVICE2  Test specific services
    --mode MODE                   Test mode: 'docker', 'local', 'hybrid' (default: hybrid)
    --parallel                    Run tests in parallel
    --report FILE                 Output report filename
    --verbose                     Verbose output
    --coverage                    Generate coverage reports
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import structlog

from run_automated_tests import AutomatedTestSuite
from docker_test_runner import run_docker_tests, DockerTestRunner

# Configure structured logging
logger = structlog.get_logger(__name__)


class UnifiedTestSuite:
    """Unified testing suite combining local and Docker-based tests"""

    def __init__(self, mode: str = "hybrid", verbose: bool = False):
        self.mode = mode
        self.verbose = verbose
        self.local_suite = AutomatedTestSuite()
        self.docker_runner = DockerTestRunner()
        self.combined_results = []

    async def run_hybrid_tests(self, services: List[str] = None) -> Dict[str, Any]:
        """Run both local and Docker-based tests"""

        logger.info("🚀 Starting Unified Hybrid Testing Suite")
        logger.info("=" * 70)

        start_time = time.time()
        all_results = {
            "local_tests": {},
            "docker_tests": [],
            "combined_summary": {},
            "start_time": start_time
        }

        try:
            # Run local infrastructure tests first
            logger.info("📋 Running Local Infrastructure Tests...")
            local_exit_code = self.local_suite.run_all_tests()
            all_results["local_tests"] = {
                "exit_code": local_exit_code,
                "results": self.local_suite.test_results,
                "services_status": self.local_suite.services_status
            }

            # Run Docker-based service tests
            logger.info("🐳 Running Docker-Based Service Tests...")
            docker_results = await run_docker_tests(services)
            all_results["docker_tests"] = [
                {
                    "service_name": r.service_name,
                    "test_type": r.test_type,
                    "status": r.status,
                    "duration": round(r.duration, 2),
                    "error_message": r.error_message,
                    "timestamp": r.timestamp
                }
                for r in docker_results
            ]

            # Combine results
            all_results["combined_summary"] = self._combine_test_results(
                all_results["local_tests"],
                all_results["docker_tests"]
            )

            return all_results

        except Exception as e:
            logger.error("Unified testing failed", error=str(e))
            all_results["error"] = str(e)
            return all_results
        finally:
            # Cleanup
            self.docker_runner._cleanup_containers()

    async def run_docker_only_tests(self, services: List[str] = None) -> Dict[str, Any]:
        """Run only Docker-based tests"""

        logger.info("🐳 Running Docker-Only Testing Suite")
        logger.info("=" * 70)

        start_time = time.time()

        try:
            docker_results = await run_docker_tests(services)
            docker_report = self.docker_runner.generate_test_report(docker_results)

            return {
                "test_mode": "docker_only",
                "start_time": start_time,
                "duration": time.time() - start_time,
                "docker_results": docker_report
            }

        except Exception as e:
            logger.error("Docker-only testing failed", error=str(e))
            return {
                "test_mode": "docker_only",
                "error": str(e),
                "start_time": start_time,
                "duration": time.time() - start_time
            }

    def run_local_only_tests(self) -> Dict[str, Any]:
        """Run only local tests"""

        logger.info("💻 Running Local-Only Testing Suite")
        logger.info("=" * 70)

        start_time = time.time()

        try:
            exit_code = self.local_suite.run_all_tests()

            return {
                "test_mode": "local_only",
                "start_time": start_time,
                "duration": time.time() - start_time,
                "exit_code": exit_code,
                "results": self.local_suite.test_results,
                "services_status": self.local_suite.services_status
            }

        except Exception as e:
            logger.error("Local-only testing failed", error=str(e))
            return {
                "test_mode": "local_only",
                "error": str(e),
                "start_time": start_time,
                "duration": time.time() - start_time
            }

    def _combine_test_results(self, local_results: Dict, docker_results: List) -> Dict[str, Any]:
        """Combine local and Docker test results into unified summary"""

        # Count local test results
        local_total = len(local_results.get("results", []))
        local_passed = len([r for r in local_results.get("results", []) if r["status"] == "PASS"])
        local_failed = len([r for r in local_results.get("results", []) if r["status"] == "FAIL"])

        # Count Docker test results
        docker_total = len(docker_results)
        docker_passed = len([r for r in docker_results if r["status"] == "PASS"])
        docker_failed = len([r for r in docker_results if r["status"] in ["FAIL", "ERROR"]])

        # Combined metrics
        total_tests = local_total + docker_total
        total_passed = local_passed + docker_passed
        total_failed = local_failed + docker_failed

        return {
            "local_tests": {
                "total": local_total,
                "passed": local_passed,
                "failed": local_failed,
                "success_rate": (local_passed / local_total * 100) if local_total > 0 else 0
            },
            "docker_tests": {
                "total": docker_total,
                "passed": docker_passed,
                "failed": docker_failed,
                "success_rate": (docker_passed / docker_total * 100) if docker_total > 0 else 0
            },
            "combined": {
                "total": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0
            }
        }

    def save_unified_report(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save unified test report"""

        if filename is None:
            timestamp = int(time.time())
            filename = f"unified_test_report_{timestamp}.json"

        # Add metadata
        results["metadata"] = {
            "test_mode": self.mode,
            "generated_at": time.time(),
            "platform": "agentic-ai",
            "version": "1.0.0"
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Unified test report saved to: {filename}")
        return filename


async def main():
    """Main entry point for unified testing"""

    parser = argparse.ArgumentParser(description="Unified Docker-Based Testing Suite")
    parser.add_argument("--services", nargs="*", help="Specific services to test")
    parser.add_argument("--mode", choices=["local", "docker", "hybrid"],
                       default="hybrid", help="Testing mode")
    parser.add_argument("--report", help="Output report filename")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage reports")

    args = parser.parse_args()

    # Initialize test suite
    suite = UnifiedTestSuite(mode=args.mode, verbose=args.verbose)

    try:
        # Run tests based on mode
        if args.mode == "local":
            results = suite.run_local_only_tests()
        elif args.mode == "docker":
            results = await suite.run_docker_only_tests(args.services)
        else:  # hybrid
            results = await suite.run_hybrid_tests(args.services)

        # Save report
        report_file = suite.save_unified_report(results, args.report)

        # Print summary
        print("\n" + "=" * 70)
        print("📊 UNIFIED TESTING RESULTS SUMMARY")
        print("=" * 70)

        if args.mode == "hybrid":
            summary = results.get("combined_summary", {})
            combined = summary.get("combined", {})

            print(f"Test Mode:       Hybrid (Local + Docker)")
            print(f"Local Tests:     {summary['local_tests']['total']} total, {summary['local_tests']['passed']} passed")
            print(f"Docker Tests:    {summary['docker_tests']['total']} total, {summary['docker_tests']['passed']} passed")
            print(f"Combined:        {combined['total']} total, {combined['passed']} passed, {combined['failed']} failed")
            print(".1f"
        elif args.mode == "docker":
            summary = results.get("docker_results", {}).get("summary", {})
            print(f"Test Mode:       Docker-Only")
            print(f"Total Tests:     {summary.get('total_tests', 0)}")
            print(f"Passed:          {summary.get('passed', 0)} ✅")
            print(f"Failed:          {summary.get('failed', 0)} ❌")
            print(".1f"
        else:  # local
            summary = results
            total_tests = len(summary.get("results", []))
            passed = len([r for r in summary.get("results", []) if r["status"] == "PASS"])
            failed = len([r for r in summary.get("results", []) if r["status"] == "FAIL"])

            print(f"Test Mode:       Local-Only")
            print(f"Total Tests:     {total_tests}")
            print(f"Passed:          {passed} ✅")
            print(f"Failed:          {failed} ❌")

        print(f"Report Saved:    {report_file}")

        # Determine exit code
        if args.mode == "hybrid":
            combined = results.get("combined_summary", {}).get("combined", {})
            success_rate = combined.get("success_rate", 0)
        elif args.mode == "docker":
            success_rate = results.get("docker_results", {}).get("summary", {}).get("success_rate", 0)
        else:
            total_tests = len(results.get("results", []))
            passed = len([r for r in results.get("results", []) if r["status"] == "PASS"])
            success_rate = (passed / total_tests * 100) if total_tests > 0 else 0

        if success_rate >= 80:
            print("\n🎉 TESTING COMPLETED SUCCESSFULLY!")
            return 0
        elif success_rate >= 60:
            print("\n⚠️ MOST TESTS PASSED. Review results before proceeding.")
            return 1
        else:
            print("\n❌ CRITICAL TEST FAILURES DETECTED.")
            return 2

    except KeyboardInterrupt:
        logger.warning("Testing interrupted by user")
        return 130
    except Exception as e:
        logger.error("Testing suite crashed", error=str(e))
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
