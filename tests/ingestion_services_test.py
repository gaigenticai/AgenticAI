#!/usr/bin/env python3
"""
Comprehensive Automated Testing Suite for Ingestion Services
Tests all ingestion services for functionality, performance, and reliability
"""

import requests
import json
import time
import pytest
from typing import Dict, List, Any
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionServiceTester:
    """Comprehensive test suite for all ingestion services"""

    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        self.services = {
            'csv': 8080,
            'json': 8085,
            'pdf': 8083,
            'excel': 8084,
            'api': 8086,
            'ui-scraper': 8087
        }
        self.test_results = {}

    def test_service_health(self, service_name: str) -> Dict[str, Any]:
        """Test service health endpoint"""
        try:
            port = self.services[service_name]
            url = f"{self.base_url}:{port}/health"

            start_time = time.time()
            response = requests.get(url, timeout=10)
            response_time = time.time() - start_time

            result = {
                'service': service_name,
                'endpoint': 'health',
                'status_code': response.status_code,
                'response_time': response_time,
                'success': response.status_code == 200,
                'data': response.json() if response.status_code == 200 else None,
                'timestamp': datetime.now().isoformat()
            }

            if response.status_code == 200:
                data = response.json()
                result.update({
                    'service_status': data.get('status'),
                    'active_jobs': data.get('active_jobs', 0),
                    'service_name': data.get('service')
                })

            return result

        except Exception as e:
            return {
                'service': service_name,
                'endpoint': 'health',
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def test_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Test service metrics endpoint"""
        try:
            port = self.services[service_name]
            url = f"{self.base_url}:{port}/metrics"

            start_time = time.time()
            response = requests.get(url, timeout=10)
            response_time = time.time() - start_time

            result = {
                'service': service_name,
                'endpoint': 'metrics',
                'status_code': response.status_code,
                'response_time': response_time,
                'success': response.status_code == 200,
                'timestamp': datetime.now().isoformat()
            }

            if response.status_code == 200:
                metrics_text = response.text
                result.update({
                    'metrics_available': len(metrics_text) > 0,
                    'metrics_lines': len(metrics_text.split('\n')),
                    'has_prometheus_metrics': 'prometheus' in metrics_text.lower()
                })

            return result

        except Exception as e:
            return {
                'service': service_name,
                'endpoint': 'metrics',
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def test_service_connectivity(self, service_name: str) -> Dict[str, Any]:
        """Test basic connectivity to service"""
        try:
            port = self.services[service_name]
            url = f"{self.base_url}:{port}"

            start_time = time.time()
            response = requests.get(url, timeout=5)
            response_time = time.time() - start_time

            return {
                'service': service_name,
                'endpoint': 'connectivity',
                'status_code': response.status_code,
                'response_time': response_time,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }

        except requests.exceptions.ConnectionError:
            return {
                'service': service_name,
                'endpoint': 'connectivity',
                'success': False,
                'error': 'Connection refused',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'service': service_name,
                'endpoint': 'connectivity',
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive tests for all services"""
        logger.info("Starting comprehensive ingestion services test...")

        results = {
            'test_start_time': datetime.now().isoformat(),
            'services_tested': [],
            'overall_success': True,
            'summary': {}
        }

        for service_name in self.services.keys():
            logger.info(f"Testing {service_name} service...")

            service_results = {
                'service': service_name,
                'tests': []
            }

            # Test connectivity
            connectivity_test = self.test_service_connectivity(service_name)
            service_results['tests'].append(connectivity_test)

            # Only test health and metrics if connectivity works
            if connectivity_test['success']:
                # Test health
                health_test = self.test_service_health(service_name)
                service_results['tests'].append(health_test)

                # Test metrics
                metrics_test = self.test_service_metrics(service_name)
                service_results['tests'].append(metrics_test)

                # Check overall service success
                service_success = all(test['success'] for test in [health_test, metrics_test])
                service_results['service_success'] = service_success

                if not service_success:
                    results['overall_success'] = False
            else:
                service_results['service_success'] = False
                results['overall_success'] = False

            results['services_tested'].append(service_results)

        # Generate summary
        results['summary'] = self.generate_summary(results)
        results['test_end_time'] = datetime.now().isoformat()

        logger.info("Comprehensive test completed")
        return results

    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary"""
        summary = {
            'total_services': len(self.services),
            'services_tested': len(results['services_tested']),
            'successful_services': 0,
            'failed_services': 0,
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'average_response_time': 0.0
        }

        response_times = []

        for service in results['services_tested']:
            if service['service_success']:
                summary['successful_services'] += 1
            else:
                summary['failed_services'] += 1

            for test in service['tests']:
                summary['total_tests'] += 1
                if test['success']:
                    summary['successful_tests'] += 1
                else:
                    summary['failed_tests'] += 1

                if 'response_time' in test:
                    response_times.append(test['response_time'])

        if response_times:
            summary['average_response_time'] = sum(response_times) / len(response_times)

        summary['success_rate'] = (summary['successful_tests'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0

        return summary

    def print_results(self, results: Dict[str, Any]):
        """Print test results in a formatted way"""
        print("\n" + "="*80)
        print("INGESTION SERVICES AUTOMATED TEST RESULTS")
        print("="*80)

        print("\n📊 OVERALL SUMMARY:")
        summary = results['summary']
        print(f"  • Total Services: {summary['total_services']}")
        print(f"  • Services Tested: {summary['services_tested']}")
        print(f"  • Successful Services: {summary['successful_services']}")
        print(f"  • Failed Services: {summary['failed_services']}")
        print(f"  • Success Rate: {summary['success_rate']:.1f}%")
        print(f"  • Average Response Time: {summary['average_response_time']:.3f}s")

        print("\n🔍 SERVICE DETAILS:")
        for service in results['services_tested']:
            status_icon = "✅" if service['service_success'] else "❌"
            print(f"\n{status_icon} {service['service'].upper()} SERVICE:")

            for test in service['tests']:
                test_status = "✅" if test['success'] else "❌"
                response_time = f" ({test.get('response_time', 0):.3f}s)" if 'response_time' in test else ""

                if test['endpoint'] == 'connectivity':
                    endpoint_name = "Connectivity"
                elif test['endpoint'] == 'health':
                    endpoint_name = "Health Check"
                elif test['endpoint'] == 'metrics':
                    endpoint_name = "Metrics"
                else:
                    endpoint_name = test['endpoint']

                print(f"  {test_status} {endpoint_name}{response_time}")

                if not test['success'] and 'error' in test:
                    print(f"    Error: {test['error']}")

        print("\n⏰ TEST TIMING:")
        print(f"  • Started: {results['test_start_time']}")
        print(f"  • Completed: {results['test_end_time']}")

        print("\n" + "="*80)


def main():
    """Main test execution function"""
    print("🚀 Starting Automated Ingestion Services Testing...")

    # Initialize tester
    tester = IngestionServiceTester()

    # Run comprehensive tests
    results = tester.run_comprehensive_test()

    # Print formatted results
    tester.print_results(results)

    # Save results to file
    results_file = f"/tmp/ingestion_services_test_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Detailed results saved to: {results_file}")

    # Return exit code based on success
    return 0 if results['overall_success'] else 1


if __name__ == "__main__":
    exit(main())
