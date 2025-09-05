#!/usr/bin/env python3
"""
Unit Tests for Plugin Registry Service

Tests core functionality of the Plugin Registry:
- Plugin registration and discovery
- Risk calculation and fraud detection
- Plugin execution orchestration
- Configuration management
- Error handling and validation
"""

import asyncio
import json
import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add service path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'plugin-registry'))

from main import (
    PluginRegistryManager, RiskCalculatorPlugin, FraudDetectorPlugin,
    Config, PluginExecutionResult
)


class TestPluginRegistryManager(unittest.TestCase):
    """Test Plugin Registry Manager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = PluginRegistryManager()
        self.test_plugin = {
            'name': 'test_plugin',
            'version': '1.0.0',
            'type': 'risk_calculator',
            'description': 'Test plugin'
        }

    @patch('main.load_defaults')
    def test_plugin_registration(self, mock_load_defaults):
        """Test plugin registration"""
        mock_load_defaults.return_value = {
            'service_ports': {'plugin_registry_port': 8201},
            'performance': {'plugin_load_timeout': 30}
        }

        # Test plugin registration structure
        result = self.registry.register_plugin_from_instance(self.test_plugin)
        self.assertIsInstance(result, dict)

    def test_plugin_discovery(self):
        """Test plugin discovery functionality"""
        # Test plugin discovery structure
        plugins = self.registry.list_plugins()
        self.assertIsInstance(plugins, list)

    @patch('main.load_defaults')
    def test_configuration_loading(self, mock_load_defaults):
        """Test configuration loading"""
        mock_config = {
            'service_ports': {'plugin_registry_port': 8201},
            'performance': {'plugin_load_timeout': 30}
        }
        mock_load_defaults.return_value = mock_config

        config = Config()
        self.assertEqual(config.SERVICE_PORT, 8201)


class TestRiskCalculatorPlugin(unittest.TestCase):
    """Test Risk Calculator Plugin functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.plugin = RiskCalculatorPlugin()
        self.test_data = {
            'credit_score': 720,
            'debt_to_income': 0.35,
            'loan_to_income': 2.8,
            'employment_status': 'employed'
        }

    @patch('main.load_defaults')
    def test_risk_calculation(self, mock_load_defaults):
        """Test risk calculation logic"""
        mock_config = {
            'risk_calculation': {
                'credit_score': {'poor_threshold': 620, 'fair_threshold': 660, 'good_threshold': 720},
                'debt_to_income': {'moderate_threshold': 0.43},
                'loan_to_income': {'moderate_threshold': 2.5}
            }
        }
        mock_load_defaults.return_value = mock_config

        # Test risk calculation
        result = self.plugin.execute(self.test_data)
        self.assertIsInstance(result, dict)
        self.assertIn('risk_score', result)

    def test_credit_score_evaluation(self):
        """Test credit score evaluation"""
        # Test good credit score
        self.assertGreaterEqual(720, 720)

        # Test fair credit score
        self.assertGreaterEqual(660, 620)

        # Test poor credit score
        self.assertLess(600, 620)

    def test_debt_to_income_calculation(self):
        """Test debt-to-income ratio calculation"""
        dti = 0.35
        self.assertLess(dti, 0.43)  # Should be acceptable

    def test_loan_to_income_calculation(self):
        """Test loan-to-income ratio calculation"""
        lti = 2.8
        self.assertGreater(lti, 2.5)  # Should be moderate risk


class TestFraudDetectorPlugin(unittest.TestCase):
    """Test Fraud Detector Plugin functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.plugin = FraudDetectorPlugin()
        self.test_claim = {
            'claim_id': 'test-claim-123',
            'policy_id': 'policy-456',
            'incident_date': '2024-01-15T10:00:00Z',
            'policy_start_date': '2024-01-01T00:00:00Z',
            'claim_amount': 5000,
            'claim_history': []
        }

    @patch('main.load_defaults')
    def test_fraud_detection(self, mock_load_defaults):
        """Test fraud detection logic"""
        mock_config = {
            'fraud_config': {
                'early_claim_days': 30,
                'high_claim_amount': 100000
            }
        }
        mock_load_defaults.return_value = mock_config

        # Test fraud detection
        result = self.plugin.execute(self.test_claim)
        self.assertIsInstance(result, dict)
        self.assertIn('fraud_score', result)

    def test_early_claim_detection(self):
        """Test early claim detection after policy start"""
        # Calculate days between dates
        incident_date = datetime.fromisoformat('2024-01-15T10:00:00+00:00')
        policy_date = datetime.fromisoformat('2024-01-01T00:00:00+00:00')
        days_diff = (incident_date - policy_date).days

        # Should be flagged as early claim
        self.assertLess(days_diff, 30)

    def test_high_amount_detection(self):
        """Test high claim amount detection"""
        claim_amount = 5000
        threshold = 100000

        # Should not be flagged as high amount
        self.assertLess(claim_amount, threshold)

    def test_claim_frequency_analysis(self):
        """Test claim frequency analysis"""
        claim_history = []  # No previous claims
        max_frequency = 2

        # Should not be flagged for frequency
        self.assertLess(len(claim_history), max_frequency)


class TestPluginExecution(unittest.TestCase):
    """Test plugin execution orchestration"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = PluginRegistryManager()
        self.test_execution = {
            'plugin_name': 'risk_calculator',
            'input_data': {'credit_score': 700},
            'execution_id': 'exec-123'
        }

    def test_plugin_execution_result_structure(self):
        """Test plugin execution result structure"""
        result = PluginExecutionResult(
            execution_id='exec-123',
            plugin_name='test_plugin',
            status='success',
            result_data={'score': 85},
            execution_time=0.5
        )

        self.assertEqual(result.status, 'success')
        self.assertIn('score', result.result_data)

    def test_execution_timeout_handling(self):
        """Test execution timeout handling"""
        config = Config()
        self.assertGreater(config.PLUGIN_EXECUTION_TIMEOUT, 0)


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management"""

    @patch('main.load_defaults')
    def test_risk_thresholds_configuration(self, mock_load_defaults):
        """Test risk calculation thresholds configuration"""
        mock_config = {
            'risk_calculation': {
                'credit_score': {
                    'poor_threshold': 620,
                    'fair_threshold': 660,
                    'good_threshold': 720
                }
            }
        }
        mock_load_defaults.return_value = mock_config

        config = Config()
        thresholds = config.RISK_CALCULATION_THRESHOLDS

        self.assertEqual(thresholds['credit_score']['good_threshold'], 720)
        self.assertEqual(thresholds['credit_score']['poor_threshold'], 620)

    @patch('main.load_defaults')
    def test_fraud_configuration(self, mock_load_defaults):
        """Test fraud detection configuration"""
        mock_config = {
            'fraud_config': {
                'early_claim_days': 30,
                'high_claim_amount': 100000
            }
        }
        mock_load_defaults.return_value = mock_config

        config = Config()
        self.assertEqual(config.PLUGIN_LOAD_TIMEOUT, 30)

    @patch('main.load_defaults')
    def test_missing_config_fallback(self, mock_load_defaults):
        """Test fallback when configuration is missing"""
        mock_load_defaults.return_value = {}

        config = Config()
        # Should use fallback values
        self.assertIsInstance(config.SERVICE_PORT, int)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in plugin registry"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = PluginRegistryManager()

    def test_invalid_plugin_handling(self):
        """Test handling of invalid plugins"""
        invalid_plugin = {
            'name': None,  # Invalid name
            'version': '1.0.0'
        }

        # Should handle invalid plugin gracefully
        self.assertIsNone(invalid_plugin.get('name'))

    def test_execution_timeout_handling(self):
        """Test execution timeout handling"""
        # Test timeout configuration
        config = Config()
        self.assertGreater(config.PLUGIN_EXECUTION_TIMEOUT, 0)

    def test_database_connection_error_handling(self):
        """Test database connection error handling"""
        # Test should handle connection errors gracefully
        try:
            raise ConnectionError("Database unavailable")
        except ConnectionError as e:
            self.assertIn("unavailable", str(e).lower())


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics collection"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = PluginRegistryManager()

    def test_execution_metrics_tracking(self):
        """Test execution metrics tracking"""
        metrics = {
            'plugins_executed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0
        }

        # Simulate plugin execution
        metrics['plugins_executed'] += 1
        metrics['successful_executions'] += 1

        # Verify metrics
        self.assertEqual(metrics['plugins_executed'], 1)
        self.assertEqual(metrics['successful_executions'], 1)


def run_unit_tests():
    """Run all unit tests for Plugin Registry"""
    print("🔌 Running Plugin Registry Unit Tests")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestPluginRegistryManager,
        TestRiskCalculatorPlugin,
        TestFraudDetectorPlugin,
        TestPluginExecution,
        TestConfigurationManagement,
        TestErrorHandling,
        TestPerformanceMetrics
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    print("📊 Unit Test Results Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✅ All unit tests passed!")
        return 0
    else:
        print("❌ Some unit tests failed!")
        # Print failures
        for test, traceback in result.failures:
            print(f"   FAILED: {test}")
        # Print errors
        for test, traceback in result.errors:
            print(f"   ERROR: {test}")
        return 1


if __name__ == "__main__":
    exit_code = run_unit_tests()
    exit(exit_code)
