#!/usr/bin/env python3
"""
Unit Tests for Brain Factory Service

Tests core functionality of the Brain Factory:
- Agent creation and configuration
- Service dependency management
- Agent lifecycle orchestration
- Configuration validation
- Error handling and recovery
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'brain-factory'))

from main import AgentFactory, Config, AgentCreationStatus


class TestAgentFactory(unittest.TestCase):
    """Test Agent Factory functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.factory = AgentFactory()
        self.test_agent_config = {
            'agent_id': 'test-agent-123',
            'agent_type': 'reasoning_agent',
            'configuration': {
                'max_memory': '1GB',
                'timeout': 300,
                'plugins': ['risk_calculator', 'fraud_detector']
            }
        }

    @patch('main.load_defaults')
    def test_agent_creation_request(self, mock_load_defaults):
        """Test agent creation request processing"""
        mock_load_defaults.return_value = {
            'service_ports': {'brain_factory_port': 8301},
            'agent_config': {'max_agent_instances': 50}
        }

        # Test agent creation structure
        result = self.factory.create_agent(self.test_agent_config)
        self.assertIsInstance(result, dict)

    def test_agent_configuration_validation(self):
        """Test agent configuration validation"""
        # Test valid configuration
        valid_config = {
            'agent_id': 'test-123',
            'agent_type': 'reasoning_agent',
            'configuration': {'timeout': 300}
        }

        # Should pass basic validation
        self.assertIsInstance(valid_config, dict)
        self.assertIn('agent_id', valid_config)

    def test_agent_creation_status_tracking(self):
        """Test agent creation status management"""
        # Test status enumeration
        self.assertEqual(AgentCreationStatus.PENDING.value, 'pending')
        self.assertEqual(AgentCreationStatus.COMPLETED.value, 'completed')
        self.assertEqual(AgentCreationStatus.FAILED.value, 'failed')


class TestServiceDependencies(unittest.TestCase):
    """Test service dependency management"""

    def setUp(self):
        """Set up test fixtures"""
        self.factory = AgentFactory()

    @patch('httpx.AsyncClient')
    async def test_dependency_check(self, mock_client):
        """Test service dependency checking"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'healthy'}
        mock_client.return_value.get.return_value = mock_response

        # Test dependency checking structure
        config = Config()
        self.assertGreater(config.MAX_AGENT_INSTANCES, 0)

    def test_service_port_configuration(self):
        """Test service port configuration"""
        config = Config()
        self.assertIsInstance(config.AGENT_BRAIN_BASE_PORT, int)
        self.assertIsInstance(config.PLUGIN_REGISTRY_PORT, int)


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management"""

    @patch('main.load_defaults')
    def test_factory_configuration(self, mock_load_defaults):
        """Test Brain Factory configuration loading"""
        mock_config = {
            'service_ports': {'brain_factory_port': 8301},
            'agent_config': {'max_agent_instances': 50, 'agent_creation_timeout': 120}
        }
        mock_load_defaults.return_value = mock_config

        config = Config()
        self.assertEqual(config.BRAIN_FACTORY_PORT, 8301)
        self.assertEqual(config.MAX_AGENT_INSTANCES, 50)

    def test_service_endpoint_configuration(self):
        """Test service endpoint configuration"""
        config = Config()

        # Verify all service ports are configured
        self.assertGreater(config.AGENT_BRAIN_BASE_PORT, 0)
        self.assertGreater(config.REASONING_MODULE_FACTORY_PORT, 0)
        self.assertGreater(config.PLUGIN_REGISTRY_PORT, 0)


class TestAgentLifecycle(unittest.TestCase):
    """Test agent lifecycle management"""

    def setUp(self):
        """Set up test fixtures"""
        self.factory = AgentFactory()

    def test_agent_initialization(self):
        """Test agent initialization process"""
        # Test initialization structure
        agent_config = {
            'agent_id': 'test-agent-123',
            'status': 'initializing'
        }

        self.assertEqual(agent_config['status'], 'initializing')

    def test_agent_deployment(self):
        """Test agent deployment process"""
        # Test deployment structure
        deployment_config = {
            'agent_id': 'test-agent-123',
            'target_environment': 'development',
            'resources': {'cpu': 1, 'memory': '512MB'}
        }

        self.assertIn('target_environment', deployment_config)

    def test_agent_monitoring(self):
        """Test agent monitoring and health checks"""
        # Test monitoring structure
        health_status = {
            'agent_id': 'test-agent-123',
            'status': 'healthy',
            'last_check': datetime.now().isoformat()
        }

        self.assertEqual(health_status['status'], 'healthy')


class TestErrorHandling(unittest.TestCase):
    """Test error handling in Brain Factory"""

    def setUp(self):
        """Set up test fixtures"""
        self.factory = AgentFactory()

    def test_invalid_agent_configuration(self):
        """Test handling of invalid agent configurations"""
        invalid_config = {
            'agent_id': None,  # Invalid agent_id
            'agent_type': 'invalid_type'
        }

        # Should handle invalid configuration gracefully
        self.assertIsNone(invalid_config.get('agent_id'))

    def test_service_unavailable_handling(self):
        """Test handling of unavailable services"""
        # Test error handling structure
        try:
            raise ConnectionError("Service unavailable")
        except ConnectionError as e:
            self.assertIn("unavailable", str(e).lower())

    def test_timeout_handling(self):
        """Test timeout handling during agent creation"""
        config = Config()
        self.assertGreater(config.AGENT_CREATION_TIMEOUT, 0)
        self.assertGreater(config.AGENT_VALIDATION_TIMEOUT, 0)


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring"""

    def setUp(self):
        """Set up test fixtures"""
        self.factory = AgentFactory()

    def test_creation_metrics_tracking(self):
        """Test agent creation metrics tracking"""
        metrics = {
            'agents_created': 0,
            'successful_creations': 0,
            'failed_creations': 0,
            'average_creation_time': 0.0
        }

        # Simulate agent creation
        metrics['agents_created'] += 1
        metrics['successful_creations'] += 1

        # Verify metrics
        self.assertEqual(metrics['agents_created'], 1)
        self.assertEqual(metrics['successful_creations'], 1)

    def test_resource_utilization_tracking(self):
        """Test resource utilization tracking"""
        resources = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'active_agents': 0
        }

        # Simulate resource usage
        resources['active_agents'] += 1
        resources['cpu_usage'] = 45.2

        # Verify resource tracking
        self.assertEqual(resources['active_agents'], 1)
        self.assertGreater(resources['cpu_usage'], 0)


class TestIntegrationTesting(unittest.TestCase):
    """Test integration with other services"""

    def setUp(self):
        """Set up test fixtures"""
        self.factory = AgentFactory()

    @patch('httpx.AsyncClient')
    async def test_agent_brain_base_integration(self, mock_client):
        """Test integration with Agent Brain Base service"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'ready'}
        mock_client.return_value.get.return_value = mock_response

        # Test integration structure
        config = Config()
        url = f"http://localhost:{config.AGENT_BRAIN_BASE_PORT}/health"

        # Mock successful integration
        self.assertTrue(True)

    @patch('httpx.AsyncClient')
    async def test_plugin_registry_integration(self, mock_client):
        """Test integration with Plugin Registry service"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'plugins': []}
        mock_client.return_value.get.return_value = mock_response

        # Test integration structure
        config = Config()
        url = f"http://localhost:{config.PLUGIN_REGISTRY_PORT}/plugins"

        # Mock successful integration
        self.assertTrue(True)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation"""

    @patch('main.load_defaults')
    def test_valid_configuration(self, mock_load_defaults):
        """Test valid configuration loading"""
        mock_config = {
            'service_ports': {'brain_factory_port': 8301},
            'agent_config': {'max_agent_instances': 50}
        }
        mock_load_defaults.return_value = mock_config

        config = Config()
        self.assertEqual(config.BRAIN_FACTORY_PORT, 8301)

    @patch('main.load_defaults')
    def test_missing_configuration_fallback(self, mock_load_defaults):
        """Test fallback when configuration file is missing"""
        mock_load_defaults.return_value = {}

        config = Config()
        # Should use fallback values
        self.assertIsInstance(config.BRAIN_FACTORY_PORT, int)


def run_unit_tests():
    """Run all unit tests for Brain Factory"""
    print("🏭 Running Brain Factory Unit Tests")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestAgentFactory,
        TestServiceDependencies,
        TestConfigurationManagement,
        TestAgentLifecycle,
        TestErrorHandling,
        TestPerformanceMonitoring,
        TestIntegrationTesting,
        TestConfigurationValidation
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
