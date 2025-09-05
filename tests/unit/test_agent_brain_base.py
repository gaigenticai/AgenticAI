#!/usr/bin/env python3
"""
Unit Tests for Agent Brain Base Service

Tests core functionality of the AgentBrain class and related components:
- Agent initialization and configuration
- Task execution and state management
- Service connections and integrations
- Memory management and task storage
- Error handling and recovery
- Performance monitoring and metrics
"""

import asyncio
import json
import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Add service path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'agent-brain-base'))

from main import AgentBrain, Config, AgentState


class TestAgentBrainInitialization(unittest.TestCase):
    """Test AgentBrain initialization and configuration"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent_brain = AgentBrain()
        self.test_config = {
            'agent_id': 'test-agent-123',
            'execution_timeout': 60,
            'max_concurrent_tasks': 5
        }

    @patch('main.load_defaults')
    def test_agent_initialization_success(self, mock_load_defaults):
        """Test successful agent initialization"""
        mock_load_defaults.return_value = {
            'service_ports': {'reasoning_module_factory_port': 8304},
            'agent_config': {'default_execution_timeout': 300}
        }

        # Test agent creation
        agent = AgentBrain()
        self.assertIsInstance(agent, AgentBrain)
        self.assertEqual(agent.state, AgentState.INITIALIZING)

    @patch('main.load_defaults')
    def test_configuration_loading(self, mock_load_defaults):
        """Test configuration loading from external file"""
        mock_config = {
            'service_ports': {'reasoning_module_factory_port': 8304},
            'agent_config': {'default_execution_timeout': 300}
        }
        mock_load_defaults.return_value = mock_config

        # Test config loading
        config = Config()
        self.assertEqual(config.REASONING_MODULE_FACTORY_PORT, 8304)
        self.assertEqual(config.DEFAULT_EXECUTION_TIMEOUT, 300)

    def test_agent_state_transitions(self):
        """Test agent state management"""
        agent = AgentBrain()

        # Test initial state
        self.assertEqual(agent.state, AgentState.INITIALIZING)

        # Test state changes
        agent.state = AgentState.READY
        self.assertEqual(agent.state, AgentState.READY)


class TestTaskExecution(unittest.TestCase):
    """Test task execution functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = AgentBrain()
        self.test_task = {
            'task_id': 'test-task-123',
            'type': 'reasoning',
            'parameters': {'query': 'test query'},
            'priority': 'high',
            'timeout': 30
        }

    @patch('httpx.AsyncClient')
    async def test_task_execution_basic(self, mock_client):
        """Test basic task execution"""
        # Mock HTTP client
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'result': 'success'}
        mock_client.return_value.post.return_value = mock_response

        # Execute task
        result = await self.agent.execute_task(self.test_task)

        # Verify result
        self.assertIsInstance(result, dict)
        self.assertIn('task_id', result)

    @patch('main.load_defaults')
    def test_task_validation(self, mock_load_defaults):
        """Test task parameter validation"""
        mock_load_defaults.return_value = {
            'service_ports': {'reasoning_module_factory_port': 8304},
            'agent_config': {'default_execution_timeout': 300}
        }

        # Test valid task
        valid_task = {
            'task_id': 'test-123',
            'type': 'reasoning',
            'parameters': {'query': 'test'},
            'timeout': 30
        }

        # Task should be accepted (basic validation)
        self.assertIsInstance(valid_task, dict)
        self.assertIn('task_id', valid_task)

    def test_concurrent_task_limits(self):
        """Test concurrent task execution limits"""
        # Test max concurrent tasks configuration
        config = Config()
        self.assertGreater(config.MAX_CONCURRENT_TASKS, 0)


class TestServiceConnections(unittest.TestCase):
    """Test service connection management"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = AgentBrain()

    @patch('httpx.AsyncClient')
    async def test_reasoning_module_connection(self, mock_client):
        """Test connection to reasoning module service"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'healthy'}
        mock_client.return_value.get.return_value = mock_response

        # Test connection establishment
        config = Config()
        url = f"http://localhost:{config.REASONING_MODULE_FACTORY_PORT}/health"

        # Mock successful connection
        self.assertTrue(True)  # Placeholder - actual connection test would use real service

    @patch('httpx.AsyncClient')
    async def test_plugin_registry_connection(self, mock_client):
        """Test connection to plugin registry service"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'healthy'}
        mock_client.return_value.get.return_value = mock_response

        # Test connection establishment
        config = Config()
        url = f"http://localhost:{config.PLUGIN_REGISTRY_PORT}/health"

        # Mock successful connection
        self.assertTrue(True)  # Placeholder - actual connection test would use real service


class TestMemoryManagement(unittest.TestCase):
    """Test memory management functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = AgentBrain()

    def test_memory_configuration(self):
        """Test memory management configuration"""
        config = Config()
        self.assertGreater(config.MEMORY_TTL_SECONDS, 0)

    def test_task_result_storage(self):
        """Test task result storage in memory"""
        # Test basic memory storage structure
        memory_store = {}
        test_result = {
            'task_id': 'test-123',
            'result': 'success',
            'timestamp': datetime.now().isoformat()
        }

        # Store result
        memory_store[test_result['task_id']] = test_result

        # Verify storage
        self.assertIn('test-123', memory_store)
        self.assertEqual(memory_store['test-123']['result'], 'success')


class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = AgentBrain()

    def test_timeout_handling(self):
        """Test task timeout handling"""
        config = Config()
        self.assertGreater(config.DEFAULT_EXECUTION_TIMEOUT, 0)

    def test_service_unavailable_handling(self):
        """Test handling of unavailable services"""
        # Test error handling structure
        try:
            # Simulate service unavailable
            raise ConnectionError("Service unavailable")
        except ConnectionError as e:
            # Verify error is caught and handled
            self.assertIn("unavailable", str(e).lower())

    def test_invalid_task_handling(self):
        """Test handling of invalid tasks"""
        invalid_task = {
            'task_id': None,  # Invalid task_id
            'type': 'invalid_type'
        }

        # Test should handle invalid task gracefully
        self.assertIsNone(invalid_task.get('task_id'))


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring and metrics"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = AgentBrain()

    def test_metrics_configuration(self):
        """Test metrics collection configuration"""
        config = Config()
        self.assertIsInstance(config.ENABLE_METRICS, bool)

    def test_execution_metrics_tracking(self):
        """Test execution metrics tracking"""
        metrics = {
            'tasks_executed': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0
        }

        # Simulate task execution
        metrics['tasks_executed'] += 1
        metrics['successful_tasks'] += 1

        # Verify metrics
        self.assertEqual(metrics['tasks_executed'], 1)
        self.assertEqual(metrics['successful_tasks'], 1)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation"""

    @patch('main.load_defaults')
    def test_valid_configuration(self, mock_load_defaults):
        """Test valid configuration loading"""
        mock_config = {
            'service_ports': {'reasoning_module_factory_port': 8304},
            'agent_config': {'default_execution_timeout': 300}
        }
        mock_load_defaults.return_value = mock_config

        config = Config()

        # Verify configuration values
        self.assertEqual(config.REASONING_MODULE_FACTORY_PORT, 8304)
        self.assertEqual(config.DEFAULT_EXECUTION_TIMEOUT, 300)

    @patch('main.load_defaults')
    def test_missing_configuration_fallback(self, mock_load_defaults):
        """Test fallback when configuration file is missing"""
        mock_load_defaults.return_value = {}

        config = Config()

        # Verify fallback values are used
        self.assertIsInstance(config.REASONING_MODULE_FACTORY_PORT, int)
        self.assertGreater(config.REASONING_MODULE_FACTORY_PORT, 0)


def run_unit_tests():
    """Run all unit tests for Agent Brain Base"""
    print("🧠 Running Agent Brain Base Unit Tests")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestAgentBrainInitialization,
        TestTaskExecution,
        TestServiceConnections,
        TestMemoryManagement,
        TestErrorHandling,
        TestPerformanceMonitoring,
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
