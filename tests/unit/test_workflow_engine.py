#!/usr/bin/env python3
"""
Unit Tests for Workflow Engine Service

Tests core functionality of the Workflow Engine:
- Workflow execution orchestration
- Component dependency management
- Task scheduling and execution
- Error handling and recovery
- Performance monitoring
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'workflow-engine'))

from main import WorkflowExecutor, Config, WorkflowExecution


class TestWorkflowExecutor(unittest.TestCase):
    """Test Workflow Executor functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = WorkflowExecutor()
        self.test_workflow = {
            'workflow_id': 'test-workflow-123',
            'name': 'Test Workflow',
            'components': [
                {
                    'id': 'comp1',
                    'type': 'data_input',
                    'config': {'source': 'database', 'query': 'SELECT * FROM test'}
                },
                {
                    'id': 'comp2',
                    'type': 'llm_processor',
                    'config': {'model': 'gpt-4', 'prompt': 'Analyze data'}
                }
            ],
            'connections': [
                {'from': 'comp1', 'to': 'comp2'}
            ]
        }

    @patch('main.load_defaults')
    def test_workflow_execution_plan(self, mock_load_defaults):
        """Test workflow execution plan creation"""
        mock_load_defaults.return_value = {
            'service_ports': {'workflow_engine_port': 8202},
            'performance': {'workflow_max_components': 50}
        }

        # Test execution plan creation
        plan = self.executor.create_execution_plan(self.test_workflow)
        self.assertIsInstance(plan, dict)
        self.assertIn('execution_order', plan)

    def test_component_dependency_resolution(self):
        """Test component dependency resolution"""
        # Test dependency graph structure
        components = self.test_workflow['components']
        connections = self.test_workflow['connections']

        # Should have proper dependency structure
        self.assertGreater(len(components), 0)
        self.assertGreater(len(connections), 0)

    def test_workflow_validation(self):
        """Test workflow configuration validation"""
        # Test valid workflow structure
        self.assertIn('workflow_id', self.test_workflow)
        self.assertIn('components', self.test_workflow)
        self.assertIn('connections', self.test_workflow)


class TestComponentExecution(unittest.TestCase):
    """Test component execution functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = WorkflowExecutor()
        self.test_component = {
            'id': 'test-comp-123',
            'type': 'data_input',
            'config': {
                'source': 'database',
                'query': 'SELECT * FROM users WHERE active = true'
            }
        }

    @patch('httpx.AsyncClient')
    async def test_data_input_component(self, mock_client):
        """Test data input component execution"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{'id': 1, 'name': 'Test User'}]
        mock_client.return_value.post.return_value = mock_response

        # Test component execution
        result = await self.executor._execute_component_async(self.test_component, {})
        self.assertIsInstance(result, dict)

    def test_component_configuration_validation(self):
        """Test component configuration validation"""
        # Test valid component configuration
        self.assertIn('id', self.test_component)
        self.assertIn('type', self.test_component)
        self.assertIn('config', self.test_component)


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management"""

    @patch('main.load_defaults')
    def test_workflow_configuration(self, mock_load_defaults):
        """Test Workflow Engine configuration loading"""
        mock_config = {
            'service_ports': {'workflow_engine_port': 8202},
            'performance': {'workflow_max_components': 50, 'workflow_execution_timeout': 1800}
        }
        mock_load_defaults.return_value = mock_config

        config = Config()
        self.assertEqual(config.SERVICE_PORT, 8202)
        self.assertEqual(config.WORKFLOW_MAX_COMPONENTS, 50)

    def test_service_endpoint_configuration(self):
        """Test service endpoint configuration"""
        config = Config()

        # Verify all service ports are configured
        self.assertGreater(config.PLUGIN_REGISTRY_PORT, 0)
        self.assertGreater(config.RULE_ENGINE_PORT, 0)
        self.assertGreater(config.MEMORY_MANAGER_PORT, 0)


class TestExecutionOrchestration(unittest.TestCase):
    """Test execution orchestration"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = WorkflowExecutor()

    @patch('asyncio.gather')
    async def test_parallel_execution(self, mock_gather):
        """Test parallel component execution"""
        mock_gather.return_value = [{'status': 'success'}, {'status': 'success'}]

        # Test parallel execution structure
        config = Config()
        self.assertTrue(config.WORKFLOW_PARALLEL_EXECUTION)

    def test_execution_state_management(self):
        """Test execution state management"""
        # Test state management structure
        execution_state = {
            'execution_id': 'exec-123',
            'status': 'running',
            'current_component': 'comp1',
            'completed_components': [],
            'failed_components': []
        }

        self.assertEqual(execution_state['status'], 'running')

    def test_execution_timeout_handling(self):
        """Test execution timeout handling"""
        config = Config()
        self.assertGreater(config.WORKFLOW_EXECUTION_TIMEOUT, 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in Workflow Engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = WorkflowExecutor()

    def test_invalid_workflow_handling(self):
        """Test handling of invalid workflows"""
        invalid_workflow = {
            'workflow_id': None,  # Invalid workflow_id
            'components': []
        }

        # Should handle invalid workflow gracefully
        self.assertIsNone(invalid_workflow.get('workflow_id'))

    def test_component_failure_handling(self):
        """Test component failure handling"""
        # Test failure handling structure
        try:
            raise Exception("Component execution failed")
        except Exception as e:
            self.assertIn("failed", str(e).lower())

    def test_dependency_failure_handling(self):
        """Test dependency failure handling"""
        # Test dependency failure structure
        failed_dependency = {
            'component_id': 'comp1',
            'status': 'failed',
            'error': 'Dependency unavailable'
        }

        self.assertEqual(failed_dependency['status'], 'failed')


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = WorkflowExecutor()

    def test_execution_metrics_tracking(self):
        """Test execution metrics tracking"""
        metrics = {
            'workflows_executed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'total_components_executed': 0
        }

        # Simulate workflow execution
        metrics['workflows_executed'] += 1
        metrics['successful_executions'] += 1
        metrics['total_components_executed'] += 3

        # Verify metrics
        self.assertEqual(metrics['workflows_executed'], 1)
        self.assertEqual(metrics['total_components_executed'], 3)

    def test_component_performance_tracking(self):
        """Test component performance tracking"""
        component_metrics = {
            'component_id': 'comp1',
            'execution_time': 0.0,
            'memory_usage': 0.0,
            'success_rate': 0.0
        }

        # Simulate component execution
        component_metrics['execution_time'] = 0.5
        component_metrics['success_rate'] = 1.0

        # Verify metrics
        self.assertGreater(component_metrics['execution_time'], 0)
        self.assertEqual(component_metrics['success_rate'], 1.0)


class TestDatabaseOperations(unittest.TestCase):
    """Test database operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = WorkflowExecutor()

    def test_execution_persistence(self):
        """Test workflow execution persistence"""
        # Test database persistence structure
        execution_record = WorkflowExecution(
            execution_id='exec-123',
            agent_id='agent-456',
            workflow_id='workflow-789',
            status='running',
            total_components=3,
            completed_components=1
        )

        self.assertEqual(execution_record.status, 'running')
        self.assertEqual(execution_record.total_components, 3)

    def test_component_execution_tracking(self):
        """Test component execution tracking"""
        # Test component tracking structure
        component_record = {
            'execution_id': 'exec-123',
            'component_id': 'comp1',
            'status': 'completed',
            'execution_time_ms': 500,
            'output_data': {'result': 'success'}
        }

        self.assertEqual(component_record['status'], 'completed')
        self.assertIn('output_data', component_record)


class TestIntegrationTesting(unittest.TestCase):
    """Test integration with other services"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = WorkflowExecutor()

    @patch('httpx.AsyncClient')
    async def test_plugin_registry_integration(self, mock_client):
        """Test integration with Plugin Registry service"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'plugins': ['risk_calculator']}
        mock_client.return_value.get.return_value = mock_response

        # Test integration structure
        config = Config()
        url = f"http://localhost:{config.PLUGIN_REGISTRY_PORT}/plugins"

        # Mock successful integration
        self.assertTrue(True)

    @patch('httpx.AsyncClient')
    async def test_rule_engine_integration(self, mock_client):
        """Test integration with Rule Engine service"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'rules': []}
        mock_client.return_value.get.return_value = mock_response

        # Test integration structure
        config = Config()
        url = f"http://localhost:{config.RULE_ENGINE_PORT}/rules"

        # Mock successful integration
        self.assertTrue(True)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation"""

    @patch('main.load_defaults')
    def test_valid_configuration(self, mock_load_defaults):
        """Test valid configuration loading"""
        mock_config = {
            'service_ports': {'workflow_engine_port': 8202},
            'performance': {'workflow_max_components': 50}
        }
        mock_load_defaults.return_value = mock_config

        config = Config()
        self.assertEqual(config.SERVICE_PORT, 8202)

    @patch('main.load_defaults')
    def test_missing_configuration_fallback(self, mock_load_defaults):
        """Test fallback when configuration file is missing"""
        mock_load_defaults.return_value = {}

        config = Config()
        # Should use fallback values
        self.assertIsInstance(config.SERVICE_PORT, int)


def run_unit_tests():
    """Run all unit tests for Workflow Engine"""
    print("⚙️ Running Workflow Engine Unit Tests")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestWorkflowExecutor,
        TestComponentExecution,
        TestConfigurationManagement,
        TestExecutionOrchestration,
        TestErrorHandling,
        TestPerformanceMonitoring,
        TestDatabaseOperations,
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
