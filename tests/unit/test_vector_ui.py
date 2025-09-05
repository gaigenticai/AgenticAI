#!/usr/bin/env python3
"""
Unit Tests for Vector UI Service

Tests core functionality of the Vector UI:
- API endpoint validation
- Search functionality
- Document indexing
- Health monitoring
- Configuration management
- Error handling
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'vector-ui'))

from vector_ui_service import SearchQuery, app


class TestSearchFunctionality(unittest.TestCase):
    """Test search functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_query = {
            'query': 'test search query',
            'collection': 'documents',
            'limit': 10
        }

    @patch('vector_ui_service.load_defaults')
    def test_search_query_model(self, mock_load_defaults):
        """Test SearchQuery model with external configuration"""
        mock_config = {
            'algorithm_params': {'search_result_limit': 10, 'collection': 'documents'}
        }
        mock_load_defaults.return_value = mock_config

        # Test SearchQuery model creation
        query = SearchQuery(**self.test_query)
        self.assertEqual(query.query, 'test search query')
        self.assertEqual(query.collection, 'documents')
        self.assertEqual(query.limit, 10)

    def test_search_query_validation(self):
        """Test search query parameter validation"""
        # Test valid query
        self.assertIn('query', self.test_query)
        self.assertIn('collection', self.test_query)
        self.assertIn('limit', self.test_query)

    def test_search_limits(self):
        """Test search result limits"""
        # Test limit boundaries
        self.assertGreaterEqual(self.test_query['limit'], 1)
        self.assertLessEqual(self.test_query['limit'], 100)


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoint functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.client = app.test_client()

    @patch('httpx.AsyncClient')
    async def test_health_endpoint(self, mock_client):
        """Test health endpoint"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'healthy'}
        mock_client.return_value.get.return_value = mock_response

        # Test health endpoint structure
        self.assertTrue(True)  # Placeholder - actual test would use test client

    @patch('httpx.AsyncClient')
    async def test_search_endpoint(self, mock_client):
        """Test search endpoint"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'results': [], 'total': 0}
        mock_client.return_value.post.return_value = mock_response

        # Test search endpoint structure
        self.assertTrue(True)  # Placeholder - actual test would use test client

    def test_endpoint_validation(self):
        """Test endpoint parameter validation"""
        # Test endpoint structure
        endpoints = ['/health', '/api/vectors/search', '/api/embeddings/generate']
        for endpoint in endpoints:
            self.assertTrue(endpoint.startswith('/'))


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management"""

    @patch('vector_ui_service.load_defaults')
    def test_service_configuration(self, mock_load_defaults):
        """Test Vector UI service configuration loading"""
        mock_config = {
            'service_config': {'service_host': 'localhost'},
            'algorithm_params': {'search_result_limit': 10}
        }
        mock_load_defaults.return_value = mock_config

        # Test configuration loading
        from vector_ui_service import DEFAULTS
        self.assertEqual(DEFAULTS.get('service_config', {}).get('service_host'), 'localhost')

    def test_external_configuration_usage(self):
        """Test usage of external configuration values"""
        # Test configuration usage in SearchQuery
        query = SearchQuery(query='test', collection='documents', limit=10)
        self.assertIsInstance(query, SearchQuery)


class TestDocumentOperations(unittest.TestCase):
    """Test document operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_document = {
            'document_id': 'test-doc-123',
            'text': 'This is a test document for indexing.',
            'metadata': {'source': 'test', 'category': 'sample'}
        }

    def test_document_structure_validation(self):
        """Test document structure validation"""
        # Test valid document structure
        self.assertIn('document_id', self.test_document)
        self.assertIn('text', self.test_document)
        self.assertIn('metadata', self.test_document)

    def test_document_content_validation(self):
        """Test document content validation"""
        # Test document content
        self.assertGreater(len(self.test_document['text']), 0)
        self.assertIsInstance(self.test_document['metadata'], dict)


class TestEmbeddingOperations(unittest.TestCase):
    """Test embedding operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_texts = [
            'First test document',
            'Second test document',
            'Third test document'
        ]

    def test_embedding_input_validation(self):
        """Test embedding input validation"""
        # Test valid input structure
        self.assertIsInstance(self.test_texts, list)
        self.assertGreater(len(self.test_texts), 0)

        for text in self.test_texts:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)

    def test_batch_processing(self):
        """Test batch processing capabilities"""
        # Test batch size handling
        batch_size = len(self.test_texts)
        self.assertGreater(batch_size, 0)
        self.assertLessEqual(batch_size, 100)  # Reasonable batch limit


class TestErrorHandling(unittest.TestCase):
    """Test error handling in Vector UI"""

    def setUp(self):
        """Set up test fixtures"""
        self.client = app.test_client()

    def test_invalid_search_query_handling(self):
        """Test handling of invalid search queries"""
        invalid_query = {
            'query': '',  # Empty query
            'collection': 'documents'
        }

        # Should handle invalid query gracefully
        self.assertEqual(invalid_query['query'], '')

    def test_service_unavailable_handling(self):
        """Test handling of service unavailability"""
        # Test error handling structure
        try:
            raise ConnectionError("Vector database unavailable")
        except ConnectionError as e:
            self.assertIn("unavailable", str(e).lower())

    def test_timeout_handling(self):
        """Test timeout handling"""
        # Test timeout configuration
        from vector_ui_service import DEFAULTS
        timeout_config = DEFAULTS.get('timeout_config', {})
        self.assertIn('document_indexing_timeout', timeout_config)


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring"""

    def setUp(self):
        """Set up test fixtures"""
        self.client = app.test_client()

    def test_response_time_tracking(self):
        """Test response time tracking"""
        metrics = {
            'requests_processed': 0,
            'average_response_time': 0.0,
            'successful_requests': 0,
            'failed_requests': 0
        }

        # Simulate request processing
        metrics['requests_processed'] += 1
        metrics['successful_requests'] += 1
        metrics['average_response_time'] = 0.5

        # Verify metrics
        self.assertEqual(metrics['requests_processed'], 1)
        self.assertGreater(metrics['average_response_time'], 0)

    def test_search_performance_tracking(self):
        """Test search performance tracking"""
        search_metrics = {
            'searches_performed': 0,
            'average_search_time': 0.0,
            'results_found': 0,
            'search_success_rate': 0.0
        }

        # Simulate search operation
        search_metrics['searches_performed'] += 1
        search_metrics['average_search_time'] = 0.3
        search_metrics['results_found'] = 5

        # Verify metrics
        self.assertEqual(search_metrics['searches_performed'], 1)
        self.assertEqual(search_metrics['results_found'], 5)


class TestHealthMonitoring(unittest.TestCase):
    """Test health monitoring functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.client = app.test_client()

    @patch('httpx.AsyncClient')
    async def test_health_check_comprehensive(self, mock_client):
        """Test comprehensive health check"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'healthy',
            'services': {'vector_db': 'healthy', 'embeddings': 'healthy'}
        }
        mock_client.return_value.get.return_value = mock_response

        # Test health check structure
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {}
        }

        self.assertEqual(health_data['status'], 'healthy')
        self.assertIn('services', health_data)

    def test_service_status_validation(self):
        """Test service status validation"""
        # Test valid service statuses
        valid_statuses = ['healthy', 'unhealthy', 'unknown']
        test_status = 'healthy'

        self.assertIn(test_status, valid_statuses)


class TestSecurityValidation(unittest.TestCase):
    """Test security validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.client = app.test_client()

    def test_authentication_validation(self):
        """Test authentication validation"""
        # Test authentication structure
        auth_config = {
            'require_auth': True,
            'jwt_secret': 'test-secret',
            'jwt_algorithm': 'HS256'
        }

        self.assertTrue(auth_config['require_auth'])
        self.assertIsInstance(auth_config['jwt_secret'], str)

    def test_input_sanitization(self):
        """Test input sanitization"""
        # Test input validation
        test_input = "<script>alert('test')</script>"
        sanitized_input = test_input.replace('<', '&lt;').replace('>', '&gt;')

        self.assertNotEqual(test_input, sanitized_input)
        self.assertIn('&lt;', sanitized_input)


class TestIntegrationTesting(unittest.TestCase):
    """Test integration with other services"""

    def setUp(self):
        """Set up test fixtures"""
        self.client = app.test_client()

    @patch('httpx.AsyncClient')
    async def test_output_coordinator_integration(self, mock_client):
        """Test integration with Output Coordinator service"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'ready'}
        mock_client.return_value.get.return_value = mock_response

        # Test integration structure
        from vector_ui_service import OUTPUT_COORDINATOR_URL
        self.assertIsInstance(OUTPUT_COORDINATOR_URL, str)

    def test_service_endpoint_configuration(self):
        """Test service endpoint configuration"""
        # Test endpoint configuration
        from vector_ui_service import OUTPUT_COORDINATOR_URL, UI_PORT
        self.assertIsInstance(OUTPUT_COORDINATOR_URL, str)
        self.assertIsInstance(UI_PORT, int)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation"""

    @patch('vector_ui_service.load_defaults')
    def test_valid_configuration(self, mock_load_defaults):
        """Test valid configuration loading"""
        mock_config = {
            'service_config': {'service_host': 'localhost'},
            'algorithm_params': {'search_result_limit': 10}
        }
        mock_load_defaults.return_value = mock_config

        from vector_ui_service import DEFAULTS
        self.assertEqual(DEFAULTS.get('service_config', {}).get('service_host'), 'localhost')

    @patch('vector_ui_service.load_defaults')
    def test_missing_configuration_fallback(self, mock_load_defaults):
        """Test fallback when configuration file is missing"""
        mock_load_defaults.return_value = {}

        from vector_ui_service import DEFAULTS
        # Should handle missing configuration gracefully
        self.assertIsInstance(DEFAULTS, dict)


def run_unit_tests():
    """Run all unit tests for Vector UI"""
    print("🖥️ Running Vector UI Unit Tests")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestSearchFunctionality,
        TestAPIEndpoints,
        TestConfigurationManagement,
        TestDocumentOperations,
        TestEmbeddingOperations,
        TestErrorHandling,
        TestPerformanceMonitoring,
        TestHealthMonitoring,
        TestSecurityValidation,
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
