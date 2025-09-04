#!/usr/bin/env python3
"""
Test Suite for Service Connector Factory Service

Comprehensive testing for the service connector factory including:
- Connector creation and management
- Connection testing and validation
- Data operation execution
- Error handling and recovery
- Performance monitoring and metrics
- Health monitoring and automatic failover
- Concurrent operations and resource management

Author: AgenticAI Platform
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json

from main import (
    ServiceConnectorFactory,
    ServiceConnection,
    DataOperationRequest,
    CSVIngestionConnector,
    PostgreSQLOutputConnector,
    ServiceType,
    ConnectionStatus,
    Config
)


class TestServiceConnectorFactory:
    """Test suite for Service Connector Factory"""

    @pytest.fixture
    def factory(self):
        """Fixture for ServiceConnectorFactory instance"""
        return ServiceConnectorFactory()

    @pytest.fixture
    def csv_config(self):
        """Fixture for CSV service configuration"""
        return ServiceConnection(
            service_type="csv",
            service_name="test_csv_ingest",
            host="localhost",
            port=8001,
            connection_timeout=30
        )

    @pytest.fixture
    def postgresql_config(self):
        """Fixture for PostgreSQL service configuration"""
        return ServiceConnection(
            service_type="postgresql",
            service_name="test_postgres",
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            connection_timeout=30
        )

    @pytest.fixture
    def operation_request(self):
        """Fixture for data operation request"""
        return DataOperationRequest(
            connection_id="test_connector",
            operation="query",
            data={"query": "SELECT * FROM test_table"},
            parameters={"limit": 100}
        )

    def test_factory_initialization(self, factory):
        """Test factory initialization"""
        assert factory is not None
        assert len(factory.connectors) == 0
        assert len(factory.connector_types) >= 2  # At least CSV and PostgreSQL
        assert "csv" in factory.connector_types
        assert "postgresql" in factory.connector_types

    @pytest.mark.asyncio
    async def test_create_csv_connector(self, factory, csv_config):
        """Test creating CSV ingestion connector"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful health check
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            connector = await factory.create_connector(csv_config)

            assert isinstance(connector, CSVIngestionConnector)
            assert connector.service_name == csv_config.service_name
            assert csv_config.service_name in factory.connectors
            assert connector.metrics.connection_status == ConnectionStatus.CONNECTED.value

    @pytest.mark.asyncio
    async def test_create_postgresql_connector(self, factory, postgresql_config):
        """Test creating PostgreSQL output connector"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful connection test
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"connection_status": "success"}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            connector = await factory.create_connector(postgresql_config)

            assert isinstance(connector, PostgreSQLOutputConnector)
            assert connector.service_name == postgresql_config.service_name
            assert postgresql_config.service_name in factory.connectors

    @pytest.mark.asyncio
    async def test_create_duplicate_connector(self, factory, csv_config):
        """Test preventing duplicate connector creation"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            # Create first connector
            await factory.create_connector(csv_config)

            # Try to create duplicate
            with pytest.raises(ValueError, match="already exists"):
                await factory.create_connector(csv_config)

    @pytest.mark.asyncio
    async def test_create_invalid_connector_type(self, factory):
        """Test creating connector with invalid service type"""
        invalid_config = ServiceConnection(
            service_type="invalid_type",
            service_name="test_invalid",
            host="localhost",
            port=9999
        )

        with pytest.raises(ValueError, match="Unsupported service type"):
            await factory.create_connector(invalid_config)

    @pytest.mark.asyncio
    async def test_get_connector(self, factory, csv_config):
        """Test retrieving existing connector"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            # Create connector first
            created_connector = await factory.create_connector(csv_config)

            # Retrieve connector
            retrieved_connector = await factory.get_connector(csv_config.service_name)

            assert retrieved_connector == created_connector

    @pytest.mark.asyncio
    async def test_get_nonexistent_connector(self, factory):
        """Test retrieving non-existent connector"""
        with pytest.raises(ValueError, match="not found"):
            await factory.get_connector("nonexistent")

    def test_list_connectors(self, factory):
        """Test listing all connectors"""
        connectors = factory.list_connectors()

        assert isinstance(connectors, list)
        assert len(connectors) == 0  # Initially empty

        # Add a mock connector
        mock_connector = Mock()
        mock_connector.config.service_type = "csv"
        mock_connector.config.host = "localhost"
        mock_connector.config.port = 8001
        mock_connector.metrics.connection_status = "connected"

        factory.connectors["test_connector"] = mock_connector

        connectors = factory.list_connectors()
        assert len(connectors) == 1
        assert connectors[0]["service_name"] == "test_connector"
        assert connectors[0]["service_type"] == "csv"

    @pytest.mark.asyncio
    async def test_remove_connector(self, factory, csv_config):
        """Test removing connector"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            # Create connector
            await factory.create_connector(csv_config)
            assert csv_config.service_name in factory.connectors

            # Remove connector
            success = await factory.remove_connector(csv_config.service_name)
            assert success is True
            assert csv_config.service_name not in factory.connectors

    @pytest.mark.asyncio
    async def test_remove_nonexistent_connector(self, factory):
        """Test removing non-existent connector"""
        success = await factory.remove_connector("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_test_connection(self, factory, csv_config):
        """Test connection testing through factory"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful connection
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            # Create connector first
            await factory.create_connector(csv_config)

            # Test connection
            test_result = await factory.test_connection(csv_config.service_name)

            assert test_result.service_name == csv_config.service_name
            assert test_result.status == "success"
            assert test_result.response_time >= 0

    @pytest.mark.asyncio
    async def test_execute_operation(self, factory, csv_config, operation_request):
        """Test operation execution through factory"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock all HTTP calls
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"records_processed": 10}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Create connector
            await factory.create_connector(csv_config)

            # Execute operation
            result = await factory.execute_operation(csv_config.service_name, operation_request)

            assert result.operation_id is not None
            assert result.status == "success"
            assert "records_processed" in result.result_data

    def test_get_metrics(self, factory):
        """Test metrics retrieval"""
        # Test with no connectors
        metrics = factory.get_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) == 0

        # Test with specific connector (non-existent)
        metric = factory.get_metrics("nonexistent")
        assert metric is None

        # Test with mock connector
        mock_connector = Mock()
        mock_connector.get_metrics.return_value = Mock(total_operations=10, successful_operations=9)
        factory.connectors["test_connector"] = mock_connector

        metrics = factory.get_metrics()
        assert len(metrics) == 1

        specific_metric = factory.get_metrics("test_connector")
        assert specific_metric is not None

    @pytest.mark.asyncio
    async def test_health_monitoring(self, factory):
        """Test health monitoring functionality"""
        # Start health monitoring
        await factory.start_health_monitoring()
        assert factory.health_monitor_task is not None
        assert not factory.health_monitor_task.done()

        # Stop health monitoring
        await factory.stop_health_monitoring()
        assert factory.health_monitor_task.done()


class TestCSVIngestionConnector:
    """Test suite for CSV Ingestion Connector"""

    @pytest.fixture
    def csv_config(self):
        """Fixture for CSV connector configuration"""
        return ServiceConnection(
            service_type="csv",
            service_name="test_csv",
            host="localhost",
            port=8001
        )

    @pytest.fixture
    def csv_connector(self, csv_config):
        """Fixture for CSV connector instance"""
        return CSVIngestionConnector(csv_config)

    @pytest.mark.asyncio
    async def test_csv_connect_success(self, csv_connector):
        """Test successful CSV connector connection"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            success = await csv_connector.connect()

            assert success is True
            assert csv_connector.metrics.connection_status == ConnectionStatus.CONNECTED.value

    @pytest.mark.asyncio
    async def test_csv_connect_failure(self, csv_connector):
        """Test CSV connector connection failure"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection failed")

            success = await csv_connector.connect()

            assert success is False
            assert csv_connector.metrics.connection_status == ConnectionStatus.ERROR.value

    @pytest.mark.asyncio
    async def test_csv_test_connection(self, csv_connector):
        """Test CSV connector connection testing"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await csv_connector.test_connection()

            assert result.service_name == csv_connector.service_name
            assert result.status == "success"
            assert result.response_time >= 0

    @pytest.mark.asyncio
    async def test_csv_ingest_operation(self, csv_connector):
        """Test CSV file ingestion operation"""
        operation_request = DataOperationRequest(
            connection_id="test_csv",
            operation="ingest",
            data={"file_data": "col1,col2\nval1,val2"},
            parameters={"has_headers": True}
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"records_processed": 1}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await csv_connector.execute_operation(operation_request)

            assert result.status == "success"
            assert result.records_processed == 1
            assert csv_connector.metrics.total_operations == 1
            assert csv_connector.metrics.successful_operations == 1

    @pytest.mark.asyncio
    async def test_csv_validate_operation(self, csv_connector):
        """Test CSV format validation operation"""
        operation_request = DataOperationRequest(
            connection_id="test_csv",
            operation="validate",
            data={"file_data": "col1,col2\nval1,val2"},
            parameters={"delimiter": ","}
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"is_valid": True, "columns": 2}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await csv_connector.execute_operation(operation_request)

            assert result.status == "success"
            assert result.result_data["is_valid"] is True

    def test_csv_service_type(self, csv_connector):
        """Test CSV connector service type"""
        assert csv_connector.get_service_type() == ServiceType.INGESTION

    @pytest.mark.asyncio
    async def test_csv_disconnect(self, csv_connector):
        """Test CSV connector disconnection"""
        # Add mock connection to pool
        csv_connector.connection_pool.append(Mock())

        success = await csv_connector.disconnect()

        assert success is True
        assert len(csv_connector.connection_pool) == 0
        assert csv_connector.metrics.connection_status == ConnectionStatus.DISCONNECTED.value


class TestPostgreSQLOutputConnector:
    """Test suite for PostgreSQL Output Connector"""

    @pytest.fixture
    def pg_config(self):
        """Fixture for PostgreSQL connector configuration"""
        return ServiceConnection(
            service_type="postgresql",
            service_name="test_postgres",
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )

    @pytest.fixture
    def pg_connector(self, pg_config):
        """Fixture for PostgreSQL connector instance"""
        return PostgreSQLOutputConnector(pg_config)

    @pytest.mark.asyncio
    async def test_postgres_connect_success(self, pg_connector):
        """Test successful PostgreSQL connector connection"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"connection_status": "success"}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            success = await pg_connector.connect()

            assert success is True
            assert pg_connector.metrics.connection_status == ConnectionStatus.CONNECTED.value

    @pytest.mark.asyncio
    async def test_postgres_test_connection(self, pg_connector):
        """Test PostgreSQL connector connection testing"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"connection_status": "success", "version": "13.4"}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await pg_connector.test_connection()

            assert result.service_name == pg_connector.service_name
            assert result.status == "success"
            assert result.test_results["connection_status"] == "success"

    @pytest.mark.asyncio
    async def test_postgres_insert_operation(self, pg_connector):
        """Test PostgreSQL data insertion operation"""
        operation_request = DataOperationRequest(
            connection_id="test_postgres",
            operation="insert",
            data={
                "table_name": "users",
                "records": [{"id": 1, "name": "John", "email": "john@example.com"}]
            },
            parameters={"batch_size": 1}
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"records_affected": 1, "inserted_rows": 1}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await pg_connector.execute_operation(operation_request)

            assert result.status == "success"
            assert result.records_affected == 1
            assert pg_connector.metrics.successful_operations == 1

    @pytest.mark.asyncio
    async def test_postgres_query_operation(self, pg_connector):
        """Test PostgreSQL query operation"""
        operation_request = DataOperationRequest(
            connection_id="test_postgres",
            operation="query",
            data={},
            parameters={
                "query": "SELECT * FROM users WHERE id = $1",
                "query_params": [1]
            }
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"rows": [{"id": 1, "name": "John"}], "row_count": 1}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await pg_connector.execute_operation(operation_request)

            assert result.status == "success"
            assert result.result_data["row_count"] == 1
            assert len(result.result_data["rows"]) == 1

    def test_postgres_service_type(self, pg_connector):
        """Test PostgreSQL connector service type"""
        assert pg_connector.get_service_type() == ServiceType.OUTPUT

    @pytest.mark.asyncio
    async def test_postgres_operation_error(self, pg_connector):
        """Test PostgreSQL operation error handling"""
        operation_request = DataOperationRequest(
            connection_id="test_postgres",
            operation="invalid_operation",
            data={},
            parameters={}
        )

        result = await pg_connector.execute_operation(operation_request)

        assert result.status == "failed"
        assert "Unsupported operation" in result.error_message
        assert pg_connector.metrics.failed_operations == 1


class TestIntegration:
    """Integration tests for service connector functionality"""

    @pytest.mark.asyncio
    async def test_full_connector_lifecycle(self):
        """Test complete connector lifecycle"""
        factory = ServiceConnectorFactory()

        config = ServiceConnection(
            service_type="csv",
            service_name="lifecycle_test",
            host="localhost",
            port=8001
        )

        with patch('httpx.AsyncClient') as mock_client:
            # Mock all HTTP responses
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "success"}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Create connector
            connector = await factory.create_connector(config)
            assert config.service_name in factory.connectors

            # Test connection
            test_result = await factory.test_connection(config.service_name)
            assert test_result.status == "success"

            # Execute operation
            operation_request = DataOperationRequest(
                connection_id=config.service_name,
                operation="ingest",
                data={"file_data": "test,data\n1,2"},
                parameters={"has_headers": False}
            )

            result = await factory.execute_operation(config.service_name, operation_request)
            assert result.status == "success"

            # Remove connector
            success = await factory.remove_connector(config.service_name)
            assert success is True
            assert config.service_name not in factory.connectors

    @pytest.mark.asyncio
    async def test_concurrent_connector_operations(self):
        """Test concurrent operations on multiple connectors"""
        factory = ServiceConnectorFactory()

        configs = []
        for i in range(3):
            config = ServiceConnection(
                service_type="csv",
                service_name=f"concurrent_csv_{i+1}",
                host="localhost",
                port=8001 + i
            )
            configs.append(config)

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "success"}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Create connectors concurrently
            create_tasks = [factory.create_connector(config) for config in configs]
            connectors = await asyncio.gather(*create_tasks)

            assert len(connectors) == 3
            assert len(factory.connectors) == 3

            # Test connections concurrently
            test_tasks = [factory.test_connection(config.service_name) for config in configs]
            test_results = await asyncio.gather(*test_tasks)

            assert all(result.status == "success" for result in test_results)

            # Execute operations concurrently
            operation_requests = []
            for config in configs:
                operation_request = DataOperationRequest(
                    connection_id=config.service_name,
                    operation="validate",
                    data={"file_data": "col1,col2\nval1,val2"},
                    parameters={"delimiter": ","}
                )
                operation_requests.append(operation_request)

            execute_tasks = [
                factory.execute_operation(config.service_name, op_request)
                for config, op_request in zip(configs, operation_requests)
            ]
            execute_results = await asyncio.gather(*execute_tasks)

            assert all(result.status == "success" for result in execute_results)

    @pytest.mark.asyncio
    async def test_error_recovery_and_retry(self):
        """Test error recovery and retry mechanisms"""
        factory = ServiceConnectorFactory()

        config = ServiceConnection(
            service_type="csv",
            service_name="retry_test",
            host="localhost",
            port=8001,
            retry_attempts=2
        )

        with patch('httpx.AsyncClient') as mock_client:
            # Mock initial failures then success
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "success"}

            # First call fails, second succeeds
            mock_client.return_value.__aenter__.return_value.get.side_effect = [
                Exception("Connection failed"),
                mock_response
            ]

            connector = await factory.create_connector(config)

            # Should succeed after retry
            assert connector is not None
            assert config.service_name in factory.connectors

    @pytest.mark.asyncio
    async def test_metrics_aggregation(self):
        """Test metrics aggregation across connectors"""
        factory = ServiceConnectorFactory()

        # Create multiple connectors with mock metrics
        for i in range(3):
            config = ServiceConnection(
                service_type="csv",
                service_name=f"metrics_test_{i+1}",
                host="localhost",
                port=8001 + i
            )

            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

                connector = await factory.create_connector(config)

                # Simulate some metrics
                connector.metrics.total_operations = 10 + i
                connector.metrics.successful_operations = 9 + i
                connector.metrics.average_response_time = 0.1 * (i + 1)

        # Get aggregated metrics
        all_metrics = factory.get_metrics()

        assert len(all_metrics) == 3
        total_operations = sum(m.total_operations for m in all_metrics)
        assert total_operations == 33  # 10+11+12

        successful_operations = sum(m.successful_operations for m in all_metrics)
        assert successful_operations == 30  # 9+10+11


class TestErrorHandling:
    """Test suite for error handling scenarios"""

    @pytest.fixture
    def factory(self):
        """Fixture for ServiceConnectorFactory instance"""
        return ServiceConnectorFactory()

    @pytest.mark.asyncio
    async def test_service_unavailable_error(self, factory):
        """Test handling of service unavailable errors"""
        config = ServiceConnection(
            service_type="csv",
            service_name="unavailable_test",
            host="localhost",
            port=8001
        )

        with patch('httpx.AsyncClient') as mock_client:
            # Mock service unavailable
            from httpx import ConnectError
            mock_client.return_value.__aenter__.return_value.get.side_effect = ConnectError("Connection refused")

            with pytest.raises(ValueError, match="Failed to establish connection"):
                await factory.create_connector(config)

    @pytest.mark.asyncio
    async def test_operation_timeout_error(self, factory):
        """Test handling of operation timeout errors"""
        config = ServiceConnection(
            service_type="postgresql",
            service_name="timeout_test",
            host="localhost",
            port=5432,
            connection_timeout=1  # Very short timeout
        )

        operation_request = DataOperationRequest(
            connection_id="timeout_test",
            operation="query",
            data={},
            parameters={"query": "SELECT * FROM large_table"}
        )

        with patch('httpx.AsyncClient') as mock_client:
            # Mock timeout
            import asyncio
            mock_client.return_value.__aenter__.return_value.post.side_effect = asyncio.TimeoutError()

            with patch('httpx.AsyncClient') as mock_client_inner:
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {"connection_status": "success"}
                mock_client_inner.return_value.__aenter__.return_value.post.return_value = mock_response

                # Create connector
                await factory.create_connector(config)

                # Execute operation that times out
                result = await factory.execute_operation("timeout_test", operation_request)

                assert result.status == "failed"
                assert "timed out" in result.error_message.lower() or "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_invalid_operation_error(self, factory):
        """Test handling of invalid operation errors"""
        config = ServiceConnection(
            service_type="csv",
            service_name="invalid_op_test",
            host="localhost",
            port=8001
        )

        operation_request = DataOperationRequest(
            connection_id="invalid_op_test",
            operation="invalid_operation_type",
            data={},
            parameters={}
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            # Create connector
            await factory.create_connector(config)

            # Execute invalid operation
            result = await factory.execute_operation("invalid_op_test", operation_request)

            assert result.status == "failed"
            assert "Unsupported operation" in result.error_message

    @pytest.mark.asyncio
    async def test_authentication_error(self, factory):
        """Test handling of authentication errors"""
        config = ServiceConnection(
            service_type="postgresql",
            service_name="auth_test",
            host="localhost",
            port=5432,
            username="invalid_user",
            password="invalid_pass"
        )

        with patch('httpx.AsyncClient') as mock_client:
            # Mock authentication failure
            from httpx import HTTPStatusError
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.raise_for_status.side_effect = HTTPStatusError(
                "Authentication failed", request=Mock(), response=mock_response
            )
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            with pytest.raises(ValueError, match="Failed to establish connection"):
                await factory.create_connector(config)


if __name__ == "__main__":
    pytest.main([__file__])
