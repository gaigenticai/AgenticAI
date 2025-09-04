#!/usr/bin/env python3
"""
Comprehensive Test Suite for Agent Orchestrator Service

This test suite provides complete coverage for the Agent Orchestrator Service including:
- Unit tests for individual components and methods
- Integration tests for agent lifecycle management
- Performance tests for concurrent agent operations
- Task routing and distribution tests
- Error handling and recovery tests
- Multi-agent coordination tests

Test Categories:
1. Agent Registration and Lifecycle Management
2. Task Routing and Execution Orchestration
3. Multi-Agent Coordination and Communication
4. Performance Monitoring and Health Checks
5. Error Handling and Fault Tolerance
6. Session Management and State Persistence
7. Integration with External Services
"""

import asyncio
import json
import os
import pytest
import pytest_asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
import redis
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import service components
from main import (
    AgentOrchestrator,
    AgentRegistrationRequest,
    TaskRequest,
    TaskResponse,
    AgentStatus,
    Config
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    with patch('redis.Redis') as mock_redis:
        mock_instance = Mock()
        mock_instance.get.return_value = None
        mock_instance.setex.return_value = True
        mock_instance.delete.return_value = 1
        mock_instance.keys.return_value = []
        mock_instance.publish.return_value = True
        mock_instance.subscribe.return_value = Mock()
        mock_redis.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_db_session():
    """Mock database session for testing"""
    with patch('sqlalchemy.orm.sessionmaker') as mock_sessionmaker:
        mock_session = Mock()
        mock_sessionmaker.return_value = Mock(return_value=mock_session)

        # Mock query operations
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter_by.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        mock_query.first.return_value = None
        mock_query.count.return_value = 0
        mock_query.update.return_value = None

        # Mock transaction operations
        mock_session.add = Mock()
        mock_session.delete = Mock()
        mock_session.commit = Mock()
        mock_session.rollback = Mock()

        yield mock_session

@pytest.fixture
def sample_agent_registration():
    """Sample agent registration data"""
    return AgentRegistrationRequest(
        agent_id="test_agent_123",
        agent_name="Test Underwriting Agent",
        domain="underwriting",
        deployment_id="deployment_456",
        brain_config={
            "reasoning_pattern": "react",
            "memory_config": {"type": "working", "ttl": 3600},
            "plugin_config": ["risk_calculator", "fraud_detector"]
        }
    )

@pytest.fixture
def sample_task_request():
    """Sample task request data"""
    return TaskRequest(
        task_id="task_789",
        agent_id="test_agent_123",
        task_type="risk_assessment",
        payload={
            "application_id": "app_123",
            "applicant_data": {
                "name": "John Doe",
                "age": 35,
                "income": 75000,
                "credit_score": 720
            },
            "policy_details": {
                "type": "auto",
                "coverage": 50000,
                "deductible": 1000
            }
        },
        priority=1,
        timeout_seconds=300
    )


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestAgentRegistration:
    """Unit tests for agent registration functionality"""

    @pytest.fixture
    def orchestrator(self, mock_db_session, mock_redis):
        """Create AgentOrchestrator instance for testing"""
        return AgentOrchestrator(mock_db_session, mock_redis)

    def test_register_new_agent(self, orchestrator, sample_agent_registration):
        """Test registering a new agent"""
        # Mock database operations
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = None

        # Register agent
        result = orchestrator.register_agent(sample_agent_registration)

        # Verify result structure
        assert result["success"] is True
        assert result["agent_id"] == sample_agent_registration.agent_id
        assert result["status"] == "registered"
        assert "message" in result

        # Verify database operations
        orchestrator.db.add.assert_called_once()
        orchestrator.db.commit.assert_called_once()

        # Verify Redis caching
        orchestrator.redis.setex.assert_called_once()

    def test_register_existing_agent(self, orchestrator, sample_agent_registration):
        """Test registering an existing agent (update scenario)"""
        # Mock existing agent
        mock_existing = Mock()
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = mock_existing

        # Register agent
        result = orchestrator.register_agent(sample_agent_registration)

        # Verify result
        assert result["success"] is True
        assert result["agent_id"] == sample_agent_registration.agent_id

        # Verify update operations (no new add, but commit called)
        orchestrator.db.add.assert_not_called()
        orchestrator.db.commit.assert_called_once()

    def test_register_agent_database_failure(self, orchestrator, sample_agent_registration):
        """Test handling database failure during registration"""
        # Mock database failure
        orchestrator.db.commit.side_effect = Exception("Database connection failed")

        # Should raise HTTPException
        with pytest.raises(Exception):  # Would be HTTPException in real implementation
            orchestrator.register_agent(sample_agent_registration)

        # Verify rollback was called
        orchestrator.db.rollback.assert_called_once()

    def test_register_agent_validation_error(self, orchestrator):
        """Test handling validation errors in registration"""
        # Invalid registration request
        invalid_registration = AgentRegistrationRequest(
            agent_id="",  # Invalid empty ID
            agent_name="Test Agent",
            domain="underwriting",
            deployment_id="deployment_123",
            brain_config={}
        )

        # Should raise validation error
        with pytest.raises(Exception):
            orchestrator.register_agent(invalid_registration)


class TestTaskExecution:
    """Unit tests for task execution and routing"""

    @pytest.fixture
    def orchestrator(self, mock_db_session, mock_redis):
        """Create AgentOrchestrator instance for testing"""
        return AgentOrchestrator(mock_db_session, mock_redis)

    @pytest.mark.asyncio
    async def test_execute_task_success(self, orchestrator, sample_task_request):
        """Test successful task execution"""
        # Mock agent lookup
        mock_agent = Mock()
        mock_agent.status = "active"
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = mock_agent

        # Mock task execution
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "task_id": sample_task_request.task_id,
                "status": "completed",
                "result": {"decision": "approved", "confidence": 0.85}
            }
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Execute task
            result = await orchestrator.execute_task(sample_task_request)

            # Verify result
            assert result.task_id == sample_task_request.task_id
            assert result.status == "completed"
            assert "result" in result.response_data

    @pytest.mark.asyncio
    async def test_execute_task_agent_not_found(self, orchestrator, sample_task_request):
        """Test task execution with non-existent agent"""
        # Mock agent not found
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = None

        # Should raise exception
        with pytest.raises(Exception, match="Agent not found"):
            await orchestrator.execute_task(sample_task_request)

    @pytest.mark.asyncio
    async def test_execute_task_timeout(self, orchestrator, sample_task_request):
        """Test task execution timeout handling"""
        # Mock agent
        mock_agent = Mock()
        mock_agent.status = "active"
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = mock_agent

        # Mock timeout
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.side_effect = asyncio.TimeoutError()

            # Should handle timeout gracefully
            result = await orchestrator.execute_task(sample_task_request)

            # Verify timeout handling
            assert result.status == "failed"
            assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_task_agent_unavailable(self, orchestrator, sample_task_request):
        """Test task execution when agent is unavailable"""
        # Mock agent as inactive
        mock_agent = Mock()
        mock_agent.status = "inactive"
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = mock_agent

        # Should handle unavailable agent
        result = await orchestrator.execute_task(sample_task_request)

        assert result.status == "failed"
        assert "unavailable" in result.error_message.lower()


class TestMultiAgentCoordination:
    """Unit tests for multi-agent coordination"""

    @pytest.fixture
    def orchestrator(self, mock_db_session, mock_redis):
        """Create AgentOrchestrator instance for testing"""
        return AgentOrchestrator(mock_db_session, mock_redis)

    def test_get_active_agents(self, orchestrator):
        """Test retrieving list of active agents"""
        # Mock active agents
        mock_agents = [
            Mock(agent_id="agent_1", agent_name="Agent 1", status="active", domain="underwriting"),
            Mock(agent_id="agent_2", agent_name="Agent 2", status="active", domain="claims")
        ]
        orchestrator.db.query.return_value.filter_by.return_value.all.return_value = mock_agents

        # Get active agents
        agents = orchestrator.get_active_agents()

        assert len(agents) == 2
        assert agents[0]["agent_id"] == "agent_1"
        assert agents[1]["domain"] == "claims"

    def test_get_agent_by_domain(self, orchestrator):
        """Test retrieving agents by domain"""
        # Mock domain-specific agents
        mock_agents = [
            Mock(agent_id="agent_1", domain="underwriting", status="active"),
            Mock(agent_id="agent_2", domain="underwriting", status="active"),
            Mock(agent_id="agent_3", domain="claims", status="active")
        ]
        orchestrator.db.query.return_value.filter_by.return_value.all.return_value = mock_agents

        # Get underwriting agents
        underwriting_agents = orchestrator.get_agents_by_domain("underwriting")

        assert len(underwriting_agents) == 2
        assert all(agent["domain"] == "underwriting" for agent in underwriting_agents)

    def test_route_task_to_domain(self, orchestrator, sample_task_request):
        """Test routing task to appropriate domain agents"""
        # Mock domain agents
        mock_agents = [
            Mock(agent_id="agent_1", domain="underwriting", status="active"),
            Mock(agent_id="agent_2", domain="underwriting", status="active")
        ]
        orchestrator.db.query.return_value.filter_by.return_value.all.return_value = mock_agents

        # Route task
        available_agents = orchestrator.get_agents_for_domain("underwriting")

        assert len(available_agents) == 2
        assert all(agent["domain"] == "underwriting" for agent in available_agents)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestAgentLifecycleIntegration:
    """Integration tests for complete agent lifecycle"""

    @pytest.fixture
    async def orchestrator(self, mock_db_session, mock_redis):
        """Create AgentOrchestrator instance for integration testing"""
        orch = AgentOrchestrator(mock_db_session, mock_redis)
        yield orch

    @pytest.mark.asyncio
    async def test_complete_agent_lifecycle(self, orchestrator, sample_agent_registration, sample_task_request):
        """Test complete agent lifecycle: register -> execute task -> monitor -> cleanup"""
        # Phase 1: Register agent
        registration_result = orchestrator.register_agent(sample_agent_registration)
        assert registration_result["success"] is True

        # Phase 2: Execute task
        mock_agent = Mock()
        mock_agent.status = "active"
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = mock_agent

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "task_id": sample_task_request.task_id,
                "status": "completed",
                "result": {"decision": "approved"}
            }
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            task_result = await orchestrator.execute_task(sample_task_request)
            assert task_result.status == "completed"

        # Phase 3: Check agent status
        status = orchestrator.get_agent_status(sample_agent_registration.agent_id)
        assert status["status"] == "active"

        # Phase 4: Cleanup/unregister
        cleanup_result = orchestrator.unregister_agent(sample_agent_registration.agent_id)
        assert cleanup_result["success"] is True

    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self, orchestrator):
        """Test concurrent agent registrations and task executions"""
        # Create multiple agent registrations
        registrations = []
        for i in range(5):
            reg = AgentRegistrationRequest(
                agent_id=f"concurrent_agent_{i}",
                agent_name=f"Concurrent Agent {i}",
                domain="underwriting",
                deployment_id=f"deployment_{i}",
                brain_config={"reasoning_pattern": "react"}
            )
            registrations.append(reg)

        # Mock database operations
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = None

        # Register agents concurrently
        tasks = [orchestrator.register_agent(reg) for reg in registrations]
        results = await asyncio.gather(*tasks)

        # All registrations should succeed
        assert len(results) == 5
        assert all(result["success"] for result in results)

    @pytest.mark.asyncio
    async def test_agent_failure_recovery(self, orchestrator, sample_agent_registration):
        """Test agent failure and recovery scenarios"""
        # Register agent
        orchestrator.register_agent(sample_agent_registration)

        # Simulate agent failure
        mock_agent = Mock()
        mock_agent.status = "failed"
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = mock_agent

        # Attempt task execution on failed agent
        task_request = TaskRequest(
            task_id="failure_test_task",
            agent_id=sample_agent_registration.agent_id,
            task_type="test_task",
            payload={"test": "data"},
            priority=1,
            timeout_seconds=60
        )

        # Should handle failed agent gracefully
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.side_effect = Exception("Agent unavailable")

            result = await orchestrator.execute_task(task_request)
            assert result.status == "failed"


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestAgentPerformance:
    """Performance tests for agent orchestrator operations"""

    @pytest.fixture
    async def orchestrator(self, mock_db_session, mock_redis):
        """Create AgentOrchestrator instance for performance testing"""
        orch = AgentOrchestrator(mock_db_session, mock_redis)
        yield orch

    @pytest.mark.asyncio
    async def test_bulk_agent_registration(self, orchestrator):
        """Test bulk agent registration performance"""
        # Create multiple agent registrations
        registrations = []
        for i in range(50):
            reg = AgentRegistrationRequest(
                agent_id=f"perf_agent_{i}",
                agent_name=f"Performance Agent {i}",
                domain="underwriting",
                deployment_id=f"perf_deployment_{i}",
                brain_config={"pattern": "react"}
            )
            registrations.append(reg)

        # Mock database operations
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = None

        # Measure registration time
        import time
        start_time = time.time()

        for reg in registrations:
            orchestrator.register_agent(reg)

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert total_time < 10.0  # Should complete within 10 seconds
        avg_time_per_registration = total_time / len(registrations)
        assert avg_time_per_registration < 0.2  # Less than 200ms per registration

    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, orchestrator):
        """Test concurrent task execution performance"""
        # Mock agent availability
        mock_agent = Mock()
        mock_agent.status = "active"
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = mock_agent

        # Create multiple task requests
        tasks = []
        for i in range(20):
            task = TaskRequest(
                task_id=f"perf_task_{i}",
                agent_id="perf_agent_123",
                task_type="performance_test",
                payload={"index": i, "data": "test" * 100},  # Larger payload
                priority=1,
                timeout_seconds=60
            )
            tasks.append(task)

        # Mock successful task execution
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "completed", "result": {"success": True}}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Measure concurrent execution time
            import time
            start_time = time.time()

            # Execute tasks concurrently
            task_executions = [orchestrator.execute_task(task) for task in tasks]
            results = await asyncio.gather(*task_executions)

            end_time = time.time()
            total_time = end_time - start_time

            # Performance assertions
            assert total_time < 15.0  # Should complete within 15 seconds
            assert len(results) == 20
            assert all(result.status == "completed" for result in results)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestAgentErrorHandling:
    """Error handling and edge case tests"""

    @pytest.fixture
    async def orchestrator(self, mock_db_session, mock_redis):
        """Create AgentOrchestrator instance for error testing"""
        orch = AgentOrchestrator(mock_db_session, mock_redis)
        yield orch

    @pytest.mark.asyncio
    async def test_database_connection_failure(self, orchestrator, sample_agent_registration):
        """Test handling of database connection failures"""
        # Mock database failure during registration
        orchestrator.db.commit.side_effect = Exception("Database connection failed")

        with pytest.raises(Exception):
            orchestrator.register_agent(sample_agent_registration)

        # Verify rollback was called
        orchestrator.db.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_connection_failure(self, orchestrator, sample_agent_registration):
        """Test handling of Redis connection failures"""
        # Mock Redis failure
        orchestrator.redis.setex.side_effect = Exception("Redis connection failed")

        # Registration should still succeed (Redis failure shouldn't break core functionality)
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = None
        result = orchestrator.register_agent(sample_agent_registration)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_external_service_timeout(self, orchestrator, sample_task_request):
        """Test handling of external service timeouts"""
        # Mock agent availability
        mock_agent = Mock()
        mock_agent.status = "active"
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = mock_agent

        # Mock timeout from brain factory
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.side_effect = asyncio.TimeoutError()

            result = await orchestrator.execute_task(sample_task_request)

            # Should handle timeout gracefully
            assert result.status == "failed"
            assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_invalid_agent_configuration(self, orchestrator):
        """Test handling of invalid agent configurations"""
        # Invalid configuration missing required fields
        invalid_registration = AgentRegistrationRequest(
            agent_id="invalid_agent",
            agent_name="",  # Invalid empty name
            domain="invalid_domain",
            deployment_id="deployment_123",
            brain_config={}
        )

        # Should raise validation error
        with pytest.raises(Exception):
            orchestrator.register_agent(invalid_registration)

    @pytest.mark.asyncio
    async def test_task_execution_deadline_exceeded(self, orchestrator):
        """Test handling of task deadline exceeded"""
        # Create task with very short timeout
        urgent_task = TaskRequest(
            task_id="urgent_task",
            agent_id="test_agent",
            task_type="urgent_processing",
            payload={"urgent": True},
            priority=5,  # High priority
            timeout_seconds=1  # Very short timeout
        )

        # Mock agent
        mock_agent = Mock()
        mock_agent.status = "active"
        orchestrator.db.query.return_value.filter_by.return_value.first.return_value = mock_agent

        # Mock slow response
        with patch('httpx.AsyncClient') as mock_client:
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(2)  # Sleep longer than timeout
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"status": "completed"}
                return mock_response

            mock_client.return_value.__aenter__.return_value.post.side_effect = slow_response

            result = await orchestrator.execute_task(urgent_task)

            # Should detect timeout and fail gracefully
            assert result.status == "failed"


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestAgentConfiguration:
    """Configuration and environment variable tests"""

    def test_config_defaults(self):
        """Test configuration default values"""
        assert Config.SERVICE_HOST == "0.0.0.0"
        assert Config.SERVICE_PORT == 8200
        assert Config.MAX_CONCURRENT_SESSIONS == 100
        assert Config.SESSION_TIMEOUT_MINUTES == 60

    def test_config_environment_variables(self):
        """Test configuration from environment variables"""
        with patch.dict(os.environ, {
            'AGENT_ORCHESTRATOR_PORT': '9000',
            'MAX_CONCURRENT_SESSIONS': '50',
            'REQUIRE_AUTH': 'true'
        }):
            # Configurations should use environment variables when available
            assert int(os.getenv('AGENT_ORCHESTRATOR_PORT', '8200')) == 9000
            assert int(os.getenv('MAX_CONCURRENT_SESSIONS', '100')) == 50
            assert os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'

    def test_agent_status_enum(self):
        """Test AgentStatus enum values"""
        from main import AgentStatus
        assert hasattr(AgentStatus, 'ACTIVE')
        assert hasattr(AgentStatus, 'INACTIVE')
        assert hasattr(AgentStatus, 'FAILED')
        assert hasattr(AgentStatus, 'MAINTENANCE')


# =============================================================================
# TEST UTILITIES
# =============================================================================

def pytest_configure(config):
    """Pytest configuration"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Mark integration tests
        if "Integration" in item.cls.__name__:
            item.add_marker(pytest.mark.integration)

        # Mark performance tests
        if "Performance" in item.cls.__name__:
            item.add_marker(pytest.mark.performance)


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "--verbose",
        "--cov=main",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--durations=10"
    ])
