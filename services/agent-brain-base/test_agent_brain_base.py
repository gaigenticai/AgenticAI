#!/usr/bin/env python3
"""
Test Suite for Agent Brain Base Class Service

Comprehensive testing for the core agent execution framework including:
- Agent lifecycle management (create, initialize, terminate)
- Task execution with various reasoning patterns
- Memory integration and state persistence
- Plugin orchestration and execution
- Performance monitoring and metrics
- Error handling and recovery mechanisms
- Concurrent task execution and resource management

Author: AgenticAI Platform
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json

from main import (
    AgentBrain,
    AgentBrainService,
    AgentConfig,
    AgentPersona,
    ReasoningConfig,
    MemoryConfig,
    PluginConfig,
    TaskRequest,
    TaskResult,
    AgentState,
    TaskStatus,
    Config
)


class TestAgentBrain:
    """Test suite for AgentBrain base class"""

    @pytest.fixture
    def sample_config(self):
        """Fixture for sample agent configuration"""
        return AgentConfig(
            agent_id="test_agent_001",
            name="Test Agent",
            persona=AgentPersona(
                name="Test Agent",
                description="A test agent for unit testing",
                domain="testing",
                expertise_level="intermediate",
                communication_style="direct"
            ),
            reasoning_config=ReasoningConfig(
                pattern="ReAct",
                confidence_threshold=0.7
            ),
            memory_config=MemoryConfig(
                working_memory_enabled=True,
                episodic_memory_enabled=True,
                semantic_memory_enabled=False,
                memory_ttl_seconds=3600
            ),
            plugin_config=PluginConfig(
                enabled_plugins=["test_plugin"],
                auto_discovery=False
            )
        )

    @pytest.fixture
    def sample_task_request(self):
        """Fixture for sample task request"""
        return TaskRequest(
            task_id="task_001",
            description="Test task for agent execution",
            input_data={"test_key": "test_value"},
            constraints={"time_limit": 60},
            priority="normal",
            timeout_seconds=30
        )

    @pytest.fixture
    def mock_dependencies(self):
        """Fixture for mock service dependencies"""
        return {
            "reasoning_factory_url": "http://localhost:8304",
            "memory_manager_url": "http://localhost:8205",
            "plugin_registry_url": "http://localhost:8201"
        }

    def test_agent_initialization(self, sample_config, mock_dependencies):
        """Test agent initialization with valid configuration"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful service connections
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            agent = AgentBrain(sample_config)

            assert agent.agent_id == sample_config.agent_id
            assert agent.name == sample_config.name
            assert agent.state == AgentState.INITIALIZING
            assert isinstance(agent.created_at, datetime)
            assert agent.metrics.total_tasks == 0

    @pytest.mark.asyncio
    async def test_agent_full_initialization(self, sample_config):
        """Test complete agent initialization process"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock all service responses
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"patterns": [{"name": "ReAct"}]}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            agent = AgentBrain(sample_config)
            success = await agent.initialize()

            assert success is True
            assert agent.state == AgentState.READY
            assert agent.metrics.uptime_seconds == 0

    @pytest.mark.asyncio
    async def test_agent_initialization_failure(self, sample_config):
        """Test agent initialization failure handling"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock service connection failure
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection failed")

            agent = AgentBrain(sample_config)
            success = await agent.initialize()

            assert success is False
            assert agent.state == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_task_execution_success(self, sample_config, sample_task_request):
        """Test successful task execution"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock all external service calls
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"response": "Task completed successfully"}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            agent = AgentBrain(sample_config)
            await agent.initialize()  # Initialize first

            # Mock the internal task execution
            with patch.object(agent, '_execute_task_internal') as mock_execute:
                mock_result = TaskResult(
                    task_id=sample_task_request.task_id,
                    status="completed",
                    result={"output": "test result"},
                    execution_time=1.5,
                    confidence_score=0.85
                )
                mock_execute.return_value = mock_result

                result = await agent.execute_task(sample_task_request)

                assert result.task_id == sample_task_request.task_id
                assert result.status == "completed"
                assert result.confidence_score == 0.85
                assert agent.state == AgentState.READY
                assert agent.metrics.total_tasks == 1
                assert agent.metrics.successful_tasks == 1

    @pytest.mark.asyncio
    async def test_task_execution_failure(self, sample_config, sample_task_request):
        """Test task execution failure handling"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            agent = AgentBrain(sample_config)
            await agent.initialize()

            with patch.object(agent, '_execute_task_internal') as mock_execute:
                mock_execute.side_effect = Exception("Task execution failed")

                result = await agent.execute_task(sample_task_request)

                assert result.status == "failed"
                assert "Task execution failed" in result.error_message
                assert agent.metrics.failed_tasks == 1
                assert agent.state == AgentState.READY

    @pytest.mark.asyncio
    async def test_task_execution_timeout(self, sample_config, sample_task_request):
        """Test task execution timeout handling"""
        agent = AgentBrain(sample_config)
        await agent.initialize()

        with patch.object(agent, '_execute_task_internal') as mock_execute:
            # Simulate timeout
            async def slow_execution():
                await asyncio.sleep(2)  # Longer than timeout
                return TaskResult(task_id=sample_task_request.task_id, status="completed")

            mock_execute.side_effect = slow_execution

            # Set very short timeout
            sample_task_request.timeout_seconds = 1

            result = await agent.execute_task(sample_task_request)

            assert result.status == "failed"
            assert "timed out" in result.error_message.lower()

    def test_agent_status_reporting(self, sample_config):
        """Test agent status reporting"""
        agent = AgentBrain(sample_config)

        status = agent.get_agent_status()

        assert status["agent_id"] == sample_config.agent_id
        assert status["name"] == sample_config.name
        assert status["state"] == AgentState.INITIALIZING.value
        assert "created_at" in status
        assert "total_tasks_executed" in status
        assert "success_rate" in status

    def test_task_history_management(self, sample_config):
        """Test task history management"""
        agent = AgentBrain(sample_config)

        # Initially empty
        history = agent.get_task_history()
        assert len(history) == 0

        # Add some mock history
        agent.task_history = [
            {"task_id": "task_001", "status": "completed", "execution_time": 1.5},
            {"task_id": "task_002", "status": "failed", "execution_time": 2.0},
            {"task_id": "task_003", "status": "completed", "execution_time": 1.0}
        ]

        # Test full history
        full_history = agent.get_task_history()
        assert len(full_history) == 3

        # Test limited history
        limited_history = agent.get_task_history(limit=2)
        assert len(limited_history) == 2

    @pytest.mark.asyncio
    async def test_agent_pause_resume(self, sample_config):
        """Test agent pause and resume functionality"""
        agent = AgentBrain(sample_config)
        await agent.initialize()

        # Test pause when ready
        paused = await agent.pause_agent()
        assert paused is True
        assert agent.state == AgentState.PAUSED

        # Test resume when paused
        resumed = await agent.resume_agent()
        assert resumed is True
        assert agent.state == AgentState.READY

        # Test pause when executing (should fail)
        agent.state = AgentState.EXECUTING
        paused = await agent.pause_agent()
        assert paused is False

    @pytest.mark.asyncio
    async def test_agent_termination(self, sample_config):
        """Test agent termination process"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            agent = AgentBrain(sample_config)
            await agent.initialize()

            # Add some mock active tasks
            agent.active_tasks = {"task_001": {}, "task_002": {}}

            terminated = await agent.terminate_agent()

            assert terminated is True
            assert agent.state == AgentState.TERMINATED
            assert len(agent.active_tasks) == 0

    def test_metrics_update(self, sample_config):
        """Test metrics update functionality"""
        agent = AgentBrain(sample_config)

        # Test initial metrics
        assert agent.metrics.total_tasks == 0
        assert agent.metrics.successful_tasks == 0
        assert agent.metrics.average_execution_time == 0.0

        # Update metrics manually for testing
        agent._update_execution_metrics(2.0, True, 0.8)

        assert agent.metrics.total_tasks == 1
        assert agent.metrics.successful_tasks == 1
        assert agent.metrics.average_execution_time == 2.0
        assert agent.metrics.average_confidence == 0.8

        # Update again to test averaging
        agent._update_execution_metrics(4.0, True, 0.9)

        assert agent.metrics.total_tasks == 2
        assert agent.metrics.average_execution_time == 3.0  # (2.0 + 4.0) / 2
        assert agent.metrics.average_confidence == 0.85  # (0.8 + 0.9) / 2

    @pytest.mark.asyncio
    async def test_memory_integration(self, sample_config, sample_task_request):
        """Test memory integration for task results"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock memory manager response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            agent = AgentBrain(sample_config)
            await agent.initialize()

            # Mock task result
            task_result = TaskResult(
                task_id=sample_task_request.task_id,
                status="completed",
                result={"analysis": "completed"},
                execution_time=1.5,
                confidence_score=0.85
            )

            # Test memory storage
            await agent._store_task_result(sample_task_request, task_result)

            # Verify memory storage was attempted
            assert mock_client.return_value.__aenter__.return_value.post.called

    def test_concurrent_task_limits(self, sample_config):
        """Test concurrent task execution limits"""
        agent = AgentBrain(sample_config)

        # Test with default config
        assert agent.execution_semaphore._value == Config.MAX_CONCURRENT_TASKS

        # Test semaphore acquisition
        async def test_semaphore():
            async with agent.execution_semaphore:
                assert agent.execution_semaphore._value == Config.MAX_CONCURRENT_TASKS - 1
            assert agent.execution_semaphore._value == Config.MAX_CONCURRENT_TASKS

        asyncio.run(test_semaphore())


class TestAgentBrainService:
    """Test suite for AgentBrainService"""

    @pytest.fixture
    def service(self):
        """Fixture for AgentBrainService instance"""
        return AgentBrainService()

    @pytest.fixture
    def sample_config(self):
        """Fixture for sample agent configuration"""
        return AgentConfig(
            agent_id="service_test_agent",
            name="Service Test Agent",
            persona=AgentPersona(
                name="Service Test Agent",
                description="Agent for service testing",
                domain="testing"
            ),
            reasoning_config=ReasoningConfig(pattern="ReAct"),
            memory_config=MemoryConfig(),
            plugin_config=PluginConfig()
        )

    def test_service_initialization(self, service):
        """Test service initialization"""
        assert service.active_agents == {}
        assert service.agent_configs == {}

        status = service.get_service_status()
        assert status["service"] == "Agent Brain Base Class"
        assert status["active_agents"] == 0
        assert status["total_configs"] == 0

    @pytest.mark.asyncio
    async def test_agent_creation_and_management(self, service, sample_config):
        """Test agent creation and management through service"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock service connections
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"patterns": [{"name": "ReAct"}]}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            # Create agent
            agent = await service.create_agent(sample_config)

            assert agent.agent_id == sample_config.agent_id
            assert agent.name == sample_config.name
            assert len(service.active_agents) == 1
            assert len(service.agent_configs) == 1

            # Test agent retrieval
            retrieved_agent = await service.get_agent(sample_config.agent_id)
            assert retrieved_agent == agent

            # Test agent termination
            terminated = await service.terminate_agent(sample_config.agent_id)
            assert terminated is True
            assert len(service.active_agents) == 0
            assert len(service.agent_configs) == 0

    @pytest.mark.asyncio
    async def test_agent_not_found(self, service):
        """Test handling of non-existent agents"""
        with pytest.raises(ValueError, match="Agent nonexistent not found"):
            await service.get_agent("nonexistent")

        terminated = await service.terminate_agent("nonexistent")
        assert terminated is False

    @pytest.mark.asyncio
    async def test_duplicate_agent_creation(self, service, sample_config):
        """Test duplicate agent creation prevention"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"patterns": [{"name": "ReAct"}]}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            # Create first agent
            await service.create_agent(sample_config)

            # Try to create duplicate
            with pytest.raises(ValueError, match="Agent service_test_agent already exists"):
                await service.create_agent(sample_config)


class TestTaskExecution:
    """Test suite for task execution scenarios"""

    @pytest.fixture
    def agent(self, sample_config):
        """Fixture for initialized agent"""
        return AgentBrain(sample_config)

    @pytest.fixture
    def sample_config(self):
        """Fixture for sample agent configuration"""
        return AgentConfig(
            agent_id="task_test_agent",
            name="Task Test Agent",
            persona=AgentPersona(name="Task Test Agent", domain="testing"),
            reasoning_config=ReasoningConfig(pattern="ReAct"),
            memory_config=MemoryConfig(),
            plugin_config=PluginConfig()
        )

    @pytest.mark.asyncio
    async def test_task_execution_with_constraints(self, agent, sample_task_request):
        """Test task execution with various constraints"""
        await agent.initialize()

        # Test with priority constraint
        sample_task_request.priority = "high"
        sample_task_request.constraints = {"max_complexity": "simple"}

        with patch.object(agent, '_execute_task_internal') as mock_execute:
            mock_result = TaskResult(
                task_id=sample_task_request.task_id,
                status="completed",
                result={"constraint_handled": True},
                execution_time=1.0,
                confidence_score=0.9
            )
            mock_execute.return_value = mock_result

            result = await agent.execute_task(sample_task_request)

            assert result.status == "completed"
            assert result.confidence_score == 0.9

    @pytest.mark.asyncio
    async def test_task_execution_with_callback(self, agent, sample_task_request):
        """Test task execution with callback URL"""
        await agent.initialize()

        sample_task_request.callback_url = "http://callback.example.com/result"

        with patch.object(agent, '_execute_task_internal') as mock_execute:
            mock_result = TaskResult(
                task_id=sample_task_request.task_id,
                status="completed",
                result={"callback_test": True},
                execution_time=1.0,
                confidence_score=0.8
            )
            mock_execute.return_value = mock_result

            result = await agent.execute_task(sample_task_request)

            assert result.status == "completed"
            # Note: In real implementation, callback would be invoked

    @pytest.mark.asyncio
    async def test_task_execution_error_recovery(self, agent, sample_task_request):
        """Test error recovery during task execution"""
        await agent.initialize()

        with patch.object(agent, '_execute_task_internal') as mock_execute:
            # First call fails
            mock_execute.side_effect = [
                Exception("Temporary failure"),
                TaskResult(
                    task_id=sample_task_request.task_id,
                    status="completed",
                    result={"recovered": True},
                    execution_time=2.0,
                    confidence_score=0.7
                )
            ]

            # In real implementation, there might be retry logic
            # For now, just test that failure is handled properly
            result = await agent.execute_task(sample_task_request)

            assert result.status == "failed"
            assert "Temporary failure" in result.error_message


class TestIntegration:
    """Integration tests for agent brain functionality"""

    @pytest.mark.asyncio
    async def test_full_agent_workflow(self):
        """Test complete agent workflow from creation to termination"""
        service = AgentBrainService()

        config = AgentConfig(
            agent_id="workflow_test_agent",
            name="Workflow Test Agent",
            persona=AgentPersona(name="Workflow Test Agent", domain="testing"),
            reasoning_config=ReasoningConfig(pattern="ReAct"),
            memory_config=MemoryConfig(),
            plugin_config=PluginConfig()
        )

        with patch('httpx.AsyncClient') as mock_client:
            # Mock all external services
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"patterns": [{"name": "ReAct"}]}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Step 1: Create agent
            agent = await service.create_agent(config)
            assert agent.state == AgentState.READY

            # Step 2: Execute multiple tasks
            tasks = []
            for i in range(3):
                task_request = TaskRequest(
                    task_id=f"task_{i+1}",
                    description=f"Test task {i+1}",
                    input_data={"task_number": i+1}
                )
                tasks.append(task_request)

            for task in tasks:
                with patch.object(agent, '_execute_task_internal') as mock_execute:
                    mock_result = TaskResult(
                        task_id=task.task_id,
                        status="completed",
                        result={"task_completed": True},
                        execution_time=1.0,
                        confidence_score=0.8
                    )
                    mock_execute.return_value = mock_result

                    result = await service.execute_task(config.agent_id, task)
                    assert result.status == "completed"

            # Step 3: Check agent status and history
            status = agent.get_agent_status()
            assert status["total_tasks_executed"] == 3
            assert status["successful_tasks"] == 3

            history = agent.get_task_history()
            assert len(history) == 3

            # Step 4: Terminate agent
            terminated = await service.terminate_agent(config.agent_id)
            assert terminated is True
            assert len(service.active_agents) == 0

    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self):
        """Test concurrent operations on multiple agents"""
        service = AgentBrainService()

        configs = []
        for i in range(3):
            config = AgentConfig(
                agent_id=f"concurrent_agent_{i+1}",
                name=f"Concurrent Agent {i+1}",
                persona=AgentPersona(name=f"Concurrent Agent {i+1}", domain="testing"),
                reasoning_config=ReasoningConfig(pattern="ReAct"),
                memory_config=MemoryConfig(),
                plugin_config=PluginConfig()
            )
            configs.append(config)

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"patterns": [{"name": "ReAct"}]}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            # Create agents concurrently
            create_tasks = [service.create_agent(config) for config in configs]
            agents = await asyncio.gather(*create_tasks)

            assert len(agents) == 3
            assert len(service.active_agents) == 3

            # Execute tasks concurrently
            task_requests = []
            for i, agent in enumerate(agents):
                task_request = TaskRequest(
                    task_id=f"concurrent_task_{i+1}",
                    description=f"Concurrent task {i+1}",
                    input_data={"agent_index": i}
                )
                task_requests.append((agent.agent_id, task_request))

            async def execute_single_task(agent_id, task_request):
                with patch.object(service.active_agents[agent_id], '_execute_task_internal') as mock_execute:
                    mock_result = TaskResult(
                        task_id=task_request.task_id,
                        status="completed",
                        result={"concurrent_execution": True},
                        execution_time=1.0,
                        confidence_score=0.8
                    )
                    mock_execute.return_value = mock_result

                    return await service.execute_task(agent_id, task_request)

            # Execute all tasks concurrently
            execution_tasks = [execute_single_task(agent_id, task) for agent_id, task in task_requests]
            results = await asyncio.gather(*execution_tasks)

            assert len(results) == 3
            assert all(result.status == "completed" for result in results)


class TestErrorHandling:
    """Test suite for error handling scenarios"""

    @pytest.fixture
    def agent(self, sample_config):
        """Fixture for agent instance"""
        return AgentBrain(sample_config)

    @pytest.fixture
    def sample_config(self):
        """Fixture for sample agent configuration"""
        return AgentConfig(
            agent_id="error_test_agent",
            name="Error Test Agent",
            persona=AgentPersona(name="Error Test Agent", domain="testing"),
            reasoning_config=ReasoningConfig(pattern="ReAct"),
            memory_config=MemoryConfig(),
            plugin_config=PluginConfig()
        )

    @pytest.mark.asyncio
    async def test_service_connection_failures(self, agent):
        """Test handling of service connection failures"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock connection failure
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection refused")

            success = await agent._initialize_service_connections()

            # Should not raise exception, but log warnings
            assert True  # If we get here, error was handled gracefully

    @pytest.mark.asyncio
    async def test_memory_storage_failure(self, agent, sample_task_request):
        """Test handling of memory storage failures"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock memory storage failure
            mock_response = Mock()
            mock_response.status_code = 500
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            task_result = TaskResult(
                task_id=sample_task_request.task_id,
                status="completed",
                result={"test": "data"},
                execution_time=1.0,
                confidence_score=0.8
            )

            # Should not raise exception
            await agent._store_task_result(sample_task_request, task_result)

            # Memory storage failure should be logged but not crash the agent
            assert True

    @pytest.mark.asyncio
    async def test_invalid_reasoning_pattern(self, sample_config):
        """Test handling of invalid reasoning pattern"""
        # Create config with invalid pattern
        invalid_config = sample_config.copy()
        invalid_config.reasoning_config.pattern = "InvalidPattern"

        agent = AgentBrain(invalid_config)

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            # Return empty patterns list
            mock_response.json.return_value = {"patterns": []}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            success = await agent.initialize()

            # Agent should still initialize but with warnings
            assert success is True
            assert agent.state == AgentState.READY


if __name__ == "__main__":
    pytest.main([__file__])
