#!/usr/bin/env python3
"""
Test Suite for Deployment Pipeline Service

Comprehensive testing for the deployment pipeline service including:
- Deployment request validation and processing
- Pre-deployment validation framework
- Multi-stage testing (functional, performance, security)
- Environment-specific deployment orchestration
- Real-time monitoring and status tracking
- Automated rollback capabilities
- Performance metrics and analytics
- Error handling and edge case management

Author: AgenticAI Platform
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json

from main import (
    DeploymentPipeline,
    DeploymentRequest,
    DeploymentStatus,
    DeploymentStage,
    ValidationSeverity,
    Config
)


class TestDeploymentPipeline:
    """Test suite for DeploymentPipeline core functionality"""

    @pytest.fixture
    def pipeline(self):
        """Fixture for DeploymentPipeline instance"""
        return DeploymentPipeline()

    @pytest.fixture
    def valid_deployment_request(self):
        """Fixture for valid deployment request"""
        return DeploymentRequest(
            agent_id="test_agent_001",
            environment="staging",
            version="1.0.0",
            deployment_options={"auto_rollback": True},
            test_options={"performance_test": True}
        )

    @pytest.fixture
    def invalid_deployment_request(self):
        """Fixture for invalid deployment request"""
        return DeploymentRequest(
            agent_id="",
            environment="invalid_env",
            deployment_options={},
            test_options={}
        )

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline is not None
        assert len(pipeline.active_deployments) == 0
        assert len(pipeline.service_endpoints) > 0
        assert "brain_factory" in pipeline.service_endpoints

    @pytest.mark.asyncio
    async def test_deploy_agent_success(self, pipeline, valid_deployment_request):
        """Test successful agent deployment"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock all service calls for successful deployment
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "agent_id": valid_deployment_request.agent_id,
                "status": "ready"
            }
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await pipeline.deploy_agent(valid_deployment_request)

            assert result.success is True
            assert result.agent_id == valid_deployment_request.agent_id
            assert result.status == DeploymentStatus.SUCCESS.value
            assert result.stage == DeploymentStage.COMPLETED.value
            assert result.deployment_id in pipeline.active_deployments

            # Check deployment metrics updated
            assert pipeline.metrics["total_deployments"] == 1
            assert pipeline.metrics["successful_deployments"] == 1

    @pytest.mark.asyncio
    async def test_deploy_agent_validation_failure(self, pipeline):
        """Test deployment failure due to validation errors"""
        invalid_request = DeploymentRequest(
            agent_id="",
            environment="staging"
        )

        result = await pipeline.deploy_agent(invalid_request)

        assert result.success is False
        assert result.status == DeploymentStatus.FAILED.value
        assert "validation failed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_deploy_agent_service_unavailable(self, pipeline, valid_deployment_request):
        """Test deployment failure due to service unavailability"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock service unavailability
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Service unavailable")

            result = await pipeline.deploy_agent(valid_deployment_request)

            assert result.success is False
            assert result.status == DeploymentStatus.FAILED.value
            assert "dependencies not available" in result.error_message

    @pytest.mark.asyncio
    async def test_deploy_agent_test_failure(self, pipeline, valid_deployment_request):
        """Test deployment failure due to test failures"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful validation but failed tests
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None

            # Mock successful health checks
            mock_response.json.return_value = {"status": "healthy"}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            # Mock failed test execution
            mock_client.return_value.__aenter__.return_value.post.side_effect = Exception("Test execution failed")

            result = await pipeline.deploy_agent(valid_deployment_request)

            assert result.success is False
            assert result.status == DeploymentStatus.FAILED.value
            assert "test failures" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_get_deployment_status(self, pipeline, valid_deployment_request):
        """Test getting deployment status"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful deployment first
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "ready"}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Create deployment
            result = await pipeline.deploy_agent(valid_deployment_request)
            deployment_id = result.deployment_id

            # Get status
            status = pipeline.get_deployment_status(deployment_id)

            assert status is not None
            assert status.deployment_id == deployment_id
            assert status.agent_id == valid_deployment_request.agent_id
            assert status.status == DeploymentStatus.SUCCESS.value

    @pytest.mark.asyncio
    async def test_get_deployment_status_not_found(self, pipeline):
        """Test getting status of non-existent deployment"""
        status = pipeline.get_deployment_status("nonexistent")
        assert status is None

    @pytest.mark.asyncio
    async def test_rollback_deployment(self, pipeline, valid_deployment_request):
        """Test deployment rollback"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful deployment first
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "ready"}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Create deployment
            result = await pipeline.deploy_agent(valid_deployment_request)
            deployment_id = result.deployment_id

            # Create rollback request
            from main import RollbackRequest
            rollback_request = RollbackRequest(
                deployment_id=deployment_id,
                rollback_reason="Test rollback"
            )

            # Execute rollback
            rollback_result = await pipeline.rollback_deployment(rollback_request)

            assert rollback_result.rollback_id is not None
            assert rollback_result.deployment_id == deployment_id
            assert rollback_result.status == "success"

    def test_get_deployment_metrics(self, pipeline):
        """Test getting deployment metrics"""
        metrics = pipeline.get_deployment_metrics()

        expected_fields = [
            "total_deployments", "successful_deployments", "failed_deployments",
            "success_rate", "average_deployment_time", "average_validation_time",
            "average_test_time", "active_deployments", "last_updated"
        ]

        for field in expected_fields:
            assert field in metrics

    @pytest.mark.asyncio
    async def test_list_deployments(self, pipeline, valid_deployment_request):
        """Test listing deployments"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful deployment
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "ready"}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Create deployment
            await pipeline.deploy_agent(valid_deployment_request)

            # List deployments
            deployments = await pipeline.list_deployments()

            assert len(deployments) == 1
            assert deployments[0]["agent_id"] == valid_deployment_request.agent_id
            assert deployments[0]["environment"] == valid_deployment_request.environment
            assert "deployment_id" in deployments[0]

    def test_update_deployment_metrics(self, pipeline):
        """Test deployment metrics update"""
        # Test successful deployment
        pipeline._update_deployment_metrics(True, 420.5)
        assert pipeline.metrics["total_deployments"] == 1
        assert pipeline.metrics["successful_deployments"] == 1
        assert pipeline.metrics["average_deployment_time"] == 420.5

        # Test failed deployment
        pipeline._update_deployment_metrics(False, 180.2)
        assert pipeline.metrics["total_deployments"] == 2
        assert pipeline.metrics["failed_deployments"] == 1
        assert pipeline.metrics["average_deployment_time"] == 300.35  # (420.5 + 180.2) / 2


class TestValidationFramework:
    """Test suite for validation framework"""

    @pytest.fixture
    def pipeline(self):
        """Fixture for DeploymentPipeline instance"""
        return DeploymentPipeline()

    @pytest.mark.asyncio
    async def test_validate_agent_status_success(self, pipeline):
        """Test successful agent status validation"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "ready"}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await pipeline._validate_agent_status("test_agent")

            assert result["passed"] is True
            assert result["severity"] == ValidationSeverity.CRITICAL.value
            assert "ready" in result["message"]

    @pytest.mark.asyncio
    async def test_validate_agent_status_failure(self, pipeline):
        """Test failed agent status validation"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await pipeline._validate_agent_status("test_agent")

            assert result["passed"] is False
            assert result["severity"] == ValidationSeverity.CRITICAL.value
            assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_validate_environment_config(self, pipeline):
        """Test environment configuration validation"""
        result = await pipeline._validate_environment_config("production")

        assert result["passed"] is True
        assert result["severity"] == ValidationSeverity.HIGH.value
        assert "production" in result["message"]

    @pytest.mark.asyncio
    async def test_validate_environment_config_invalid(self, pipeline):
        """Test invalid environment validation"""
        result = await pipeline._validate_environment_config("invalid_env")

        assert result["passed"] is False
        assert result["severity"] == ValidationSeverity.HIGH.value
        assert "validation failed" in result["message"]

    @pytest.mark.asyncio
    async def test_validate_service_dependencies_success(self, pipeline):
        """Test successful service dependency validation"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await pipeline._validate_service_dependencies("test_agent")

            assert result["passed"] is True
            assert result["severity"] == ValidationSeverity.HIGH.value
            assert "validated successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_validate_service_dependencies_failure(self, pipeline):
        """Test failed service dependency validation"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await pipeline._validate_service_dependencies("test_agent")

            assert result["passed"] is False
            assert result["severity"] == ValidationSeverity.HIGH.value
            assert "not healthy" in result["message"]

    @pytest.mark.asyncio
    async def test_validate_security_config(self, pipeline):
        """Test security configuration validation"""
        result = await pipeline._validate_security_config("test_agent", "production")

        assert result["passed"] is True
        assert result["severity"] == ValidationSeverity.HIGH.value
        assert "security validation passed" in result["message"]

    @pytest.mark.asyncio
    async def test_validate_resource_requirements(self, pipeline):
        """Test resource requirements validation"""
        result = await pipeline._validate_resource_requirements("test_agent", "staging")

        assert result["passed"] is True
        assert result["severity"] == ValidationSeverity.MEDIUM.value
        assert "resource requirements validated" in result["message"]


class TestTestingFramework:
    """Test suite for testing framework"""

    @pytest.fixture
    def pipeline(self):
        """Fixture for DeploymentPipeline instance"""
        return DeploymentPipeline()

    @pytest.mark.asyncio
    async def test_run_functional_test_success(self, pipeline):
        """Test successful functional test execution"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await pipeline._run_functional_test("test_agent")

            assert result["passed"] is True
            assert result["test_type"] == "functional_test"
            assert "successfully" in result["message"]
            assert "execution_time" in result

    @pytest.mark.asyncio
    async def test_run_functional_test_failure(self, pipeline):
        """Test failed functional test execution"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await pipeline._run_functional_test("test_agent")

            assert result["passed"] is False
            assert result["test_type"] == "functional_test"
            assert result["status"] == "failed"
            assert "failed" in result["message"]

    @pytest.mark.asyncio
    async def test_run_performance_test_success(self, pipeline):
        """Test successful performance test execution"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await pipeline._run_performance_test("test_agent")

            assert result["passed"] is True
            assert result["test_type"] == "performance_test"
            assert "completed" in result["message"]
            assert "metrics" in result
            assert "average_response_time" in result["metrics"]

    @pytest.mark.asyncio
    async def test_run_performance_test_failure(self, pipeline):
        """Test failed performance test execution"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Performance test failed")

            result = await pipeline._run_performance_test("test_agent")

            assert result["passed"] is False
            assert result["test_type"] == "performance_test"
            assert result["status"] == "failed"
            assert "failed" in result["message"]

    @pytest.mark.asyncio
    async def test_run_load_test_success(self, pipeline):
        """Test successful load test execution"""
        test_options = {"concurrent_users": 10, "duration": 30}

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await pipeline._run_load_test("test_agent", test_options)

            assert result["passed"] is True
            assert result["test_type"] == "load_test"
            assert "completed" in result["message"]
            assert "metrics" in result
            assert "concurrent_users" in result["metrics"]

    @pytest.mark.asyncio
    async def test_run_security_test_success(self, pipeline):
        """Test successful security test execution"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await pipeline._run_security_test("test_agent")

            assert result["passed"] is True
            assert result["test_type"] == "security_test"
            assert "completed" in result["message"]
            assert "metrics" in result
            assert "passed_checks" in result["metrics"]

    @pytest.mark.asyncio
    async def test_run_deployment_tests(self, pipeline):
        """Test running comprehensive deployment test suite"""
        test_options = {
            "performance_test": True,
            "load_test": True,
            "security_test": True
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            results = await pipeline._run_deployment_tests("test_agent", test_options)

            assert len(results) >= 3  # functional + performance + load + security
            assert all("test_name" in result for result in results)
            assert all("test_type" in result for result in results)


class TestDeploymentExecution:
    """Test suite for deployment execution"""

    @pytest.fixture
    def pipeline(self):
        """Fixture for DeploymentPipeline instance"""
        return DeploymentPipeline()

    @pytest.mark.asyncio
    async def test_execute_deployment_success(self, pipeline):
        """Test successful deployment execution"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "registered"}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await pipeline._execute_deployment("test_agent", "staging", {})

            assert result["success"] is True
            assert "agent_endpoint" in result
            assert "staging" in result["environment"]

    @pytest.mark.asyncio
    async def test_execute_deployment_failure(self, pipeline):
        """Test failed deployment execution"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await pipeline._execute_deployment("test_agent", "staging", {})

            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_monitor_deployment_success(self, pipeline):
        """Test successful deployment monitoring"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "status": "ready",
                "services_status": {"brain_base": "connected"}
            }
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await pipeline._monitor_deployment("test_agent", "staging")

            assert result["healthy"] is True
            assert result["status"] == "ready"
            assert "services_status" in result

    @pytest.mark.asyncio
    async def test_monitor_deployment_failure(self, pipeline):
        """Test failed deployment monitoring"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await pipeline._monitor_deployment("test_agent", "staging")

            assert result["healthy"] is False
            assert "error" in result


class TestErrorHandling:
    """Test suite for error handling scenarios"""

    @pytest.fixture
    def pipeline(self):
        """Fixture for DeploymentPipeline instance"""
        return DeploymentPipeline()

    @pytest.fixture
    def valid_deployment_request(self):
        """Fixture for valid deployment request"""
        return DeploymentRequest(
            agent_id="test_agent_001",
            environment="staging"
        )

    @pytest.mark.asyncio
    async def test_service_timeout_error(self, pipeline, valid_deployment_request):
        """Test handling of service timeout errors"""
        with patch('httpx.AsyncClient') as mock_client:
            import asyncio
            mock_client.return_value.__aenter__.return_value.get.side_effect = asyncio.TimeoutError()

            result = await pipeline.deploy_agent(valid_deployment_request)

            assert result.success is False
            assert result.status == DeploymentStatus.FAILED.value
            assert "timed out" in result.error_message or "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_network_connectivity_error(self, pipeline, valid_deployment_request):
        """Test handling of network connectivity errors"""
        with patch('httpx.AsyncClient') as mock_client:
            from httpx import ConnectError
            mock_client.return_value.__aenter__.return_value.get.side_effect = ConnectError("Connection refused")

            result = await pipeline.deploy_agent(valid_deployment_request)

            assert result.success is False
            assert result.status == DeploymentStatus.FAILED.value
            assert "dependencies not available" in result.error_message

    @pytest.mark.asyncio
    async def test_partial_validation_failure(self, pipeline, valid_deployment_request):
        """Test handling of partial validation failures"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock some validations passing, others failing
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None

            call_count = 0
            def mock_get(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:  # Every other call fails
                    raise Exception("Service temporarily unavailable")
                return mock_response

            mock_client.return_value.__aenter__.return_value.get.side_effect = mock_get

            result = await pipeline.deploy_agent(valid_deployment_request)

            assert result.success is False
            assert len(result.validation_results) > 0

    @pytest.mark.asyncio
    async def test_concurrent_deployment_limit(self, pipeline):
        """Test enforcement of concurrent deployment limits"""
        # This test would verify that the semaphore properly limits concurrent deployments
        # In practice, this would require running multiple deployment tasks simultaneously
        assert hasattr(pipeline, 'deployment_semaphore')
        assert isinstance(pipeline.deployment_semaphore, asyncio.Semaphore)

    @pytest.mark.asyncio
    async def test_deployment_state_consistency(self, pipeline, valid_deployment_request):
        """Test that deployment state remains consistent during failures"""
        with patch('httpx.AsyncClient') as mock_client:
            # Force a failure during deployment
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Forced failure")

            result = await pipeline.deploy_agent(valid_deployment_request)

            # Verify that deployment state is properly updated
            assert result.deployment_id in pipeline.active_deployments
            deployment = pipeline.active_deployments[result.deployment_id]

            assert deployment["status"] == DeploymentStatus.FAILED.value
            assert "error_message" in deployment


class TestIntegration:
    """Integration tests for deployment pipeline"""

    @pytest.fixture
    def pipeline(self):
        """Fixture for DeploymentPipeline instance"""
        return DeploymentPipeline()

    @pytest.mark.asyncio
    async def test_complete_deployment_workflow(self, pipeline):
        """Test complete deployment workflow from start to finish"""
        request = DeploymentRequest(
            agent_id="integration_test_agent",
            environment="staging",
            test_options={"performance_test": False, "load_test": False, "security_test": False}
        )

        with patch('httpx.AsyncClient') as mock_client:
            # Mock all service interactions for successful deployment
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "ready"}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Execute deployment
            result = await pipeline.deploy_agent(request)

            # Verify complete workflow
            assert result.success is True
            assert result.status == DeploymentStatus.SUCCESS.value
            assert result.stage == DeploymentStage.COMPLETED.value
            assert result.progress_percentage == 100

            # Verify all stages completed
            assert len(result.validation_results) > 0
            assert len(result.test_results) > 0
            assert result.deployment_metadata is not None

            # Verify deployment tracking
            assert result.deployment_id in pipeline.active_deployments
            deployment = pipeline.active_deployments[result.deployment_id]
            assert deployment["stage"] == DeploymentStage.COMPLETED.value

    @pytest.mark.asyncio
    async def test_deployment_rollback_workflow(self, pipeline):
        """Test deployment rollback workflow"""
        # First create a successful deployment
        request = DeploymentRequest(
            agent_id="rollback_test_agent",
            environment="staging"
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "ready"}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Create deployment
            result = await pipeline.deploy_agent(request)
            assert result.success is True

            # Now test rollback
            from main import RollbackRequest
            rollback_request = RollbackRequest(
                deployment_id=result.deployment_id,
                rollback_reason="Testing rollback functionality"
            )

            rollback_result = await pipeline.rollback_deployment(rollback_request)

            # Verify rollback completed
            assert rollback_result.status == "success"
            assert rollback_result.rollback_id is not None
            assert rollback_result.deployment_id == result.deployment_id

    @pytest.mark.asyncio
    async def test_deployment_metrics_accumulation(self, pipeline):
        """Test deployment metrics accumulation over multiple deployments"""
        initial_total = pipeline.metrics["total_deployments"]

        # Execute multiple deployments
        for i in range(3):
            request = DeploymentRequest(
                agent_id=f"metrics_test_agent_{i}",
                environment="staging"
            )

            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {"status": "ready"}
                mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
                mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

                result = await pipeline.deploy_agent(request)
                assert result.success is True

        # Verify metrics accumulation
        final_total = pipeline.metrics["total_deployments"]
        assert final_total == initial_total + 3
        assert pipeline.metrics["successful_deployments"] == initial_total + 3
        assert pipeline.metrics["failed_deployments"] == 0

        # Verify success rate calculation
        success_rate = pipeline.metrics["successful_deployments"] / pipeline.metrics["total_deployments"]
        assert success_rate == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
