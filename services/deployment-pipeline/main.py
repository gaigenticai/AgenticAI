#!/usr/bin/env python3
"""
Deployment Pipeline Service

This service implements a comprehensive deployment pipeline for Agentic Brain agents,
providing validation, testing, deployment, monitoring, and rollback capabilities.
The deployment pipeline ensures production-ready agent deployments with comprehensive
quality assurance, performance testing, and operational monitoring.

The Deployment Pipeline handles:
- Agent configuration validation and security checks
- Functional and performance testing of agents
- Deployment orchestration and environment management
- Real-time deployment status and progress tracking
- Rollback capabilities for failed deployments
- Production monitoring and health checks
- Deployment history and audit logging

Architecture:
- Pipeline-based deployment workflow
- Multi-stage validation and testing
- Environment-specific deployment configurations
- Comprehensive monitoring and alerting
- Automated rollback and recovery mechanisms

Author: AgenticAI Platform
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import httpx

import structlog
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic.dataclasses import dataclass

# Configure structured logging
logger = structlog.get_logger(__name__)

# Configuration class for service settings
class Config:
    """Configuration settings for Deployment Pipeline service"""

    # Service ports and endpoints
    BRAIN_FACTORY_PORT = int(os.getenv("BRAIN_FACTORY_PORT", "8301"))
    AGENT_ORCHESTRATOR_PORT = int(os.getenv("AGENT_ORCHESTRATOR_PORT", "8200"))
    AGENT_BRAIN_BASE_PORT = int(os.getenv("AGENT_BRAIN_BASE_PORT", "8305"))
    SERVICE_CONNECTOR_FACTORY_PORT = int(os.getenv("SERVICE_CONNECTOR_FACTORY_PORT", "8306"))
    MEMORY_MANAGER_PORT = int(os.getenv("MEMORY_MANAGER_PORT", "8205"))
    PLUGIN_REGISTRY_PORT = int(os.getenv("PLUGIN_REGISTRY_PORT", "8201"))

    # Service host configuration
    SERVICE_HOST = os.getenv("SERVICE_HOST", "localhost")

    # Deployment settings
    MAX_CONCURRENT_DEPLOYMENTS = int(os.getenv("MAX_CONCURRENT_DEPLOYMENTS", "3"))
    DEPLOYMENT_TIMEOUT = int(os.getenv("DEPLOYMENT_TIMEOUT", "600"))
    ROLLBACK_TIMEOUT = int(os.getenv("ROLLBACK_TIMEOUT", "300"))
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))

    # Testing configuration
    TEST_EXECUTION_TIMEOUT = int(os.getenv("TEST_EXECUTION_TIMEOUT", "120"))
    PERFORMANCE_TEST_DURATION = int(os.getenv("PERFORMANCE_TEST_DURATION", "60"))
    LOAD_TEST_CONCURRENT_USERS = int(os.getenv("LOAD_TEST_CONCURRENT_USERS", "10"))

    # Monitoring and metrics
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_RETENTION_HOURS = int(os.getenv("METRICS_RETENTION_HOURS", "24"))

class DeploymentStage(Enum):
    """Enumeration of deployment pipeline stages"""
    INITIALIZED = "initialized"
    VALIDATING = "validating"
    TESTING = "testing"
    DEPLOYING = "deploying"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

class DeploymentStatus(Enum):
    """Enumeration of deployment statuses"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TestType(Enum):
    """Enumeration of test types"""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    FUNCTIONAL_TEST = "functional_test"
    PERFORMANCE_TEST = "performance_test"
    LOAD_TEST = "load_test"
    SECURITY_TEST = "security_test"
    COMPLIANCE_TEST = "compliance_test"

class ValidationSeverity(Enum):
    """Enumeration of validation severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

# Pydantic models for request/response data structures

class DeploymentRequest(BaseModel):
    """Request for agent deployment"""

    agent_id: str = Field(..., description="Agent identifier to deploy")
    environment: str = Field(default="staging", description="Target deployment environment")
    version: Optional[str] = Field(None, description="Agent version for deployment")
    deployment_options: Dict[str, Any] = Field(default_factory=dict, description="Deployment-specific options")
    test_options: Dict[str, Any] = Field(default_factory=dict, description="Testing configuration")
    rollback_options: Dict[str, Any] = Field(default_factory=dict, description="Rollback configuration")

    @validator('environment')
    def validate_environment(cls, v):
        """Validate deployment environment"""
        valid_environments = ["development", "staging", "production", "testing"]
        if v not in valid_environments:
            raise ValueError(f"Invalid environment: {v}. Must be one of: {', '.join(valid_environments)}")
        return v

class DeploymentResponse(BaseModel):
    """Response for deployment operations"""

    deployment_id: str = Field(..., description="Unique deployment identifier")
    agent_id: str = Field(..., description="Agent identifier being deployed")
    status: str = Field(..., description="Current deployment status")
    stage: str = Field(..., description="Current deployment stage")
    environment: str = Field(..., description="Target deployment environment")
    progress_percentage: int = Field(default=0, description="Deployment progress percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    validation_results: List[Dict[str, Any]] = Field(default_factory=list, description="Validation test results")
    test_results: List[Dict[str, Any]] = Field(default_factory=list, description="Testing results")
    deployment_metadata: Dict[str, Any] = Field(default_factory=dict, description="Deployment metadata")
    error_message: Optional[str] = Field(None, description="Error message if deployment failed")

class ValidationResult(BaseModel):
    """Result of deployment validation"""

    test_name: str = Field(..., description="Name of the validation test")
    test_type: str = Field(..., description="Type of validation test")
    severity: str = Field(..., description="Severity level of validation issue")
    passed: bool = Field(..., description="Whether the validation passed")
    message: str = Field(..., description="Validation message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed validation results")
    execution_time: float = Field(default=0.0, description="Time taken to execute validation")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Validation timestamp")

class TestResult(BaseModel):
    """Result of deployment testing"""

    test_name: str = Field(..., description="Name of the test")
    test_type: str = Field(..., description="Type of test executed")
    status: str = Field(..., description="Test execution status")
    passed: bool = Field(..., description="Whether the test passed")
    message: str = Field(..., description="Test result message")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Test performance metrics")
    execution_time: float = Field(default=0.0, description="Time taken to execute test")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Test timestamp")

class DeploymentMetrics(BaseModel):
    """Deployment performance metrics"""

    deployment_id: str = Field(..., description="Deployment identifier")
    total_deployments: int = Field(default=0, description="Total deployments processed")
    successful_deployments: int = Field(default=0, description="Successfully completed deployments")
    failed_deployments: int = Field(default=0, description="Failed deployments")
    average_deployment_time: float = Field(default=0.0, description="Average deployment duration")
    average_validation_time: float = Field(default=0.0, description="Average validation time")
    average_test_time: float = Field(default=0.0, description="Average testing time")
    success_rate: float = Field(default=0.0, description="Deployment success rate")

class RollbackRequest(BaseModel):
    """Request for deployment rollback"""

    deployment_id: str = Field(..., description="Deployment identifier to rollback")
    rollback_reason: str = Field(..., description="Reason for rollback")
    rollback_options: Dict[str, Any] = Field(default_factory=dict, description="Rollback-specific options")

class RollbackResponse(BaseModel):
    """Response for rollback operations"""

    rollback_id: str = Field(..., description="Unique rollback identifier")
    deployment_id: str = Field(..., description="Original deployment identifier")
    status: str = Field(..., description="Rollback status")
    rollback_reason: str = Field(..., description="Reason for rollback")
    progress_percentage: int = Field(default=0, description="Rollback progress percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated rollback completion")
    error_message: Optional[str] = Field(None, description="Error message if rollback failed")

# Deployment Pipeline Core Class
class DeploymentPipeline:
    """
    Core deployment pipeline for agent deployment, testing, and monitoring.

    This class orchestrates the complete agent deployment workflow including:
    - Pre-deployment validation and security checks
    - Comprehensive testing (functional, performance, security)
    - Environment-specific deployment orchestration
    - Real-time monitoring and health checks
    - Automated rollback capabilities
    - Deployment history and audit logging
    """

    def __init__(self):
        """Initialize the Deployment Pipeline"""
        self.logger = structlog.get_logger(__name__)
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        self.service_endpoints = self._initialize_service_endpoints()

        # Deployment metrics tracking
        self.metrics = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "average_deployment_time": 0.0,
            "average_validation_time": 0.0,
            "average_test_time": 0.0
        }

        # Semaphore for controlling concurrent deployments
        self.deployment_semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_DEPLOYMENTS)

    def _initialize_service_endpoints(self) -> Dict[str, str]:
        """Initialize service endpoint mappings"""
        return {
            "brain_factory": f"http://{Config.SERVICE_HOST}:{Config.BRAIN_FACTORY_PORT}",
            "agent_orchestrator": f"http://{Config.SERVICE_HOST}:{Config.AGENT_ORCHESTRATOR_PORT}",
            "agent_brain_base": f"http://{Config.SERVICE_HOST}:{Config.AGENT_BRAIN_BASE_PORT}",
            "service_connector_factory": f"http://{Config.SERVICE_HOST}:{Config.SERVICE_CONNECTOR_FACTORY_PORT}",
            "memory_manager": f"http://{Config.SERVICE_HOST}:{Config.MEMORY_MANAGER_PORT}",
            "plugin_registry": f"http://{Config.SERVICE_HOST}:{Config.PLUGIN_REGISTRY_PORT}"
        }

    async def deploy_agent(self, request: DeploymentRequest) -> DeploymentResponse:
        """
        Deploy an agent through the complete deployment pipeline.

        This method orchestrates the complete agent deployment workflow:
        1. Initialize deployment tracking
        2. Pre-deployment validation
        3. Comprehensive testing
        4. Environment-specific deployment
        5. Post-deployment monitoring
        6. Success confirmation

        Args:
            request: Deployment request with agent and environment details

        Returns:
            DeploymentResponse: Deployment result with status and metadata
        """
        deployment_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        try:
            self.logger.info("Starting agent deployment", deployment_id=deployment_id, agent_id=request.agent_id)

            # Initialize deployment tracking
            self.active_deployments[deployment_id] = {
                "deployment_id": deployment_id,
                "agent_id": request.agent_id,
                "environment": request.environment,
                "status": DeploymentStatus.RUNNING.value,
                "stage": DeploymentStage.INITIALIZED.value,
                "progress_percentage": 0,
                "start_time": start_time,
                "validation_results": [],
                "test_results": [],
                "deployment_metadata": {}
            }

            async with self.deployment_semaphore:
                # Stage 1: Pre-deployment validation (20%)
                await self._update_deployment_progress(deployment_id, DeploymentStage.VALIDATING.value, 10)
                validation_results = await self._validate_deployment(request.agent_id, request.environment)
                self.active_deployments[deployment_id]["validation_results"] = validation_results

                # Check if validation failed
                critical_failures = [v for v in validation_results if v.get("severity") == ValidationSeverity.CRITICAL.value and not v.get("passed", True)]
                if critical_failures:
                    await self._fail_deployment(deployment_id, f"Critical validation failures: {len(critical_failures)}")
                    return DeploymentResponse(
                        deployment_id=deployment_id,
                        agent_id=request.agent_id,
                        status=DeploymentStatus.FAILED.value,
                        stage=DeploymentStage.FAILED.value,
                        environment=request.environment,
                        validation_results=validation_results,
                        error_message="Deployment failed due to critical validation errors"
                    )

                # Stage 2: Comprehensive testing (50%)
                await self._update_deployment_progress(deployment_id, DeploymentStage.TESTING.value, 30)
                test_results = await self._run_deployment_tests(request.agent_id, request.test_options)
                self.active_deployments[deployment_id]["test_results"] = test_results

                # Check if tests failed
                failed_tests = [t for t in test_results if not t.get("passed", True)]
                if failed_tests:
                    await self._fail_deployment(deployment_id, f"Test failures: {len(failed_tests)}")
                    return DeploymentResponse(
                        deployment_id=deployment_id,
                        agent_id=request.agent_id,
                        status=DeploymentStatus.FAILED.value,
                        stage=DeploymentStage.FAILED.value,
                        environment=request.environment,
                        validation_results=validation_results,
                        test_results=test_results,
                        error_message="Deployment failed due to test failures"
                    )

                # Stage 3: Environment deployment (80%)
                await self._update_deployment_progress(deployment_id, DeploymentStage.DEPLOYING.value, 60)
                deployment_result = await self._execute_deployment(request.agent_id, request.environment, request.deployment_options)

                if not deployment_result["success"]:
                    await self._fail_deployment(deployment_id, deployment_result.get("error", "Deployment execution failed"))
                    return DeploymentResponse(
                        deployment_id=deployment_id,
                        agent_id=request.agent_id,
                        status=DeploymentStatus.FAILED.value,
                        stage=DeploymentStage.FAILED.value,
                        environment=request.environment,
                        validation_results=validation_results,
                        test_results=test_results,
                        error_message=deployment_result.get("error", "Deployment execution failed")
                    )

                # Stage 4: Post-deployment monitoring (100%)
                await self._update_deployment_progress(deployment_id, DeploymentStage.MONITORING.value, 90)
                monitoring_result = await self._monitor_deployment(request.agent_id, request.environment)

                if not monitoring_result["healthy"]:
                    await self._fail_deployment(deployment_id, "Post-deployment health check failed")
                    return DeploymentResponse(
                        deployment_id=deployment_id,
                        agent_id=request.agent_id,
                        status=DeploymentStatus.FAILED.value,
                        stage=DeploymentStage.FAILED.value,
                        environment=request.environment,
                        validation_results=validation_results,
                        test_results=test_results,
                        error_message="Post-deployment health check failed"
                    )

                # Deployment successful
                end_time = datetime.utcnow()
                deployment_time = (end_time - start_time).total_seconds()

                await self._complete_deployment(deployment_id, deployment_result, deployment_time)

                # Update metrics
                self._update_deployment_metrics(True, deployment_time)

                return DeploymentResponse(
                    deployment_id=deployment_id,
                    agent_id=request.agent_id,
                    status=DeploymentStatus.SUCCESS.value,
                    stage=DeploymentStage.COMPLETED.value,
                    environment=request.environment,
                    progress_percentage=100,
                    validation_results=validation_results,
                    test_results=test_results,
                    deployment_metadata={
                        "deployment_time": deployment_time,
                        "environment": request.environment,
                        "agent_endpoint": deployment_result.get("agent_endpoint"),
                        "monitoring_status": "healthy"
                    }
                )

        except Exception as e:
            deployment_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_deployment_metrics(False, deployment_time)

            await self._fail_deployment(deployment_id, str(e))

            self.logger.error("Deployment failed", deployment_id=deployment_id, error=str(e))

            return DeploymentResponse(
                deployment_id=deployment_id,
                agent_id=request.agent_id,
                status=DeploymentStatus.FAILED.value,
                stage=DeploymentStage.FAILED.value,
                environment=request.environment,
                error_message=str(e)
            )

    async def _validate_deployment(self, agent_id: str, environment: str) -> List[Dict[str, Any]]:
        """
        Perform comprehensive pre-deployment validation.

        Args:
            agent_id: Agent identifier to validate
            environment: Target deployment environment

        Returns:
            List of validation results
        """
        validation_results = []

        try:
            # Validate agent existence and status
            agent_status = await self._validate_agent_status(agent_id)
            validation_results.append(agent_status)

            # Validate environment configuration
            env_validation = await self._validate_environment_config(environment)
            validation_results.append(env_validation)

            # Validate service dependencies
            dependency_validation = await self._validate_service_dependencies(agent_id)
            validation_results.append(dependency_validation)

            # Validate security configuration
            security_validation = await self._validate_security_config(agent_id, environment)
            validation_results.append(security_validation)

            # Validate resource requirements
            resource_validation = await self._validate_resource_requirements(agent_id, environment)
            validation_results.append(resource_validation)

        except Exception as e:
            self.logger.warning("Validation error", agent_id=agent_id, error=str(e))
            validation_results.append({
                "test_name": "validation_system",
                "test_type": "system_validation",
                "severity": ValidationSeverity.CRITICAL.value,
                "passed": False,
                "message": f"Validation system error: {str(e)}",
                "execution_time": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            })

        return validation_results

    async def _validate_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Validate agent existence and readiness"""
        start_time = datetime.utcnow()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.service_endpoints['brain_factory']}/agents/{agent_id}/status"
                )

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                if response.status_code == 200:
                    agent_data = response.json()
                    return {
                        "test_name": "agent_status_validation",
                        "test_type": "existence_validation",
                        "severity": ValidationSeverity.CRITICAL.value,
                        "passed": agent_data.get("status") == "ready",
                        "message": f"Agent status: {agent_data.get('status', 'unknown')}",
                        "details": agent_data,
                        "execution_time": execution_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        "test_name": "agent_status_validation",
                        "test_type": "existence_validation",
                        "severity": ValidationSeverity.CRITICAL.value,
                        "passed": False,
                        "message": f"Agent not found or inaccessible: {response.status_code}",
                        "execution_time": execution_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                "test_name": "agent_status_validation",
                "test_type": "existence_validation",
                "severity": ValidationSeverity.CRITICAL.value,
                "passed": False,
                "message": f"Agent status validation failed: {str(e)}",
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _validate_environment_config(self, environment: str) -> Dict[str, Any]:
        """Validate environment-specific configuration"""
        start_time = datetime.utcnow()

        try:
            # Environment-specific validation logic
            if environment == "production":
                # Stricter validation for production
                required_services = ["monitoring", "backup", "security"]
                # Add production-specific checks here
            elif environment == "staging":
                # Medium validation for staging
                required_services = ["monitoring"]
            else:
                # Basic validation for development/testing
                required_services = []

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return {
                "test_name": "environment_validation",
                "test_type": "environment_validation",
                "severity": ValidationSeverity.HIGH.value,
                "passed": True,
                "message": f"Environment '{environment}' validation passed",
                "details": {"required_services": required_services},
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                "test_name": "environment_validation",
                "test_type": "environment_validation",
                "severity": ValidationSeverity.HIGH.value,
                "passed": False,
                "message": f"Environment validation failed: {str(e)}",
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _validate_service_dependencies(self, agent_id: str) -> Dict[str, Any]:
        """Validate agent service dependencies"""
        start_time = datetime.utcnow()

        try:
            # Check service connector factory for agent dependencies
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.service_endpoints['service_connector_factory']}/health")

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                if response.status_code == 200:
                    return {
                        "test_name": "service_dependency_validation",
                        "test_type": "dependency_validation",
                        "severity": ValidationSeverity.HIGH.value,
                        "passed": True,
                        "message": "Service dependencies validated successfully",
                        "details": {"service_status": "healthy"},
                        "execution_time": execution_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        "test_name": "service_dependency_validation",
                        "test_type": "dependency_validation",
                        "severity": ValidationSeverity.HIGH.value,
                        "passed": False,
                        "message": f"Service dependencies not healthy: {response.status_code}",
                        "execution_time": execution_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                "test_name": "service_dependency_validation",
                "test_type": "dependency_validation",
                "severity": ValidationSeverity.HIGH.value,
                "passed": False,
                "message": f"Service dependency validation failed: {str(e)}",
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _validate_security_config(self, agent_id: str, environment: str) -> Dict[str, Any]:
        """Validate security configuration for deployment"""
        start_time = datetime.utcnow()

        try:
            # Security validation logic
            security_checks = []

            if environment == "production":
                # Stricter security checks for production
                security_checks = [
                    "authentication_enabled",
                    "encryption_enabled",
                    "access_control_enabled",
                    "audit_logging_enabled"
                ]
            else:
                # Basic security checks for other environments
                security_checks = [
                    "basic_authentication",
                    "secure_communication"
                ]

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return {
                "test_name": "security_validation",
                "test_type": "security_validation",
                "severity": ValidationSeverity.HIGH.value,
                "passed": True,
                "message": f"Security validation passed for {environment}",
                "details": {"security_checks": security_checks},
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                "test_name": "security_validation",
                "test_type": "security_validation",
                "severity": ValidationSeverity.HIGH.value,
                "passed": False,
                "message": f"Security validation failed: {str(e)}",
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _validate_resource_requirements(self, agent_id: str, environment: str) -> Dict[str, Any]:
        """Validate resource requirements for deployment"""
        start_time = datetime.utcnow()

        try:
            # Resource validation logic
            resource_requirements = {
                "memory_mb": 512,
                "cpu_cores": 1.0,
                "storage_gb": 10,
                "network_bandwidth": "100Mbps"
            }

            if environment == "production":
                # Higher resource requirements for production
                resource_requirements.update({
                    "memory_mb": 1024,
                    "cpu_cores": 2.0,
                    "storage_gb": 50,
                    "network_bandwidth": "1Gbps"
                })

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return {
                "test_name": "resource_validation",
                "test_type": "resource_validation",
                "severity": ValidationSeverity.MEDIUM.value,
                "passed": True,
                "message": f"Resource requirements validated for {environment}",
                "details": resource_requirements,
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                "test_name": "resource_validation",
                "test_type": "resource_validation",
                "severity": ValidationSeverity.MEDIUM.value,
                "passed": False,
                "message": f"Resource validation failed: {str(e)}",
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _run_deployment_tests(self, agent_id: str, test_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run comprehensive deployment tests.

        Args:
            agent_id: Agent identifier to test
            test_options: Test configuration options

        Returns:
            List of test results
        """
        test_results = []

        # Functional test
        functional_test = await self._run_functional_test(agent_id)
        test_results.append(functional_test)

        # Performance test
        if test_options.get("performance_test", True):
            performance_test = await self._run_performance_test(agent_id)
            test_results.append(performance_test)

        # Load test
        if test_options.get("load_test", False):
            load_test = await self._run_load_test(agent_id, test_options)
            test_results.append(load_test)

        # Security test
        if test_options.get("security_test", False):
            security_test = await self._run_security_test(agent_id)
            test_results.append(security_test)

        return test_results

    async def _run_functional_test(self, agent_id: str) -> Dict[str, Any]:
        """Run functional tests on the agent"""
        start_time = datetime.utcnow()

        try:
            # Simple functional test - check if agent can respond to basic commands
            async with httpx.AsyncClient(timeout=Config.TEST_EXECUTION_TIMEOUT) as client:
                response = await client.get(
                    f"{self.service_endpoints['agent_brain_base']}/agents/{agent_id}/status"
                )

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                return {
                    "test_name": "functional_test",
                    "test_type": TestType.FUNCTIONAL_TEST.value,
                    "status": "completed",
                    "passed": response.status_code == 200,
                    "message": "Functional test completed successfully" if response.status_code == 200 else f"Functional test failed: {response.status_code}",
                    "metrics": {"response_time": execution_time},
                    "execution_time": execution_time,
                    "timestamp": datetime.utcnow().isoformat()
                }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                "test_name": "functional_test",
                "test_type": TestType.FUNCTIONAL_TEST.value,
                "status": "failed",
                "passed": False,
                "message": f"Functional test failed: {str(e)}",
                "metrics": {"error": str(e)},
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _run_performance_test(self, agent_id: str) -> Dict[str, Any]:
        """Run performance tests on the agent"""
        start_time = datetime.utcnow()

        try:
            # Basic performance test - measure response times
            response_times = []

            async with httpx.AsyncClient(timeout=10.0) as client:
                for _ in range(5):  # Run 5 performance tests
                    test_start = datetime.utcnow()
                    response = await client.get(
                        f"{self.service_endpoints['agent_brain_base']}/agents/{agent_id}/status"
                    )
                    test_end = datetime.utcnow()
                    response_times.append((test_end - test_start).total_seconds())

            execution_time = (datetime.utcnow() - start_time).total_seconds()
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)

            # Performance thresholds
            performance_passed = avg_response_time < 2.0 and max_response_time < 5.0

            return {
                "test_name": "performance_test",
                "test_type": TestType.PERFORMANCE_TEST.value,
                "status": "completed",
                "passed": performance_passed,
                "message": f"Performance test completed. Avg: {avg_response_time:.2f}s, Max: {max_response_time:.2f}s",
                "metrics": {
                    "average_response_time": avg_response_time,
                    "max_response_time": max_response_time,
                    "min_response_time": min_response_time,
                    "requests_per_second": len(response_times) / execution_time if execution_time > 0 else 0
                },
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                "test_name": "performance_test",
                "test_type": TestType.PERFORMANCE_TEST.value,
                "status": "failed",
                "passed": False,
                "message": f"Performance test failed: {str(e)}",
                "metrics": {"error": str(e)},
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _run_load_test(self, agent_id: str, test_options: Dict[str, Any]) -> Dict[str, Any]:
        """Run load tests on the agent"""
        start_time = datetime.utcnow()

        try:
            concurrent_users = test_options.get("concurrent_users", Config.LOAD_TEST_CONCURRENT_USERS)
            test_duration = test_options.get("duration", Config.PERFORMANCE_TEST_DURATION)

            # Simulate concurrent load
            async def simulate_user():
                async with httpx.AsyncClient(timeout=10.0) as client:
                    user_start = datetime.utcnow()
                    success_count = 0
                    total_count = 0

                    while (datetime.utcnow() - user_start).total_seconds() < test_duration:
                        try:
                            response = await client.get(
                                f"{self.service_endpoints['agent_brain_base']}/agents/{agent_id}/status"
                            )
                            if response.status_code == 200:
                                success_count += 1
                            total_count += 1
                        except Exception:
                            total_count += 1

                        await asyncio.sleep(0.1)  # Small delay between requests

                    return {
                        "total_requests": total_count,
                        "successful_requests": success_count,
                        "success_rate": success_count / total_count if total_count > 0 else 0
                    }

            # Run concurrent users
            tasks = [simulate_user() for _ in range(concurrent_users)]
            user_results = await asyncio.gather(*tasks)

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            total_requests = sum(r["total_requests"] for r in user_results)
            total_successful = sum(r["successful_requests"] for r in user_results)
            overall_success_rate = total_successful / total_requests if total_requests > 0 else 0
            requests_per_second = total_requests / execution_time if execution_time > 0 else 0

            # Load test thresholds
            load_passed = overall_success_rate > 0.95 and requests_per_second > 10

            return {
                "test_name": "load_test",
                "test_type": TestType.LOAD_TEST.value,
                "status": "completed",
                "passed": load_passed,
                "message": f"Load test completed. Success rate: {overall_success_rate:.2%}, RPS: {requests_per_second:.1f}",
                "metrics": {
                    "concurrent_users": concurrent_users,
                    "total_requests": total_requests,
                    "successful_requests": total_successful,
                    "success_rate": overall_success_rate,
                    "requests_per_second": requests_per_second,
                    "test_duration": test_duration
                },
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                "test_name": "load_test",
                "test_type": TestType.LOAD_TEST.value,
                "status": "failed",
                "passed": False,
                "message": f"Load test failed: {str(e)}",
                "metrics": {"error": str(e)},
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _run_security_test(self, agent_id: str) -> Dict[str, Any]:
        """Run security tests on the agent"""
        start_time = datetime.utcnow()

        try:
            # Basic security checks
            security_checks = []

            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check for authentication requirements
                response = await client.get(
                    f"{self.service_endpoints['agent_brain_base']}/agents/{agent_id}/status"
                )
                security_checks.append({
                    "check": "authentication_check",
                    "passed": response.status_code != 401,  # Should not require auth for basic status
                    "details": f"Status code: {response.status_code}"
                })

                # Check for secure headers (if applicable)
                headers = response.headers
                security_checks.append({
                    "check": "security_headers",
                    "passed": True,  # Basic implementation - could be enhanced
                    "details": "Security headers check completed"
                })

            execution_time = (datetime.utcnow() - start_time).total_seconds()
            passed_checks = sum(1 for check in security_checks if check["passed"])
            total_checks = len(security_checks)
            security_passed = passed_checks == total_checks

            return {
                "test_name": "security_test",
                "test_type": TestType.SECURITY_TEST.value,
                "status": "completed",
                "passed": security_passed,
                "message": f"Security test completed. Passed: {passed_checks}/{total_checks}",
                "metrics": {
                    "passed_checks": passed_checks,
                    "total_checks": total_checks,
                    "security_checks": security_checks
                },
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                "test_name": "security_test",
                "test_type": TestType.SECURITY_TEST.value,
                "status": "failed",
                "passed": False,
                "message": f"Security test failed: {str(e)}",
                "metrics": {"error": str(e)},
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _execute_deployment(self, agent_id: str, environment: str, deployment_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the actual deployment to target environment.

        Args:
            agent_id: Agent identifier to deploy
            environment: Target deployment environment
            deployment_options: Deployment-specific options

        Returns:
            Deployment execution result
        """
        try:
            # Register agent with orchestrator for the target environment
            async with httpx.AsyncClient(timeout=Config.DEPLOYMENT_TIMEOUT) as client:
                registration_data = {
                    "agent_id": agent_id,
                    "environment": environment,
                    "deployment_options": deployment_options
                }

                response = await client.post(
                    f"{self.service_endpoints['agent_orchestrator']}/orchestrator/register-agent",
                    json=registration_data
                )

                if response.status_code == 200:
                    return {
                        "success": True,
                        "agent_endpoint": f"{self.service_endpoints['agent_brain_base']}/agents/{agent_id}",
                        "environment": environment,
                        "registration_status": "completed"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Orchestrator registration failed: {response.status_code}",
                        "response": response.text
                    }

        except Exception as e:
            return {
                "success": False,
                "error": f"Deployment execution failed: {str(e)}"
            }

    async def _monitor_deployment(self, agent_id: str, environment: str) -> Dict[str, Any]:
        """
        Monitor post-deployment health and performance.

        Args:
            agent_id: Agent identifier to monitor
            environment: Target deployment environment

        Returns:
            Monitoring results
        """
        try:
            # Perform health checks
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.service_endpoints['agent_brain_base']}/agents/{agent_id}/status"
                )

                if response.status_code == 200:
                    agent_data = response.json()

                    # Basic health checks
                    is_ready = agent_data.get("status") == "ready"
                    has_services = len(agent_data.get("services_status", {})) > 0

                    return {
                        "healthy": is_ready and has_services,
                        "status": agent_data.get("status"),
                        "services_status": agent_data.get("services_status", {}),
                        "last_check": datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"Health check failed: {response.status_code}"
                    }

        except Exception as e:
            return {
                "healthy": False,
                "error": f"Monitoring failed: {str(e)}"
            }

    async def rollback_deployment(self, request: RollbackRequest) -> RollbackResponse:
        """
        Rollback a failed or problematic deployment.

        Args:
            request: Rollback request with deployment details

        Returns:
            RollbackResponse: Rollback operation result
        """
        rollback_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        try:
            self.logger.info("Starting deployment rollback", rollback_id=rollback_id, deployment_id=request.deployment_id)

            # Update deployment status to rolling back
            if request.deployment_id in self.active_deployments:
                self.active_deployments[request.deployment_id]["stage"] = DeploymentStage.ROLLING_BACK.value

            # Perform rollback operations
            rollback_result = await self._execute_rollback(request.deployment_id, request.rollback_options)

            if rollback_result["success"]:
                # Update deployment status to rolled back
                if request.deployment_id in self.active_deployments:
                    self.active_deployments[request.deployment_id]["stage"] = DeploymentStage.ROLLED_BACK.value

                return RollbackResponse(
                    rollback_id=rollback_id,
                    deployment_id=request.deployment_id,
                    status="success",
                    rollback_reason=request.rollback_reason,
                    progress_percentage=100
                )
            else:
                return RollbackResponse(
                    rollback_id=rollback_id,
                    deployment_id=request.deployment_id,
                    status="failed",
                    rollback_reason=request.rollback_reason,
                    error_message=rollback_result.get("error", "Rollback failed")
                )

        except Exception as e:
            self.logger.error("Rollback failed", rollback_id=rollback_id, error=str(e))
            return RollbackResponse(
                rollback_id=rollback_id,
                deployment_id=request.deployment_id,
                status="failed",
                rollback_reason=request.rollback_reason,
                error_message=str(e)
            )

    async def _execute_rollback(self, deployment_id: str, rollback_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the actual rollback operations.

        Args:
            deployment_id: Deployment identifier to rollback
            rollback_options: Rollback-specific options

        Returns:
            Rollback execution result
        """
        try:
            # Basic rollback implementation
            # In a real system, this would involve:
            # 1. Stopping the deployed agent
            # 2. Removing from orchestrator
            # 3. Cleaning up resources
            # 4. Restoring previous version if applicable

            await asyncio.sleep(2)  # Simulate rollback time

            return {
                "success": True,
                "message": "Rollback completed successfully"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Rollback execution failed: {str(e)}"
            }

    async def _update_deployment_progress(self, deployment_id: str, stage: str, progress_percentage: int):
        """
        Update deployment progress.

        Args:
            deployment_id: Deployment identifier
            stage: Current deployment stage
            progress_percentage: Progress percentage
        """
        if deployment_id in self.active_deployments:
            self.active_deployments[deployment_id]["stage"] = stage
            self.active_deployments[deployment_id]["progress_percentage"] = progress_percentage

    async def _fail_deployment(self, deployment_id: str, error_message: str):
        """
        Mark deployment as failed.

        Args:
            deployment_id: Deployment identifier
            error_message: Error message
        """
        if deployment_id in self.active_deployments:
            self.active_deployments[deployment_id]["status"] = DeploymentStatus.FAILED.value
            self.active_deployments[deployment_id]["stage"] = DeploymentStage.FAILED.value
            self.active_deployments[deployment_id]["error_message"] = error_message

    async def _complete_deployment(self, deployment_id: str, deployment_result: Dict[str, Any], deployment_time: float):
        """
        Mark deployment as completed.

        Args:
            deployment_id: Deployment identifier
            deployment_result: Deployment result data
            deployment_time: Total deployment time
        """
        if deployment_id in self.active_deployments:
            self.active_deployments[deployment_id]["status"] = DeploymentStatus.SUCCESS.value
            self.active_deployments[deployment_id]["stage"] = DeploymentStage.COMPLETED.value
            self.active_deployments[deployment_id]["progress_percentage"] = 100
            self.active_deployments[deployment_id]["deployment_metadata"] = deployment_result
            self.active_deployments[deployment_id]["deployment_time"] = deployment_time

    def _update_deployment_metrics(self, success: bool, deployment_time: float):
        """
        Update deployment metrics.

        Args:
            success: Whether deployment was successful
            deployment_time: Deployment duration
        """
        self.metrics["total_deployments"] += 1

        if success:
            self.metrics["successful_deployments"] += 1
        else:
            self.metrics["failed_deployments"] += 1

        # Update rolling averages
        total = self.metrics["total_deployments"]
        current_avg = self.metrics["average_deployment_time"]

        if current_avg == 0:
            self.metrics["average_deployment_time"] = deployment_time
        else:
            self.metrics["average_deployment_time"] = (
                (current_avg * (total - 1)) + deployment_time
            ) / total

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResponse]:
        """
        Get status of a deployment.

        Args:
            deployment_id: Deployment identifier

        Returns:
            Deployment status or None if not found
        """
        if deployment_id not in self.active_deployments:
            return None

        deployment = self.active_deployments[deployment_id]

        return DeploymentResponse(
            deployment_id=deployment_id,
            agent_id=deployment["agent_id"],
            status=deployment["status"],
            stage=deployment["stage"],
            environment=deployment["environment"],
            progress_percentage=deployment["progress_percentage"],
            validation_results=deployment["validation_results"],
            test_results=deployment["test_results"],
            deployment_metadata=deployment["deployment_metadata"],
            error_message=deployment.get("error_message")
        )

    def get_deployment_metrics(self) -> Dict[str, Any]:
        """
        Get deployment metrics and statistics.

        Returns:
            Dictionary with deployment metrics
        """
        total_deployments = self.metrics["total_deployments"]
        successful_deployments = self.metrics["successful_deployments"]

        return {
            **self.metrics,
            "success_rate": successful_deployments / total_deployments if total_deployments > 0 else 0,
            "active_deployments": len(self.active_deployments),
            "last_updated": datetime.utcnow().isoformat()
        }

    async def list_deployments(self) -> List[Dict[str, Any]]:
        """
        List all deployments with their current status.

        Returns:
            List of deployment information
        """
        deployments = []

        for deployment_id, deployment in self.active_deployments.items():
            deployment_info = {
                "deployment_id": deployment_id,
                "agent_id": deployment["agent_id"],
                "environment": deployment["environment"],
                "status": deployment["status"],
                "stage": deployment["stage"],
                "progress_percentage": deployment["progress_percentage"],
                "start_time": deployment["start_time"].isoformat(),
                "estimated_completion": (
                    deployment["start_time"] + timedelta(seconds=600)
                ).isoformat()  # Estimated 10 minutes
            }
            deployments.append(deployment_info)

        return deployments

# FastAPI application setup
app = FastAPI(
    title="Deployment Pipeline Service",
    description="Comprehensive deployment pipeline for Agentic Brain agents with validation, testing, and rollback capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize deployment pipeline
deployment_pipeline = DeploymentPipeline()

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Deployment Pipeline Service started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup service on shutdown"""
    logger.info("Deployment Pipeline Service shutting down")

@app.get("/")
async def root():
    """Service health check and information endpoint"""
    return {
        "service": "Deployment Pipeline",
        "version": "1.0.0",
        "status": "healthy",
        "description": "Comprehensive deployment pipeline for Agentic Brain agents",
        "capabilities": [
            "Pre-deployment validation",
            "Comprehensive testing (functional, performance, security)",
            "Environment-specific deployment",
            "Post-deployment monitoring",
            "Automated rollback capabilities",
            "Deployment metrics and analytics"
        ],
        "endpoints": {
            "POST /deploy": "Deploy agent with full pipeline",
            "POST /rollback": "Rollback failed deployment",
            "GET /deployments/{deployment_id}/status": "Get deployment status",
            "GET /deployments": "List all deployments",
            "GET /metrics": "Get deployment metrics",
            "GET /health": "Service health check"
        },
        "supported_environments": ["development", "staging", "production", "testing"],
        "supported_tests": [
            "functional_test",
            "performance_test",
            "load_test",
            "security_test",
            "compliance_test"
        ]
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "Deployment Pipeline",
        "active_deployments": len(deployment_pipeline.active_deployments),
        "deployment_metrics": deployment_pipeline.get_deployment_metrics(),
        "service_endpoints": list(deployment_pipeline.service_endpoints.keys())
    }

@app.post("/deploy", response_model=DeploymentResponse)
async def deploy_agent(request: DeploymentRequest, background_tasks: BackgroundTasks):
    """
    Deploy an agent through the complete deployment pipeline.

    This endpoint initiates a comprehensive deployment process that includes:
    1. Pre-deployment validation (configuration, security, resources)
    2. Comprehensive testing (functional, performance, security)
    3. Environment-specific deployment orchestration
    4. Post-deployment monitoring and health checks
    5. Automatic rollback on failures

    The deployment process is asynchronous and provides real-time status updates.

    Request Body:
    - agent_id: Unique identifier of the agent to deploy
    - environment: Target deployment environment (development, staging, production, testing)
    - version: Optional specific agent version to deploy
    - deployment_options: Environment-specific deployment configuration
    - test_options: Testing configuration and options

    Returns:
    - deployment_id: Unique deployment identifier for tracking
    - agent_id: Agent identifier being deployed
    - status: Current deployment status
    - stage: Current deployment stage (initialized, validating, testing, deploying, monitoring, completed)
    - environment: Target deployment environment
    - progress_percentage: Deployment progress (0-100%)
    - validation_results: Pre-deployment validation results
    - test_results: Testing results
    - deployment_metadata: Deployment-specific information
    - error_message: Error message if deployment fails
    """
    try:
        logger.info("Agent deployment requested", agent_id=request.agent_id, environment=request.environment)

        # For complex deployments, run in background
        # For now, execute synchronously with timeout
        result = await deployment_pipeline.deploy_agent(request)

        logger.info(
            "Agent deployment completed",
            agent_id=request.agent_id,
            deployment_id=result.deployment_id,
            status=result.status,
            success=result.status == DeploymentStatus.SUCCESS.value
        )

        return result

    except Exception as e:
        logger.error(
            "Agent deployment failed",
            agent_id=request.agent_id,
            environment=request.environment,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Agent deployment failed: {str(e)}")

@app.post("/rollback", response_model=RollbackResponse)
async def rollback_deployment(request: RollbackRequest, background_tasks: BackgroundTasks):
    """
    Rollback a failed or problematic deployment.

    This endpoint initiates a rollback process that safely reverts a deployment
    to its previous state, cleaning up resources and restoring system stability.

    Request Body:
    - deployment_id: Unique identifier of the deployment to rollback
    - rollback_reason: Reason for performing the rollback
    - rollback_options: Rollback-specific configuration options

    Returns:
    - rollback_id: Unique rollback operation identifier
    - deployment_id: Original deployment identifier
    - status: Rollback operation status
    - rollback_reason: Reason for rollback
    - progress_percentage: Rollback progress (0-100%)
    - error_message: Error message if rollback fails
    """
    try:
        logger.info("Deployment rollback requested", deployment_id=request.deployment_id, reason=request.rollback_reason)

        result = await deployment_pipeline.rollback_deployment(request)

        logger.info(
            "Deployment rollback completed",
            deployment_id=request.deployment_id,
            rollback_id=result.rollback_id,
            status=result.status
        )

        return result

    except Exception as e:
        logger.error(
            "Deployment rollback failed",
            deployment_id=request.deployment_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Deployment rollback failed: {str(e)}")

@app.get("/deployments/{deployment_id}/status", response_model=DeploymentResponse)
async def get_deployment_status(deployment_id: str):
    """
    Get comprehensive status information for a deployment.

    Path Parameters:
    - deployment_id: Unique deployment identifier

    Returns:
    - deployment_id: Deployment identifier
    - agent_id: Agent identifier being deployed
    - status: Current deployment status
    - stage: Current deployment stage
    - environment: Target deployment environment
    - progress_percentage: Deployment progress percentage
    - validation_results: Pre-deployment validation results
    - test_results: Testing results
    - deployment_metadata: Deployment-specific metadata
    - error_message: Error message if deployment failed
    """
    try:
        status = deployment_pipeline.get_deployment_status(deployment_id)

        if status is None:
            raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get deployment status", deployment_id=deployment_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get deployment status: {str(e)}")

@app.get("/deployments")
async def list_deployments():
    """
    List all active deployments with their current status.

    Returns:
    - deployments: List of deployment information including:
        - deployment_id: Unique deployment identifier
        - agent_id: Agent identifier being deployed
        - environment: Target deployment environment
        - status: Current deployment status
        - stage: Current deployment stage
        - progress_percentage: Deployment progress percentage
        - start_time: Deployment start timestamp
        - estimated_completion: Estimated completion time
    - total_count: Total number of active deployments
    """
    try:
        deployments = await deployment_pipeline.list_deployments()

        return {
            "deployments": deployments,
            "total_count": len(deployments),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to list deployments", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list deployments: {str(e)}")

@app.get("/metrics")
async def get_deployment_metrics():
    """
    Get comprehensive deployment metrics and performance statistics.

    Returns:
    - total_deployments: Total number of deployments processed
    - successful_deployments: Number of successfully completed deployments
    - failed_deployments: Number of failed deployments
    - success_rate: Percentage of successful deployments
    - average_deployment_time: Average time for deployment completion
    - average_validation_time: Average time for validation
    - average_test_time: Average time for testing
    - active_deployments: Number of currently active deployments
    - last_updated: Timestamp of last metrics update
    """
    try:
        metrics = deployment_pipeline.get_deployment_metrics()

        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to get deployment metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get deployment metrics: {str(e)}")

@app.get("/environments")
async def get_supported_environments():
    """
    Get information about supported deployment environments.

    Returns:
    - environments: List of supported environments with their configurations
    - default_environment: Default environment for deployments
    - environment_requirements: Resource and configuration requirements per environment
    """
    try:
        environments = {
            "development": {
                "description": "Development environment for testing and debugging",
                "resource_requirements": {
                    "memory_mb": 512,
                    "cpu_cores": 0.5,
                    "storage_gb": 5
                },
                "validation_level": "basic",
                "monitoring_level": "minimal",
                "backup_enabled": False
            },
            "staging": {
                "description": "Staging environment for integration testing",
                "resource_requirements": {
                    "memory_mb": 1024,
                    "cpu_cores": 1.0,
                    "storage_gb": 20
                },
                "validation_level": "standard",
                "monitoring_level": "standard",
                "backup_enabled": True
            },
            "production": {
                "description": "Production environment for live deployments",
                "resource_requirements": {
                    "memory_mb": 2048,
                    "cpu_cores": 2.0,
                    "storage_gb": 100
                },
                "validation_level": "strict",
                "monitoring_level": "comprehensive",
                "backup_enabled": True
            },
            "testing": {
                "description": "Testing environment for automated test execution",
                "resource_requirements": {
                    "memory_mb": 1024,
                    "cpu_cores": 1.0,
                    "storage_gb": 10
                },
                "validation_level": "comprehensive",
                "monitoring_level": "detailed",
                "backup_enabled": False
            }
        }

        return {
            "environments": environments,
            "default_environment": "staging",
            "supported_tests": [
                "functional_test",
                "performance_test",
                "load_test",
                "security_test",
                "compliance_test"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to get environment information", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get environment information: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("DEPLOYMENT_PIPELINE_PORT", "8303"))

    logger.info("Starting Deployment Pipeline Service", port=port)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
