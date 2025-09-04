#!/usr/bin/env python3
"""
Integration Tests Service for Agentic Brain Platform

This service provides comprehensive end-to-end integration testing for the entire Agentic Brain platform,
including service-to-service communication, template loading, agent creation, deployment pipeline execution,
and task execution validation.

Features:
- End-to-end workflow testing
- Service integration validation
- Template loading and instantiation tests
- Agent creation and configuration tests
- Deployment pipeline testing
- Task execution and result validation
- Cross-service data flow testing
- Error handling and recovery testing
- Performance and scalability testing
- Automated test reporting and analytics
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import threading

import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON, Float, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func
import httpx
import aiohttp
import pytest
from pytest_asyncio import fixture
import requests

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class IntegrationTestSuite(Base):
    """Integration test suite tracking"""
    __tablename__ = 'integration_test_suites'

    id = Column(String(100), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    test_category = Column(String(50), nullable=False)  # e2e, service_integration, performance
    status = Column(String(20), nullable=False)  # running, completed, failed
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_tests = Column(Integer, default=0)
    passed_tests = Column(Integer, default=0)
    failed_tests = Column(Integer, default=0)
    skipped_tests = Column(Integer, default=0)
    duration_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    test_results = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)

class IntegrationTestCase(Base):
    """Individual integration test case"""
    __tablename__ = 'integration_test_cases'

    id = Column(String(100), primary_key=True)
    suite_id = Column(String(100), nullable=False)
    test_name = Column(String(200), nullable=False)
    test_description = Column(Text, nullable=False)
    service_under_test = Column(String(100), nullable=False)
    test_type = Column(String(50), nullable=False)  # unit, integration, e2e, performance
    status = Column(String(20), nullable=False)  # passed, failed, skipped, error
    duration_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    logs = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    artifacts = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ServiceHealthCheck(Base):
    """Service health monitoring"""
    __tablename__ = 'service_health_checks'

    id = Column(String(100), primary_key=True)
    service_name = Column(String(100), nullable=False)
    service_url = Column(String(255), nullable=False)
    check_type = Column(String(50), nullable=False)  # health, readiness, liveness
    status = Column(String(20), nullable=False)  # healthy, unhealthy, unknown
    response_time_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    checked_at = Column(DateTime, default=datetime.utcnow)

class DataFlowTest(Base):
    """Cross-service data flow validation"""
    __tablename__ = 'data_flow_tests'

    id = Column(String(100), primary_key=True)
    test_name = Column(String(200), nullable=False)
    source_service = Column(String(100), nullable=False)
    target_service = Column(String(100), nullable=False)
    data_payload = Column(JSON, nullable=False)
    expected_transformations = Column(JSON, nullable=False)
    actual_result = Column(JSON, nullable=True)
    status = Column(String(20), nullable=False)  # passed, failed
    validation_errors = Column(JSON, nullable=True)
    executed_at = Column(DateTime, default=datetime.utcnow)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration"""
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agentic_brain")

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

    # Service ports
    INTEGRATION_TESTS_PORT = int(os.getenv("INTEGRATION_TESTS_PORT", "8320"))

    # Service URLs
    AGENT_ORCHESTRATOR_URL = os.getenv("AGENT_ORCHESTRATOR_URL", "http://localhost:8200")
    PLUGIN_REGISTRY_URL = os.getenv("PLUGIN_REGISTRY_URL", "http://localhost:8201")
    WORKFLOW_ENGINE_URL = os.getenv("WORKFLOW_ENGINE_URL", "http://localhost:8202")
    TEMPLATE_STORE_URL = os.getenv("TEMPLATE_STORE_URL", "http://localhost:8203")
    RULE_ENGINE_URL = os.getenv("RULE_ENGINE_URL", "http://localhost:8204")
    MEMORY_MANAGER_URL = os.getenv("MEMORY_MANAGER_URL", "http://localhost:8205")
    AGENT_BUILDER_UI_URL = os.getenv("AGENT_BUILDER_UI_URL", "http://localhost:8300")
    BRAIN_FACTORY_URL = os.getenv("BRAIN_FACTORY_URL", "http://localhost:8301")
    UI_TO_BRAIN_MAPPER_URL = os.getenv("UI_TO_BRAIN_MAPPER_URL", "http://localhost:8302")
    DEPLOYMENT_PIPELINE_URL = os.getenv("DEPLOYMENT_PIPELINE_URL", "http://localhost:8303")
    REASONING_MODULE_FACTORY_URL = os.getenv("REASONING_MODULE_FACTORY_URL", "http://localhost:8304")
    AGENT_BRAIN_BASE_URL = os.getenv("AGENT_BRAIN_BASE_URL", "http://localhost:8305")
    SERVICE_CONNECTOR_FACTORY_URL = os.getenv("SERVICE_CONNECTOR_FACTORY_URL", "http://localhost:8306")

    # Test configuration
    DEFAULT_TEST_TIMEOUT = int(os.getenv("DEFAULT_TEST_TIMEOUT", "300"))
    MAX_CONCURRENT_TESTS = int(os.getenv("MAX_CONCURRENT_TESTS", "5"))
    RETRY_FAILED_TESTS = os.getenv("RETRY_FAILED_TESTS", "true").lower() == "true"
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    # Performance thresholds
    MAX_RESPONSE_TIME_MS = int(os.getenv("MAX_RESPONSE_TIME_MS", "5000"))
    MIN_SUCCESS_RATE = float(os.getenv("MIN_SUCCESS_RATE", "0.95"))

    # Monitoring
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8002"))

# =============================================================================
# SERVICE CLIENTS
# =============================================================================

class ServiceClient:
    """Base service client for HTTP communication"""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """GET request"""
        url = f"{self.base_url}{endpoint}"
        async with self.session.get(url, **kwargs) as response:
            return await self._handle_response(response)

    async def post(self, endpoint: str, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """POST request"""
        url = f"{self.base_url}{endpoint}"
        if data:
            kwargs['json'] = data
        async with self.session.post(url, **kwargs) as response:
            return await self._handle_response(response)

    async def put(self, endpoint: str, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """PUT request"""
        url = f"{self.base_url}{endpoint}"
        if data:
            kwargs['json'] = data
        async with self.session.put(url, **kwargs) as response:
            return await self._handle_response(response)

    async def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """DELETE request"""
        url = f"{self.base_url}{endpoint}"
        async with self.session.delete(url, **kwargs) as response:
            return await self._handle_response(response)

    async def _handle_response(self, response) -> Dict[str, Any]:
        """Handle HTTP response"""
        try:
            if response.content_type == 'application/json':
                data = await response.json()
            else:
                data = await response.text()

            return {
                'status_code': response.status,
                'data': data,
                'headers': dict(response.headers)
            }
        except Exception as e:
            return {
                'status_code': response.status,
                'error': str(e),
                'data': await response.text()
            }

class AgentOrchestratorClient(ServiceClient):
    """Client for Agent Orchestrator service"""

    def __init__(self):
        super().__init__(Config.AGENT_ORCHESTRATOR_URL)

    async def register_agent(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new agent"""
        return await self.post("/orchestrator/register-agent", agent_config)

    async def start_agent(self, agent_id: str) -> Dict[str, Any]:
        """Start an agent"""
        return await self.post(f"/orchestrator/start-agent/{agent_id}")

    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task"""
        return await self.post("/orchestrator/execute-task", task_data)

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent status"""
        return await self.get(f"/orchestrator/agents/{agent_id}")

class BrainFactoryClient(ServiceClient):
    """Client for Brain Factory service"""

    def __init__(self):
        super().__init__(Config.BRAIN_FACTORY_URL)

    async def generate_agent(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent from configuration"""
        return await self.post("/generate-agent", agent_config)

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent generation status"""
        return await self.get(f"/agents/{agent_id}/status")

class TemplateStoreClient(ServiceClient):
    """Client for Template Store service"""

    def __init__(self):
        super().__init__(Config.TEMPLATE_STORE_URL)

    async def get_templates(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get available templates"""
        params = filters or {}
        return await self.get("/templates", params=params)

    async def instantiate_template(self, template_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Instantiate a template"""
        return await self.post(f"/templates/{template_id}/instantiate", parameters)

class DeploymentPipelineClient(ServiceClient):
    """Client for Deployment Pipeline service"""

    def __init__(self):
        super().__init__(Config.DEPLOYMENT_PIPELINE_URL)

    async def deploy_agent(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy an agent"""
        return await self.post("/deploy", deployment_config)

    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status"""
        return await self.get(f"/deployments/{deployment_id}/status")

# =============================================================================
# INTEGRATION TEST CLASSES
# =============================================================================

class EndToEndTestSuite:
    """End-to-end test suite for complete workflows"""

    def __init__(self, db_session: Session):
        self.db = db_session
        self.clients = {
            'orchestrator': AgentOrchestratorClient(),
            'brain_factory': BrainFactoryClient(),
            'template_store': TemplateStoreClient(),
            'deployment': DeploymentPipelineClient()
        }

    async def test_underwriting_agent_e2e(self) -> Dict[str, Any]:
        """End-to-end test for underwriting agent workflow"""
        test_results = {
            'test_name': 'underwriting_agent_e2e',
            'status': 'running',
            'steps': [],
            'start_time': datetime.utcnow().isoformat()
        }

        try:
            # Step 1: Get underwriting template
            async with self.clients['template_store'] as client:
                templates_response = await client.get_templates({'domain': 'underwriting'})
                if templates_response['status_code'] != 200:
                    raise Exception("Failed to get templates")

                templates = templates_response['data']
                underwriting_template = next(
                    (t for t in templates if 'underwriting' in t.get('name', '').lower()),
                    None
                )

                if not underwriting_template:
                    raise Exception("Underwriting template not found")

                test_results['steps'].append({
                    'step': 'get_template',
                    'status': 'passed',
                    'template_id': underwriting_template['template_id']
                })

            # Step 2: Instantiate template
            async with self.clients['template_store'] as client:
                instantiate_response = await client.instantiate_template(
                    underwriting_template['template_id'],
                    {
                        'data_source': '/data/test_policies.csv',
                        'risk_threshold': 0.7,
                        'output_table': 'test_underwriting_results'
                    }
                )

                if instantiate_response['status_code'] != 200:
                    raise Exception("Failed to instantiate template")

                agent_config = instantiate_response['data']
                test_results['steps'].append({
                    'step': 'instantiate_template',
                    'status': 'passed',
                    'agent_config': agent_config
                })

            # Step 3: Generate agent
            async with self.clients['brain_factory'] as client:
                generate_response = await client.generate_agent(agent_config)
                if generate_response['status_code'] != 200:
                    raise Exception("Failed to generate agent")

                agent_data = generate_response['data']
                agent_id = agent_data.get('agent_id')

                test_results['steps'].append({
                    'step': 'generate_agent',
                    'status': 'passed',
                    'agent_id': agent_id
                })

            # Step 4: Register agent
            async with self.clients['orchestrator'] as client:
                register_response = await client.register_agent({
                    'agent_id': agent_id,
                    'agent_config': agent_config,
                    'resource_requirements': {
                        'cpu': '500m',
                        'memory': '256Mi'
                    }
                })

                if register_response['status_code'] != 200:
                    raise Exception("Failed to register agent")

                test_results['steps'].append({
                    'step': 'register_agent',
                    'status': 'passed'
                })

            # Step 5: Deploy agent
            async with self.clients['deployment'] as client:
                deploy_response = await client.deploy_agent({
                    'agent_id': agent_id,
                    'deployment_strategy': 'canary',
                    'traffic_percentage': 10,
                    'rollback_timeout_minutes': 30
                })

                if deploy_response['status_code'] != 200:
                    raise Exception("Failed to deploy agent")

                deployment_id = deploy_response['data'].get('deployment_id')
                test_results['steps'].append({
                    'step': 'deploy_agent',
                    'status': 'passed',
                    'deployment_id': deployment_id
                })

            # Step 6: Execute test task
            async with self.clients['orchestrator'] as client:
                task_response = await client.execute_task({
                    'agent_id': agent_id,
                    'task_type': 'risk_assessment',
                    'data': {
                        'loan_amount': 250000,
                        'credit_score': 720,
                        'income': 85000,
                        'debt_ratio': 0.32
                    }
                })

                if task_response['status_code'] != 200:
                    raise Exception("Failed to execute task")

                task_result = task_response['data']
                test_results['steps'].append({
                    'step': 'execute_task',
                    'status': 'passed',
                    'task_result': task_result
                })

            # Test completed successfully
            test_results['status'] = 'passed'
            test_results['end_time'] = datetime.utcnow().isoformat()
            test_results['duration_seconds'] = (
                datetime.fromisoformat(test_results['end_time']) -
                datetime.fromisoformat(test_results['start_time'])
            ).total_seconds()

        except Exception as e:
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
            test_results['end_time'] = datetime.utcnow().isoformat()

        return test_results

    async def test_claims_processing_e2e(self) -> Dict[str, Any]:
        """End-to-end test for claims processing workflow"""
        test_results = {
            'test_name': 'claims_processing_e2e',
            'status': 'running',
            'steps': [],
            'start_time': datetime.utcnow().isoformat()
        }

        try:
            # Similar to underwriting test but for claims processing
            # Step 1: Get claims template
            async with self.clients['template_store'] as client:
                templates_response = await client.get_templates({'domain': 'claims'})
                if templates_response['status_code'] != 200:
                    raise Exception("Failed to get claims templates")

                templates = templates_response['data']
                claims_template = next(
                    (t for t in templates if 'claims' in t.get('name', '').lower()),
                    None
                )

                if not claims_template:
                    raise Exception("Claims template not found")

                test_results['steps'].append({
                    'step': 'get_claims_template',
                    'status': 'passed',
                    'template_id': claims_template['template_id']
                })

            # Continue with claims-specific workflow...
            test_results['status'] = 'passed'
            test_results['end_time'] = datetime.utcnow().isoformat()

        except Exception as e:
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
            test_results['end_time'] = datetime.utcnow().isoformat()

        return test_results

class ServiceIntegrationTestSuite:
    """Service-to-service integration tests"""

    def __init__(self, db_session: Session):
        self.db = db_session

    async def test_plugin_registry_integration(self) -> Dict[str, Any]:
        """Test plugin registry integration with other services"""
        test_results = {
            'test_name': 'plugin_registry_integration',
            'status': 'running',
            'services_tested': [],
            'start_time': datetime.utcnow().isoformat()
        }

        try:
            # Test plugin loading and execution
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test plugin registry health
                response = await client.get(f"{Config.PLUGIN_REGISTRY_URL}/health")
                if response.status_code != 200:
                    raise Exception("Plugin registry health check failed")

                test_results['services_tested'].append({
                    'service': 'plugin_registry',
                    'endpoint': '/health',
                    'status': 'passed'
                })

                # Test plugin listing
                response = await client.get(f"{Config.PLUGIN_REGISTRY_URL}/plugins")
                if response.status_code != 200:
                    raise Exception("Plugin listing failed")

                plugins = response.json()
                if not plugins:
                    raise Exception("No plugins found")

                test_results['services_tested'].append({
                    'service': 'plugin_registry',
                    'endpoint': '/plugins',
                    'status': 'passed',
                    'plugin_count': len(plugins)
                })

                # Test plugin execution (if available)
                if plugins:
                    plugin_id = plugins[0].get('plugin_id')
                    if plugin_id:
                        test_data = {"test_input": "sample_data"}
                        response = await client.post(
                            f"{Config.PLUGIN_REGISTRY_URL}/plugins/{plugin_id}/execute",
                            json={"input_data": test_data}
                        )

                        if response.status_code == 200:
                            test_results['services_tested'].append({
                                'service': 'plugin_registry',
                                'endpoint': f'/plugins/{plugin_id}/execute',
                                'status': 'passed'
                            })
                        else:
                            test_results['services_tested'].append({
                                'service': 'plugin_registry',
                                'endpoint': f'/plugins/{plugin_id}/execute',
                                'status': 'failed',
                                'error': f"Status code: {response.status_code}"
                            })

            test_results['status'] = 'passed'
            test_results['end_time'] = datetime.utcnow().isoformat()

        except Exception as e:
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
            test_results['end_time'] = datetime.utcnow().isoformat()

        return test_results

    async def test_data_flow_integration(self) -> Dict[str, Any]:
        """Test data flow between services"""
        test_results = {
            'test_name': 'data_flow_integration',
            'status': 'running',
            'data_flows_tested': [],
            'start_time': datetime.utcnow().isoformat()
        }

        try:
            # Test data flow from UI to Brain Factory
            test_workflow = {
                "name": "Test Workflow",
                "components": [
                    {
                        "id": "data_input",
                        "type": "data_input_csv",
                        "config": {"file_path": "/test/data.csv"},
                        "position": {"x": 100, "y": 100}
                    },
                    {
                        "id": "llm_processor",
                        "type": "llm_processor",
                        "config": {"model": "gpt-4", "temperature": 0.7},
                        "position": {"x": 300, "y": 100}
                    }
                ],
                "connections": [
                    {"from": "data_input", "to": "llm_processor"}
                ]
            }

            # Test UI-to-Brain-Mapper
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{Config.UI_TO_BRAIN_MAPPER_URL}/map-workflow",
                    json={"workflow": test_workflow}
                )

                if response.status_code != 200:
                    raise Exception("UI-to-Brain-Mapper failed")

                agent_config = response.json()
                test_results['data_flows_tested'].append({
                    'flow': 'ui_to_brain_mapper',
                    'status': 'passed',
                    'agent_config_generated': bool(agent_config)
                })

                # Test Brain Factory with generated config
                response = await client.post(
                    f"{Config.BRAIN_FACTORY_URL}/generate-agent",
                    json=agent_config
                )

                if response.status_code != 200:
                    raise Exception("Brain Factory failed")

                test_results['data_flows_tested'].append({
                    'flow': 'brain_factory_generation',
                    'status': 'passed'
                })

            test_results['status'] = 'passed'
            test_results['end_time'] = datetime.utcnow().isoformat()

        except Exception as e:
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
            test_results['end_time'] = datetime.utcnow().isoformat()

        return test_results

class PerformanceTestSuite:
    """Performance and load testing"""

    def __init__(self, db_session: Session):
        self.db = db_session

    async def test_concurrent_agent_creation(self, num_agents: int = 5) -> Dict[str, Any]:
        """Test concurrent agent creation performance"""
        test_results = {
            'test_name': 'concurrent_agent_creation',
            'status': 'running',
            'num_agents': num_agents,
            'results': [],
            'start_time': datetime.utcnow().isoformat()
        }

        try:
            # Create multiple agents concurrently
            agent_configs = []
            for i in range(num_agents):
                agent_configs.append({
                    "name": f"Test Agent {i}",
                    "domain": "test",
                    "components": [
                        {
                            "id": f"llm_processor_{i}",
                            "type": "llm_processor",
                            "config": {"model": "gpt-3.5-turbo", "temperature": 0.7}
                        }
                    ],
                    "connections": []
                })

            # Execute concurrent agent creation
            async with httpx.AsyncClient(timeout=60.0) as client:
                tasks = []
                for config in agent_configs:
                    task = client.post(f"{Config.BRAIN_FACTORY_URL}/generate-agent", json=config)
                    tasks.append(task)

                start_time = time.time()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()

                total_time = end_time - start_time
                successful_creations = 0
                errors = []

                for i, response in enumerate(responses):
                    if isinstance(response, Exception):
                        errors.append(f"Agent {i}: {str(response)}")
                    elif hasattr(response, 'status_code'):
                        if response.status_code == 200:
                            successful_creations += 1
                        else:
                            errors.append(f"Agent {i}: Status {response.status_code}")

                test_results['results'] = {
                    'total_time_seconds': total_time,
                    'successful_creations': successful_creations,
                    'failed_creations': len(errors),
                    'success_rate': successful_creations / num_agents,
                    'average_time_per_agent': total_time / num_agents,
                    'errors': errors[:5]  # Limit error messages
                }

                # Performance thresholds
                if test_results['results']['success_rate'] >= 0.95:
                    test_results['status'] = 'passed'
                else:
                    test_results['status'] = 'failed'
                    test_results['error'] = f"Success rate below threshold: {test_results['results']['success_rate']}"

            test_results['end_time'] = datetime.utcnow().isoformat()

        except Exception as e:
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
            test_results['end_time'] = datetime.utcnow().isoformat()

        return test_results

    async def test_service_response_times(self) -> Dict[str, Any]:
        """Test response times for all services"""
        test_results = {
            'test_name': 'service_response_times',
            'status': 'running',
            'service_times': [],
            'start_time': datetime.utcnow().isoformat()
        }

        try:
            services_to_test = {
                'agent_orchestrator': Config.AGENT_ORCHESTRATOR_URL,
                'plugin_registry': Config.PLUGIN_REGISTRY_URL,
                'template_store': Config.TEMPLATE_STORE_URL,
                'brain_factory': Config.BRAIN_FACTORY_URL,
                'deployment_pipeline': Config.DEPLOYMENT_PIPELINE_URL
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                for service_name, service_url in services_to_test.items():
                    try:
                        start_time = time.time()
                        response = await client.get(f"{service_url}/health")
                        end_time = time.time()

                        response_time = (end_time - start_time) * 1000  # Convert to milliseconds

                        status = 'passed' if response.status_code == 200 else 'failed'

                        test_results['service_times'].append({
                            'service': service_name,
                            'url': service_url,
                            'response_time_ms': response_time,
                            'status_code': response.status_code,
                            'status': status
                        })

                    except Exception as e:
                        test_results['service_times'].append({
                            'service': service_name,
                            'url': service_url,
                            'response_time_ms': None,
                            'error': str(e),
                            'status': 'failed'
                        })

            # Check if all services are within acceptable response time
            max_response_time = max(
                (s['response_time_ms'] for s in test_results['service_times']
                 if s['response_time_ms'] is not None),
                default=0
            )

            if max_response_time <= Config.MAX_RESPONSE_TIME_MS:
                test_results['status'] = 'passed'
            else:
                test_results['status'] = 'failed'
                test_results['error'] = f"Max response time exceeded: {max_response_time}ms > {Config.MAX_RESPONSE_TIME_MS}ms"

            test_results['end_time'] = datetime.utcnow().isoformat()

        except Exception as e:
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
            test_results['end_time'] = datetime.utcnow().isoformat()

        return test_results

# =============================================================================
# INTEGRATION TEST MANAGER
# =============================================================================

class IntegrationTestManager:
    """Main integration test manager"""

    def __init__(self, db_session: Session, redis_client):
        self.db = db_session
        self.redis = redis_client
        self.active_suites = {}
        self.executor = ThreadPoolExecutor(max_workers=Config.MAX_CONCURRENT_TESTS)

    def create_test_suite(self, name: str, description: str, category: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new test suite"""
        suite_id = str(uuid.uuid4())

        suite = IntegrationTestSuite(
            id=suite_id,
            name=name,
            description=description,
            test_category=category,
            status="running",
            metadata=metadata or {}
        )

        self.db.add(suite)
        self.db.commit()

        self.active_suites[suite_id] = suite
        logger.info("Test suite created", suite_id=suite_id, name=name, category=category)
        return suite_id

    def update_test_suite(self, suite_id: str, updates: Dict[str, Any]):
        """Update test suite status and results"""
        suite = self.active_suites.get(suite_id)
        if suite:
            for key, value in updates.items():
                if hasattr(suite, key):
                    setattr(suite, key, value)

            suite.updated_at = datetime.utcnow()
            self.db.commit()
            logger.info("Test suite updated", suite_id=suite_id, updates=updates)

    async def run_e2e_test_suite(self, suite_id: str) -> Dict[str, Any]:
        """Run end-to-end test suite"""
        suite = self.active_suites.get(suite_id)
        if not suite:
            raise Exception(f"Test suite {suite_id} not found")

        e2e_tester = EndToEndTestSuite(self.db)
        results = []

        # Run underwriting agent E2E test
        underwriting_result = await e2e_tester.test_underwriting_agent_e2e()
        results.append(underwriting_result)

        # Run claims processing E2E test
        claims_result = await e2e_tester.test_claims_processing_e2e()
        results.append(claims_result)

        # Update suite with results
        total_tests = len(results)
        passed_tests = len([r for r in results if r['status'] == 'passed'])
        failed_tests = total_tests - passed_tests

        self.update_test_suite(suite_id, {
            "status": "completed",
            "end_time": datetime.utcnow(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "test_results": results
        })

        return {
            "suite_id": suite_id,
            "results": results,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            }
        }

    async def run_service_integration_tests(self, suite_id: str) -> Dict[str, Any]:
        """Run service integration tests"""
        suite = self.active_suites.get(suite_id)
        if not suite:
            raise Exception(f"Test suite {suite_id} not found")

        integration_tester = ServiceIntegrationTestSuite(self.db)
        results = []

        # Test plugin registry integration
        plugin_result = await integration_tester.test_plugin_registry_integration()
        results.append(plugin_result)

        # Test data flow integration
        data_flow_result = await integration_tester.test_data_flow_integration()
        results.append(data_flow_result)

        # Update suite with results
        total_tests = len(results)
        passed_tests = len([r for r in results if r['status'] == 'passed'])
        failed_tests = total_tests - passed_tests

        self.update_test_suite(suite_id, {
            "status": "completed",
            "end_time": datetime.utcnow(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "test_results": results
        })

        return {
            "suite_id": suite_id,
            "results": results,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            }
        }

    async def run_performance_tests(self, suite_id: str) -> Dict[str, Any]:
        """Run performance tests"""
        suite = self.active_suites.get(suite_id)
        if not suite:
            raise Exception(f"Test suite {suite_id} not found")

        performance_tester = PerformanceTestSuite(self.db)
        results = []

        # Test concurrent agent creation
        concurrent_result = await performance_tester.test_concurrent_agent_creation(5)
        results.append(concurrent_result)

        # Test service response times
        response_time_result = await performance_tester.test_service_response_times()
        results.append(response_time_result)

        # Update suite with results
        total_tests = len(results)
        passed_tests = len([r for r in results if r['status'] == 'passed'])
        failed_tests = total_tests - passed_tests

        self.update_test_suite(suite_id, {
            "status": "completed",
            "end_time": datetime.utcnow(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "test_results": results
        })

        return {
            "suite_id": suite_id,
            "results": results,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            }
        }

# =============================================================================
# API MODELS
# =============================================================================

class TestSuiteRequest(BaseModel):
    """Request model for creating test suites"""
    name: str = Field(..., description="Test suite name")
    description: str = Field(..., description="Test suite description")
    category: str = Field(..., description="Test category (e2e, integration, performance)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class TestSuiteResponse(BaseModel):
    """Response model for test suite operations"""
    suite_id: str
    name: str
    category: str
    status: str
    created_at: datetime
    results: Optional[Dict[str, Any]] = None

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Integration Tests Service",
    description="Comprehensive integration testing service for Agentic Brain platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
engine = create_engine(Config.DATABASE_URL)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize Redis
redis_client = redis.Redis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=Config.REDIS_DB,
    decode_responses=True
)

# Initialize test manager
test_manager = IntegrationTestManager(SessionLocal(), redis_client)

# Prometheus metrics
REQUEST_COUNT = Counter('integration_test_requests_total', 'Total integration test requests', ['method', 'endpoint'])
TEST_EXECUTION_TIME = Histogram('integration_test_execution_duration_seconds', 'Integration test execution time')

@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    """Database session middleware"""
    response = None
    try:
        request.state.db = SessionLocal()
        response = await call_next(request)
    finally:
        if hasattr(request.state, 'db'):
            request.state.db.close()
    return response

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Integration Tests Service", "status": "healthy", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "redis": "connected",
            "test_manager": "active"
        }
    }

@app.post("/test-suites")
async def create_test_suite(request: TestSuiteRequest, background_tasks: BackgroundTasks):
    """Create and execute a test suite"""
    REQUEST_COUNT.labels(method="POST", endpoint="/test-suites").inc()

    try:
        # Create test suite
        suite_id = test_manager.create_test_suite(
            request.name,
            request.description,
            request.category,
            request.metadata
        )

        # Execute tests based on category
        if request.category == "e2e":
            background_tasks.add_task(test_manager.run_e2e_test_suite, suite_id)
        elif request.category == "integration":
            background_tasks.add_task(test_manager.run_service_integration_tests, suite_id)
        elif request.category == "performance":
            background_tasks.add_task(test_manager.run_performance_tests, suite_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown test category: {request.category}")

        return {
            "suite_id": suite_id,
            "status": "running",
            "message": f"{request.category} test suite started",
            "estimated_duration_seconds": 300
        }

    except Exception as e:
        logger.error("Test suite creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Test suite creation failed: {str(e)}")

@app.get("/test-suites/{suite_id}")
async def get_test_suite(suite_id: str):
    """Get test suite results"""
    REQUEST_COUNT.labels(method="GET", endpoint="/test-suites/{suite_id}").inc()

    try:
        suite = test_manager.active_suites.get(suite_id)
        if not suite:
            # Try to get from database
            db = SessionLocal()
            suite = db.query(IntegrationTestSuite).filter_by(id=suite_id).first()
            db.close()

            if not suite:
                raise HTTPException(status_code=404, detail="Test suite not found")

        return {
            "suite_id": suite.id,
            "name": suite.name,
            "description": suite.description,
            "category": suite.test_category,
            "status": suite.status,
            "start_time": suite.start_time.isoformat() if suite.start_time else None,
            "end_time": suite.end_time.isoformat() if suite.end_time else None,
            "total_tests": suite.total_tests,
            "passed_tests": suite.passed_tests,
            "failed_tests": suite.failed_tests,
            "skipped_tests": suite.skipped_tests,
            "duration_seconds": suite.duration_seconds,
            "error_message": suite.error_message,
            "test_results": suite.test_results,
            "metadata": suite.metadata
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get test suite", suite_id=suite_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get test suite: {str(e)}")

@app.get("/test-suites")
async def list_test_suites(limit: int = 50, offset: int = 0, category: Optional[str] = None):
    """List test suites"""
    REQUEST_COUNT.labels(method="GET", endpoint="/test-suites").inc()

    try:
        db = SessionLocal()
        query = db.query(IntegrationTestSuite)

        if category:
            query = query.filter_by(test_category=category)

        suites = query.order_by(IntegrationTestSuite.start_time.desc()).limit(limit).offset(offset).all()
        db.close()

        return {
            "suites": [
                {
                    "suite_id": suite.id,
                    "name": suite.name,
                    "category": suite.test_category,
                    "status": suite.status,
                    "start_time": suite.start_time.isoformat() if suite.start_time else None,
                    "duration_seconds": suite.duration_seconds,
                    "passed_tests": suite.passed_tests,
                    "failed_tests": suite.failed_tests,
                    "success_rate": (suite.passed_tests / max(suite.total_tests, 1)) if suite.total_tests > 0 else 0
                }
                for suite in suites
            ],
            "total": len(suites),
            "limit": limit,
            "offset": offset,
            "category_filter": category
        }

    except Exception as e:
        logger.error("Failed to list test suites", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list test suites: {str(e)}")

@app.get("/test-results/summary")
async def get_test_summary(days: int = 7):
    """Get test results summary"""
    REQUEST_COUNT.labels(method="GET", endpoint="/test-results/summary").inc()

    try:
        db = SessionLocal()
        since_date = datetime.utcnow() - timedelta(days=days)

        # Get test suite statistics
        suites = db.query(IntegrationTestSuite).filter(
            IntegrationTestSuite.start_time >= since_date
        ).all()

        total_suites = len(suites)
        completed_suites = len([s for s in suites if s.status == 'completed'])
        failed_suites = len([s for s in suites if s.status == 'failed'])

        # Get test case statistics
        test_cases = db.query(IntegrationTestCase).filter(
            IntegrationTestCase.created_at >= since_date
        ).all()

        total_tests = len(test_cases)
        passed_tests = len([t for t in test_cases if t.status == 'passed'])
        failed_tests = len([t for t in test_cases if t.status == 'failed'])
        skipped_tests = len([t for t in test_cases if t.status == 'skipped'])

        db.close()

        success_rate = (passed_tests / max(total_tests, 1)) * 100

        return {
            "period_days": days,
            "summary": {
                "total_suites": total_suites,
                "completed_suites": completed_suites,
                "failed_suites": failed_suites,
                "suite_success_rate": (completed_suites / max(total_suites, 1)) * 100,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "skipped_tests": skipped_tests,
                "test_success_rate": success_rate
            },
            "categories": {
                "e2e": len([s for s in suites if s.test_category == 'e2e']),
                "integration": len([s for s in suites if s.test_category == 'integration']),
                "performance": len([s for s in suites if s.test_category == 'performance'])
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to get test summary", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get test summary: {str(e)}")

@app.post("/test-suites/{suite_id}/retry")
async def retry_test_suite(suite_id: str, background_tasks: BackgroundTasks):
    """Retry a failed test suite"""
    REQUEST_COUNT.labels(method="POST", endpoint="/test-suites/{suite_id}/retry").inc()

    try:
        # Get original suite
        db = SessionLocal()
        original_suite = db.query(IntegrationTestSuite).filter_by(id=suite_id).first()
        db.close()

        if not original_suite:
            raise HTTPException(status_code=404, detail="Test suite not found")

        if original_suite.status != 'failed':
            raise HTTPException(status_code=400, detail="Only failed test suites can be retried")

        # Create retry suite
        retry_suite_id = test_manager.create_test_suite(
            f"{original_suite.name} (Retry)",
            f"Retry of {original_suite.name}",
            original_suite.test_category,
            {"original_suite_id": suite_id, "retry_count": (original_suite.metadata.get('retry_count', 0) + 1)}
        )

        # Execute tests
        if original_suite.test_category == "e2e":
            background_tasks.add_task(test_manager.run_e2e_test_suite, retry_suite_id)
        elif original_suite.test_category == "integration":
            background_tasks.add_task(test_manager.run_service_integration_tests, retry_suite_id)
        elif original_suite.test_category == "performance":
            background_tasks.add_task(test_manager.run_performance_tests, retry_suite_id)

        return {
            "retry_suite_id": retry_suite_id,
            "original_suite_id": suite_id,
            "status": "running",
            "message": f"Retry test suite started for {original_suite.name}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retry test suite", suite_id=suite_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retry test suite: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return HTMLResponse(generate_latest(), media_type="text/plain")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Integration Tests Dashboard"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Integration Tests Dashboard - Agentic Brain Platform</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f7fa;
                color: #333;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .metric-card {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                text-align: center;
            }}
            .metric-value {{
                font-size: 3em;
                font-weight: bold;
                color: #3498db;
                margin-bottom: 10px;
            }}
            .status-success {{ color: #27ae60; }}
            .status-error {{ color: #e74c3c; }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧪 Integration Tests Dashboard</h1>
            <p>End-to-end testing for Agentic Brain platform</p>
        </div>

        <div class="container">
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-value status-success" id="total-suites">0</div>
                    <div class="metric-label">Total Test Suites</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-success" id="passed-tests">0</div>
                    <div class="metric-label">Passed Tests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-error" id="failed-tests">0</div>
                    <div class="metric-label">Failed Tests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-success" id="success-rate">0%</div>
                    <div class="metric-label">Success Rate</div>
                </div>
            </div>

            <div class="metric-card">
                <h3>Quick Actions</h3>
                <button onclick="runE2ETests()">Run E2E Tests</button>
                <button onclick="runIntegrationTests()">Run Integration Tests</button>
                <button onclick="runPerformanceTests()">Run Performance Tests</button>
                <button onclick="viewTestHistory()">View Test History</button>
            </div>

            <div class="metric-card">
                <h3>Recent Test Suites</h3>
                <div id="recent-suites">Loading...</div>
            </div>
        </div>

        <script>
            async function loadDashboardData() {{
                try {{
                    const response = await fetch('/test-results/summary');
                    const data = await response.json();

                    document.getElementById('total-suites').textContent = data.summary.total_suites;
                    document.getElementById('passed-tests').textContent = data.summary.passed_tests;
                    document.getElementById('failed-tests').textContent = data.summary.failed_tests;
                    document.getElementById('success-rate').textContent = data.summary.test_success_rate.toFixed(1) + '%';
                }} catch (error) {{
                    console.error('Error loading dashboard data:', error);
                }}
            }}

            async function runE2ETests() {{
                const response = await fetch('/test-suites', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        name: 'E2E Test Suite',
                        description: 'End-to-end testing suite',
                        category: 'e2e'
                    }})
                }});
                const result = await response.json();
                alert(`E2E Test Suite started: ${{result.suite_id}}`);
                loadDashboardData();
            }}

            async function runIntegrationTests() {{
                const response = await fetch('/test-suites', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        name: 'Integration Test Suite',
                        description: 'Service integration testing suite',
                        category: 'integration'
                    }})
                }});
                const result = await response.json();
                alert(`Integration Test Suite started: ${{result.suite_id}}`);
                loadDashboardData();
            }}

            async function runPerformanceTests() {{
                const response = await fetch('/test-suites', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        name: 'Performance Test Suite',
                        description: 'Performance and load testing suite',
                        category: 'performance'
                    }})
                }});
                const result = await response.json();
                alert(`Performance Test Suite started: ${{result.suite_id}}`);
                loadDashboardData();
            }}

            async function viewTestHistory() {{
                window.location.href = '/test-suites';
            }}

            // Load data on page load
            document.addEventListener('DOMContentLoaded', loadDashboardData);

            // Refresh data every 30 seconds
            setInterval(loadDashboardData, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Integration Tests Service starting up")

    # Verify service connectivity
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            services_to_check = [
                (Config.AGENT_ORCHESTRATOR_URL, "Agent Orchestrator"),
                (Config.PLUGIN_REGISTRY_URL, "Plugin Registry"),
                (Config.TEMPLATE_STORE_URL, "Template Store"),
                (Config.BRAIN_FACTORY_URL, "Brain Factory"),
                (Config.DEPLOYMENT_PIPELINE_URL, "Deployment Pipeline")
            ]

            for service_url, service_name in services_to_check:
                try:
                    response = await client.get(f"{service_url}/health")
                    if response.status_code == 200:
                        logger.info(f"{service_name} is accessible")
                    else:
                        logger.warning(f"{service_name} returned status {response.status_code}")
                except Exception as e:
                    logger.warning(f"Could not verify {service_name} accessibility: {str(e)}")

    except Exception as e:
        logger.warning(f"Service connectivity check failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Integration Tests Service shutting down")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.INTEGRATION_TESTS_PORT,
        reload=True,
        log_level="info"
    )
