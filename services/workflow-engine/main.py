#!/usr/bin/env python3
"""
Workflow Engine Service for Agentic Brain Platform

This service executes agent workflows composed of drag-and-drop components including:
- Data Input components (CSV, API, database sources)
- Processing components (LLM processors, rule engines)
- Decision components (conditional logic, branching)
- Output components (database writes, email notifications, file exports)
- Multi-agent coordination components

Features:
- Workflow execution orchestration
- Component dependency management
- Parallel execution support
- Error handling and recovery
- State persistence and monitoring
- Integration with plugin registry
- Real-time execution tracking
- RESTful API for workflow operations
- Comprehensive monitoring and metrics
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import networkx as nx

import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON, Float, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func
import httpx
import uvicorn

# JWT support for authentication
try:
    import jwt
except ImportError:
    jwt = None

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for Workflow Engine Service"""

    # Database Configuration
    DB_HOST = os.getenv('POSTGRES_HOST', 'postgresql_ingestion')
    DB_PORT = os.getenv('POSTGRES_PORT', '5432')
    DB_NAME = os.getenv('POSTGRES_DB', 'agentic_ingestion')
    DB_USER = os.getenv('POSTGRES_USER', 'agentic_user')
    DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'agentic123')
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Redis Configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis_ingestion')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB = 2  # Use DB 2 for workflow engine

    # Service Configuration
    SERVICE_HOST = os.getenv('WORKFLOW_ENGINE_HOST', '0.0.0.0')
    SERVICE_PORT = int(os.getenv('WORKFLOW_ENGINE_PORT', '8202'))

    # Authentication Configuration
    REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-jwt-key-change-in-production')
    JWT_ALGORITHM = 'HS256'

    # Plugin Registry Configuration
    PLUGIN_REGISTRY_HOST = os.getenv('PLUGIN_REGISTRY_HOST', 'plugin-registry')
    PLUGIN_REGISTRY_PORT = int(os.getenv('PLUGIN_REGISTRY_PORT', '8201'))

    # Rule Engine Configuration
    RULE_ENGINE_HOST = os.getenv('RULE_ENGINE_HOST', 'rule-engine')
    RULE_ENGINE_PORT = int(os.getenv('RULE_ENGINE_PORT', '8204'))

    # Memory Manager Configuration
    MEMORY_MANAGER_HOST = os.getenv('MEMORY_MANAGER_HOST', 'memory-manager')
    MEMORY_MANAGER_PORT = int(os.getenv('MEMORY_MANAGER_PORT', '8205'))

    # Workflow Configuration
    WORKFLOW_MAX_COMPONENTS = int(os.getenv('WORKFLOW_MAX_COMPONENTS', '50'))
    WORKFLOW_EXECUTION_TIMEOUT = int(os.getenv('WORKFLOW_EXECUTION_TIMEOUT', '1800'))
    WORKFLOW_PARALLEL_EXECUTION = os.getenv('WORKFLOW_PARALLEL_EXECUTION', 'true').lower() == 'true'
    WORKFLOW_ERROR_RECOVERY_ENABLED = os.getenv('WORKFLOW_ERROR_RECOVERY_ENABLED', 'true').lower() == 'true'

    # Monitoring Configuration
    METRICS_ENABLED = os.getenv('AGENT_METRICS_ENABLED', 'true').lower() == 'true'

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class WorkflowExecution(Base):
    """Database model for workflow execution tracking"""
    __tablename__ = 'workflow_executions'

    id = Column(Integer, primary_key=True)
    execution_id = Column(String(100), unique=True, nullable=False)
    agent_id = Column(String(100), nullable=False)
    workflow_id = Column(String(100), nullable=False)
    status = Column(String(50), default='running')  # running, completed, failed, paused
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    total_components = Column(Integer, default=0)
    completed_components = Column(Integer, default=0)
    failed_components = Column(Integer, default=0)
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)
    execution_plan = Column(JSON)  # Component execution order and dependencies
    metadata = Column(JSON, default=dict)

class ComponentExecution(Base):
    """Database model for individual component execution"""
    __tablename__ = 'component_executions'

    id = Column(Integer, primary_key=True)
    execution_id = Column(String(100), nullable=False)
    component_id = Column(String(100), nullable=False)
    component_type = Column(String(50), nullable=False)
    status = Column(String(50), default='pending')  # pending, running, completed, failed, skipped
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    dependencies = Column(JSON, default=list)  # List of component IDs this depends on
    metadata = Column(JSON, default=dict)

# =============================================================================
# WORKFLOW COMPONENT INTERFACES
# =============================================================================

class ComponentType(Enum):
    """Enumeration of supported component types"""
    DATA_INPUT = "data_input"
    LLM_PROCESSOR = "llm_processor"
    RULE_ENGINE = "rule_engine"
    DECISION_NODE = "decision_node"
    DATABASE_OUTPUT = "database_output"
    EMAIL_OUTPUT = "email_output"
    PDF_REPORT = "pdf_report"
    MULTI_AGENT_COORDINATOR = "multi_agent_coordinator"
    PLUGIN_EXECUTOR = "plugin_executor"

class WorkflowComponent:
    """Base class for workflow components"""

    def __init__(self, component_id: str, config: Dict[str, Any]):
        self.component_id = component_id
        self.config = config
        self.execution_context = {}

    @property
    def component_type(self) -> ComponentType:
        """Return the component type"""
        raise NotImplementedError

    def validate_config(self) -> bool:
        """Validate component configuration"""
        return True

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the component with input data"""
        raise NotImplementedError

    def get_dependencies(self) -> List[str]:
        """Get list of component IDs this component depends on"""
        return []

    def get_execution_estimate_ms(self) -> int:
        """Estimate execution time in milliseconds"""
        return 1000

class DataInputComponent(WorkflowComponent):
    """Data input component for various data sources"""

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.DATA_INPUT

    def validate_config(self) -> bool:
        """Validate data input configuration"""
        required_fields = ['source_type']
        return all(field in self.config for field in required_fields)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data input component"""
        source_type = self.config.get('source_type')
        connection_config = self.config.get('connection_config', {})

        try:
            if source_type == 'csv':
                # Mock CSV data retrieval
                result = {
                    'data': [
                        {'id': 1, 'field1': 'value1', 'field2': 'value2'},
                        {'id': 2, 'field1': 'value3', 'field2': 'value4'}
                    ],
                    'row_count': 2,
                    'columns': ['id', 'field1', 'field2'],
                    'source': 'csv_file'
                }
            elif source_type == 'api':
                # Mock API data retrieval
                result = {
                    'data': [{'user_id': 1, 'name': 'John Doe', 'email': 'john@example.com'}],
                    'row_count': 1,
                    'endpoint': connection_config.get('endpoint', '/api/data'),
                    'source': 'api_endpoint'
                }
            elif source_type == 'database':
                # Mock database query
                result = {
                    'data': [{'policy_id': 123, 'amount': 50000, 'status': 'active'}],
                    'row_count': 1,
                    'table': connection_config.get('table', 'policies'),
                    'source': 'database_table'
                }
            else:
                result = {
                    'data': [],
                    'row_count': 0,
                    'error': f'Unsupported source type: {source_type}',
                    'source': 'unknown'
                }

            return result

        except Exception as e:
            return {
                'data': [],
                'row_count': 0,
                'error': f'Data input failed: {str(e)}',
                'source': source_type
            }

class LLMProcessorComponent(WorkflowComponent):
    """LLM processing component"""

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.LLM_PROCESSOR

    def validate_config(self) -> bool:
        """Validate LLM processor configuration"""
        required_fields = ['model', 'prompt_template']
        return all(field in self.config for field in required_fields)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM processing"""
        try:
            model = self.config.get('model', 'gpt-4')
            prompt_template = self.config.get('prompt_template', '')
            temperature = self.config.get('temperature', 0.7)

            # Mock LLM processing
            await asyncio.sleep(0.5)  # Simulate API call delay

            # Format prompt with input data
            prompt = prompt_template
            if isinstance(input_data, dict) and 'data' in input_data:
                # Use first data item for context
                data_item = input_data['data'][0] if input_data['data'] else {}
                for key, value in data_item.items():
                    prompt = prompt.replace(f"{{{key}}}", str(value))

            # Mock LLM response
            result = {
                'processed_text': f"Analysis of input data using {model}",
                'confidence_score': 0.85,
                'tokens_used': len(prompt.split()) * 1.5,
                'model': model,
                'temperature': temperature,
                'processing_time_seconds': 0.5,
                'recommendation': 'PROCEED',
                'analysis': {
                    'risk_level': 'LOW',
                    'key_insights': ['Data appears valid', 'No anomalies detected'],
                    'next_steps': ['Continue processing', 'Generate report']
                }
            }

            return result

        except Exception as e:
            return {
                'error': f'LLM processing failed: {str(e)}',
                'processed_text': '',
                'confidence_score': 0.0,
                'tokens_used': 0,
                'model': self.config.get('model', 'unknown')
            }

class RuleEngineComponent(WorkflowComponent):
    """Rule engine component"""

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.RULE_ENGINE

    def validate_config(self) -> bool:
        """Validate rule engine configuration"""
        required_fields = ['rule_set']
        return all(field in self.config for field in required_fields)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rule engine processing"""
        try:
            rule_set = self.config.get('rule_set', 'default')

            # Mock rule engine processing
            await asyncio.sleep(0.2)  # Simulate rule evaluation

            # Apply mock rules
            if rule_set == 'fraud_rules':
                fraud_score = 0
                flags = []

                # Mock fraud detection rules
                data = input_data.get('data', [{}])[0] if input_data.get('data') else {}

                if data.get('amount', 0) > 10000:
                    fraud_score += 20
                    flags.append('High transaction amount')

                if data.get('frequency', 0) > 5:
                    fraud_score += 15
                    flags.append('High transaction frequency')

                result = {
                    'rule_set': rule_set,
                    'fraud_score': fraud_score,
                    'risk_level': 'HIGH' if fraud_score > 25 else 'LOW',
                    'flags': flags,
                    'recommendation': 'REVIEW' if fraud_score > 25 else 'APPROVE'
                }

            elif rule_set == 'compliance_rules':
                # Mock compliance rules
                result = {
                    'rule_set': rule_set,
                    'compliance_score': 95,
                    'violations': [],
                    'recommendation': 'COMPLIANT'
                }

            else:
                result = {
                    'rule_set': rule_set,
                    'evaluation_result': 'PASSED',
                    'matched_rules': ['rule1', 'rule2'],
                    'recommendation': 'CONTINUE'
                }

            return result

        except Exception as e:
            return {
                'error': f'Rule engine execution failed: {str(e)}',
                'rule_set': self.config.get('rule_set', 'unknown'),
                'evaluation_result': 'ERROR'
            }

class DecisionNodeComponent(WorkflowComponent):
    """Decision node component for conditional branching"""

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.DECISION_NODE

    def validate_config(self) -> bool:
        """Validate decision node configuration"""
        required_fields = ['condition_field', 'threshold']
        return all(field in self.config for field in required_fields)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute decision logic"""
        try:
            condition_field = self.config.get('condition_field')
            threshold = self.config.get('threshold', 0)
            operator = self.config.get('operator', 'greater_than')

            # Extract value from input data
            data = input_data.get('data', [{}])[0] if input_data.get('data') else {}
            value = data.get(condition_field, 0)

            # Evaluate condition
            if operator == 'greater_than':
                condition_met = value > threshold
            elif operator == 'less_than':
                condition_met = value < threshold
            elif operator == 'equals':
                condition_met = value == threshold
            elif operator == 'not_equals':
                condition_met = value != threshold
            else:
                condition_met = False

            result = {
                'condition_field': condition_field,
                'field_value': value,
                'threshold': threshold,
                'operator': operator,
                'condition_met': condition_met,
                'decision': 'TRUE_BRANCH' if condition_met else 'FALSE_BRANCH',
                'evaluation_time_ms': 10
            }

            return result

        except Exception as e:
            return {
                'error': f'Decision evaluation failed: {str(e)}',
                'condition_field': self.config.get('condition_field'),
                'condition_met': False,
                'decision': 'ERROR_BRANCH'
            }

class DatabaseOutputComponent(WorkflowComponent):
    """Database output component"""

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.DATABASE_OUTPUT

    def validate_config(self) -> bool:
        """Validate database output configuration"""
        required_fields = ['table_name', 'connection_config']
        return all(field in self.config for field in required_fields)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database output"""
        try:
            table_name = self.config.get('table_name')
            connection_config = self.config.get('connection_config', {})

            # Mock database write operation
            await asyncio.sleep(0.3)  # Simulate database write

            data = input_data.get('data', [])
            records_processed = len(data) if isinstance(data, list) else 1

            result = {
                'table_name': table_name,
                'records_processed': records_processed,
                'operation': 'INSERT',
                'success': True,
                'execution_time_ms': 300,
                'transaction_id': str(uuid.uuid4())
            }

            return result

        except Exception as e:
            return {
                'error': f'Database output failed: {str(e)}',
                'table_name': self.config.get('table_name'),
                'records_processed': 0,
                'success': False
            }

class PluginExecutorComponent(WorkflowComponent):
    """Plugin executor component"""

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.PLUGIN_EXECUTOR

    def validate_config(self) -> bool:
        """Validate plugin executor configuration"""
        required_fields = ['plugin_id']
        return all(field in self.config for field in required_fields)

    def get_dependencies(self) -> List[str]:
        """Plugin executor depends on plugin registry"""
        return ['plugin-registry']

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plugin via plugin registry"""
        try:
            plugin_id = self.config.get('plugin_id')

            # This would typically call the plugin registry service
            # For now, return mock result
            await asyncio.sleep(0.4)  # Simulate plugin execution

            result = {
                'plugin_id': plugin_id,
                'execution_result': 'SUCCESS',
                'output_data': input_data,
                'execution_time_ms': 400,
                'plugin_version': '1.0.0'
            }

            return result

        except Exception as e:
            return {
                'error': f'Plugin execution failed: {str(e)}',
                'plugin_id': self.config.get('plugin_id'),
                'execution_result': 'FAILED'
            }

# =============================================================================
# API MODELS
# =============================================================================

class ComponentConfig(BaseModel):
    """Model for component configuration"""
    component_id: str
    component_type: str
    config: Dict[str, Any]
    position: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class WorkflowDefinition(BaseModel):
    """Model for workflow definition"""
    workflow_id: str
    agent_id: str
    name: str
    description: Optional[str] = None
    components: List[ComponentConfig]
    connections: List[Dict[str, str]]
    metadata: Optional[Dict[str, Any]] = None

class WorkflowExecutionRequest(BaseModel):
    """Request model for workflow execution"""
    workflow_id: str
    agent_id: str
    input_data: Dict[str, Any]
    execution_config: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = 1800

class WorkflowExecutionStatus(BaseModel):
    """Response model for workflow execution status"""
    execution_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    total_components: int
    completed_components: int
    failed_components: int
    progress_percentage: float
    current_component: Optional[str]
    error_message: Optional[str]

# =============================================================================
# BUSINESS LOGIC CLASSES
# =============================================================================

class WorkflowExecutor:
    """Handles workflow execution orchestration"""

    def __init__(self, db_session: Session, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.component_factory = ComponentFactory()
        self.active_executions = {}  # Track active workflow executions

    def create_execution_plan(self, components: List[Dict], connections: List[Dict]) -> Dict[str, Any]:
        """Create execution plan with dependency graph"""
        try:
            # Build dependency graph
            graph = nx.DiGraph()

            # Add all components as nodes
            for component in components:
                graph.add_node(component['component_id'], data=component)

            # Add connections as edges
            for connection in connections:
                from_component = connection['from']
                to_component = connection['to']
                graph.add_edge(from_component, to_component)

            # Find execution order using topological sort
            try:
                execution_order = list(nx.topological_sort(graph))
            except nx.NetworkXError:
                # Handle cycles by using a different approach
                execution_order = list(graph.nodes())

            # Group components by dependency level
            execution_levels = {}
            for component_id in execution_order:
                predecessors = list(graph.predecessors(component_id))
                level = max([execution_levels.get(pred, 0) for pred in predecessors] + [0]) + 1
                execution_levels[component_id] = level

            # Group by levels for parallel execution
            levels = {}
            for component_id, level in execution_levels.items():
                if level not in levels:
                    levels[level] = []
                levels[level].append(component_id)

            return {
                'execution_order': execution_order,
                'execution_levels': levels,
                'total_components': len(components),
                'parallel_execution_possible': len(levels) > 1,
                'estimated_duration_ms': sum([self._estimate_component_duration(c) for c in components])
            }

        except Exception as e:
            logger.error(f"Failed to create execution plan: {str(e)}")
            return {
                'execution_order': [c['component_id'] for c in components],
                'execution_levels': {1: [c['component_id'] for c in components]},
                'total_components': len(components),
                'parallel_execution_possible': False,
                'error': str(e)
            }

    def _estimate_component_duration(self, component: Dict) -> int:
        """Estimate component execution duration"""
        component_type = component.get('component_type', '')

        # Base estimates in milliseconds
        estimates = {
            'data_input': 500,
            'llm_processor': 2000,
            'rule_engine': 300,
            'decision_node': 100,
            'database_output': 400,
            'plugin_executor': 800
        }

        return estimates.get(component_type, 1000)

    async def execute_workflow(self, request: WorkflowExecutionRequest) -> str:
        """Execute a workflow asynchronously"""
        execution_id = str(uuid.uuid4())

        try:
            # Get workflow definition from database
            workflow = self.db.query(AgentWorkflow).filter_by(
                workflow_id=request.workflow_id,
                agent_id=request.agent_id,
                is_active=True
            ).first()

            if not workflow:
                raise HTTPException(status_code=404, detail=f"Workflow {request.workflow_id} not found")

            # Create execution record
            execution = WorkflowExecution(
                execution_id=execution_id,
                agent_id=request.agent_id,
                workflow_id=request.workflow_id,
                status='running',
                total_components=len(workflow.components or []),
                input_data=request.input_data,
                execution_plan=self.create_execution_plan(workflow.components or [], workflow.connections or [])
            )
            self.db.add(execution)
            self.db.commit()

            # Start async execution
            asyncio.create_task(self._execute_workflow_async(execution_id, workflow, request.input_data))

            return execution_id

        except Exception as e:
            logger.error(f"Failed to start workflow execution: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Workflow execution failed to start: {str(e)}")

    async def _execute_workflow_async(self, execution_id: str, workflow: Any, input_data: Dict[str, Any]):
        """Execute workflow asynchronously"""
        try:
            start_time = time.time()

            # Update execution status
            execution = self.db.query(WorkflowExecution).filter_by(execution_id=execution_id).first()
            if execution:
                execution.started_at = datetime.utcnow()

            # Execute components
            results = {}
            component_results = {}

            components = workflow.components or []
            connections = workflow.connections or []

            # Build component lookup
            component_lookup = {c['component_id']: c for c in components}

            # Execute components in dependency order
            execution_plan = self.create_execution_plan(components, connections)

            for level, component_ids in execution_plan['execution_levels'].items():
                # Execute components in this level (potentially in parallel)
                tasks = []
                for component_id in component_ids:
                    if component_id in component_lookup:
                        component_config = component_lookup[component_id]
                        task = self._execute_component_async(
                            execution_id,
                            component_config,
                            input_data,
                            results
                        )
                        tasks.append(task)

                # Wait for all components in this level to complete
                if tasks:
                    level_results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process results
                    for i, result in enumerate(level_results):
                        component_id = component_ids[i]
                        if isinstance(result, Exception):
                            component_results[component_id] = {'error': str(result)}
                            logger.error(f"Component {component_id} failed: {str(result)}")
                        else:
                            component_results[component_id] = result
                            results[component_id] = result

            # Update execution record
            duration = time.time() - start_time
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = duration
            execution.status = 'completed'
            execution.output_data = component_results
            execution.completed_components = len([r for r in component_results.values() if 'error' not in r])

            self.db.commit()

            logger.info(f"Workflow {execution_id} completed in {duration:.2f} seconds")

        except Exception as e:
            # Update execution with error
            execution = self.db.query(WorkflowExecution).filter_by(execution_id=execution_id).first()
            if execution:
                execution.status = 'failed'
                execution.completed_at = datetime.utcnow()
                execution.error_message = str(e)
                self.db.commit()

            logger.error(f"Workflow {execution_id} failed: {str(e)}")

    async def _execute_component_async(self, execution_id: str, component_config: Dict,
                                     input_data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single component"""
        component_id = component_config['component_id']

        try:
            # Create component execution record
            component_exec = ComponentExecution(
                execution_id=execution_id,
                component_id=component_id,
                component_type=component_config.get('component_type', 'unknown'),
                status='running',
                started_at=datetime.utcnow(),
                input_data=input_data
            )
            self.db.add(component_exec)
            self.db.commit()

            # Create component instance
            component = self.component_factory.create_component(component_config)

            # Prepare input data (combine workflow input with previous component results)
            component_input = dict(input_data)  # Copy input data
            component_input.update(previous_results)  # Add previous results

            # Execute component
            start_time = time.time()
            result = await component.execute(component_input)
            duration_ms = int((time.time() - start_time) * 1000)

            # Update component execution record
            component_exec.status = 'completed'
            component_exec.completed_at = datetime.utcnow()
            component_exec.duration_ms = duration_ms
            component_exec.output_data = result
            self.db.commit()

            return result

        except Exception as e:
            # Update component execution with error
            component_exec.status = 'failed'
            component_exec.completed_at = datetime.utcnow()
            component_exec.error_message = str(e)
            self.db.commit()

            raise e

    def get_execution_status(self, execution_id: str) -> WorkflowExecutionStatus:
        """Get workflow execution status"""
        execution = self.db.query(WorkflowExecution).filter_by(execution_id=execution_id).first()
        if not execution:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

        # Calculate progress
        progress_percentage = 0.0
        if execution.total_components > 0:
            progress_percentage = (execution.completed_components / execution.total_components) * 100

        # Get current component
        current_component = None
        if execution.status == 'running':
            running_components = self.db.query(ComponentExecution).filter_by(
                execution_id=execution_id,
                status='running'
            ).all()
            if running_components:
                current_component = running_components[0].component_id

        return WorkflowExecutionStatus(
            execution_id=execution_id,
            status=execution.status,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            duration_seconds=execution.duration_seconds,
            total_components=execution.total_components,
            completed_components=execution.completed_components,
            failed_components=execution.failed_components,
            progress_percentage=progress_percentage,
            current_component=current_component,
            error_message=execution.error_message
        )

class ComponentFactory:
    """Factory for creating workflow components"""

    def __init__(self):
        self.component_classes = {
            'data_input': DataInputComponent,
            'llm_processor': LLMProcessorComponent,
            'rule_engine': RuleEngineComponent,
            'decision_node': DecisionNodeComponent,
            'database_output': DatabaseOutputComponent,
            'plugin_executor': PluginExecutorComponent
        }

    def create_component(self, component_config: Dict) -> WorkflowComponent:
        """Create a component instance from configuration"""
        component_type = component_config.get('component_type')
        component_id = component_config.get('component_id')
        config = component_config.get('config', {})

        if component_type not in self.component_classes:
            raise ValueError(f"Unknown component type: {component_type}")

        component_class = self.component_classes[component_type]
        return component_class(component_id, config)

# =============================================================================
# MONITORING & METRICS
# =============================================================================

class MetricsCollector:
    """Collects and exposes Prometheus metrics"""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Workflow metrics
        self.active_workflows = Gauge('workflow_engine_active_workflows', 'Number of active workflow executions', registry=self.registry)
        self.workflow_executions = Counter('workflow_engine_executions_total', 'Total workflow executions', ['status'], registry=self.registry)
        self.workflow_execution_time = Histogram('workflow_engine_execution_duration_seconds', 'Workflow execution duration', registry=self.registry)
        self.component_executions = Counter('workflow_engine_component_executions_total', 'Total component executions', ['component_type', 'status'], registry=self.registry)
        self.component_execution_time = Histogram('workflow_engine_component_duration_seconds', 'Component execution duration', ['component_type'], registry=self.registry)

        # Performance metrics
        self.request_count = Counter('workflow_engine_requests_total', 'Total number of requests', ['method', 'endpoint'], registry=self.registry)
        self.request_duration = Histogram('workflow_engine_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'], registry=self.registry)
        self.error_count = Counter('workflow_engine_errors_total', 'Total number of errors', ['type'], registry=self.registry)

    def update_workflow_metrics(self, workflow_executor: WorkflowExecutor):
        """Update workflow-related metrics"""
        try:
            # This would typically query the database for active workflows
            # For now, use a simple estimate
            self.active_workflows.set(len(workflow_executor.active_executions))

        except Exception as e:
            logger.error(f"Failed to update workflow metrics: {str(e)}")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Workflow Engine Service",
    description="Executes agent workflows with drag-and-drop components",
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

# Global instances
db_engine = create_engine(Config.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=Config.REDIS_DB, decode_responses=True)
metrics_collector = MetricsCollector()

# Dependency injection
def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_workflow_executor(db: Session = Depends(get_db)):
    """Workflow executor dependency"""
    return WorkflowExecutor(db, redis_client)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "workflow-engine",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    if not Config.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    return generate_latest(metrics_collector.registry)

@app.post("/workflows/execute")
async def execute_workflow(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    workflow_executor: WorkflowExecutor = Depends(get_workflow_executor)
):
    """Execute a workflow"""
    metrics_collector.request_count.labels(method='POST', endpoint='/workflows/execute').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/workflows/execute').time():
        execution_id = await workflow_executor.execute_workflow(request)

    return {
        "execution_id": execution_id,
        "status": "started",
        "message": "Workflow execution started successfully"
    }

@app.get("/workflows/executions/{execution_id}")
async def get_execution_status(
    execution_id: str,
    workflow_executor: WorkflowExecutor = Depends(get_workflow_executor)
):
    """Get workflow execution status"""
    metrics_collector.request_count.labels(method='GET', endpoint='/workflows/executions/{execution_id}').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/workflows/executions/{execution_id}').time():
        status = workflow_executor.get_execution_status(execution_id)

    return status

@app.get("/workflows/components")
async def list_component_types():
    """List available component types"""
    components = [
        {
            "type": "data_input",
            "name": "Data Input",
            "description": "Import data from various sources (CSV, API, Database)",
            "category": "input"
        },
        {
            "type": "llm_processor",
            "name": "LLM Processor",
            "description": "Process data using Large Language Models",
            "category": "processing"
        },
        {
            "type": "rule_engine",
            "name": "Rule Engine",
            "description": "Apply business rules and logic",
            "category": "processing"
        },
        {
            "type": "decision_node",
            "name": "Decision Node",
            "description": "Conditional branching based on data values",
            "category": "logic"
        },
        {
            "type": "database_output",
            "name": "Database Output",
            "description": "Write results to database tables",
            "category": "output"
        },
        {
            "type": "plugin_executor",
            "name": "Plugin Executor",
            "description": "Execute registered plugins",
            "category": "processing"
        }
    ]

    return {"components": components}

@app.post("/workflows/validate")
async def validate_workflow(workflow: WorkflowDefinition):
    """Validate a workflow definition"""
    metrics_collector.request_count.labels(method='POST', endpoint='/workflows/validate').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/workflows/validate').time():
        try:
            # Basic validation
            errors = []

            if not workflow.components:
                errors.append("Workflow must have at least one component")

            if len(workflow.components) > Config.WORKFLOW_MAX_COMPONENTS:
                errors.append(f"Workflow exceeds maximum components limit of {Config.WORKFLOW_MAX_COMPONENTS}")

            # Validate component configurations
            component_ids = set()
            for component in workflow.components:
                if component.component_id in component_ids:
                    errors.append(f"Duplicate component ID: {component.component_id}")
                component_ids.add(component.component_id)

                # Validate component type
                valid_types = ['data_input', 'llm_processor', 'rule_engine', 'decision_node', 'database_output', 'plugin_executor']
                if component.component_type not in valid_types:
                    errors.append(f"Invalid component type: {component.component_type}")

            # Validate connections
            for connection in workflow.connections:
                if connection['from'] not in component_ids:
                    errors.append(f"Connection from unknown component: {connection['from']}")
                if connection['to'] not in component_ids:
                    errors.append(f"Connection to unknown component: {connection['to']}")

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "component_count": len(workflow.components),
                "connection_count": len(workflow.connections)
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "component_count": 0,
                "connection_count": 0
            }

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    metrics_collector.error_count.labels(type='validation').inc()

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "message": "Invalid request data"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    metrics_collector.error_count.labels(type='http').inc()

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    metrics_collector.error_count.labels(type='general').inc()

    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": str(uuid.uuid4())
        }
    )

# =============================================================================
# LIFECYCLE MANAGEMENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Starting Workflow Engine Service...")

    # Create database tables
    try:
        Base.metadata.create_all(bind=db_engine)
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise

    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        raise

    logger.info(f"Workflow Engine Service started on {Config.SERVICE_HOST}:{Config.SERVICE_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down Workflow Engine Service...")

    # Close Redis connection
    try:
        redis_client.close()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {str(e)}")

    logger.info("Workflow Engine Service shutdown complete")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.SERVICE_HOST,
        port=Config.SERVICE_PORT,
        reload=False,
        log_level="info",
        access_log=True
    )
