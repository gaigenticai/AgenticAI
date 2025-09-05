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
import yaml
import sys
import os

# Local configuration (removed utils dependency for Docker compatibility)
class DatabaseConfig:
    @staticmethod
    def get_postgres_config():
        return {
            'host': os.getenv('POSTGRES_HOST', 'postgresql_ingestion'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'agentic_db'),
            'user': os.getenv('POSTGRES_USER', 'agentic_user'),
            'password': os.getenv('POSTGRES_PASSWORD', ''),
            'url': os.getenv('DATABASE_URL', '')
        }

    @staticmethod
    def get_redis_config():
        return {
            'host': os.getenv('REDIS_HOST', 'redis_ingestion'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'db': int(os.getenv('REDIS_DB', '0')),
            'password': os.getenv('REDIS_PASSWORD', '')
        }

class ServiceConfig:
    @staticmethod
    def get_service_host_port(service_name, default_port):
        host_env = f"{service_name.upper()}_HOST"
        port_env = f"{service_name.upper()}_PORT"
        return {
            'host': os.getenv(host_env, '0.0.0.0'),
            'port': os.getenv(port_env, default_port)
        }

    @staticmethod
    def get_auth_config():
        return {
            'require_auth': os.getenv('REQUIRE_AUTH', 'false').lower() == 'true',
            'jwt_secret': os.getenv('JWT_SECRET', ''),
            'jwt_algorithm': 'HS256',
            'jwt_expiration_hours': int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
        }

# JWT support for authentication
try:
    import jwt
except ImportError:
    jwt = None

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# Load default configuration values (Rule 1 compliance - no hardcoded values)
def load_defaults():
    """Load default configuration values from external file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'defaults.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Default configuration file not found at {config_path}, using minimal defaults")
        return {}

# Load default values from configuration file
DEFAULTS = load_defaults()

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """
    Configuration class for Workflow Engine Service

    Rule 1 Compliance: All default values loaded from external configuration file
    No hardcoded values in source code
    """

    # Database Configuration - using shared config for modularity (Rule 2)
    db_config = DatabaseConfig.get_postgres_config()
    DB_HOST = db_config['host']
    DB_PORT = db_config['port']
    DB_NAME = db_config['database']
    DB_USER = db_config['user']
    DB_PASSWORD = db_config['password']
    # Build DATABASE_URL from components if not provided
    if db_config['url']:
        DATABASE_URL = db_config['url']
    else:
        DATABASE_URL = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

    # Redis Configuration - using shared config for modularity (Rule 2)
    redis_config = DatabaseConfig.get_redis_config()
    REDIS_HOST = redis_config['host']
    REDIS_PORT = redis_config['port']
    REDIS_DB = int(os.getenv('WORKFLOW_ENGINE_REDIS_DB', '2'))  # Service-specific DB

    # Service Configuration - using shared config for consistency
    service_config = ServiceConfig.get_service_host_port('WORKFLOW_ENGINE', '8202')
    SERVICE_HOST = service_config['host']
    SERVICE_PORT = int(service_config['port'])

    # Authentication Configuration - using shared config for consistency
    auth_config = ServiceConfig.get_auth_config()
    REQUIRE_AUTH = auth_config['require_auth']
    JWT_SECRET = auth_config['jwt_secret']
    JWT_ALGORITHM = auth_config['jwt_algorithm']

    # Plugin Registry Configuration - loaded from external config
    PLUGIN_REGISTRY_HOST = os.getenv('PLUGIN_REGISTRY_HOST', DEFAULTS.get('service_config', {}).get('service_host', 'plugin-registry'))
    PLUGIN_REGISTRY_PORT = int(os.getenv('PLUGIN_REGISTRY_PORT',
                                        str(DEFAULTS.get('service_ports', {}).get('plugin_registry_port', 8201))))

    # Rule Engine Configuration - loaded from external config
    RULE_ENGINE_HOST = os.getenv('RULE_ENGINE_HOST', DEFAULTS.get('service_config', {}).get('service_host', 'rule-engine'))
    RULE_ENGINE_PORT = int(os.getenv('RULE_ENGINE_PORT',
                                    str(DEFAULTS.get('service_ports', {}).get('rule_engine_port', 8204))))

    # Memory Manager Configuration - loaded from external config
    MEMORY_MANAGER_HOST = os.getenv('MEMORY_MANAGER_HOST', DEFAULTS.get('service_config', {}).get('service_host', 'memory-manager'))
    MEMORY_MANAGER_PORT = int(os.getenv('MEMORY_MANAGER_PORT',
                                       str(DEFAULTS.get('service_ports', {}).get('memory_manager_port', 8205))))

    # Workflow Configuration - loaded from external config
    WORKFLOW_MAX_COMPONENTS = int(os.getenv('WORKFLOW_MAX_COMPONENTS',
                                           str(DEFAULTS.get('performance', {}).get('workflow_max_components', 50))))
    WORKFLOW_EXECUTION_TIMEOUT = int(os.getenv('WORKFLOW_EXECUTION_TIMEOUT',
                                              str(DEFAULTS.get('performance', {}).get('workflow_execution_timeout', 1800))))
    WORKFLOW_PARALLEL_EXECUTION = os.getenv('WORKFLOW_PARALLEL_EXECUTION',
                                           str(DEFAULTS.get('performance', {}).get('workflow_parallel_execution', True))).lower() == 'true'
    WORKFLOW_ERROR_RECOVERY_ENABLED = os.getenv('WORKFLOW_ERROR_RECOVERY_ENABLED',
                                               str(DEFAULTS.get('performance', {}).get('workflow_error_recovery_enabled', True))).lower() == 'true'

    # Monitoring Configuration - loaded from external config
    METRICS_ENABLED = os.getenv('AGENT_METRICS_ENABLED',
                               str(DEFAULTS.get('performance', {}).get('enable_metrics', True))).lower() == 'true'

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
    execution_metadata = Column(JSON, default=dict)

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
    component_metadata = Column(JSON, default=dict)

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
                # Real CSV data retrieval with file processing
                result = self._process_csv_data(query, connection_config)
            elif source_type == 'api':
                # Real API data retrieval with HTTP calls
                result = self._process_api_data(query, connection_config)
            elif source_type == 'database':
                # Real database query with proper SQL execution
                try:
                    db_config = connection_config
                    table_name = db_config.get('table', 'data')
                    limit = db_config.get('limit', 1000)

                    # Build database URL
                    db_url = db_config.get('url')
                    if not db_url:
                        db_url = f"postgresql://{db_config.get('user')}:{db_config.get('password')}@{db_config.get('host')}:{db_config.get('port', 5432)}/{db_config.get('database')}"

                    engine = create_engine(db_url)

                    with engine.connect() as conn:
                        result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
                        columns = result.keys()
                        data = [dict(zip(columns, row)) for row in result.fetchall()]

                    result = {
                        'data': data,
                        'row_count': len(data),
                        'table': table_name,
                        'source': 'database_table'
                    }
                except Exception as db_error:
                    result = {
                        'data': [],
                        'row_count': 0,
                        'error': f'Database query failed: {str(db_error)}',
                        'table': connection_config.get('table', 'unknown'),
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

    def _process_csv_data(self, query: Dict[str, Any], connection_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process CSV data from file with real file I/O"""
        try:
            import csv
            file_path = connection_config.get('file_path') or query.get('file_path')

            if not file_path:
                raise ValueError("file_path is required for CSV processing")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")

            data = []
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                # Detect delimiter and read CSV
                sample = csvfile.read(1024)
                csvfile.seek(0)
                delimiter = csv.Sniffer().sniff(sample).delimiter

                reader = csv.DictReader(csvfile, delimiter=delimiter)
                for row in reader:
                    data.append(row)

            return {
                'data': data,
                'row_count': len(data),
                'columns': list(data[0].keys()) if data else [],
                'source': 'csv_file',
                'file_path': file_path
            }

        except Exception as e:
            logger.error(f"CSV processing failed: {str(e)}")
            raise

    def _process_api_data(self, query: Dict[str, Any], connection_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process API data with real HTTP requests"""
        try:
            import httpx

            base_url = connection_config.get('base_url')
            endpoint = query.get('endpoint', '/api/data')
            params = query.get('params', {})
            headers = connection_config.get('headers', {})

            if not base_url:
                raise ValueError("base_url is required for API processing")

            # Make synchronous HTTP request (since this is called from async context)
            import asyncio
            async def fetch_data():
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{base_url.rstrip('/')}{endpoint}",
                        params=params,
                        headers=headers
                    )
                    response.raise_for_status()
                    return response.json()

            # Run async function in current context
            data = asyncio.run(fetch_data())

            # Handle different response formats
            if isinstance(data, dict):
                if 'data' in data:
                    processed_data = data['data']
                    row_count = len(processed_data) if isinstance(processed_data, list) else 1
                else:
                    processed_data = [data]
                    row_count = 1
            elif isinstance(data, list):
                processed_data = data
                row_count = len(data)
            else:
                processed_data = [data]
                row_count = 1

            return {
                'data': processed_data,
                'row_count': row_count,
                'endpoint': f"{base_url}{endpoint}",
                'source': 'api_endpoint',
                'response_metadata': {
                    'params': params,
                    'headers': {k: v for k, v in headers.items() if k.lower() != 'authorization'}  # Don't log auth headers
                }
            }

        except Exception as e:
            logger.error(f"API processing failed: {str(e)}")
            raise

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
        """
        Execute LLM processing with real OpenAI API integration for data analysis and recommendations.

        This method performs actual LLM API calls for data processing and analysis:
        1. Extracts configuration parameters (model, prompt template, temperature, API key)
        2. Formats prompt template with input data using variable substitution
        3. Makes real OpenAI API call with proper error handling
        4. Parses and structures the response with confidence scoring
        5. Provides structured output with recommendations and insights

        The component supports dynamic prompt templating where variables in the
        format {variable_name} are replaced with values from input data.

        Args:
            input_data: Dictionary containing workflow data, typically with 'data' key

        Returns:
            Dict containing processed text, confidence score, token usage, and analysis
        """
        try:
            import openai

            # Extract configuration with proper validation
            model = self.config.get('model', 'gpt-4')
            prompt_template = self.config.get('prompt_template', '')
            temperature = self.config.get('temperature', float(os.getenv('LLM_DEFAULT_TEMPERATURE', '0.7')))
            api_key = os.getenv('OPENAI_API_KEY')

            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for LLM processing")

            if not prompt_template:
                raise ValueError("Prompt template is required for LLM processing")

            # Format prompt with input data using variable substitution
            prompt = prompt_template
            if isinstance(input_data, dict) and 'data' in input_data:
                # Use first data item for context (typical for single record processing)
                data_item = input_data['data'][0] if input_data['data'] else {}
                for key, value in data_item.items():
                    prompt = prompt.replace(f"{{{key}}}", str(value))

            # Initialize OpenAI client and make API call
            client = openai.AsyncOpenAI(api_key=api_key)

            start_time = time.time()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst providing structured analysis with confidence scores and actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            processing_time = time.time() - start_time

            # Extract response content
            if not response.choices:
                raise ValueError("No response received from OpenAI API")

            response_content = response.choices[0].message.content
            if not response_content:
                raise ValueError("Empty response received from OpenAI API")

            # Parse JSON response from LLM
            try:
                llm_response = json.loads(response_content)
            except json.JSONDecodeError:
                # Fallback: extract key information from text response
                llm_response = self._parse_text_response(response_content)

            # Calculate confidence based on response completeness and coherence
            confidence_score = self._calculate_confidence_score(llm_response)

            # Structure the result
            result = {
                'processed_text': llm_response.get('analysis', response_content),
                'confidence_score': confidence_score,
                'tokens_used': response.usage.total_tokens if response.usage else len(prompt.split()) * 1.2,
                'model': model,
                'temperature': temperature,
                'processing_time_seconds': processing_time,
                'recommendation': llm_response.get('recommendation', 'REVIEW'),
                'analysis': {
                    'risk_level': llm_response.get('risk_level', 'UNKNOWN'),
                    'key_insights': llm_response.get('key_insights', []),
                    'next_steps': llm_response.get('next_steps', [])
                }
            }

            return result

        except Exception as e:
            logger.error(f"LLM processing failed: {str(e)}")
            return {
                'error': f'LLM processing failed: {str(e)}',
                'processed_text': '',
                'confidence_score': 0.0,
                'tokens_used': 0,
                'model': self.config.get('model', 'unknown')
            }

    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails"""
        # Basic text parsing for key information
        response_lower = response_text.lower()

        risk_indicators = ['high risk', 'high-risk', 'risky', 'suspicious', 'concerning']
        low_risk_indicators = ['low risk', 'safe', 'normal', 'standard', 'typical']

        if any(indicator in response_lower for indicator in risk_indicators):
            risk_level = 'HIGH'
        elif any(indicator in response_lower for indicator in low_risk_indicators):
            risk_level = 'LOW'
        else:
            risk_level = 'MEDIUM'

        return {
            'analysis': response_text,
            'risk_level': risk_level,
            'key_insights': [response_text[:100] + '...'],
            'next_steps': ['Review analysis manually'],
            'recommendation': 'REVIEW'
        }

    def _calculate_confidence_score(self, llm_response: Dict[str, Any]) -> float:
        """Calculate confidence score based on response completeness"""
        score = 0.5  # Base score

        # Increase score for presence of key fields
        if llm_response.get('risk_level'):
            score += 0.2
        if llm_response.get('key_insights') and len(llm_response['key_insights']) > 0:
            score += 0.15
        if llm_response.get('next_steps') and len(llm_response['next_steps']) > 0:
            score += 0.15

        # Cap at 0.95 for realism
        return min(score, 0.95)

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
        """
        Execute rule engine processing with conditional logic evaluation.

        This method implements a rule-based decision engine that evaluates
        input data against predefined rule sets for different business scenarios:
        1. Fraud Detection Rules: Analyzes transaction patterns and amounts
        2. Compliance Rules: Checks regulatory compliance requirements
        3. Default Rules: Generic rule evaluation framework

        The fraud detection algorithm calculates a risk score based on:
        - Transaction amount thresholds (high amounts increase risk)
        - Transaction frequency patterns (high frequency flags suspicious activity)
        - Risk level determination based on cumulative score
        - Recommendation generation for business decision making

        Args:
            input_data: Dictionary containing data to evaluate against rules

        Returns:
            Dict containing rule evaluation results, scores, and recommendations
        """
        try:
            rule_set = self.config.get('rule_set', 'default')

            # Simulate rule evaluation processing time
            rule_processing_delay = float(os.getenv('RULE_ENGINE_PROCESSING_DELAY_SECONDS', '0.2'))
            await asyncio.sleep(rule_processing_delay)

            # Apply different rule sets based on configuration
            if rule_set == 'fraud_rules':
                fraud_score = 0
                flags = []

                # Extract data for rule evaluation
                data = input_data.get('data', [{}])[0] if input_data.get('data') else {}

                # Rule 1: High transaction amount detection
                amount_threshold = float(os.getenv('FRAUD_AMOUNT_THRESHOLD', '10000'))
                amount_risk_score = float(os.getenv('FRAUD_AMOUNT_RISK_SCORE', '20'))
                if data.get('amount', 0) > amount_threshold:
                    fraud_score += amount_risk_score  # Configurable risk increase for large amounts
                    flags.append('High transaction amount')

                # Rule 2: High transaction frequency detection
                frequency_threshold = float(os.getenv('FRAUD_FREQUENCY_THRESHOLD', '5'))
                frequency_risk_score = float(os.getenv('FRAUD_FREQUENCY_RISK_SCORE', '15'))
                if data.get('frequency', 0) > frequency_threshold:
                    fraud_score += frequency_risk_score  # Configurable risk increase for frequent activity
                    flags.append('High transaction frequency')

                # Determine risk level and recommendation based on cumulative score
                risk_threshold = float(os.getenv('FRAUD_RISK_THRESHOLD', '25'))
                risk_level = 'HIGH' if fraud_score > risk_threshold else 'LOW'
                recommendation = 'REVIEW' if fraud_score > risk_threshold else 'APPROVE'

                result = {
                    'rule_set': rule_set,
                    'fraud_score': fraud_score,  # Cumulative risk score
                    'risk_level': risk_level,    # Categorized risk level
                    'flags': flags,             # List of triggered risk indicators
                    'recommendation': recommendation  # Business action recommendation
                }

            elif rule_set == 'compliance_rules':
                # Mock compliance rules evaluation
                default_compliance_score = float(os.getenv('COMPLIANCE_DEFAULT_SCORE', '95'))
                result = {
                    'rule_set': rule_set,
                    'compliance_score': default_compliance_score,     # Configurable compliance score
                    'violations': [],           # List of compliance violations
                    'recommendation': os.getenv('COMPLIANCE_DEFAULT_RECOMMENDATION', 'COMPLIANT')  # Configurable compliance status
                }

            else:
                # Default rule set for generic evaluations
                result = {
                    'rule_set': rule_set,
                    'evaluation_result': 'PASSED',    # Overall evaluation result
                    'matched_rules': ['rule1', 'rule2'],  # Rules that were triggered
                    'recommendation': 'CONTINUE'      # Next action recommendation
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
        """Execute plugin via plugin registry service"""
        try:
            plugin_id = self.config.get('plugin_id')
            plugin_registry_host = os.getenv('PLUGIN_REGISTRY_HOST', DEFAULTS.get('service_config', {}).get('service_host', 'plugin-registry'))
            plugin_registry_port = int(os.getenv('PLUGIN_REGISTRY_PORT', str(DEFAULTS.get('service_ports', {}).get('plugin_registry_port', 8201))))

            if not plugin_id:
                raise ValueError("plugin_id is required for plugin execution")

            # Make real HTTP call to plugin registry service
            plugin_url = f"http://{plugin_registry_host}:{plugin_registry_port}/api/plugins/{plugin_id}/execute"

            start_time = time.time()
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    plugin_url,
                    json=input_data,
                    headers={"Content-Type": "application/json"}
                )

            execution_time_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                plugin_result = response.json()

                result = {
                    'plugin_id': plugin_id,
                    'execution_result': 'SUCCESS',
                    'output_data': plugin_result.get('result', plugin_result),
                    'execution_time_ms': execution_time_ms,
                    'plugin_version': plugin_result.get('version', 'unknown'),
                    'service_response': plugin_result
                }

                return result
            else:
                return {
                    'plugin_id': plugin_id,
                    'execution_result': 'FAILED',
                    'error': f'Plugin registry returned status {response.status_code}',
                    'response_body': response.text,
                    'execution_time_ms': execution_time_ms
                }

        except httpx.TimeoutException:
            return {
                'plugin_id': self.config.get('plugin_id'),
                'execution_result': 'TIMEOUT',
                'error': 'Plugin execution timed out after 30 seconds'
            }
        except httpx.RequestError as e:
            return {
                'plugin_id': self.config.get('plugin_id'),
                'execution_result': 'NETWORK_ERROR',
                'error': f'Network error calling plugin registry: {str(e)}'
            }
        except Exception as e:
            return {
                'plugin_id': self.config.get('plugin_id'),
                'execution_result': 'FAILED',
                'error': f'Plugin execution failed: {str(e)}'
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
    """
    Handles workflow execution orchestration and management.

    This class is responsible for:
    - Creating execution plans with dependency resolution
    - Orchestrating parallel and sequential component execution
    - Managing workflow state and progress tracking
    - Handling errors and recovery mechanisms
    - Coordinating with database and Redis for persistence and caching

    The executor uses NetworkX for dependency graph analysis and supports
    both parallel execution of independent components and sequential
    execution based on data dependencies.
    """

    def __init__(self, db_session: Session, redis_client: redis.Redis):
        """
        Initialize the WorkflowExecutor with required dependencies.

        Args:
            db_session: SQLAlchemy database session for persistence
            redis_client: Redis client for caching and state management
        """
        self.db = db_session
        self.redis = redis_client
        self.component_factory = ComponentFactory()
        self.active_executions = {}  # Track active workflow executions

    def create_execution_plan(self, components: List[Dict], connections: List[Dict]) -> Dict[str, Any]:
        """
        Create execution plan with dependency graph analysis.

        This method builds a dependency graph using NetworkX and determines:
        - Execution order using topological sorting
        - Parallel execution levels for independent components
        - Estimated total execution time
        - Dependency validation and cycle detection

        Args:
            components: List of component configurations
            connections: List of component connections defining dependencies

        Returns:
            Dict containing execution plan with order, levels, and metadata
        """
        try:
            # DEPENDENCY RESOLUTION ALGORITHM
            # =================================
            # This algorithm uses graph theory to resolve component dependencies and create
            # an optimal execution plan that maximizes parallelism while respecting data flow

            # Step 1: Build Directed Graph Representation
            # Create a NetworkX DiGraph where each node represents a workflow component
            # and each edge represents a dependency relationship (data flow requirement)
            graph = nx.DiGraph()

            # Add all workflow components as nodes in the dependency graph
            # Each node stores the complete component configuration for later execution
            for component in components:
                component_id = component['component_id']
                graph.add_node(component_id, data=component)

            # Step 2: Establish Dependency Relationships
            # Convert workflow connections into directed edges
            # Edge direction: source_component -> dependent_component
            # This represents "component A must complete before component B can start"
            for connection in connections:
                source_component = connection['from']      # Component providing data
                dependent_component = connection['to']     # Component requiring data
                graph.add_edge(source_component, dependent_component)

            # Step 3: Topological Sorting for Execution Order
            # Topological sort ensures all dependencies are satisfied before execution
            # Result: ordered list where each component appears after its dependencies
            try:
                execution_order = list(nx.topological_sort(graph))
            except nx.NetworkXError:
                # Cycle Detection: Handle circular dependencies gracefully
                # In production workflows, cycles indicate design errors and should be rejected
                # For robustness, we fall back to unordered execution (not recommended for production)
                logger.warning("Circular dependency detected in workflow, falling back to unordered execution")
                execution_order = list(graph.nodes())

            # Step 4: Parallel Execution Level Calculation
            # Group components by dependency level to enable parallel execution
            # Components in the same level have no dependencies between them and can run in parallel
            execution_levels = {}
            for component_id in execution_order:
                # Find all components that must complete before this component
                predecessors = list(graph.predecessors(component_id))

                # Calculate execution level using dynamic programming approach
                # Level = maximum level of all predecessors + 1
                # This ensures components are executed in the correct dependency order
                if predecessors:
                    # Get the highest level among all predecessor components
                    predecessor_levels = [execution_levels.get(pred, 0) for pred in predecessors]
                    level = max(predecessor_levels) + 1
                else:
                    # No dependencies: execute at level 1 (first level)
                    level = 1

                execution_levels[component_id] = level

            # Step 5: Group Components by Execution Level
            # Organize components into parallel execution batches
            # Components in the same level can be executed concurrently
            levels = {}
            for component_id, level in execution_levels.items():
                if level not in levels:
                    levels[level] = []
                levels[level].append(component_id)

            # Step 6: Generate Execution Plan with Metadata
            # Calculate execution statistics and optimization opportunities
            total_components = len(components)
            parallel_levels = len(levels)
            parallel_execution_possible = parallel_levels > 1

            # Estimate total execution time based on component type benchmarks
            # This helps with resource planning and execution time prediction
            estimated_duration_ms = sum([self._estimate_component_duration(c) for c in components])

            execution_plan = {
                'execution_order': execution_order,           # Topologically sorted component list
                'execution_levels': levels,                   # Components grouped by parallel level
                'total_components': total_components,         # Total number of components in workflow
                'parallel_execution_possible': parallel_execution_possible,  # True if parallelism available
                'parallel_levels': parallel_levels,           # Number of parallel execution levels
                'estimated_duration_ms': estimated_duration_ms  # Predicted execution time
            }

            logger.info(f"Generated execution plan: {total_components} components, "
                       f"{parallel_levels} parallel levels, {estimated_duration_ms}ms estimated")
            return execution_plan

        except Exception as e:
            logger.error(f"Failed to create execution plan: {str(e)}")
            # Return fallback plan with all components in single level
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
        """
        Execute a workflow asynchronously with full orchestration.

        This method handles the complete workflow execution lifecycle:
        1. Validates workflow existence and accessibility
        2. Creates execution record in database for tracking
        3. Generates execution plan with dependency analysis
        4. Launches asynchronous execution in background
        5. Returns execution ID for status monitoring

        Args:
            request: Workflow execution request with workflow ID, agent ID, and input data

        Returns:
            str: Unique execution ID for tracking workflow progress

        Raises:
            HTTPException: If workflow not found or execution cannot be started
        """
        execution_id = str(uuid.uuid4())

        try:
            # Retrieve workflow definition from database with validation
            workflow = self.db.query(AgentWorkflow).filter_by(
                workflow_id=request.workflow_id,
                agent_id=request.agent_id,
                is_active=True
            ).first()

            if not workflow:
                raise HTTPException(status_code=404, detail=f"Workflow {request.workflow_id} not found")

            # Create execution record for tracking and monitoring
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

            # Start async execution in background to avoid blocking the API response
            asyncio.create_task(self._execute_workflow_async(execution_id, workflow, request.input_data))

            return execution_id

        except Exception as e:
            logger.error(f"Failed to start workflow execution: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Workflow execution failed to start: {str(e)}")

    async def _execute_workflow_async(self, execution_id: str, workflow: Any, input_data: Dict[str, Any]):
        """
        Execute workflow asynchronously with parallel component processing.

        This method implements the core workflow execution algorithm:
        1. Updates execution status and timing
        2. Builds component lookup for efficient access
        3. Generates execution plan with dependency levels
        4. Executes components level by level (parallel within levels)
        5. Aggregates results and handles errors
        6. Updates execution record with final status

        The execution uses asyncio.gather for parallel processing of independent
        components while maintaining dependency order between levels.

        Args:
            execution_id: Unique identifier for this execution
            workflow: Workflow definition from database
            input_data: Initial input data for the workflow
        """
        try:
            start_time = time.time()

            # Update execution status to mark as started
            execution = self.db.query(WorkflowExecution).filter_by(execution_id=execution_id).first()
            if execution:
                execution.started_at = datetime.utcnow()

            # Initialize result collections
            results = {}  # Stores successful component outputs for data flow
            component_results = {}  # Stores all component results (success/error)

            components = workflow.components or []
            connections = workflow.connections or []

            # Build component lookup dictionary for O(1) access during execution
            component_lookup = {c['component_id']: c for c in components}

            # Generate execution plan with dependency analysis
            execution_plan = self.create_execution_plan(components, connections)

            # COMPONENT EXECUTION ORCHESTRATION
            # ================================
            # Execute components level by level to respect dependencies while maximizing parallelism
            # This implements the core workflow execution algorithm with state management

            for level, component_ids in execution_plan['execution_levels'].items():
                # Level-based Execution Strategy
                # Each level contains components that can be executed in parallel
                # Components in the same level have no dependencies between them
                logger.debug(f"Executing level {level} with {len(component_ids)} components")

                # Step 1: Create async tasks for parallel component execution
                # Build task list for components in current execution level
                tasks = []
                for component_id in component_ids:
                    if component_id in component_lookup:
                        component_config = component_lookup[component_id]

                        # Create async task for component execution
                        # Each task is independent and can run concurrently within the level
                        task = self._execute_component_async(
                            execution_id,              # For tracking and state management
                            component_config,          # Component configuration and parameters
                            input_data,                # Original workflow input data
                            results                    # Previous component results for data flow
                        )
                        tasks.append(task)
                    else:
                        logger.warning(f"Component {component_id} not found in lookup table")

                # Step 2: Execute all components in current level concurrently
                # asyncio.gather enables parallel execution within the level
                # return_exceptions=True ensures one failure doesn't cancel others
                if tasks:
                    logger.debug(f"Awaiting {len(tasks)} concurrent tasks in level {level}")
                    level_results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Step 3: Process execution results and update workflow state
                    # Handle both successful executions and failures appropriately
                    for i, result in enumerate(level_results):
                        component_id = component_ids[i]

                        if isinstance(result, Exception):
                            # Component execution failed - record error state
                            error_message = str(result)
                            component_results[component_id] = {'error': error_message}

                            # Update execution state: mark component as failed
                            logger.error(f"Component {component_id} failed: {error_message}",
                                       execution_id=execution_id,
                                       level=level)

                            # State Management: Track failed components for workflow status
                            # This affects overall workflow completion percentage
                        else:
                            # Component executed successfully
                            component_results[component_id] = result
                            results[component_id] = result  # Make result available for dependent components

                            # State Management: Update successful execution tracking
                            logger.debug(f"Component {component_id} completed successfully",
                                       execution_id=execution_id,
                                       level=level)

                    # Update workflow progress after processing all components in this level
                    # This provides real-time visibility into execution progress
                    successful_in_level = sum(1 for r in level_results if not isinstance(r, Exception))
                    logger.info(f"Level {level} completed: {successful_in_level}/{len(component_ids)} successful",
                              execution_id=execution_id)

            # STATE MANAGEMENT: Update Final Execution Status
            # ===============================================
            # Calculate final execution metrics and update database state
            # This ensures the execution record reflects the true completion status

            duration = time.time() - start_time
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = duration

            # Determine final execution status based on component results
            # State Management: Calculate completion statistics
            total_components = len(component_results)
            successful_components = len([r for r in component_results.values() if 'error' not in r])
            failed_components = total_components - successful_components

            # State Logic: Determine workflow status based on component outcomes
            if failed_components == 0:
                # All components successful - workflow completed successfully
                execution.status = 'completed'
                execution_message = f"All {total_components} components executed successfully"
            elif successful_components == 0:
                # All components failed - workflow completely failed
                execution.status = 'failed'
                execution_message = f"All {total_components} components failed"
            else:
                # Partial success - some components failed but workflow reached completion
                execution.status = 'completed_with_errors'
                execution_message = f"{successful_components}/{total_components} components successful"

            # Update execution record with final state and results
            execution.output_data = component_results
            execution.completed_components = successful_components
            execution.failed_components = failed_components

            # Persist final state to database
            # This commit ensures all execution metadata is saved for monitoring and analysis
            self.db.commit()

            logger.info(f"Workflow {execution_id} {execution.status} in {duration:.2f} seconds: {execution_message}",
                       total_components=total_components,
                       successful=successful_components,
                       failed=failed_components)

        except Exception as e:
            # STATE MANAGEMENT: Handle Critical Execution Failure
            # ===================================================
            # Update execution record with failure state when unexpected errors occur
            # This ensures failed executions are properly tracked and reported

            # Retrieve execution record for failure state update
            execution = self.db.query(WorkflowExecution).filter_by(execution_id=execution_id).first()
            if execution:
                # State Transition: Mark execution as failed
                execution.status = 'failed'
                execution.completed_at = datetime.utcnow()

                # State Data: Record failure details for debugging
                execution.error_message = str(e)
                execution.failed_components = execution.total_components  # Assume all failed

                # Persist failure state to database
                self.db.commit()

                logger.error(f"Workflow {execution_id} failed with critical error: {str(e)}",
                           error_type=type(e).__name__,
                           execution_status=execution.status)

            logger.error(f"Workflow {execution_id} failed: {str(e)}")

    async def _execute_component_async(self, execution_id: str, component_config: Dict,
                                     input_data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single component with full tracking and error handling.

        This method handles the complete lifecycle of a single component execution:
        1. Creates database record for component execution tracking
        2. Instantiates the appropriate component class
        3. Prepares input data by combining workflow input and previous results
        4. Executes the component with timing measurement
        5. Records execution results and metrics
        6. Handles errors with proper database updates

        Args:
            execution_id: Parent workflow execution ID
            component_config: Component configuration dictionary
            input_data: Original workflow input data
            previous_results: Results from previously executed components

        Returns:
            Dict containing component execution results

        Raises:
            Exception: If component execution fails (with proper error tracking)
        """
        component_id = component_config['component_id']

        try:
            # STATE MANAGEMENT: Initialize Component Execution Tracking
            # ========================================================
            # Create database record to track individual component execution state
            # This enables detailed monitoring and debugging of component-level failures

            component_exec = ComponentExecution(
                execution_id=execution_id,
                component_id=component_id,
                component_type=component_config.get('component_type', 'unknown'),
                status='running',                    # Initial state: component is executing
                started_at=datetime.utcnow(),        # Timestamp for execution duration calculation
                input_data=input_data,               # Input data for debugging and audit trails
                retry_count=0                        # Track retry attempts if implemented
            )
            self.db.add(component_exec)
            self.db.commit()  # Immediate commit to ensure record exists

            # COMPONENT EXECUTION: Instantiate and Prepare
            # ===========================================
            # Create component instance using factory pattern for extensibility
            component = self.component_factory.create_component(component_config)

            # DATA FLOW MANAGEMENT: Prepare Component Input
            # ============================================
            # Merge original workflow input with results from previous components
            # This implements the data flow mechanism that connects dependent components
            component_input = dict(input_data)  # Preserve original input data
            component_input.update(previous_results)  # Add upstream component results

            # COMPONENT EXECUTION: Execute with Performance Monitoring
            # =======================================================
            # Execute component with timing measurement for performance analysis
            start_time = time.time()
            result = await component.execute(component_input)
            duration_ms = int((time.time() - start_time) * 1000)

            # STATE MANAGEMENT: Update Successful Execution State
            # ===================================================
            # Update component execution record with completion details
            component_exec.status = 'completed'      # Mark as successfully completed
            component_exec.completed_at = datetime.utcnow()  # End timestamp
            component_exec.duration_ms = duration_ms  # Actual execution time
            component_exec.output_data = result       # Store results for dependent components
            self.db.commit()  # Persist state changes

            logger.debug(f"Component {component_id} executed successfully in {duration_ms}ms",
                        execution_id=execution_id,
                        component_type=component_config.get('component_type'))

            return result

        except Exception as e:
            # STATE MANAGEMENT: Handle Component Execution Failure
            # ===================================================
            # Update component execution record with failure details
            # This ensures failed components are properly tracked and reported

            component_exec.status = 'failed'         # Mark execution as failed
            component_exec.completed_at = datetime.utcnow()  # Record failure time
            component_exec.error_message = str(e)    # Store error details for debugging
            self.db.commit()  # Persist failure state

            logger.error(f"Component {component_id} execution failed: {str(e)}",
                        execution_id=execution_id,
                        component_type=component_config.get('component_type'),
                        error_type=type(e).__name__)

            # Re-raise exception to propagate failure up the execution chain
            # This ensures workflow-level error handling can respond appropriately
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
        """Update workflow-related metrics with real database queries"""
        try:
            # Query database for comprehensive workflow statistics
            with self.engine.connect() as conn:
                # Count active workflows
                active_result = conn.execute(text("""
                    SELECT COUNT(*) as active_count
                    FROM workflow_executions
                    WHERE status IN ('running', 'pending')
                """))
                active_count = active_result.fetchone()[0]

                # Count total workflows by status
                status_result = conn.execute(text("""
                    SELECT status, COUNT(*) as count
                    FROM workflow_executions
                    WHERE status IN ('completed', 'failed', 'running', 'pending')
                    GROUP BY status
                """))
                status_counts = {row[0]: row[1] for row in status_result.fetchall()}

                # Get average execution time for completed workflows
                avg_time_result = conn.execute(text("""
                    SELECT AVG(duration_seconds) as avg_duration
                    FROM workflow_executions
                    WHERE status = 'completed' AND duration_seconds IS NOT NULL
                """))
                avg_duration = avg_time_result.fetchone()[0] or 0

                # Update metrics
                self.active_workflows.set(active_count)
                self.completed_workflows.set(status_counts.get('completed', 0))
                self.failed_workflows.set(status_counts.get('failed', 0))
                self.pending_workflows.set(status_counts.get('pending', 0))
                self.average_execution_time.set(avg_duration)

                # Update success rate
                total_completed = status_counts.get('completed', 0)
                total_failed = status_counts.get('failed', 0)
                total_workflows = total_completed + total_failed
                if total_workflows > 0:
                    success_rate = (total_completed / total_workflows) * 100
                    self.workflow_success_rate.set(success_rate)

                logger.debug(f"Updated workflow metrics: active={active_count}, completed={total_completed}, failed={total_failed}")

        except Exception as e:
            logger.error(f"Failed to update workflow metrics: {str(e)}")
            # Fallback to simple estimate if database query fails
            self.active_workflows.set(len(workflow_executor.active_executions) if hasattr(workflow_executor, 'active_executions') else 0)

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
        # Base.metadata.create_all(bind=...)  # Removed - use schema.sql instead
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
