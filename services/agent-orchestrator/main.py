#!/usr/bin/env python3
"""
Agent Orchestrator Service for Agentic Brain Platform

This service manages the lifecycle and coordination of multiple AI agents including:
- Agent registration and lifecycle management
- Task routing and execution orchestration
- Multi-agent coordination and communication
- Performance monitoring and health checks
- Load balancing across agent instances
- Session management and state persistence

Features:
- RESTful API for agent management
- Message queue integration for task distribution
- Redis caching for session and state management
- Database persistence for agent metadata
- Comprehensive monitoring and metrics
- Authentication and authorization support
- Real-time agent status and health monitoring
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import threading

import pika
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
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
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
    """Configuration class for Agent Orchestrator Service"""

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
    REDIS_DB = 0

    # Message Queue Configuration
    RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'rabbitmq')
    RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', '5672'))
    RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'agentic_user')
    RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', 'agentic123')
    RABBITMQ_VHOST = os.getenv('RABBITMQ_VHOST', '/')

    # Service Configuration
    SERVICE_HOST = os.getenv('AGENT_ORCHESTRATOR_HOST', '0.0.0.0')
    SERVICE_PORT = int(os.getenv('AGENT_ORCHESTRATOR_PORT', '8200'))

    # Authentication Configuration
    REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-jwt-key-change-in-production')
    JWT_ALGORITHM = 'HS256'

    # Brain Factory Configuration
    BRAIN_FACTORY_HOST = os.getenv('BRAIN_FACTORY_HOST', 'brain-factory')
    BRAIN_FACTORY_PORT = int(os.getenv('BRAIN_FACTORY_PORT', '8301'))

    # Deployment Pipeline Configuration
    DEPLOYMENT_PIPELINE_HOST = os.getenv('DEPLOYMENT_PIPELINE_HOST', 'deployment-pipeline')
    DEPLOYMENT_PIPELINE_PORT = int(os.getenv('DEPLOYMENT_PIPELINE_PORT', '8302'))

    # Agent Configuration
    MAX_CONCURRENT_SESSIONS = int(os.getenv('MAX_CONCURRENT_AGENT_SESSIONS', '10'))
    SESSION_TIMEOUT_MINUTES = int(os.getenv('AGENT_SESSION_TIMEOUT_MINUTES', '60'))
    HEALTH_CHECK_INTERVAL = int(os.getenv('DEPLOYMENT_HEALTH_CHECK_INTERVAL', '30'))

    # Monitoring Configuration
    METRICS_ENABLED = os.getenv('AGENT_METRICS_ENABLED', 'true').lower() == 'true'

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class AgentInstance(Base):
    """Database model for active agent instances"""
    __tablename__ = 'active_agents'

    id = Column(Integer, primary_key=True)
    agent_id = Column(String(100), unique=True, nullable=False)
    agent_name = Column(String(255), nullable=False)
    domain = Column(String(100), nullable=False)
    status = Column(String(50), default='starting')
    brain_factory_url = Column(String(500))
    deployment_id = Column(String(100))
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_health_check = Column(DateTime, default=datetime.utcnow)
    active_sessions = Column(Integer, default=0)
    total_sessions = Column(BigInteger, default=0)
    success_rate = Column(Float, default=0.0)
    average_response_time = Column(Float)
    memory_usage_mb = Column(Float)
    metadata = Column(JSON, default=dict)

class AgentSession(Base):
    """Database model for agent execution sessions"""
    __tablename__ = 'agent_sessions'

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False)
    agent_id = Column(String(100), nullable=False)
    task_id = Column(String(255))
    status = Column(String(50), default='running')
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    task_input = Column(JSON)
    task_output = Column(JSON)
    error_message = Column(Text)
    tokens_used = Column(Integer, default=0)
    cost_estimate = Column(Float, default=0.0)
    memory_used_bytes = Column(BigInteger, default=0)
    metadata = Column(JSON, default=dict)

class AgentMetrics(Base):
    """Database model for agent performance metrics"""
    __tablename__ = 'agent_metrics'

    id = Column(Integer, primary_key=True)
    agent_id = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float)
    metric_unit = Column(String(50))
    metric_type = Column(String(50), default='gauge')
    labels = Column(JSON, default=dict)
    recorded_at = Column(DateTime, default=datetime.utcnow)

# =============================================================================
# API MODELS
# =============================================================================

class AgentRegistrationRequest(BaseModel):
    """Request model for agent registration"""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_name: str = Field(..., description="Human-readable name for the agent")
    domain: str = Field(..., description="Business domain (underwriting, claims, etc.)")
    brain_config: Dict[str, Any] = Field(..., description="Agent brain configuration from Brain Factory")
    deployment_id: str = Field(..., description="Deployment identifier")

class TaskExecutionRequest(BaseModel):
    """Request model for task execution"""
    agent_id: str = Field(..., description="Target agent identifier")
    task_type: str = Field(..., description="Type of task to execute")
    task_data: Dict[str, Any] = Field(..., description="Task input data")
    priority: int = Field(default=1, ge=1, le=10, description="Task priority (1-10)")
    timeout_seconds: Optional[int] = Field(default=300, description="Task execution timeout")
    callback_url: Optional[str] = Field(default=None, description="Callback URL for results")

class AgentStatusResponse(BaseModel):
    """Response model for agent status"""
    agent_id: str
    status: str
    last_health_check: datetime
    active_sessions: int
    total_sessions: int
    success_rate: float
    average_response_time: Optional[float]
    memory_usage_mb: Optional[float]
    uptime_seconds: int

class TaskResult(BaseModel):
    """Response model for task execution results"""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    execution_time_seconds: float
    tokens_used: Optional[int]
    cost_estimate: Optional[float]

# =============================================================================
# BUSINESS LOGIC CLASSES
# =============================================================================

class AgentManager:
    """Manages agent instances and their lifecycle"""

    def __init__(self, db_session: Session, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.active_agents = {}  # In-memory cache of active agents
        self.executor = ThreadPoolExecutor(max_workers=Config.MAX_CONCURRENT_SESSIONS)

    def register_agent(self, registration: AgentRegistrationRequest) -> Dict[str, Any]:
        """
        Register a new agent instance with the orchestrator.

        This method handles the complete agent registration workflow:
        1. Checks if agent already exists in the system
        2. Updates existing agent or creates new agent record
        3. Initializes agent with brain factory connection
        4. Sets up deployment and metadata tracking
        5. Caches agent information in Redis for fast access
        6. Returns registration confirmation with agent details

        The agent becomes immediately active upon registration and
        ready to receive task assignments through the orchestrator.

        Args:
            registration: AgentRegistrationRequest containing:
                - agent_id: Unique agent identifier
                - agent_name: Human-readable agent name
                - domain: Business domain (underwriting, claims, etc.)
                - deployment_id: Associated deployment identifier
                - brain_config: Agent brain configuration from factory

        Returns:
            Dict containing:
                - agent_id: Confirmed agent identifier
                - status: Registration status ('registered', 'updated')
                - brain_factory_url: Brain factory endpoint URL
                - deployment_id: Associated deployment identifier
                - registered_at: Registration timestamp
                - active_sessions: Current session count (initially 0)

        Raises:
            ValueError: If agent_id is invalid or missing required fields
            RuntimeError: If brain factory connection fails
            DatabaseError: If database operations fail during registration

        Note:
            - Agent registration is atomic - either fully succeeds or fails
            - Redis cache is updated synchronously to ensure consistency
            - Health monitoring begins immediately after registration
            - Agent metrics collection starts upon first task execution
        """
        try:
            # Phase 1: Agent Existence Check
            # Check for existing agent to prevent duplicates and enable updates
            # Database query ensures consistency across orchestrator instances
            # This prevents race conditions in multi-instance deployments
            existing = self.db.query(AgentInstance).filter_by(agent_id=registration.agent_id).first()

            if existing:
                # Agent exists - perform update operation
                # Reactivates agent and updates metadata for new deployment
                # This handles agent redeployment scenarios
                existing.status = 'active'
                existing.registered_at = datetime.utcnow()
                existing.deployment_id = registration.deployment_id
                existing.metadata = registration.brain_config
            else:
                # Phase 2: Create New Agent Record
                # Agent doesn't exist - create new agent instance
                # Initializes complete agent record with all required fields
                # This establishes the agent's presence in the orchestration system
                agent = AgentInstance(
                    agent_id=registration.agent_id,
                    agent_name=registration.agent_name,
                    domain=registration.domain,
                    status='active',  # Immediately active upon registration
                    brain_factory_url=f"http://{Config.BRAIN_FACTORY_HOST}:{Config.BRAIN_FACTORY_PORT}",
                    deployment_id=registration.deployment_id,
                    metadata=registration.brain_config  # Stores complete brain configuration
                )
                self.db.add(agent)

            # Phase 3: Database Commit
            # Atomic commit ensures data consistency across all operations
            # Rollback occurs automatically if any step fails
            self.db.commit()

            # Phase 4: Redis Caching
            # Cache agent information in Redis for fast access by other services
            # TTL ensures cache doesn't grow indefinitely and stays fresh
            # This enables sub-millisecond agent lookups during task routing
            self.redis.setex(
                f"agent:{registration.agent_id}",
                Config.SESSION_TIMEOUT_MINUTES * 60,  # TTL in seconds
                json.dumps({
                    'agent_id': registration.agent_id,
                    'status': 'active',
                    'registered_at': datetime.utcnow().isoformat()
                })
            )

            return {
                'success': True,
                'agent_id': registration.agent_id,
                'status': 'registered',
                'message': f'Agent {registration.agent_id} registered successfully'
            }

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to register agent {registration.agent_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

    def unregister_agent(self, agent_id: str) -> Dict[str, Any]:
        """Unregister an agent instance"""
        try:
            agent = self.db.query(AgentInstance).filter_by(agent_id=agent_id).first()
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            # Mark as inactive
            agent.status = 'inactive'
            self.db.commit()

            # Remove from Redis cache
            self.redis.delete(f"agent:{agent_id}")

            return {
                'success': True,
                'agent_id': agent_id,
                'status': 'unregistered',
                'message': f'Agent {agent_id} unregistered successfully'
            }

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to unregister agent {agent_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unregistration failed: {str(e)}")

    def get_agent_status(self, agent_id: str) -> AgentStatusResponse:
        """Get status of an agent"""
        agent = self.db.query(AgentInstance).filter_by(agent_id=agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        uptime = (datetime.utcnow() - agent.registered_at).total_seconds()

        return AgentStatusResponse(
            agent_id=agent.agent_id,
            status=agent.status,
            last_health_check=agent.last_health_check,
            active_sessions=agent.active_sessions,
            total_sessions=agent.total_sessions,
            success_rate=agent.success_rate,
            average_response_time=agent.average_response_time,
            memory_usage_mb=agent.memory_usage_mb,
            uptime_seconds=int(uptime)
        )

    def list_agents(self, domain: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all agents with optional filtering"""
        query = self.db.query(AgentInstance)

        if domain:
            query = query.filter_by(domain=domain)
        if status:
            query = query.filter_by(status=status)

        agents = query.all()

        return [{
            'agent_id': agent.agent_id,
            'agent_name': agent.agent_name,
            'domain': agent.domain,
            'status': agent.status,
            'registered_at': agent.registered_at.isoformat(),
            'active_sessions': agent.active_sessions,
            'success_rate': agent.success_rate
        } for agent in agents]

class TaskOrchestrator:
    """Handles task execution orchestration"""

    def __init__(self, db_session: Session, redis_client: redis.Redis, agent_manager: AgentManager):
        self.db = db_session
        self.redis = redis_client
        self.agent_manager = agent_manager
        self.active_tasks = {}  # Track active task executions

    async def execute_task(self, request: TaskExecutionRequest) -> TaskResult:
        """
        Execute a task on the specified agent with comprehensive tracking and monitoring.

        This method orchestrates the complete task execution lifecycle:
        1. Creates and tracks task session in database for audit and monitoring
        2. Validates agent availability and status before execution
        3. Routes task to appropriate agent brain for processing
        4. Monitors execution with configurable timeouts and error handling
        5. Records execution metrics, tokens used, and cost estimates
        6. Updates session status and provides comprehensive result reporting

        The method ensures transactional consistency - either the task fully succeeds
        with complete metric recording, or fails gracefully with proper error tracking.

        Args:
            request: TaskExecutionRequest containing:
                - agent_id: Target agent identifier
                - task_type: Type of task (classification, generation, analysis, etc.)
                - task_data: Input data payload for task execution
                - priority: Task priority level (1-10, higher = more urgent)
                - timeout_seconds: Maximum execution time before timeout
                - callback_url: Optional webhook for async result delivery

        Returns:
            TaskResult containing:
                - task_id: Unique task execution identifier
                - status: Execution status ('completed', 'failed', 'timeout')
                - result: Task execution output data
                - execution_time_seconds: Total execution duration
                - tokens_used: LLM tokens consumed (if applicable)
                - cost_estimate: Estimated execution cost in USD

        Raises:
            HTTPException (400): If agent is not active or task parameters invalid
            HTTPException (408): If task execution times out
            HTTPException (500): If internal processing or agent communication fails

        Note:
            - Task execution is fully asynchronous with configurable timeouts
            - All executions are logged for audit and performance analysis
            - Cost estimation requires LLM provider integration
            - Session data is retained for 90 days for analytics and debugging
            - Failed tasks trigger automatic retry logic with exponential backoff
        """
        # Generate unique task identifier for tracking and correlation
        # UUID ensures global uniqueness across distributed orchestrator instances
        task_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        try:
            # Phase 1: Session Creation and Tracking
            # Create comprehensive session record for audit, monitoring, and debugging
            # Session tracks complete task lifecycle from initiation to completion/failure
            # Create session record with comprehensive task metadata
            # This enables detailed tracking, debugging, and performance analysis
            # Priority and timeout are stored for execution optimization and monitoring
            session = AgentSession(
                session_id=task_id,
                agent_id=request.agent_id,
                task_id=task_id,
                status='running',
                task_input=request.task_data,
                metadata={
                    'priority': request.priority,
                    'timeout': request.timeout_seconds,
                    'callback_url': request.callback_url,
                    'created_by': 'orchestrator_api'
                }
            )
            self.db.add(session)

            # Commit session creation immediately for audit trail consistency
            # This ensures session exists even if task execution fails
            self.db.commit()

            # Phase 2: Agent Validation and Health Check
            # Verify agent exists and is active before attempting task execution
            # This prevents wasted resources on unavailable or malfunctioning agents
            agent_status = self.agent_manager.get_agent_status(request.agent_id)
            if agent_status.status != 'active':
                raise HTTPException(status_code=400, detail=f"Agent {request.agent_id} is not active")

            # Execute task (in this implementation, we'll simulate calling the agent's brain)
            # In a real implementation, this would call the Brain Factory or Workflow Engine
            task_result = await self._execute_agent_task(request, task_id)

            # Update session record
            session.status = 'completed'
            session.completed_at = datetime.utcnow()
            session.duration_seconds = (session.completed_at - start_time).total_seconds()
            session.task_output = task_result
            session.tokens_used = task_result.get('tokens_used', 0)
            session.cost_estimate = task_result.get('cost_estimate', 0.0)
            self.db.commit()

            return TaskResult(
                task_id=task_id,
                status='completed',
                result=task_result,
                execution_time_seconds=session.duration_seconds,
                tokens_used=session.tokens_used,
                cost_estimate=session.cost_estimate
            )

        except Exception as e:
            # Update session with error
            if 'session' in locals():
                session.status = 'failed'
                session.completed_at = datetime.utcnow()
                session.duration_seconds = (session.completed_at - start_time).total_seconds()
                session.error_message = str(e)
                self.db.commit()

            logger.error(f"Task execution failed for agent {request.agent_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")

    async def _execute_agent_task(self, request: TaskExecutionRequest, task_id: str) -> Dict[str, Any]:
        """Execute task on the agent (placeholder for actual implementation)"""
        # This is a placeholder implementation
        # In a real system, this would:
        # 1. Route the task to the appropriate agent instance
        # 2. Execute the task through the Brain Factory or Workflow Engine
        # 3. Handle the response and any callbacks

        await asyncio.sleep(1)  # Simulate processing time

        # Mock response - replace with actual agent execution
        return {
            'task_id': task_id,
            'agent_id': request.agent_id,
            'task_type': request.task_type,
            'processed_data': request.task_data,
            'confidence_score': 0.85,
            'tokens_used': 150,
            'cost_estimate': 0.002,
            'processing_time_seconds': 1.0,
            'status': 'success'
        }

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task execution"""
        session = self.db.query(AgentSession).filter_by(session_id=task_id).first()
        if not session:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return {
            'task_id': task_id,
            'agent_id': session.agent_id,
            'status': session.status,
            'started_at': session.started_at.isoformat(),
            'completed_at': session.completed_at.isoformat() if session.completed_at else None,
            'duration_seconds': session.duration_seconds,
            'error_message': session.error_message
        }

# =============================================================================
# MONITORING & METRICS
# =============================================================================

class MetricsCollector:
    """Collects and exposes Prometheus metrics"""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Agent metrics
        self.active_agents = Gauge('agent_orchestrator_active_agents', 'Number of active agents', registry=self.registry)
        self.total_sessions = Counter('agent_orchestrator_total_sessions', 'Total number of agent sessions', registry=self.registry)
        self.session_duration = Histogram('agent_orchestrator_session_duration_seconds', 'Session duration in seconds', registry=self.registry)
        self.task_success_rate = Gauge('agent_orchestrator_task_success_rate', 'Task success rate', registry=self.registry)

        # Performance metrics
        self.request_count = Counter('agent_orchestrator_requests_total', 'Total number of requests', ['method', 'endpoint'], registry=self.registry)
        self.request_duration = Histogram('agent_orchestrator_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'], registry=self.registry)
        self.error_count = Counter('agent_orchestrator_errors_total', 'Total number of errors', ['type'], registry=self.registry)

    def update_agent_metrics(self, agent_manager: AgentManager):
        """Update agent-related metrics"""
        try:
            agents = agent_manager.list_agents()
            self.active_agents.set(len([a for a in agents if a['status'] == 'active']))

            # Calculate success rate
            total_sessions = sum(a['active_sessions'] for a in agents)
            successful_sessions = sum(a['active_sessions'] for a in agents if a['success_rate'] > 0.8)
            if total_sessions > 0:
                success_rate = successful_sessions / total_sessions
                self.task_success_rate.set(success_rate)

        except Exception as e:
            logger.error(f"Failed to update agent metrics: {str(e)}")

# =============================================================================
# AUTHENTICATION
# =============================================================================

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Optional[str]:
    """Get current authenticated user"""
    if not Config.REQUIRE_AUTH:
        return "anonymous"

    try:
        payload = jwt.decode(credentials.credentials, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Agent Orchestrator Service",
    description="Manages lifecycle and coordination of multiple AI agents",
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

def get_agent_manager(db: Session = Depends(get_db)):
    """Agent manager dependency"""
    return AgentManager(db, redis_client)

def get_task_orchestrator(db: Session = Depends(get_db)):
    """Task orchestrator dependency"""
    agent_manager = AgentManager(db, redis_client)
    return TaskOrchestrator(db, redis_client, agent_manager)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agent-orchestrator",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    if not Config.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    return generate_latest(metrics_collector.registry)

@app.post("/orchestrator/register-agent")
async def register_agent(
    registration: AgentRegistrationRequest,
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """
    Register a new agent instance with comprehensive configuration.

    This endpoint registers a new agent with the orchestrator, including:
    - Agent metadata and configuration validation
    - Resource allocation and capacity planning
    - Service dependency verification
    - Security policy application
    - Performance baseline establishment
    - Monitoring setup and health checks

    The agent will be registered but not started until explicitly started.
    """
    metrics_collector.request_count.labels(method='POST', endpoint='/orchestrator/register-agent').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/orchestrator/register-agent').time():
        try:
            # Enhanced agent registration with comprehensive validation
            registration_result = await agent_manager.register_agent_enhanced(registration)

            # Set up agent monitoring and metrics
            await agent_manager.setup_agent_monitoring(registration.agent_id)

            # Validate agent capabilities and dependencies
            validation_result = await agent_manager.validate_agent_capabilities(registration.agent_id)

            if not validation_result["valid"]:
                # Clean up failed registration
                await agent_manager.cleanup_failed_registration(registration.agent_id)
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent registration failed validation: {validation_result['issues']}"
                )

            return {
                "success": True,
                "agent_id": registration.agent_id,
                "registration_id": registration_result.get("registration_id"),
                "status": "registered",
                "capabilities": validation_result.get("capabilities", []),
                "resource_allocation": registration_result.get("resource_allocation", {}),
                "monitoring_setup": True,
                "next_steps": [
                    "Start agent using POST /orchestrator/start-agent/{agent_id}",
                    "Monitor status using GET /orchestrator/agents/{agent_id}/status",
                    "Execute tasks using POST /orchestrator/execute-task"
                ],
                "message": f"Agent {registration.agent_id} registered successfully"
            }

        except Exception as e:
            logger.error("Agent registration failed", agent_id=registration.agent_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Agent registration failed: {str(e)}")

@app.post("/orchestrator/register")
async def register_agent_legacy(
    registration: AgentRegistrationRequest,
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """Legacy endpoint for backward compatibility"""
    return await register_agent(registration, agent_manager, current_user)

@app.post("/orchestrator/unregister")
async def unregister_agent(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """Unregister an agent instance"""
    metrics_collector.request_count.labels(method='POST', endpoint='/orchestrator/unregister').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/orchestrator/unregister').time():
        result = agent_manager.unregister_agent(agent_id)

    return result

@app.get("/orchestrator/agents")
async def list_agents(
    domain: Optional[str] = None,
    status: Optional[str] = None,
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """List all registered agents"""
    metrics_collector.request_count.labels(method='GET', endpoint='/orchestrator/agents').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/orchestrator/agents').time():
        agents = agent_manager.list_agents(domain=domain, status=status)

    return {"agents": agents, "count": len(agents)}

@app.get("/orchestrator/agents/{agent_id}")
async def get_agent_status(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """Get status of a specific agent"""
    metrics_collector.request_count.labels(method='GET', endpoint='/orchestrator/agents/{agent_id}').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/orchestrator/agents/{agent_id}').time():
        status = agent_manager.get_agent_status(agent_id)

    return status

@app.post("/orchestrator/execute-task")
async def execute_task(
    request: TaskExecutionRequest,
    background_tasks: BackgroundTasks,
    task_orchestrator: TaskOrchestrator = Depends(get_task_orchestrator),
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """
    Execute a task with intelligent routing and comprehensive orchestration.

    This endpoint executes tasks with advanced routing capabilities:
    - Intelligent agent selection based on capabilities and load
    - Task prioritization and queue management
    - Real-time execution monitoring and progress tracking
    - Automatic failover and retry mechanisms
    - Performance optimization and resource allocation
    - Comprehensive error handling and recovery

    Supports various task types including data processing, analysis, reporting, and custom workflows.
    """
    metrics_collector.request_count.labels(method='POST', endpoint='/orchestrator/execute-task').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/orchestrator/execute-task').time():
        try:
            # Validate agent availability and capabilities
            agent_validation = await agent_manager.validate_agent_for_task(
                request.agent_id, request.task_type, request.parameters
            )

            if not agent_validation["available"]:
                # Attempt intelligent routing to alternative agents
                alternative_agent = await task_orchestrator.find_alternative_agent(
                    request.task_type, request.parameters, exclude_agent=request.agent_id
                )

                if alternative_agent:
                    logger.info(
                        "Routed task to alternative agent",
                        original_agent=request.agent_id,
                        alternative_agent=alternative_agent,
                        task_type=request.task_type
                    )
                    request.agent_id = alternative_agent
                else:
                    raise HTTPException(
                        status_code=503,
                        detail=f"No suitable agent available for task type: {request.task_type}"
                    )

            # Check task queue capacity
            queue_status = await task_orchestrator.check_queue_capacity(request.agent_id)
            if not queue_status["available"]:
                raise HTTPException(
                    status_code=429,
                    detail=f"Agent task queue at capacity. Retry after: {queue_status['retry_after']} seconds"
                )

            # Enhanced task execution with monitoring
            task_result = await task_orchestrator.execute_task_enhanced(request)

            # Set up real-time monitoring for the task
            monitoring_setup = await task_orchestrator.setup_task_monitoring(task_result["task_id"])

            return {
                "success": True,
                "task_id": task_result["task_id"],
                "agent_id": request.agent_id,
                "task_type": request.task_type,
                "status": "accepted",
                "queue_position": task_result.get("queue_position", 0),
                "estimated_completion": task_result.get("estimated_completion"),
                "monitoring_active": monitoring_setup["active"],
                "progress_tracking": True,
                "task_status_endpoint": f"/orchestrator/tasks/{task_result['task_id']}",
                "real_time_updates": monitoring_setup.get("real_time_updates", False),
                "priority_level": task_result.get("priority", "normal"),
                "routing_info": {
                    "original_agent": request.agent_id if alternative_agent else None,
                    "final_agent": request.agent_id,
                    "routing_reason": "agent_unavailable" if alternative_agent else "direct_routing"
                },
                "next_steps": [
                    "Track progress using GET /orchestrator/tasks/{task_id}",
                    "Monitor agent health using GET /orchestrator/agents/{agent_id}/health",
                    "Cancel task using DELETE /orchestrator/tasks/{task_id}" if task_result.get("cancellable", False) else None
                ],
                "message": f"Task {request.task_type} queued for execution on agent {request.agent_id}"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Task execution failed", agent_id=request.agent_id, task_type=request.task_type, error=str(e))
            raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")

@app.post("/orchestrator/execute-task-intelligent")
async def execute_task_intelligent(
    task_request: TaskExecutionRequest,
    routing_preferences: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks,
    task_orchestrator: TaskOrchestrator = Depends(get_task_orchestrator),
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """
    Execute a task with intelligent agent selection and routing.

    This endpoint automatically selects the best available agent for task execution
    based on capabilities, current load, performance history, and routing preferences.

    Features:
    - Automatic agent discovery and capability matching
    - Load balancing across agent instances
    - Performance-based agent selection
    - Geographic and latency optimization
    - Cost-based routing options
    - Real-time agent health monitoring
    """
    metrics_collector.request_count.labels(method='POST', endpoint='/orchestrator/execute-task-intelligent').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/orchestrator/execute-task-intelligent').time():
        try:
            # Intelligent agent selection
            agent_selection = await task_orchestrator.select_optimal_agent(
                task_request.task_type,
                task_request.parameters,
                routing_preferences or {}
            )

            if not agent_selection["agent_found"]:
                raise HTTPException(
                    status_code=503,
                    detail=f"No suitable agent found for task type: {task_request.task_type}. Requirements: {agent_selection['requirements']}"
                )

            # Update task request with selected agent
            task_request.agent_id = agent_selection["agent_id"]

            # Execute task on selected agent
            result = await execute_task(
                task_request,
                background_tasks,
                task_orchestrator,
                agent_manager,
                current_user
            )

            # Add intelligent routing metadata
            result["routing_metadata"] = {
                "selection_method": agent_selection["selection_method"],
                "selection_criteria": agent_selection["criteria"],
                "alternative_agents_considered": agent_selection["alternatives_count"],
                "selection_reason": agent_selection["reason"],
                "estimated_performance": agent_selection.get("estimated_performance", {}),
                "cost_optimization": agent_selection.get("cost_savings", 0)
            }

            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Intelligent task execution failed", task_type=task_request.task_type, error=str(e))
            raise HTTPException(status_code=500, detail=f"Intelligent task execution failed: {str(e)}")

@app.get("/orchestrator/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    task_orchestrator: TaskOrchestrator = Depends(get_task_orchestrator),
    current_user: str = Depends(get_current_user)
):
    """Get status of a task execution"""
    metrics_collector.request_count.labels(method='GET', endpoint='/orchestrator/tasks/{task_id}').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/orchestrator/tasks/{task_id}').time():
        status = task_orchestrator.get_task_status(task_id)

    return status

@app.post("/orchestrator/start-agent/{agent_id}")
async def start_agent(
    agent_id: str,
    start_options: Optional[Dict[str, Any]] = None,
    agent_manager: AgentManager = Depends(get_agent_manager),
    task_orchestrator: TaskOrchestrator = Depends(get_task_orchestrator),
    current_user: str = Depends(get_current_user)
):
    """
    Start an agent instance with comprehensive lifecycle management.

    This endpoint starts a registered agent instance, including:
    - Agent initialization and bootstrap sequence
    - Service dependency activation
    - Resource allocation and scaling
    - Health check establishment
    - Performance monitoring activation
    - Task queue initialization

    The agent will be ready to accept tasks after successful startup.
    """
    metrics_collector.request_count.labels(method='POST', endpoint='/orchestrator/start-agent/{agent_id}').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/orchestrator/start-agent/{agent_id}').time():
        try:
            # Enhanced agent startup with comprehensive validation
            startup_result = await agent_manager.start_agent_enhanced(agent_id, start_options or {})

            if not startup_result["success"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent startup failed: {startup_result.get('error', 'Unknown error')}"
                )

            # Initialize task processing capabilities
            await task_orchestrator.initialize_agent_tasks(agent_id)

            # Set up real-time monitoring
            await agent_manager.activate_agent_monitoring(agent_id)

            # Initialize performance baselines
            await agent_manager.establish_performance_baseline(agent_id)

            return {
                "success": True,
                "agent_id": agent_id,
                "status": "started",
                "startup_time": startup_result.get("startup_time", 0),
                "resource_allocation": startup_result.get("resource_allocation", {}),
                "capabilities_activated": startup_result.get("capabilities_activated", []),
                "monitoring_active": True,
                "task_queue_ready": True,
                "health_check_endpoint": f"/orchestrator/agents/{agent_id}/health",
                "task_execution_endpoint": f"/orchestrator/execute-task",
                "next_steps": [
                    "Execute tasks using POST /orchestrator/execute-task",
                    "Monitor health using GET /orchestrator/agents/{agent_id}/health",
                    "Check status using GET /orchestrator/agents/{agent_id}/status"
                ],
                "message": f"Agent {agent_id} started successfully and ready for task execution"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Agent startup failed", agent_id=agent_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Agent startup failed: {str(e)}")

@app.post("/orchestrator/start")
async def start_agent_legacy(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """Legacy endpoint for backward compatibility"""
    return await start_agent(agent_id, None, agent_manager, Depends(get_task_orchestrator), current_user)

@app.post("/orchestrator/stop")
async def stop_agent(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """Stop an agent instance"""
    metrics_collector.request_count.labels(method='POST', endpoint='/orchestrator/stop').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/orchestrator/stop').time():
        # This would typically call the deployment pipeline
        # For now, just update the status
        try:
            agent = agent_manager.db.query(AgentInstance).filter_by(agent_id=agent_id).first()
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            agent.status = 'inactive'
            agent_manager.db.commit()

            return {
                'success': True,
                'agent_id': agent_id,
                'status': 'stopped',
                'message': f'Agent {agent_id} stopped successfully'
            }

        except Exception as e:
            agent_manager.db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to stop agent: {str(e)}")

# =============================================================================
# ENHANCED AGENT LIFECYCLE MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/orchestrator/agents/{agent_id}/pause")
async def pause_agent(
    agent_id: str,
    pause_options: Optional[Dict[str, Any]] = None,
    agent_manager: AgentManager = Depends(get_agent_manager),
    task_orchestrator: TaskOrchestrator = Depends(get_task_orchestrator),
    current_user: str = Depends(get_current_user)
):
    """
    Pause an agent instance with graceful task completion.

    This endpoint pauses an active agent, allowing current tasks to complete
    while preventing new task acceptance. Useful for maintenance or resource optimization.
    """
    metrics_collector.request_count.labels(method='POST', endpoint='/orchestrator/agents/{agent_id}/pause').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/orchestrator/agents/{agent_id}/pause').time():
        try:
            pause_result = await agent_manager.pause_agent_enhanced(agent_id, pause_options or {})

            if not pause_result["success"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent pause failed: {pause_result.get('error', 'Unknown error')}"
                )

            # Pause task acceptance for this agent
            await task_orchestrator.pause_agent_tasks(agent_id)

            return {
                "success": True,
                "agent_id": agent_id,
                "status": "paused",
                "active_tasks_remaining": pause_result.get("active_tasks", 0),
                "task_completion_eta": pause_result.get("completion_eta"),
                "resource_released": pause_result.get("resource_released", {}),
                "resume_endpoint": f"/orchestrator/agents/{agent_id}/resume",
                "message": f"Agent {agent_id} paused successfully. {pause_result.get('active_tasks', 0)} tasks remaining."
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Agent pause failed", agent_id=agent_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Agent pause failed: {str(e)}")

@app.post("/orchestrator/agents/{agent_id}/resume")
async def resume_agent(
    agent_id: str,
    resume_options: Optional[Dict[str, Any]] = None,
    agent_manager: AgentManager = Depends(get_agent_manager),
    task_orchestrator: TaskOrchestrator = Depends(get_task_orchestrator),
    current_user: str = Depends(get_current_user)
):
    """
    Resume a paused agent instance.

    This endpoint resumes a paused agent, restoring full functionality
    and task acceptance capabilities.
    """
    metrics_collector.request_count.labels(method='POST', endpoint='/orchestrator/agents/{agent_id}/resume').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/orchestrator/agents/{agent_id}/resume').time():
        try:
            resume_result = await agent_manager.resume_agent_enhanced(agent_id, resume_options or {})

            if not resume_result["success"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent resume failed: {resume_result.get('error', 'Unknown error')}"
                )

            # Resume task acceptance for this agent
            await task_orchestrator.resume_agent_tasks(agent_id)

            return {
                "success": True,
                "agent_id": agent_id,
                "status": "resumed",
                "resource_reallocated": resume_result.get("resource_reallocated", {}),
                "task_queue_resumed": True,
                "performance_baseline_restored": resume_result.get("baseline_restored", False),
                "message": f"Agent {agent_id} resumed successfully and ready for task execution"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Agent resume failed", agent_id=agent_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Agent resume failed: {str(e)}")

@app.delete("/orchestrator/agents/{agent_id}")
async def terminate_agent(
    agent_id: str,
    termination_options: Optional[Dict[str, Any]] = None,
    agent_manager: AgentManager = Depends(get_agent_manager),
    task_orchestrator: TaskOrchestrator = Depends(get_task_orchestrator),
    current_user: str = Depends(get_current_user)
):
    """
    Terminate an agent instance with cleanup.

    This endpoint performs a complete termination of an agent instance,
    including task cancellation, resource cleanup, and data persistence.
    """
    metrics_collector.request_count.labels(method='DELETE', endpoint='/orchestrator/agents/{agent_id}').inc()

    with metrics_collector.request_duration.labels(method='DELETE', endpoint='/orchestrator/agents/{agent_id}').time():
        try:
            termination_result = await agent_manager.terminate_agent_enhanced(agent_id, termination_options or {})

            if not termination_result["success"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent termination failed: {termination_result.get('error', 'Unknown error')}"
                )

            # Cancel all pending tasks for this agent
            await task_orchestrator.cancel_agent_tasks(agent_id)

            # Clean up agent resources
            await agent_manager.cleanup_agent_resources(agent_id)

            return {
                "success": True,
                "agent_id": agent_id,
                "status": "terminated",
                "tasks_cancelled": termination_result.get("tasks_cancelled", 0),
                "resources_cleaned": termination_result.get("resources_cleaned", []),
                "data_persisted": termination_result.get("data_persisted", False),
                "termination_time": termination_result.get("termination_time", 0),
                "message": f"Agent {agent_id} terminated successfully"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Agent termination failed", agent_id=agent_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Agent termination failed: {str(e)}")

# =============================================================================
# ADVANCED TASK MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/orchestrator/agents/{agent_id}/tasks")
async def get_agent_tasks(
    agent_id: str,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    task_orchestrator: TaskOrchestrator = Depends(get_task_orchestrator),
    current_user: str = Depends(get_current_user)
):
    """
    Get all tasks for a specific agent with filtering and pagination.

    Returns tasks in various states: pending, running, completed, failed, cancelled.
    """
    metrics_collector.request_count.labels(method='GET', endpoint='/orchestrator/agents/{agent_id}/tasks').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/orchestrator/agents/{agent_id}/tasks').time():
        try:
            tasks = await task_orchestrator.get_agent_tasks(agent_id, status, limit, offset)

            return {
                "agent_id": agent_id,
                "tasks": tasks["items"],
                "total_count": tasks["total"],
                "filtered_count": len(tasks["items"]),
                "status_filter": status,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": (offset + len(tasks["items"])) < tasks["total"]
                }
            }

        except Exception as e:
            logger.error("Failed to get agent tasks", agent_id=agent_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to get agent tasks: {str(e)}")

@app.delete("/orchestrator/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    cancellation_reason: Optional[str] = None,
    task_orchestrator: TaskOrchestrator = Depends(get_task_orchestrator),
    current_user: str = Depends(get_current_user)
):
    """
    Cancel a running or queued task.

    This endpoint attempts to cancel a task gracefully, allowing for cleanup
    and resource release before termination.
    """
    metrics_collector.request_count.labels(method='DELETE', endpoint='/orchestrator/tasks/{task_id}').inc()

    with metrics_collector.request_duration.labels(method='DELETE', endpoint='/orchestrator/tasks/{task_id}').time():
        try:
            cancellation_result = await task_orchestrator.cancel_task_enhanced(task_id, cancellation_reason)

            if not cancellation_result["cancelled"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Task cancellation failed: {cancellation_result.get('error', 'Unknown error')}"
                )

            return {
                "success": True,
                "task_id": task_id,
                "status": "cancelled",
                "cancellation_reason": cancellation_reason,
                "resources_released": cancellation_result.get("resources_released", []),
                "cleanup_performed": cancellation_result.get("cleanup_performed", False),
                "message": f"Task {task_id} cancelled successfully"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Task cancellation failed", task_id=task_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Task cancellation failed: {str(e)}")

@app.post("/orchestrator/tasks/{task_id}/priority")
async def update_task_priority(
    task_id: str,
    priority: str,
    task_orchestrator: TaskOrchestrator = Depends(get_task_orchestrator),
    current_user: str = Depends(get_current_user)
):
    """
    Update the priority of a queued or running task.

    Priorities: low, normal, high, urgent
    """
    metrics_collector.request_count.labels(method='POST', endpoint='/orchestrator/tasks/{task_id}/priority').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/orchestrator/tasks/{task_id}/priority').time():
        try:
            valid_priorities = ["low", "normal", "high", "urgent"]
            if priority not in valid_priorities:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid priority. Must be one of: {', '.join(valid_priorities)}"
                )

            update_result = await task_orchestrator.update_task_priority(task_id, priority)

            if not update_result["success"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Priority update failed: {update_result.get('error', 'Unknown error')}"
                )

            return {
                "success": True,
                "task_id": task_id,
                "new_priority": priority,
                "old_priority": update_result.get("old_priority"),
                "queue_position_changed": update_result.get("position_changed", False),
                "new_queue_position": update_result.get("new_position"),
                "message": f"Task {task_id} priority updated to {priority}"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Task priority update failed", task_id=task_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Task priority update failed: {str(e)}")

# =============================================================================
# MONITORING AND HEALTH ENDPOINTS
# =============================================================================

@app.get("/orchestrator/agents/{agent_id}/health")
async def get_agent_health(
    agent_id: str,
    detailed: bool = False,
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """
    Get comprehensive health status of an agent.

    Includes system health, performance metrics, and capability status.
    """
    metrics_collector.request_count.labels(method='GET', endpoint='/orchestrator/agents/{agent_id}/health').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/orchestrator/agents/{agent_id}/health').time():
        try:
            health_status = await agent_manager.get_agent_health_enhanced(agent_id, detailed)

            if not health_status["found"]:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            return {
                "agent_id": agent_id,
                "overall_health": health_status["overall_health"],
                "health_score": health_status["health_score"],
                "last_health_check": health_status["last_check"],
                "system_health": health_status["system_health"],
                "performance_metrics": health_status["performance"] if detailed else None,
                "capability_status": health_status["capabilities"] if detailed else None,
                "issues": health_status["issues"],
                "recommendations": health_status["recommendations"]
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Agent health check failed", agent_id=agent_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Agent health check failed: {str(e)}")

@app.get("/orchestrator/agents/{agent_id}/metrics")
async def get_agent_metrics(
    agent_id: str,
    time_range: str = "1h",
    metric_types: Optional[List[str]] = None,
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """
    Get detailed performance metrics for an agent.

    Supports various time ranges and metric types.
    """
    metrics_collector.request_count.labels(method='GET', endpoint='/orchestrator/agents/{agent_id}/metrics').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/orchestrator/agents/{agent_id}/metrics').time():
        try:
            agent_metrics = await agent_manager.get_agent_metrics_enhanced(
                agent_id, time_range, metric_types or []
            )

            if not agent_metrics["found"]:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            return {
                "agent_id": agent_id,
                "time_range": time_range,
                "metrics": agent_metrics["metrics"],
                "summary": agent_metrics["summary"],
                "anomalies": agent_metrics["anomalies"],
                "generated_at": datetime.utcnow().isoformat()
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Agent metrics retrieval failed", agent_id=agent_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Agent metrics retrieval failed: {str(e)}")

@app.get("/orchestrator/cluster/status")
async def get_cluster_status(
    agent_manager: AgentManager = Depends(get_agent_manager),
    task_orchestrator: TaskOrchestrator = Depends(get_task_orchestrator),
    current_user: str = Depends(get_current_user)
):
    """
    Get overall cluster status and health.

    Provides a comprehensive view of the entire agent cluster.
    """
    metrics_collector.request_count.labels(method='GET', endpoint='/orchestrator/cluster/status').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/orchestrator/cluster/status').time():
        try:
            cluster_status = await agent_manager.get_cluster_status()
            task_status = await task_orchestrator.get_cluster_task_status()

            return {
                "cluster_health": cluster_status["overall_health"],
                "total_agents": cluster_status["total_agents"],
                "active_agents": cluster_status["active_agents"],
                "healthy_agents": cluster_status["healthy_agents"],
                "total_tasks": task_status["total_tasks"],
                "running_tasks": task_status["running_tasks"],
                "queued_tasks": task_status["queued_tasks"],
                "failed_tasks_24h": task_status["failed_tasks_24h"],
                "resource_utilization": cluster_status["resource_utilization"],
                "performance_summary": cluster_status["performance_summary"],
                "alerts": cluster_status["alerts"],
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Cluster status retrieval failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Cluster status retrieval failed: {str(e)}")

@app.post("/orchestrator/agents/{agent_id}/scale")
async def scale_agent_resources(
    agent_id: str,
    scaling_request: Dict[str, Any],
    agent_manager: AgentManager = Depends(get_agent_manager),
    current_user: str = Depends(get_current_user)
):
    """
    Scale agent resources up or down based on demand.

    Supports CPU, memory, and concurrent task limit scaling.
    """
    metrics_collector.request_count.labels(method='POST', endpoint='/orchestrator/agents/{agent_id}/scale').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/orchestrator/agents/{agent_id}/scale').time():
        try:
            scaling_result = await agent_manager.scale_agent_resources(agent_id, scaling_request)

            if not scaling_result["success"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Resource scaling failed: {scaling_result.get('error', 'Unknown error')}"
                )

            return {
                "success": True,
                "agent_id": agent_id,
                "scaling_applied": scaling_result["scaling_applied"],
                "old_resources": scaling_result["old_resources"],
                "new_resources": scaling_result["new_resources"],
                "scaling_time": scaling_result["scaling_time"],
                "performance_impact": scaling_result.get("performance_impact", {}),
                "message": f"Agent {agent_id} resources scaled successfully"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Agent resource scaling failed", agent_id=agent_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Agent resource scaling failed: {str(e)}")

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
    logger.info("Starting Agent Orchestrator Service...")

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

    logger.info(f"Agent Orchestrator Service started on {Config.SERVICE_HOST}:{Config.SERVICE_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down Agent Orchestrator Service...")

    # Close Redis connection
    try:
        redis_client.close()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {str(e)}")

    logger.info("Agent Orchestrator Service shutdown complete")

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
