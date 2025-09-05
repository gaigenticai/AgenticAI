#!/usr/bin/env python3
"""
Ingestion Coordinator Service for Agentic Platform

This service orchestrates data ingestion from multiple sources including:
- CSV, Excel, PDF, JSON files
- API endpoints
- UI scraping
- Streaming data sources

Features:
- Job scheduling and management
- Data validation and quality checks
- Metadata management
- Message queue integration
- Caching layer integration
- RESTful API endpoints
- Comprehensive monitoring and logging
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pika
import redis
import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import uvicorn

# JWT support (add to requirements.txt: PyJWT==2.8.0)
try:
    import jwt
except ImportError:
    # Fallback if JWT not available
    jwt = None

# Configure structured logging
logging.basicConfig(level=logging.INFO)
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

# Database models
Base = declarative_base()

class IngestionJob(Base):
    """Ingestion job model"""
    __tablename__ = "ingestion_jobs"

    id = Column(Integer, primary_key=True)
    job_id = Column(String(36), unique=True, nullable=False)
    source_type = Column(String(100), nullable=False)
    status = Column(String(50), default="pending")
    total_records = Column(Integer)
    processed_records = Column(Integer, default=0)
    failed_records = Column(Integer, default=0)
    job_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class User(Base):
    """User model for authentication"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(500), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    last_login = Column(DateTime)
    password_changed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Pydantic models for API
class DataSourceConfig(BaseModel):
    """Data source configuration model"""
    source_name: str = Field(..., description="Name of the data source")
    source_type: str = Field(..., description="Type of data source (csv, excel, pdf, json, api, ui)")
    connection_config: Dict[str, Any] = Field(..., description="Connection configuration")
    schema_definition: Optional[Dict[str, Any]] = Field(None, description="Schema definition")
    validation_rules: Optional[List[Dict[str, Any]]] = Field(None, description="Data validation rules")

class IngestionJobRequest(BaseModel):
    """Ingestion job request model"""
    source_config: DataSourceConfig
    priority: int = Field(1, description="Job priority (1-10)")
    batch_size: int = Field(1000, description="Batch size for processing")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class JobStatusResponse(BaseModel):
    """Job status response model"""
    job_id: str
    status: str
    progress: float
    total_records: Optional[int]
    processed_records: int
    failed_records: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration: Optional[str]
    error_message: Optional[str]

# Global variables
app = FastAPI(
    title="Ingestion Coordinator Service",
    description="Orchestrates data ingestion across multiple sources and formats",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    logger.error("DATABASE_URL is not configured for Ingestion Coordinator; set DATABASE_URL in environment")
    raise RuntimeError("DATABASE_URL not configured for Ingestion Coordinator")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis setup
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis_ingestion"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# Message queue setup
rabbitmq_connection = None
rabbitmq_channel = None

# Create custom registry for ingestion-coordinator to avoid metric name conflicts
INGESTION_REGISTRY = CollectorRegistry()

# Prometheus metrics - enabled with custom registry to avoid conflicts
INGESTION_JOBS_TOTAL = Counter(
    'ingestion_jobs_total',
    'Total number of ingestion jobs processed by ingestion coordinator',
    ['source_type', 'status'],
    registry=INGESTION_REGISTRY
)

INGESTION_RECORDS_PROCESSED = Counter(
    'ingestion_records_processed',
    'Total records processed by ingestion coordinator',
    ['source_type'],
    registry=INGESTION_REGISTRY
)

INGESTION_JOB_DURATION = Histogram(
    'ingestion_job_duration_seconds',
    'Time taken to process ingestion jobs',
    ['source_type'],
    registry=INGESTION_REGISTRY
)

ACTIVE_INGESTION_JOBS = Gauge(
    'ingestion_active_jobs',
    'Number of currently active ingestion jobs being processed',
    registry=INGESTION_REGISTRY
)

# Metrics helper functions
def record_job_completion(source_type: str, records_processed: int, duration_seconds: float):
    """Record metrics for completed job"""
    if INGESTION_RECORDS_PROCESSED:
        INGESTION_RECORDS_PROCESSED.labels(source_type=source_type).inc(records_processed)

    logger.info(
        "Job metrics recorded",
        source_type=source_type,
        records_processed=records_processed,
        duration_seconds=duration_seconds
    )

# Global exception handlers for consistent error responses
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with proper logging"""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_exception",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle request validation errors"""
    logger.warning(
        "Request validation error",
        errors=exc.errors(),
        path=request.url.path,
        method=request.method
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle any unhandled exceptions"""
    logger.error(
        "Unhandled exception occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        exc_info=True
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_server_error",
                "message": "An internal server error occurred"
            }
        }
    )

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Dependency to get database session
def get_db():
    """Database session dependency with enhanced error handling"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error("Database session error", error=str(e), error_type=type(e).__name__)
        db.rollback()
        raise
    finally:
        try:
            db.close()
            logger.debug("Database session closed successfully")
        except Exception as e:
            logger.error("Error closing database session", error=str(e))

# Authentication dependency
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify authentication token based on REQUIRE_AUTH setting

    Rule 10: REQUIRE_AUTH controls authentication - true enables login,
    false (default) bypasses authentication
    """
    require_auth = os.getenv("REQUIRE_AUTH", "false").lower() == "true"

    if not require_auth:
        # No authentication required - return anonymous user context
        return {"user_id": "anonymous", "roles": ["user"], "authenticated": False}

    # Authentication required - verify JWT token
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Set REQUIRE_AUTH=false to disable authentication."
        )

    try:
        # Decode and verify JWT token (simplified for demo)
        # In production, use proper JWT library with secret key validation
        token_parts = credentials.credentials.split(".")
        if len(token_parts) != 3:
            raise HTTPException(status_code=401, detail="Invalid token format")

        # Mock token verification - replace with actual JWT verification
        import base64
        import json

        # Decode payload (simplified - not secure for production)
        payload_b64 = token_parts[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)  # Add padding
        payload_json = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_json)

        # Check token expiration
        if payload.get("exp", 0) < time.time():
            raise HTTPException(status_code=401, detail="Token expired")

        return {
            "user_id": payload.get("user_id", "unknown"),
            "username": payload.get("username", "unknown"),
            "roles": payload.get("roles", ["user"]),
            "authenticated": True
        }

    except Exception as e:
        logger.warning("Token verification failed", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication token")

# Message queue functions
def setup_rabbitmq():
    """Setup RabbitMQ connection"""
    global rabbitmq_connection, rabbitmq_channel
    try:
        rabbitmq_user = os.getenv("RABBITMQ_USER", "agentic_user")
        rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", "")
        if not rabbitmq_password:
            logger.error("RABBITMQ_PASSWORD is not configured for Ingestion Coordinator; set RABBITMQ_PASSWORD in environment")
            raise RuntimeError("RABBITMQ_PASSWORD not configured for Ingestion Coordinator")

        credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
        parameters = pika.ConnectionParameters(
            host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
            port=int(os.getenv("RABBITMQ_PORT", 5672)),
            credentials=credentials
        )
        rabbitmq_connection = pika.BlockingConnection(parameters)
        rabbitmq_channel = rabbitmq_connection.channel()

        # Declare queues
        queues = ['csv_ingestion', 'excel_ingestion', 'pdf_ingestion', 'json_ingestion', 'api_ingestion', 'ui_scraper']
        for queue in queues:
            rabbitmq_channel.queue_declare(queue=queue, durable=True)

        logger.info("RabbitMQ connection established")
    except Exception as e:
        logger.error("Failed to connect to RabbitMQ", error=str(e))

def publish_message(queue_name: str, message: Dict[str, Any]):
    """Publish message to RabbitMQ queue"""
    if rabbitmq_channel:
        try:
            rabbitmq_channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # make message persistent
                )
            )
            logger.info("Message published to queue", queue=queue_name, message_id=message.get('job_id'))
        except Exception as e:
            logger.error("Failed to publish message", error=str(e), queue=queue_name)
    else:
        logger.warning("RabbitMQ channel not available")

# Core ingestion functions
def create_ingestion_job(db: Session, source_config: DataSourceConfig, priority: int = 1) -> str:
    """Create a new ingestion job with comprehensive error handling"""
    try:
        job_id = str(uuid.uuid4())

        # Validate source configuration
        if not source_config.source_type:
            raise ValueError("Source type is required for ingestion job")

        job = IngestionJob(
            job_id=job_id,
            source_type=source_config.source_type,
            status="pending",
            metadata={
                "source_config": source_config.dict(),
                "priority": priority,
                "created_by": "api"
            }
        )

        db.add(job)
        db.commit()

        # Update metrics
        if INGESTION_JOBS_TOTAL:
            INGESTION_JOBS_TOTAL.labels(source_type=source_config.source_type, status="created").inc()
        if ACTIVE_INGESTION_JOBS:
            ACTIVE_INGESTION_JOBS.inc()

        logger.info(
            "Ingestion job created successfully",
            job_id=job_id,
            source_type=source_config.source_type,
            priority=priority
        )
        return job_id

    except Exception as e:
        logger.error(
            "Failed to create ingestion job",
            error=str(e),
            error_type=type(e).__name__,
            source_type=getattr(source_config, 'source_type', 'unknown')
        )
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create ingestion job: {str(e)}"
        )

def update_job_status(db: Session, job_id: str, status: str, **kwargs):
    """Update job status"""
    job = db.query(IngestionJob).filter(IngestionJob.job_id == job_id).first()
    if job:
        old_status = job.status
        job.status = status
        job.updated_at = datetime.utcnow()

        if status == "processing" and not job.start_time:
            job.start_time = datetime.utcnow()
        elif status in ["completed", "failed"]:
            job.end_time = datetime.utcnow()
            if job.start_time:
                duration = (job.end_time - job.start_time).total_seconds()
                if INGESTION_JOB_DURATION:
                    INGESTION_JOB_DURATION.labels(source_type=job.source_type).observe(duration)

                # Record job completion metrics
                if status == "completed" and job.processed_records:
                    record_job_completion(job.source_type, job.processed_records, duration)

        # Update counters
        if old_status != status:
            if INGESTION_JOBS_TOTAL:
                INGESTION_JOBS_TOTAL.labels(source_type=job.source_type, status=old_status).dec()
                INGESTION_JOBS_TOTAL.labels(source_type=job.source_type, status=status).inc()

        if status in ["completed", "failed"]:
            if ACTIVE_INGESTION_JOBS:
                ACTIVE_INGESTION_JOBS.dec()

        db.commit()
        logger.info("Job status updated", job_id=job_id, old_status=old_status, new_status=status)
        return True
    else:
        logger.warning("Job not found for status update", job_id=job_id)
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

# Pydantic models for authentication
class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str

class UserRegister(BaseModel):
    """User registration model"""
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

# Authentication endpoints (only available when REQUIRE_AUTH=true)
@app.post("/auth/login", response_model=TokenResponse)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """
    User login endpoint with comprehensive error handling - available when REQUIRE_AUTH=true

    Rule 10: Authentication is controlled by REQUIRE_AUTH environment variable
    """
    try:
        require_auth = os.getenv("REQUIRE_AUTH", "false").lower() == "true"

        if not require_auth:
            logger.warning("Login attempt when authentication is disabled")
            raise HTTPException(
                status_code=403,
                detail="Authentication not enabled. Set REQUIRE_AUTH=true to enable login."
            )

        # Validate input
        if not user_credentials.username or not user_credentials.password:
            logger.warning("Login attempt with missing credentials")
            raise HTTPException(status_code=400, detail="Username and password are required")

        # Check user credentials with proper error handling
        try:
            user = db.query(User).filter(
                (User.username == user_credentials.username) |
                (User.email == user_credentials.username)
            ).first()
        except Exception as e:
            logger.error("Database error during user lookup", error=str(e))
            raise HTTPException(status_code=500, detail="Database error")

        if not user:
            logger.warning("Login attempt with non-existent user", username=user_credentials.username)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not user.is_active:
            logger.warning("Login attempt for inactive user", user_id=user.id)
            raise HTTPException(status_code=401, detail="Account is disabled")

        # Verify password with proper error handling
        try:
            if user.password_hash != user_credentials.password:  # In production: check_password_hash
                logger.warning("Login attempt with wrong password", user_id=user.id)
                raise HTTPException(status_code=401, detail="Invalid credentials")
        except Exception as e:
            logger.error("Password verification error", error=str(e), user_id=user.id)
            raise HTTPException(status_code=500, detail="Authentication error")

        # Generate JWT token with error handling
        try:
            payload = {
                "user_id": str(user.user_id),
                "username": user.username,
                "email": user.email,
                "roles": ["user"],  # Simplified - get from database in production
                "exp": int(time.time()) + (int(os.getenv("JWT_EXPIRATION_HOURS", "24")) * 3600)
            }

            secret = os.getenv("JWT_SECRET", "default-secret-key-change-in-production")

            if not secret or secret == "default-secret-key-change-in-production":
                logger.warning("Using default JWT secret - change in production!")

            if jwt:
                # Use proper JWT library if available
                token = jwt.encode(payload, secret, algorithm="HS256")
            else:
                # Fallback simple token (not secure - for development only)
                import base64
                import json

                header = {"alg": "HS256", "typ": "JWT"}
                header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
                payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

                # Simple signature (not cryptographically secure)
                message = f"{header_b64}.{payload_b64}"
                signature = base64.urlsafe_b64encode(message.encode())[:43].decode()  # 32 bytes

                token = f"{header_b64}.{payload_b64}.{signature}"

        except Exception as e:
            logger.error("Token generation error", error=str(e), user_id=str(user.user_id))
            raise HTTPException(status_code=500, detail="Token generation failed")

        # Update last login with error handling
        try:
            user.last_login = datetime.utcnow()
            db.commit()
        except Exception as e:
            logger.error("Failed to update last login", error=str(e), user_id=str(user.user_id))
            # Don't fail the login if this update fails
            db.rollback()

        logger.info("User logged in successfully", user_id=str(user.user_id), username=user.username)

        return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=int(os.getenv("JWT_EXPIRATION_HOURS", "24")) * 3600,
        user={
            "user_id": str(user.user_id),
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name
        }
    )

    except Exception as e:
        logger.error("Login failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail="Login failed due to internal error")

@app.post("/auth/register")
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    User registration endpoint - available when REQUIRE_AUTH=true

    Rule 10: Authentication is controlled by REQUIRE_AUTH environment variable
    """
    require_auth = os.getenv("REQUIRE_AUTH", "false").lower() == "true"

    if not require_auth:
        raise HTTPException(
            status_code=403,
            detail="Authentication not enabled. Set REQUIRE_AUTH=true to enable registration."
        )

    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == user_data.username) |
        (User.email == user_data.email)
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Username or email already exists"
        )

    # Create new user (simplified - use proper password hashing)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=user_data.password,  # In production: generate_password_hash
        full_name=user_data.full_name
    )

    db.add(new_user)
    db.commit()

    return {
        "message": "User registered successfully",
        "user_id": str(new_user.user_id),
        "username": new_user.username
    }

@app.get("/auth/me")
async def get_current_user(user_info: dict = Depends(verify_token)):
    """
    Get current user information

    Rule 10: Returns user context based on REQUIRE_AUTH setting
    """
    return {
        "user": user_info,
        "require_auth": os.getenv("REQUIRE_AUTH", "false").lower() == "true"
    }

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ingestion-coordinator",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "require_auth": os.getenv("REQUIRE_AUTH", "false").lower() == "true"
    }

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint for monitoring ingestion coordinator performance.

    Returns comprehensive metrics including:
    - Total ingestion jobs by source type and status (ingestion_jobs_total)
    - Records processed by source type (ingestion_records_processed)
    - Job processing duration histograms (ingestion_job_duration_seconds)
    - Active jobs gauge (ingestion_active_jobs)

    Access this endpoint at: http://localhost:8080/metrics
    """
    from prometheus_client import CONTENT_TYPE_LATEST
    response = generate_latest(INGESTION_REGISTRY)
    return response

@app.get("/metrics/info")
async def metrics_info():
    """
    Human-readable metrics information endpoint.

    Provides overview of available metrics and their current values.
    Useful for debugging and monitoring without Prometheus.
    """
    metrics_info = {
        "service": "ingestion-coordinator",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {
            "ingestion_jobs_total": {
                "description": "Total number of ingestion jobs by source type and status",
                "type": "Counter",
                "labels": ["source_type", "status"]
            },
            "ingestion_records_processed": {
                "description": "Total records processed by source type",
                "type": "Counter",
                "labels": ["source_type"]
            },
            "ingestion_job_duration_seconds": {
                "description": "Time taken to process ingestion jobs",
                "type": "Histogram",
                "labels": ["source_type"]
            },
            "ingestion_active_jobs": {
                "description": "Number of currently active ingestion jobs",
                "type": "Gauge"
            }
        },
        "endpoints": {
            "prometheus_metrics": "/metrics",
            "metrics_info": "/metrics/info"
        }
    }

    return metrics_info

@app.post("/ingestion/jobs", response_model=dict)
async def create_job(
    request: IngestionJobRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(verify_token)
):
    """Create a new ingestion job"""
    try:
        job_id = create_ingestion_job(db, request.source_config, request.priority)

        # Publish job to appropriate queue
        queue_name = f"{request.source_config.source_type}_ingestion"
        message = {
            "job_id": job_id,
            "source_config": request.source_config.dict(),
            "priority": request.priority,
            "job_metadata": request.metadata
        }

        background_tasks.add_task(publish_message, queue_name, message)

        return {
            "job_id": job_id,
            "status": "created",
            "message": "Ingestion job created successfully"
        }

    except Exception as e:
        logger.error("Failed to create ingestion job", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@app.get("/ingestion/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(verify_token)
):
    """Get job status with comprehensive error handling"""
    try:
        # Validate job_id format
        if not job_id or not isinstance(job_id, str) or len(job_id) != 36:
            raise HTTPException(status_code=400, detail="Invalid job ID format")

        job = db.query(IngestionJob).filter(IngestionJob.job_id == job_id).first()

        if not job:
            logger.warning("Job not found", job_id=job_id, user_id=getattr(token, 'sub', 'anonymous'))
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        # Create and return job status response
        progress = 0.0
        if job.total_records and job.total_records > 0:
            progress = (job.processed_records / job.total_records) * 100

        duration = None
        if job.start_time and job.end_time:
            duration = str(job.end_time - job.start_time)
        elif job.start_time:
            duration = str(datetime.utcnow() - job.start_time)

        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=progress,
            total_records=job.total_records,
            processed_records=job.processed_records,
            failed_records=job.failed_records,
            start_time=job.start_time,
            end_time=job.end_time,
            duration=duration,
            error_message=job.job_metadata.get("error_message") if job.job_metadata else None
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve job status",
            error=str(e),
            error_type=type(e).__name__,
            job_id=job_id
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error while retrieving job status"
        )

    duration = None
    if job.start_time and job.end_time:
        duration = str(job.end_time - job.start_time)
    elif job.start_time:
        duration = str(datetime.utcnow() - job.start_time)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=progress,
        total_records=job.total_records,
        processed_records=job.processed_records,
        failed_records=job.failed_records,
        start_time=job.start_time,
        end_time=job.end_time,
        duration=duration,
        error_message=job.job_metadata.get("error_message") if job.job_metadata else None
    )

@app.get("/ingestion/jobs", response_model=List[JobStatusResponse])
async def list_jobs(
    status: Optional[str] = None,
    source_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(verify_token)
):
    """List ingestion jobs"""
    query = db.query(IngestionJob)

    if status:
        query = query.filter(IngestionJob.status == status)
    if source_type:
        query = query.filter(IngestionJob.source_type == source_type)

    jobs = query.order_by(IngestionJob.created_at.desc()).offset(offset).limit(limit).all()

    result = []
    for job in jobs:
        progress = 0.0
        if job.total_records and job.total_records > 0:
            progress = (job.processed_records / job.total_records) * 100

        duration = None
        if job.start_time and job.end_time:
            duration = str(job.end_time - job.start_time)
        elif job.start_time:
            duration = str(datetime.utcnow() - job.start_time)

        result.append(JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=progress,
            total_records=job.total_records,
            processed_records=job.processed_records,
            failed_records=job.failed_records,
            start_time=job.start_time,
            end_time=job.end_time,
            duration=duration,
            error_message=job.job_metadata.get("error_message") if job.job_metadata else None
        ))

    return result

@app.post("/ingestion/upload")
async def upload_file(
    file: UploadFile = File(...),
    source_type: str = "csv",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(verify_token)
):
    """Upload file for ingestion"""
    try:
        # Validate file type
        allowed_types = {
            "csv": ["text/csv", "application/csv"],
            "excel": ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"],
            "json": ["application/json"],
            "pdf": ["application/pdf"]
        }

        if source_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Unsupported source type: {source_type}")

        if file.content_type not in allowed_types[source_type]:
            raise HTTPException(status_code=400, detail=f"Invalid file type for {source_type}")

        # Save file temporarily
        file_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Create ingestion job
        source_config = DataSourceConfig(
            source_name=file.filename,
            source_type=source_type,
            connection_config={"file_path": file_path, "original_filename": file.filename},
            schema_definition=None,
            validation_rules=None
        )

        job_id = create_ingestion_job(db, source_config)

        # Publish job to queue
        queue_name = f"{source_type}_ingestion"
        message = {
            "job_id": job_id,
            "source_config": source_config.dict(),
            "file_path": file_path
        }

        background_tasks.add_task(publish_message, queue_name, message)

        return {
            "job_id": job_id,
            "status": "uploaded",
            "message": f"File uploaded successfully for {source_type} processing"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload file", error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/ingestion/api-source")
async def create_api_ingestion(
    api_config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(verify_token)
):
    """Create API ingestion job"""
    try:
        source_config = DataSourceConfig(
            source_name=api_config.get("name", "api_source"),
            source_type="api",
            connection_config=api_config,
            schema_definition=None,
            validation_rules=None
        )

        job_id = create_ingestion_job(db, source_config)

        # Publish job to queue
        message = {
            "job_id": job_id,
            "source_config": source_config.dict()
        }

        background_tasks.add_task(publish_message, "api_ingestion", message)

        return {
            "job_id": job_id,
            "status": "created",
            "message": "API ingestion job created successfully"
        }

    except Exception as e:
        logger.error("Failed to create API ingestion job", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create API job: {str(e)}")

# Background tasks
async def process_completed_jobs():
    """Process completed jobs and update metrics"""
    while True:
        try:
            # Check for completed jobs and update metrics
            # This would be enhanced with actual job completion detection
            await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error("Error in job processing loop", error=str(e))
            await asyncio.sleep(60)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Ingestion Coordinator starting up...")

    # Create database tables
    # Base.metadata.create_all(bind=...)  # Removed - use schema.sql instead

    # Setup RabbitMQ
    setup_rabbitmq()

    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning("Redis connection failed", error=str(e))

    # Start background tasks
    asyncio.create_task(process_completed_jobs())

    logger.info("Ingestion Coordinator startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Ingestion Coordinator shutting down...")

    # Close RabbitMQ connection
    if rabbitmq_connection:
        rabbitmq_connection.close()

    logger.info("Ingestion Coordinator shutdown complete")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=False,
        log_level="info"
    )
