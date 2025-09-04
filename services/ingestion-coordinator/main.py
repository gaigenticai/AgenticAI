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
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest
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
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://agentic_user:agentic123@postgresql_ingestion:5432/agentic_ingestion")
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

# Prometheus metrics - temporarily disabled to avoid conflicts
# INGESTION_JOBS_TOTAL = Counter('ingestion_coordinator_jobs_total', 'Total number of ingestion jobs', ['source_type', 'status'])
# INGESTION_RECORDS_PROCESSED = Counter('ingestion_coordinator_records_processed', 'Total records processed', ['source_type'])
# INGESTION_JOB_DURATION = Histogram('ingestion_coordinator_job_duration_seconds', 'Job duration in seconds', ['source_type'])
# ACTIVE_INGESTION_JOBS = Gauge('ingestion_coordinator_active_jobs', 'Number of active ingestion jobs')

# Placeholder metrics (will be enabled after fixing registry conflicts)
INGESTION_JOBS_TOTAL = None
INGESTION_RECORDS_PROCESSED = None
INGESTION_JOB_DURATION = None
ACTIVE_INGESTION_JOBS = None

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
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
        credentials = pika.PlainCredentials(
            os.getenv("RABBITMQ_USER", "agentic_user"),
            os.getenv("RABBITMQ_PASSWORD", "agentic123")
        )
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
    """Create a new ingestion job"""
    job_id = str(uuid.uuid4())

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

    if INGESTION_JOBS_TOTAL:
        INGESTION_JOBS_TOTAL.labels(source_type=source_config.source_type, status="created").inc()
    if ACTIVE_INGESTION_JOBS:
        ACTIVE_INGESTION_JOBS.inc()

    logger.info("Ingestion job created", job_id=job_id, source_type=source_config.source_type)
    return job_id

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
    User login endpoint - available when REQUIRE_AUTH=true

    Rule 10: Authentication is controlled by REQUIRE_AUTH environment variable
    """
    require_auth = os.getenv("REQUIRE_AUTH", "false").lower() == "true"

    if not require_auth:
        raise HTTPException(
            status_code=403,
            detail="Authentication not enabled. Set REQUIRE_AUTH=true to enable login."
        )

    # Check user credentials (simplified - use proper password hashing in production)
    user = db.query(User).filter(
        (User.username == user_credentials.username) |
        (User.email == user_credentials.username)
    ).first()

    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Verify password (simplified - use proper password verification)
    if user.password_hash != user_credentials.password:  # In production: check_password_hash
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate JWT token
    payload = {
        "user_id": str(user.user_id),
        "username": user.username,
        "email": user.email,
        "roles": ["user"],  # Simplified - get from database in production
        "exp": int(time.time()) + (int(os.getenv("JWT_EXPIRATION_HOURS", "24")) * 3600)
    }

    secret = os.getenv("JWT_SECRET", "default-secret-key-change-in-production")

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

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

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
    """Prometheus metrics endpoint"""
    return generate_latest()

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
    """Get job status"""
    job = db.query(IngestionJob).filter(IngestionJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

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
    Base.metadata.create_all(bind=engine)

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
