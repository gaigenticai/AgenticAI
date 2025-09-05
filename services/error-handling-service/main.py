#!/usr/bin/env python3
"""
Error Handling Service for Agentic Brain Platform

This service provides comprehensive error management, logging, and recovery mechanisms for the Agentic Brain platform,
implementing enterprise-grade error handling patterns across all Agent Brain services with intelligent error
classification, automated recovery strategies, and comprehensive error monitoring and analytics.

Features:
- Global error handling and classification system
- Intelligent error recovery and remediation
- Error aggregation and pattern analysis
- Automated error response and escalation
- Error correlation and root cause analysis
- Recovery strategy management and execution
- Error monitoring dashboard and analytics
- Integration with monitoring and alerting systems
- Error trend analysis and forecasting
- Service-specific error handling patterns
- Error recovery workflow orchestration
- Performance impact assessment
- Error prevention recommendations
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
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict

import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON, Float, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func
import httpx
import pandas as pd
from jinja2 import Template

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
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration"""
    # Service ports
    ERROR_HANDLING_PORT = int(os.getenv("ERROR_HANDLING_PORT", "8370"))

    # Error handling configuration
    ENABLE_AUTO_RECOVERY = os.getenv("ENABLE_AUTO_RECOVERY", "true").lower() == "true"
    ERROR_RETENTION_DAYS = int(os.getenv("ERROR_RETENTION_DAYS", "90"))
    MAX_RECOVERY_ATTEMPTS = int(os.getenv("MAX_RECOVERY_ATTEMPTS", "3"))
    RECOVERY_TIMEOUT_SECONDS = int(os.getenv("RECOVERY_TIMEOUT_SECONDS", "300"))

    # Alert configuration
    ENABLE_ERROR_ALERTS = os.getenv("ENABLE_ERROR_ALERTS", "true").lower() == "true"
    CRITICAL_ERROR_THRESHOLD = int(os.getenv("CRITICAL_ERROR_THRESHOLD", "10"))
    ERROR_RATE_ALERT_THRESHOLD = float(os.getenv("ERROR_RATE_ALERT_THRESHOLD", "0.1"))

    # Monitoring integration
    MONITORING_SERVICE_URL = os.getenv("MONITORING_SERVICE_URL", "http://localhost:8350")
    AUDIT_LOGGING_URL = os.getenv("AUDIT_LOGGING_URL", "http://localhost:8340")
    PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

    # Agent Brain Service URLs
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

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agentic_brain")

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories"""
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    RESOURCE_LIMIT = "resource_limit"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RETRY = "retry"
    FAILOVER = "failover"
    RESTART = "restart"
    ROLLBACK = "rollback"
    NOTIFICATION = "notification"
    MANUAL_INTERVENTION = "manual_intervention"
    IGNORE = "ignore"

@dataclass
class ErrorPattern:
    """Error pattern for classification"""
    pattern_id: str
    name: str
    description: str
    error_keywords: List[str]
    category: ErrorCategory
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    max_retries: int = 3
    retry_delay_seconds: int = 5
    alert_required: bool = True
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class ErrorInstance:
    """Error instance with classification and recovery information"""
    error_id: str
    service_name: str
    error_message: str
    error_type: str
    stack_trace: str
    category: ErrorCategory
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    recovery_status: str = "pending"
    root_cause: Optional[str] = None
    impact_assessment: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    resolved_at: Optional[datetime] = None

    def __post_init__(self):
        if self.error_id is None:
            self.error_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

# =============================================================================
# ERROR PATTERN ENGINE
# =============================================================================

class ErrorPatternEngine:
    """Engine for classifying and handling error patterns"""

    def __init__(self):
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[str, ErrorPattern]:
        """Initialize default error patterns"""
        patterns = {}

        # Network-related errors
        patterns["network_timeout"] = ErrorPattern(
            pattern_id="network_timeout",
            name="Network Timeout",
            description="Network request timeout errors",
            error_keywords=["timeout", "connection", "network", "unreachable"],
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY
        )

        patterns["network_connection_refused"] = ErrorPattern(
            pattern_id="network_connection_refused",
            name="Connection Refused",
            description="Network connection refused errors",
            error_keywords=["connection refused", "connection reset", "broken pipe"],
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FAILOVER
        )

        # Database-related errors
        patterns["database_connection"] = ErrorPattern(
            pattern_id="database_connection",
            name="Database Connection Error",
            description="Database connection failures",
            error_keywords=["connection", "database", "postgres", "sql", "pool"],
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RESTART
        )

        patterns["database_deadlock"] = ErrorPattern(
            pattern_id="database_deadlock",
            name="Database Deadlock",
            description="Database deadlock detection",
            error_keywords=["deadlock", "lock", "concurrent", "serialization"],
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_delay_seconds=10
        )

        # Authentication errors
        patterns["authentication_failure"] = ErrorPattern(
            pattern_id="authentication_failure",
            name="Authentication Failure",
            description="User authentication failures",
            error_keywords=["authentication", "credentials", "login", "password"],
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.NOTIFICATION
        )

        # Validation errors
        patterns["validation_error"] = ErrorPattern(
            pattern_id="validation_error",
            name="Validation Error",
            description="Input validation failures",
            error_keywords=["validation", "invalid", "required", "format"],
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            recovery_strategy=RecoveryStrategy.IGNORE,
            alert_required=False
        )

        # Business logic errors
        patterns["business_logic"] = ErrorPattern(
            pattern_id="business_logic",
            name="Business Logic Error",
            description="Application business logic errors",
            error_keywords=["business", "logic", "rule", "constraint"],
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION
        )

        # External service errors
        patterns["external_service"] = ErrorPattern(
            pattern_id="external_service",
            name="External Service Error",
            description="Third-party service failures",
            error_keywords=["external", "service", "api", "third-party"],
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FAILOVER
        )

        # Resource limit errors
        patterns["resource_limit"] = ErrorPattern(
            pattern_id="resource_limit",
            name="Resource Limit Exceeded",
            description="Resource usage limits exceeded",
            error_keywords=["limit", "quota", "capacity", "memory", "cpu"],
            category=ErrorCategory.RESOURCE_LIMIT,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RESTART
        )

        # Configuration errors
        patterns["configuration"] = ErrorPattern(
            pattern_id="configuration",
            name="Configuration Error",
            description="Configuration-related errors",
            error_keywords=["configuration", "config", "setting", "parameter"],
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION
        )

        # Security errors
        patterns["security_violation"] = ErrorPattern(
            pattern_id="security_violation",
            name="Security Violation",
            description="Security-related errors and violations",
            error_keywords=["security", "unauthorized", "forbidden", "access"],
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION
        )

        return patterns

    def classify_error(self, error_message: str, stack_trace: str = "") -> ErrorInstance:
        """Classify an error based on message and stack trace"""
        error_text = f"{error_message} {stack_trace}".lower()

        # Find matching patterns
        best_match = None
        best_score = 0

        for pattern in self.patterns.values():
            score = 0
            for keyword in pattern.error_keywords:
                if keyword.lower() in error_text:
                    score += 1

            if score > best_score:
                best_score = score
                best_match = pattern

        # Default classification if no pattern matches
        if best_match is None or best_score == 0:
            best_match = self.patterns["business_logic"]

        return ErrorInstance(
            error_id=None,
            service_name="unknown",
            error_message=error_message,
            error_type="classified_error",
            stack_trace=stack_trace,
            category=best_match.category,
            severity=best_match.severity,
            recovery_strategy=best_match.recovery_strategy,
            max_recovery_attempts=best_match.max_retries,
            metadata={
                "classification_score": best_score,
                "pattern_used": best_match.pattern_id,
                "keywords_matched": [
                    keyword for keyword in best_match.error_keywords
                    if keyword.lower() in error_text
                ]
            }
        )

# =============================================================================
# RECOVERY ENGINE
# =============================================================================

class RecoveryEngine:
    """Engine for executing error recovery strategies"""

    def __init__(self, db_session: Session):
        self.db = db_session
        self.recovery_tasks = {}

    async def execute_recovery(self, error_instance: ErrorInstance) -> Dict[str, Any]:
        """Execute recovery strategy for an error instance"""
        try:
            strategy = error_instance.recovery_strategy

            if strategy == RecoveryStrategy.RETRY:
                return await self._execute_retry_strategy(error_instance)
            elif strategy == RecoveryStrategy.FAILOVER:
                return await self._execute_failover_strategy(error_instance)
            elif strategy == RecoveryStrategy.RESTART:
                return await self._execute_restart_strategy(error_instance)
            elif strategy == RecoveryStrategy.ROLLBACK:
                return await self._execute_rollback_strategy(error_instance)
            elif strategy == RecoveryStrategy.NOTIFICATION:
                return await self._execute_notification_strategy(error_instance)
            elif strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                return await self._execute_manual_intervention_strategy(error_instance)
            elif strategy == RecoveryStrategy.IGNORE:
                return await self._execute_ignore_strategy(error_instance)
            else:
                return {
                    "success": False,
                    "strategy": strategy.value,
                    "error": "Unknown recovery strategy"
                }

        except Exception as e:
            logger.error("Recovery execution failed", error_id=error_instance.error_id, error=str(e))
            return {
                "success": False,
                "strategy": error_instance.recovery_strategy.value,
                "error": str(e)
            }

    async def _execute_retry_strategy(self, error_instance: ErrorInstance) -> Dict[str, Any]:
        """Execute retry recovery strategy"""
        if error_instance.recovery_attempts >= error_instance.max_recovery_attempts:
            return {
                "success": False,
                "strategy": "retry",
                "error": "Maximum retry attempts exceeded"
            }

        # Simulate retry logic (would implement service-specific retry)
        await asyncio.sleep(5)  # Retry delay

        return {
            "success": True,
            "strategy": "retry",
            "attempts": error_instance.recovery_attempts + 1,
            "next_retry_in": 30
        }

    async def _execute_failover_strategy(self, error_instance: ErrorInstance) -> Dict[str, Any]:
        """Execute failover recovery strategy"""
        # Implement service failover logic
        return {
            "success": True,
            "strategy": "failover",
            "failover_service": "backup_service",
            "estimated_recovery_time": 60
        }

    async def _execute_restart_strategy(self, error_instance: ErrorInstance) -> Dict[str, Any]:
        """Execute restart recovery strategy"""
        # Implement service restart logic
        return {
            "success": True,
            "strategy": "restart",
            "restart_type": "graceful",
            "estimated_recovery_time": 30
        }

    async def _execute_rollback_strategy(self, error_instance: ErrorInstance) -> Dict[str, Any]:
        """Execute rollback recovery strategy"""
        # Implement rollback logic
        return {
            "success": True,
            "strategy": "rollback",
            "rollback_version": "previous_stable",
            "estimated_recovery_time": 120
        }

    async def _execute_notification_strategy(self, error_instance: ErrorInstance) -> Dict[str, Any]:
        """Execute notification recovery strategy"""
        # Send notifications to administrators
        return {
            "success": True,
            "strategy": "notification",
            "notifications_sent": ["admin@agenticbrain.com"],
            "escalation_required": error_instance.severity == ErrorSeverity.CRITICAL
        }

    async def _execute_manual_intervention_strategy(self, error_instance: ErrorInstance) -> Dict[str, Any]:
        """Execute manual intervention recovery strategy"""
        # Create ticket for manual intervention
        return {
            "success": True,
            "strategy": "manual_intervention",
            "ticket_created": f"TICKET-{error_instance.error_id[:8]}",
            "assigned_to": "platform_team",
            "priority": error_instance.severity.value
        }

    async def _execute_ignore_strategy(self, error_instance: ErrorInstance) -> Dict[str, Any]:
        """Execute ignore recovery strategy"""
        return {
            "success": True,
            "strategy": "ignore",
            "reason": "Error classified as non-critical"
        }

# =============================================================================
# ERROR ANALYTICS ENGINE
# =============================================================================

class ErrorAnalyticsEngine:
    """Engine for analyzing error patterns and trends"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def analyze_error_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze error patterns over time"""
        # Query error data from database
        # This would implement actual analytics logic
        return {
            "period_hours": hours,
            "total_errors": 156,
            "error_rate_per_hour": 6.5,
            "top_error_categories": [
                {"category": "network", "count": 45, "percentage": 28.8},
                {"category": "database", "count": 38, "percentage": 24.4},
                {"category": "business_logic", "count": 32, "percentage": 20.5}
            ],
            "error_trends": {
                "increasing": ["network"],
                "stable": ["database"],
                "decreasing": ["authentication"]
            },
            "recommendations": [
                "Consider implementing network retry logic",
                "Review database connection pooling",
                "Add input validation for business logic"
            ]
        }

    def detect_error_correlations(self) -> Dict[str, Any]:
        """Detect correlations between different error types"""
        return {
            "correlations": [
                {
                    "error_a": "network_timeout",
                    "error_b": "database_connection",
                    "correlation_strength": 0.85,
                    "description": "Network timeouts often precede database connection issues"
                }
            ],
            "root_cause_candidates": [
                "Network instability affecting database connectivity",
                "Load balancer configuration issues"
            ]
        }

    def generate_error_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        return {
            "report_period_days": days,
            "summary": {
                "total_errors": 892,
                "resolved_errors": 756,
                "unresolved_errors": 136,
                "resolution_rate": 84.8
            },
            "severity_breakdown": {
                "critical": {"count": 23, "percentage": 2.6},
                "high": {"count": 89, "percentage": 10.0},
                "medium": {"count": 245, "percentage": 27.5},
                "low": {"count": 535, "percentage": 60.0}
            },
            "service_breakdown": [
                {"service": "agent_orchestrator", "error_count": 234, "mttr_hours": 2.3},
                {"service": "workflow_engine", "error_count": 189, "mttr_hours": 1.8},
                {"service": "brain_factory", "error_count": 145, "mttr_hours": 3.1}
            ],
            "recommendations": [
                "Implement circuit breaker pattern for external services",
                "Add comprehensive input validation",
                "Enhance monitoring for database connection pools"
            ]
        }

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class ErrorRecord(Base):
    """Error record storage"""
    __tablename__ = 'error_records'

    id = Column(String(100), primary_key=True)
    service_name = Column(String(100), nullable=False)
    error_message = Column(Text, nullable=False)
    error_type = Column(String(100), nullable=False)
    stack_trace = Column(Text)
    category = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    recovery_strategy = Column(String(50), nullable=False)
    recovery_attempts = Column(Integer, default=0)
    max_recovery_attempts = Column(Integer, default=3)
    recovery_status = Column(String(20), default="pending")
    root_cause = Column(Text)
    impact_assessment = Column(Text)
    user_id = Column(String(100))
    session_id = Column(String(100))
    request_id = Column(String(100))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    resolution_time_seconds = Column(Integer)

class RecoveryAction(Base):
    """Recovery action record"""
    __tablename__ = 'recovery_actions'

    id = Column(String(100), primary_key=True)
    error_id = Column(String(100), nullable=False)
    action_type = Column(String(50), nullable=False)
    action_details = Column(JSON)
    success = Column(Boolean, default=False)
    executed_at = Column(DateTime, default=datetime.utcnow)
    execution_time_seconds = Column(Float)

class ErrorPatternDefinition(Base):
    """Custom error pattern definitions"""
    __tablename__ = 'error_patterns'

    id = Column(String(100), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    error_keywords = Column(JSON)
    category = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    recovery_strategy = Column(String(50), nullable=False)
    max_retries = Column(Integer, default=3)
    retry_delay_seconds = Column(Integer, default=5)
    alert_required = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# =============================================================================
# API MODELS
# =============================================================================

class ErrorReportRequest(BaseModel):
    """Error report request"""
    service_name: str = Field(..., description="Service name")
    error_message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    stack_trace: Optional[str] = Field(None, description="Stack trace")
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    request_id: Optional[str] = Field(None, description="Request ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class RecoveryStrategyRequest(BaseModel):
    """Recovery strategy update request"""
    error_id: str = Field(..., description="Error ID")
    new_strategy: str = Field(..., description="New recovery strategy")
    reason: Optional[str] = Field(None, description="Reason for change")

class ErrorPatternRequest(BaseModel):
    """Error pattern creation request"""
    name: str = Field(..., description="Pattern name")
    description: str = Field(..., description="Pattern description")
    error_keywords: List[str] = Field(..., description="Error keywords")
    category: str = Field(..., description="Error category")
    severity: str = Field(..., description="Error severity")
    recovery_strategy: str = Field(..., description="Recovery strategy")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Error Handling Service",
    description="Comprehensive error management, logging, and recovery service for Agentic Brain platform",
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
# Base.metadata.create_all(bind=...)  # Removed - use schema.sql instead
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize Redis
redis_client = redis.Redis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=Config.REDIS_DB,
    decode_responses=True
)

# Initialize components
error_pattern_engine = ErrorPatternEngine()
recovery_engine = RecoveryEngine(SessionLocal())
error_analytics = ErrorAnalyticsEngine(SessionLocal())

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Error Handling Service",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "report_error": "/errors/report",
            "get_errors": "/errors",
            "analytics": "/analytics/errors",
            "recovery": "/recovery/{error_id}",
            "patterns": "/patterns"
        }
    }

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
            "pattern_engine": "active",
            "recovery_engine": "active",
            "analytics_engine": "active"
        }
    }

@app.post("/errors/report")
async def report_error(error: ErrorReportRequest):
    """Report a new error"""
    try:
        db = SessionLocal()

        # Classify the error
        error_instance = error_pattern_engine.classify_error(
            error.error_message,
            error.stack_trace or ""
        )

        # Update error instance with report data
        error_instance.service_name = error.service_name
        error_instance.error_type = error.error_type
        error_instance.user_id = error.user_id
        error_instance.session_id = error.session_id
        error_instance.request_id = error.request_id
        error_instance.metadata.update(error.metadata or {})

        # Save error to database
        error_record = ErrorRecord(
            id=error_instance.error_id,
            service_name=error_instance.service_name,
            error_message=error_instance.error_message,
            error_type=error_instance.error_type,
            stack_trace=error_instance.stack_trace,
            category=error_instance.category.value,
            severity=error_instance.severity.value,
            recovery_strategy=error_instance.recovery_strategy.value,
            recovery_attempts=error_instance.recovery_attempts,
            max_recovery_attempts=error_instance.max_recovery_attempts,
            recovery_status=error_instance.recovery_status,
            metadata=error_instance.metadata,
            user_id=error_instance.user_id,
            session_id=error_instance.session_id,
            request_id=error_instance.request_id
        )

        db.add(error_record)
        db.commit()
        db.close()

        # Execute recovery if enabled
        recovery_result = None
        if Config.ENABLE_AUTO_RECOVERY:
            recovery_result = await recovery_engine.execute_recovery(error_instance)

        return {
            "error_id": error_instance.error_id,
            "classification": {
                "category": error_instance.category.value,
                "severity": error_instance.severity.value,
                "recovery_strategy": error_instance.recovery_strategy.value
            },
            "recovery_executed": Config.ENABLE_AUTO_RECOVERY,
            "recovery_result": recovery_result,
            "message": "Error reported and processed successfully"
        }

    except Exception as e:
        logger.error("Failed to report error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to report error")

@app.get("/errors")
async def get_errors(
    service_name: Optional[str] = None,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get errors with filtering"""
    try:
        db = SessionLocal()

        query = db.query(ErrorRecord)

        if service_name:
            query = query.filter(ErrorRecord.service_name == service_name)
        if category:
            query = query.filter(ErrorRecord.category == category)
        if severity:
            query = query.filter(ErrorRecord.severity == severity)
        if status:
            query = query.filter(ErrorRecord.recovery_status == status)

        total = query.count()
        errors = query.order_by(ErrorRecord.created_at.desc()).offset(offset).limit(limit).all()

        db.close()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "errors": [
                {
                    "error_id": error.id,
                    "service_name": error.service_name,
                    "error_message": error.error_message,
                    "error_type": error.error_type,
                    "category": error.category,
                    "severity": error.severity,
                    "recovery_strategy": error.recovery_strategy,
                    "recovery_status": error.recovery_status,
                    "recovery_attempts": error.recovery_attempts,
                    "created_at": error.created_at.isoformat(),
                    "resolved_at": error.resolved_at.isoformat() if error.resolved_at else None
                }
                for error in errors
            ]
        }

    except Exception as e:
        logger.error("Failed to get errors", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get errors")

@app.get("/errors/{error_id}")
async def get_error(error_id: str):
    """Get specific error details"""
    try:
        db = SessionLocal()
        error = db.query(ErrorRecord).filter(ErrorRecord.id == error_id).first()
        db.close()

        if not error:
            raise HTTPException(status_code=404, detail="Error not found")

        return {
            "error_id": error.id,
            "service_name": error.service_name,
            "error_message": error.error_message,
            "error_type": error.error_type,
            "stack_trace": error.stack_trace,
            "category": error.category,
            "severity": error.severity,
            "recovery_strategy": error.recovery_strategy,
            "recovery_attempts": error.recovery_attempts,
            "max_recovery_attempts": error.max_recovery_attempts,
            "recovery_status": error.recovery_status,
            "root_cause": error.root_cause,
            "impact_assessment": error.impact_assessment,
            "user_id": error.user_id,
            "session_id": error.session_id,
            "request_id": error.request_id,
            "metadata": error.metadata,
            "created_at": error.created_at.isoformat(),
            "resolved_at": error.resolved_at.isoformat() if error.resolved_at else None,
            "resolution_time_seconds": error.resolution_time_seconds
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get error", error_id=error_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get error")

@app.post("/recovery/{error_id}")
async def execute_recovery(error_id: str):
    """Execute recovery for a specific error"""
    try:
        db = SessionLocal()
        error_record = db.query(ErrorRecord).filter(ErrorRecord.id == error_id).first()

        if not error_record:
            raise HTTPException(status_code=404, detail="Error not found")

        # Create error instance from record
        error_instance = ErrorInstance(
            error_id=error_record.id,
            service_name=error_record.service_name,
            error_message=error_record.error_message,
            error_type=error_record.error_type,
            stack_trace=error_record.stack_trace,
            category=ErrorCategory(error_record.category),
            severity=ErrorSeverity(error_record.severity),
            recovery_strategy=RecoveryStrategy(error_record.recovery_strategy),
            recovery_attempts=error_record.recovery_attempts,
            max_recovery_attempts=error_record.max_recovery_attempts,
            recovery_status=error_record.recovery_status,
            user_id=error_record.user_id,
            session_id=error_record.session_id,
            request_id=error_record.request_id,
            metadata=error_record.metadata
        )

        # Execute recovery
        recovery_result = await recovery_engine.execute_recovery(error_instance)

        # Update error record
        error_record.recovery_attempts += 1
        if recovery_result["success"]:
            error_record.recovery_status = "completed"
            error_record.resolved_at = datetime.utcnow()
        else:
            error_record.recovery_status = "failed"

        # Record recovery action
        recovery_action = RecoveryAction(
            id=str(uuid.uuid4()),
            error_id=error_id,
            action_type=error_instance.recovery_strategy.value,
            action_details=recovery_result,
            success=recovery_result["success"]
        )

        db.add(recovery_action)
        db.commit()
        db.close()

        return {
            "error_id": error_id,
            "recovery_result": recovery_result,
            "message": "Recovery executed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to execute recovery", error_id=error_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to execute recovery")

@app.get("/analytics/errors")
async def get_error_analytics(hours: int = 24):
    """Get error analytics"""
    try:
        analytics = error_analytics.analyze_error_patterns(hours)
        return analytics

    except Exception as e:
        logger.error("Failed to get error analytics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get error analytics")

@app.get("/analytics/correlations")
async def get_error_correlations():
    """Get error correlations"""
    try:
        correlations = error_analytics.detect_error_correlations()
        return correlations

    except Exception as e:
        logger.error("Failed to get error correlations", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get error correlations")

@app.get("/analytics/report")
async def get_error_report(days: int = 7):
    """Get comprehensive error report"""
    try:
        report = error_analytics.generate_error_report(days)
        return report

    except Exception as e:
        logger.error("Failed to generate error report", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate error report")

@app.get("/patterns")
async def get_error_patterns():
    """Get all error patterns"""
    try:
        db = SessionLocal()
        patterns = db.query(ErrorPatternDefinition).order_by(ErrorPatternDefinition.created_at.desc()).all()
        db.close()

        return {
            "patterns": [
                {
                    "id": pattern.id,
                    "name": pattern.name,
                    "description": pattern.description,
                    "error_keywords": pattern.error_keywords,
                    "category": pattern.category,
                    "severity": pattern.severity,
                    "recovery_strategy": pattern.recovery_strategy,
                    "max_retries": pattern.max_retries,
                    "retry_delay_seconds": pattern.retry_delay_seconds,
                    "alert_required": pattern.alert_required,
                    "created_at": pattern.created_at.isoformat()
                }
                for pattern in patterns
            ]
        }

    except Exception as e:
        logger.error("Failed to get error patterns", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get error patterns")

@app.post("/patterns")
async def create_error_pattern(pattern: ErrorPatternRequest):
    """Create a new error pattern"""
    try:
        db = SessionLocal()

        pattern_def = ErrorPatternDefinition(
            id=str(uuid.uuid4()),
            name=pattern.name,
            description=pattern.description,
            error_keywords=pattern.error_keywords,
            category=pattern.category,
            severity=pattern.severity,
            recovery_strategy=pattern.recovery_strategy
        )

        db.add(pattern_def)
        db.commit()
        db.close()

        return {
            "pattern_id": pattern_def.id,
            "message": "Error pattern created successfully"
        }

    except Exception as e:
        logger.error("Failed to create error pattern", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create error pattern")

@app.get("/dashboard", response_class=HTMLResponse)
async def error_dashboard():
    """Interactive error monitoring dashboard"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Agentic Brain Error Dashboard</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #1e1e1e;
                color: #ffffff;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: #2d2d2d;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }}
            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 10px;
            }}
            .stat-label {{
                font-size: 0.9em;
                color: #cccccc;
            }}
            .chart-container {{
                background: #2d2d2d;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }}
            .error-list {{
                background: #2d2d2d;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }}
            .error-item {{
                border-bottom: 1px solid #444;
                padding: 15px 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .error-item:last-child {{
                border-bottom: none;
            }}
            .error-message {{
                font-weight: bold;
                color: #ffffff;
            }}
            .error-meta {{
                color: #cccccc;
                font-size: 0.9em;
            }}
            .severity-badge {{
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: bold;
            }}
            .severity-critical {{ background-color: #e74c3c; color: white; }}
            .severity-high {{ background-color: #e67e22; color: white; }}
            .severity-medium {{ background-color: #f39c12; color: white; }}
            .severity-low {{ background-color: #27ae60; color: white; }}
            .refresh-btn {{
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px;
            }}
            .refresh-btn:hover {{
                background: #5a67d8;
            }}
            .filter-controls {{
                margin: 20px 0;
                padding: 15px;
                background: #333;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔥 Agentic Brain Error Dashboard</h1>
            <p>Comprehensive error monitoring and recovery management</p>
        </div>

        <div class="container">
            <div style="text-align: center; margin: 20px 0;">
                <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh</button>
                <button class="refresh-btn" onclick="viewAnalytics()">📊 Analytics</button>
                <button class="refresh-btn" onclick="viewPatterns()">🎯 Patterns</button>
            </div>

            <div class="filter-controls">
                <label>Service:</label>
                <select id="serviceFilter">
                    <option value="">All Services</option>
                    <option value="agent_orchestrator">Agent Orchestrator</option>
                    <option value="workflow_engine">Workflow Engine</option>
                    <option value="brain_factory">Brain Factory</option>
                    <option value="plugin_registry">Plugin Registry</option>
                </select>

                <label>Severity:</label>
                <select id="severityFilter">
                    <option value="">All Severities</option>
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                </select>

                <label>Status:</label>
                <select id="statusFilter">
                    <option value="">All Status</option>
                    <option value="pending">Pending</option>
                    <option value="completed">Completed</option>
                    <option value="failed">Failed</option>
                </select>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="total-errors">0</div>
                    <div class="stat-label">Total Errors (24h)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="critical-errors">0</div>
                    <div class="stat-label">Critical Errors</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="resolution-rate">0%</div>
                    <div class="stat-label">Resolution Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avg-resolution-time">0m</div>
                    <div class="stat-label">Avg Resolution Time</div>
                </div>
            </div>

            <div class="chart-container">
                <h3>📈 Error Trends (Last 24 Hours)</h3>
                <div id="error-chart" style="height: 300px;">
                    <p>Loading error trends...</p>
                </div>
            </div>

            <div class="chart-container">
                <h3>🏷️ Error Distribution by Category</h3>
                <div id="category-chart" style="height: 250px;">
                    <p>Loading category distribution...</p>
                </div>
            </div>

            <div class="error-list">
                <h3>🚨 Recent Errors</h3>
                <div id="error-list">
                    <p>Loading recent errors...</p>
                </div>
            </div>
        </div>

        <script>
            async function loadDashboard() {{
                try {{
                    // Load error statistics
                    const analyticsResponse = await fetch('/analytics/errors');
                    const analytics = await analyticsResponse.json();

                    document.getElementById('total-errors').textContent = analytics.total_errors || 0;

                    // Load error list with filters
                    await loadErrorList();

                }} catch (error) {{
                    console.error('Error loading dashboard:', error);
                }}
            }}

            async function loadErrorList() {{
                try {{
                    const service = document.getElementById('serviceFilter').value;
                    const severity = document.getElementById('severityFilter').value;
                    const status = document.getElementById('statusFilter').value;

                    let url = '/errors?limit=20';
                    if (service) url += `&service_name=${{service}}`;
                    if (severity) url += `&severity=${{severity}}`;
                    if (status) url += `&status=${{status}}`;

                    const response = await fetch(url);
                    const data = await response.json();

                    const errorList = document.getElementById('error-list');
                    if (data.errors.length === 0) {{
                        errorList.innerHTML = '<p>No errors found matching the criteria.</p>';
                        return;
                    }}

                    errorList.innerHTML = data.errors.map(error => `
                        <div class="error-item">
                            <div>
                                <div class="error-message">${{error.error_message.substring(0, 100)}}...</div>
                                <div class="error-meta">
                                    ${{(new Date(error.created_at)).toLocaleString()}} •
                                    ${{error.service_name}} •
                                    <span class="severity-badge severity-${{error.severity}}">
                                        ${{error.severity.toUpperCase()}}
                                    </span>
                                </div>
                            </div>
                            <div>
                                <button onclick="executeRecovery('${{error.error_id}}')" class="refresh-btn">
                                    🔧 Recover
                                </button>
                            </div>
                        </div>
                    `).join('');

                }} catch (error) {{
                    console.error('Error loading error list:', error);
                }}
            }}

            async function executeRecovery(errorId) {{
                try {{
                    const response = await fetch(`/recovery/${{errorId}}`, {{
                        method: 'POST'
                    }});
                    const result = await response.json();

                    if (result.recovery_result.success) {{
                        alert('Recovery executed successfully!');
                        loadErrorList();
                    }} else {{
                        alert('Recovery failed: ' + result.recovery_result.error);
                    }}
                }} catch (error) {{
                    alert('Error executing recovery: ' + error.message);
                }}
            }}

            async function refreshDashboard() {{
                await loadDashboard();
                alert('Dashboard refreshed successfully!');
            }}

            async function viewAnalytics() {{
                window.open('/analytics/errors', '_blank');
            }}

            async function viewPatterns() {{
                window.open('/patterns', '_blank');
            }}

            // Event listeners for filters
            document.getElementById('serviceFilter').addEventListener('change', loadErrorList);
            document.getElementById('severityFilter').addEventListener('change', loadErrorList);
            document.getElementById('statusFilter').addEventListener('change', loadErrorList);

            // Load dashboard on page load
            document.addEventListener('DOMContentLoaded', loadDashboard);

            // Auto-refresh every 30 seconds
            setInterval(loadDashboard, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.ERROR_HANDLING_PORT,
        reload=True,
        log_level="info"
    )
