#!/usr/bin/env python3
"""
Monitoring Metrics Service for Agentic Brain Platform

This service provides comprehensive monitoring and metrics collection for the Agentic Brain platform,
integrating with Prometheus to collect and expose detailed metrics from all Agent Brain services.
Includes agent performance metrics, workflow analytics, task completion rates, and plugin usage statistics.

Features:
- Prometheus metrics collection and exposure
- Real-time agent performance monitoring
- Workflow execution analytics
- Task completion and error rate tracking
- Plugin usage and performance metrics
- Service health and availability monitoring
- Custom business metrics collection
- Alert generation and notification
- Historical metrics storage and analysis
- Metrics aggregation and correlation
- Performance bottleneck detection
- SLA monitoring and reporting
"""

import asyncio
import json
import logging
import os
import sys
import time
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
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import httpx
import aiohttp
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON, Float, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func

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

class MetricsSnapshot(Base):
    """Historical metrics snapshots for trend analysis"""
    __tablename__ = 'metrics_snapshots'

    id = Column(String(100), primary_key=True)
    snapshot_time = Column(DateTime, default=datetime.utcnow)
    metric_type = Column(String(50), nullable=False)  # agent, workflow, task, service
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    labels = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AlertDefinition(Base):
    """Custom alert definitions for metrics"""
    __tablename__ = 'alert_definitions'

    id = Column(String(100), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    metric_name = Column(String(100), nullable=False)
    condition = Column(String(50), nullable=False)  # gt, lt, eq, ne
    threshold = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False)
    enabled = Column(Boolean, default=True)
    labels = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class SLACompliance(Base):
    """SLA compliance tracking and reporting"""
    __tablename__ = 'sla_compliance'

    id = Column(String(100), primary_key=True)
    service_name = Column(String(100), nullable=False)
    sla_metric = Column(String(100), nullable=False)
    target_value = Column(Float, nullable=False)
    actual_value = Column(Float, nullable=False)
    compliance_percentage = Column(Float, nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    status = Column(String(20), nullable=False)  # compliant, warning, breach
    created_at = Column(DateTime, default=datetime.utcnow)

class PerformanceBaseline(Base):
    """Performance baseline metrics for anomaly detection"""
    __tablename__ = 'performance_baselines'

    id = Column(String(100), primary_key=True)
    metric_name = Column(String(100), nullable=False)
    baseline_value = Column(Float, nullable=False)
    standard_deviation = Column(Float, nullable=False)
    sample_size = Column(Integer, nullable=False)
    calculation_period_days = Column(Integer, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

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
    METRICS_SERVICE_PORT = int(os.getenv("METRICS_SERVICE_PORT", "8350"))

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
    REASONING_MODULE_FACTORY_URL = os.getenv("REASONING_MODULE_FACTORY_URL", "http://localhost:8304")
    AGENT_BRAIN_BASE_URL = os.getenv("AGENT_BRAIN_BASE_URL", "http://localhost:8305")
    SERVICE_CONNECTOR_FACTORY_URL = os.getenv("SERVICE_CONNECTOR_FACTORY_URL", "http://localhost:8306")
    AUTHENTICATION_SERVICE_URL = os.getenv("AUTHENTICATION_SERVICE_URL", "http://localhost:8330")
    AUDIT_LOGGING_URL = os.getenv("AUDIT_LOGGING_URL", "http://localhost:8340")
    INTEGRATION_TESTS_URL = os.getenv("INTEGRATION_TESTS_URL", "http://localhost:8320")

    # Metrics collection
    COLLECTION_INTERVAL_SECONDS = int(os.getenv("COLLECTION_INTERVAL_SECONDS", "30"))
    METRICS_RETENTION_DAYS = int(os.getenv("METRICS_RETENTION_DAYS", "90"))
    ENABLE_HISTORICAL_STORAGE = os.getenv("ENABLE_HISTORICAL_STORAGE", "true").lower() == "true"

    # Alert configuration
    ALERT_ENABLED = os.getenv("ALERT_ENABLED", "true").lower() == "true"
    ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
    ALERT_EMAIL_ENABLED = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"

    # Performance monitoring
    ENABLE_PERFORMANCE_BASELINES = os.getenv("ENABLE_PERFORMANCE_BASELINES", "true").lower() == "true"
    BASELINE_CALCULATION_DAYS = int(os.getenv("BASELINE_CALCULATION_DAYS", "30"))

    # Monitoring
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8005"))

# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """Collects metrics from all Agent Brain services"""

    def __init__(self, db_session: Session, redis_client):
        self.db = db_session
        self.redis = redis_client
        self.last_collection = {}
        self.collection_running = True

        # Initialize Prometheus metrics
        self._initialize_prometheus_metrics()

    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics collectors"""

        # Agent metrics
        self.agent_count = Gauge('agent_brain_agents_total', 'Total number of agents', ['status'])
        self.agent_creation_rate = Counter('agent_brain_agent_creations_total', 'Total agent creations', ['template_type'])
        self.agent_execution_time = Histogram('agent_brain_agent_execution_duration_seconds', 'Agent execution time', ['agent_type'])

        # Workflow metrics
        self.workflow_count = Gauge('agent_brain_workflows_total', 'Total number of workflows', ['status'])
        self.workflow_execution_rate = Counter('agent_brain_workflow_executions_total', 'Total workflow executions')
        self.workflow_success_rate = Gauge('agent_brain_workflow_success_rate', 'Workflow success rate')
        self.workflow_execution_time = Histogram('agent_brain_workflow_execution_duration_seconds', 'Workflow execution time', ['workflow_type'])

        # Task metrics
        self.task_count = Gauge('agent_brain_tasks_total', 'Total number of tasks', ['status'])
        self.task_completion_rate = Counter('agent_brain_task_completions_total', 'Total task completions', ['task_type'])
        self.task_error_rate = Counter('agent_brain_task_errors_total', 'Total task errors', ['error_type'])
        self.task_execution_time = Histogram('agent_brain_task_execution_duration_seconds', 'Task execution time', ['task_type'])

        # Plugin metrics
        self.plugin_usage_count = Counter('agent_brain_plugin_usage_total', 'Total plugin usage', ['plugin_name', 'plugin_type'])
        self.plugin_execution_time = Histogram('agent_brain_plugin_execution_duration_seconds', 'Plugin execution time', ['plugin_name'])
        self.plugin_error_rate = Counter('agent_brain_plugin_errors_total', 'Total plugin errors', ['plugin_name'])

        # Service health metrics
        self.service_health_status = Gauge('agent_brain_service_health_status', 'Service health status', ['service_name'])
        self.service_response_time = Histogram('agent_brain_service_response_time_seconds', 'Service response time', ['service_name', 'endpoint'])

        # Memory and performance metrics
        self.memory_usage = Gauge('agent_brain_memory_usage_bytes', 'Memory usage', ['service_name'])
        self.cpu_usage = Gauge('agent_brain_cpu_usage_percent', 'CPU usage', ['service_name'])
        self.active_connections = Gauge('agent_brain_active_connections', 'Active connections', ['service_name'])

        # Business metrics
        self.user_sessions = Gauge('agent_brain_user_sessions_total', 'Total user sessions')
        self.api_requests = Counter('agent_brain_api_requests_total', 'Total API requests', ['service_name', 'endpoint', 'method'])
        self.data_processed = Counter('agent_brain_data_processed_total', 'Total data processed', ['data_type'])

        # Alert metrics
        self.alerts_triggered = Counter('agent_brain_alerts_triggered_total', 'Total alerts triggered', ['alert_type', 'severity'])

        logger.info("Prometheus metrics initialized")

    async def collect_all_metrics(self):
        """Collect metrics from all services"""
        try:
            # Collect agent metrics
            await self._collect_agent_metrics()

            # Collect workflow metrics
            await self._collect_workflow_metrics()

            # Collect task metrics
            await self._collect_task_metrics()

            # Collect plugin metrics
            await self._collect_plugin_metrics()

            # Collect service health metrics
            await self._collect_service_health_metrics()

            # Collect performance metrics
            await self._collect_performance_metrics()

            # Store historical metrics if enabled
            if Config.ENABLE_HISTORICAL_STORAGE:
                await self._store_historical_metrics()

            # Update performance baselines
            if Config.ENABLE_PERFORMANCE_BASELINES:
                await self._update_performance_baselines()

            logger.info("Metrics collection completed")

        except Exception as e:
            logger.error("Error collecting metrics", error=str(e))

    async def _collect_agent_metrics(self):
        """Collect agent-related metrics"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get agent count from orchestrator
                response = await client.get(f"{Config.AGENT_ORCHESTRATOR_URL}/orchestrator/agents")
                if response.status_code == 200:
                    agents = response.json()
                    active_count = len([a for a in agents if a.get('status') == 'active'])
                    inactive_count = len([a for a in agents if a.get('status') != 'active'])

                    self.agent_count.labels(status='active').set(active_count)
                    self.agent_count.labels(status='inactive').set(inactive_count)

                # Get agent creation metrics
                response = await client.get(f"{Config.AGENT_ORCHESTRATOR_URL}/orchestrator/metrics/creations")
                if response.status_code == 200:
                    creation_data = response.json()
                    for template_type, count in creation_data.items():
                        self.agent_creation_rate.labels(template_type=template_type).inc(count)

        except Exception as e:
            logger.error("Error collecting agent metrics", error=str(e))

    async def _collect_workflow_metrics(self):
        """Collect workflow-related metrics"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get workflow metrics from workflow engine
                response = await client.get(f"{Config.WORKFLOW_ENGINE_URL}/workflows/metrics")
                if response.status_code == 200:
                    workflow_data = response.json()

                    self.workflow_count.labels(status='active').set(workflow_data.get('active_workflows', 0))
                    self.workflow_count.labels(status='completed').set(workflow_data.get('completed_workflows', 0))
                    self.workflow_count.labels(status='failed').set(workflow_data.get('failed_workflows', 0))

                    # Calculate success rate
                    total = workflow_data.get('total_workflows', 0)
                    successful = workflow_data.get('successful_workflows', 0)
                    if total > 0:
                        success_rate = successful / total
                        self.workflow_success_rate.set(success_rate)

        except Exception as e:
            logger.error("Error collecting workflow metrics", error=str(e))

    async def _collect_task_metrics(self):
        """Collect task-related metrics"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get task metrics from orchestrator
                response = await client.get(f"{Config.AGENT_ORCHESTRATOR_URL}/orchestrator/metrics/tasks")
                if response.status_code == 200:
                    task_data = response.json()

                    self.task_count.labels(status='running').set(task_data.get('running_tasks', 0))
                    self.task_count.labels(status='completed').set(task_data.get('completed_tasks', 0))
                    self.task_count.labels(status='failed').set(task_data.get('failed_tasks', 0))

                    # Task completion rates
                    for task_type, count in task_data.get('completions_by_type', {}).items():
                        self.task_completion_rate.labels(task_type=task_type).inc(count)

                    # Task error rates
                    for error_type, count in task_data.get('errors_by_type', {}).items():
                        self.task_error_rate.labels(error_type=error_type).inc(count)

        except Exception as e:
            logger.error("Error collecting task metrics", error=str(e))

    async def _collect_plugin_metrics(self):
        """Collect plugin-related metrics"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get plugin metrics from plugin registry
                response = await client.get(f"{Config.PLUGIN_REGISTRY_URL}/plugins/metrics")
                if response.status_code == 200:
                    plugin_data = response.json()

                    # Plugin usage counts
                    for plugin_name, usage_count in plugin_data.get('usage_counts', {}).items():
                        plugin_type = plugin_data.get('plugin_types', {}).get(plugin_name, 'unknown')
                        self.plugin_usage_count.labels(plugin_name=plugin_name, plugin_type=plugin_type).inc(usage_count)

                    # Plugin error rates
                    for plugin_name, error_count in plugin_data.get('error_counts', {}).items():
                        self.plugin_error_rate.labels(plugin_name=plugin_name).inc(error_count)

        except Exception as e:
            logger.error("Error collecting plugin metrics", error=str(e))

    async def _collect_service_health_metrics(self):
        """Collect service health and response time metrics"""
        services_to_check = {
            'agent_orchestrator': Config.AGENT_ORCHESTRATOR_URL,
            'plugin_registry': Config.PLUGIN_REGISTRY_URL,
            'workflow_engine': Config.WORKFLOW_ENGINE_URL,
            'template_store': Config.TEMPLATE_STORE_URL,
            'brain_factory': Config.BRAIN_FACTORY_URL,
            'deployment_pipeline': Config.DEPLOYMENT_PIPELINE_URL,
            'authentication_service': Config.AUTHENTICATION_SERVICE_URL,
            'audit_logging': Config.AUDIT_LOGGING_URL
        }

        for service_name, service_url in services_to_check.items():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    start_time = time.time()
                    response = await client.get(f"{service_url}/health")
                    response_time = time.time() - start_time

                    # Record response time
                    self.service_response_time.labels(
                        service_name=service_name,
                        endpoint='health'
                    ).observe(response_time)

                    # Record health status (1 for healthy, 0 for unhealthy)
                    is_healthy = response.status_code == 200
                    self.service_health_status.labels(service_name=service_name).set(1 if is_healthy else 0)

            except Exception as e:
                # Service is unhealthy
                self.service_health_status.labels(service_name=service_name).set(0)
                logger.warning(f"Service health check failed for {service_name}", error=str(e))

    async def _collect_performance_metrics(self):
        """Collect performance metrics from services"""
        try:
            # Collect from Redis (if services publish performance data)
            performance_keys = self.redis.keys("performance:*")
            for key in performance_keys:
                try:
                    data = json.loads(self.redis.get(key))
                    service_name = data.get('service_name', 'unknown')
                    self.memory_usage.labels(service_name=service_name).set(data.get('memory_mb', 0) * 1024 * 1024)
                    self.cpu_usage.labels(service_name=service_name).set(data.get('cpu_percent', 0))
                    self.active_connections.labels(service_name=service_name).set(data.get('active_connections', 0))
                except Exception as e:
                    logger.error(f"Error processing performance data for key {key}", error=str(e))

        except Exception as e:
            logger.error("Error collecting performance metrics", error=str(e))

    async def _store_historical_metrics(self):
        """Store historical metrics snapshots"""
        try:
            snapshot_time = datetime.utcnow()

            # Store key metrics snapshots
            metrics_to_snapshot = [
                ('agent_count_active', self.agent_count.labels(status='active')._value),
                ('agent_count_inactive', self.agent_count.labels(status='inactive')._value),
                ('workflow_success_rate', self.workflow_success_rate._value),
                ('task_count_running', self.task_count.labels(status='running')._value),
                ('task_count_completed', self.task_count.labels(status='completed')._value),
                ('task_count_failed', self.task_count.labels(status='failed')._value)
            ]

            for metric_name, metric_value in metrics_to_snapshot:
                if metric_value is not None:
                    snapshot = MetricsSnapshot(
                        id=f"{metric_name}_{int(snapshot_time.timestamp())}",
                        snapshot_time=snapshot_time,
                        metric_type='agent_brain',
                        metric_name=metric_name,
                        metric_value=float(metric_value)
                    )
                    self.db.add(snapshot)

            self.db.commit()

        except Exception as e:
            logger.error("Error storing historical metrics", error=str(e))
            self.db.rollback()

    async def _update_performance_baselines(self):
        """Update performance baselines for anomaly detection"""
        try:
            baseline_period = timedelta(days=Config.BASELINE_CALCULATION_DAYS)

            # Get recent metrics for baseline calculation
            recent_snapshots = self.db.query(MetricsSnapshot).filter(
                MetricsSnapshot.snapshot_time >= datetime.utcnow() - baseline_period
            ).all()

            # Group by metric name and calculate baselines
            from collections import defaultdict
            metric_values = defaultdict(list)

            for snapshot in recent_snapshots:
                metric_values[snapshot.metric_name].append(snapshot.metric_value)

            # Calculate and store baselines
            for metric_name, values in metric_values.items():
                if len(values) >= 10:  # Need minimum sample size
                    import numpy as np
                    mean_value = np.mean(values)
                    std_dev = np.std(values)

                    baseline = PerformanceBaseline(
                        id=f"baseline_{metric_name}",
                        metric_name=metric_name,
                        baseline_value=mean_value,
                        standard_deviation=std_dev,
                        sample_size=len(values),
                        calculation_period_days=Config.BASELINE_CALCULATION_DAYS
                    )

                    # Upsert baseline
                    existing = self.db.query(PerformanceBaseline).filter_by(metric_name=metric_name).first()
                    if existing:
                        existing.baseline_value = mean_value
                        existing.standard_deviation = std_dev
                        existing.sample_size = len(values)
                        existing.last_updated = datetime.utcnow()
                    else:
                        self.db.add(baseline)

            self.db.commit()

        except Exception as e:
            logger.error("Error updating performance baselines", error=str(e))
            self.db.rollback()

    async def start_collection_loop(self):
        """Start the metrics collection loop"""
        while self.collection_running:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(Config.COLLECTION_INTERVAL_SECONDS)
            except Exception as e:
                logger.error("Error in metrics collection loop", error=str(e))
                await asyncio.sleep(5)  # Wait before retrying

    def stop_collection(self):
        """Stop the metrics collection loop"""
        self.collection_running = False

# =============================================================================
# ALERT MANAGER
# =============================================================================

class AlertManager:
    """Manages alerts and notifications based on metrics"""

    def __init__(self, db_session: Session):
        self.db = db_session

    async def check_alerts(self):
        """Check for alert conditions and trigger alerts"""
        try:
            # Get enabled alert definitions
            alert_definitions = self.db.query(AlertDefinition).filter_by(enabled=True).all()

            for alert_def in alert_definitions:
                await self._check_alert_condition(alert_def)

        except Exception as e:
            logger.error("Error checking alerts", error=str(e))

    async def _check_alert_condition(self, alert_def: AlertDefinition):
        """Check if an alert condition is met"""
        try:
            # This would integrate with the actual metrics collectors
            # For now, this is a placeholder implementation
            should_trigger = await self._evaluate_alert_condition(alert_def)

            if should_trigger:
                await self._trigger_alert(alert_def)

        except Exception as e:
            logger.error(f"Error checking alert condition for {alert_def.name}", error=str(e))

    async def _evaluate_alert_condition(self, alert_def: AlertDefinition) -> bool:
        """Evaluate if an alert condition is met"""
        # Placeholder implementation - would integrate with actual metrics
        return False

    async def _trigger_alert(self, alert_def: AlertDefinition):
        """Trigger an alert notification"""
        try:
            # Create alert record
            alert = AlertDefinition(
                id=str(uuid.uuid4()),
                name=alert_def.name,
                description=alert_def.description,
                metric_name=alert_def.metric_name,
                condition=alert_def.condition,
                threshold=alert_def.threshold,
                severity=alert_def.severity,
                enabled=True
            )

            self.db.add(alert)
            self.db.commit()

            # Send notifications
            await self._send_alert_notifications(alert, alert_def)

            logger.warning("Alert triggered", alert_name=alert_def.name, severity=alert_def.severity)

        except Exception as e:
            logger.error(f"Error triggering alert {alert_def.name}", error=str(e))

    async def _send_alert_notifications(self, alert: AlertDefinition, alert_def: AlertDefinition):
        """Send alert notifications"""
        # Placeholder - would implement email, webhook, etc. notifications
        logger.info(f"Alert notification would be sent for {alert.name}")

# =============================================================================
# API MODELS
# =============================================================================

class MetricsQueryRequest(BaseModel):
    """Metrics query request"""
    metric_name: str = Field(..., description="Name of the metric to query")
    start_time: Optional[datetime] = Field(None, description="Start time for query")
    end_time: Optional[datetime] = Field(None, description="End time for query")
    labels: Optional[Dict[str, str]] = Field(None, description="Metric labels to filter by")
    aggregation: Optional[str] = Field("avg", description="Aggregation function (avg, sum, min, max)")

class AlertDefinitionRequest(BaseModel):
    """Alert definition request"""
    name: str = Field(..., description="Alert name")
    description: str = Field(..., description="Alert description")
    metric_name: str = Field(..., description="Metric name to monitor")
    condition: str = Field(..., description="Alert condition (gt, lt, eq, ne)")
    threshold: float = Field(..., description="Alert threshold value")
    severity: str = Field(..., description="Alert severity")
    labels: Optional[Dict[str, str]] = Field(None, description="Alert labels")

class SLADefinitionRequest(BaseModel):
    """SLA definition request"""
    service_name: str = Field(..., description="Service name")
    sla_metric: str = Field(..., description="SLA metric name")
    target_value: float = Field(..., description="Target SLA value")
    period_days: int = Field(30, description="SLA period in days")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Monitoring Metrics Service",
    description="Comprehensive monitoring and metrics collection service for Agentic Brain platform",
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

# Initialize components
metrics_collector = MetricsCollector(SessionLocal(), redis_client)
alert_manager = AlertManager(SessionLocal())

# Background task for metrics collection
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Monitoring Metrics Service starting up")

    # Start metrics collection loop
    asyncio.create_task(metrics_collector.start_collection_loop())

    # Start alert checking loop
    if Config.ALERT_ENABLED:
        asyncio.create_task(alert_checking_loop())

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Monitoring Metrics Service shutting down")

    # Stop metrics collection
    metrics_collector.stop_collection()

async def alert_checking_loop():
    """Background loop for checking alerts"""
    while True:
        try:
            await alert_manager.check_alerts()
            await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error("Error in alert checking loop", error=str(e))
            await asyncio.sleep(60)

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
    return {"message": "Monitoring Metrics Service", "status": "healthy", "version": "1.0.0"}

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
            "metrics_collector": "active",
            "alert_manager": "active"
        }
    }

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/metrics/agent")
async def get_agent_metrics():
    """Get current agent metrics"""
    return {
        "active_agents": metrics_collector.agent_count.labels(status='active')._value,
        "inactive_agents": metrics_collector.agent_count.labels(status='inactive')._value,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics/workflow")
async def get_workflow_metrics():
    """Get current workflow metrics"""
    return {
        "active_workflows": metrics_collector.workflow_count.labels(status='active')._value,
        "completed_workflows": metrics_collector.workflow_count.labels(status='completed')._value,
        "failed_workflows": metrics_collector.workflow_count.labels(status='failed')._value,
        "success_rate": metrics_collector.workflow_success_rate._value,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics/task")
async def get_task_metrics():
    """Get current task metrics"""
    return {
        "running_tasks": metrics_collector.task_count.labels(status='running')._value,
        "completed_tasks": metrics_collector.task_count.labels(status='completed')._value,
        "failed_tasks": metrics_collector.task_count.labels(status='failed')._value,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics/plugin")
async def get_plugin_metrics():
    """Get current plugin metrics"""
    # This would return aggregated plugin metrics
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "note": "Plugin metrics available via Prometheus /metrics endpoint"
    }

@app.get("/metrics/service-health")
async def get_service_health_metrics():
    """Get current service health metrics"""
    services = [
        'agent_orchestrator', 'plugin_registry', 'workflow_engine',
        'template_store', 'brain_factory', 'deployment_pipeline',
        'authentication_service', 'audit_logging'
    ]

    health_data = {}
    for service in services:
        try:
            health_data[service] = {
                "status": "healthy" if metrics_collector.service_health_status.labels(service_name=service)._value == 1 else "unhealthy"
            }
        except:
            health_data[service] = {"status": "unknown"}

    return {
        "services": health_data,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics/performance")
async def get_performance_metrics():
    """Get current performance metrics"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "note": "Performance metrics available via Prometheus /metrics endpoint"
    }

@app.get("/metrics/business")
async def get_business_metrics():
    """Get current business metrics"""
    return {
        "user_sessions": metrics_collector.user_sessions._value,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics/historical")
async def get_historical_metrics(metric_name: str, days: int = 7):
    """Get historical metrics data"""
    try:
        db = SessionLocal()
        start_date = datetime.utcnow() - timedelta(days=days)

        snapshots = db.query(MetricsSnapshot).filter(
            MetricsSnapshot.metric_name == metric_name,
            MetricsSnapshot.snapshot_time >= start_date
        ).order_by(MetricsSnapshot.snapshot_time).all()

        db.close()

        return {
            "metric_name": metric_name,
            "period_days": days,
            "data_points": [
                {
                    "timestamp": snapshot.snapshot_time.isoformat(),
                    "value": snapshot.metric_value
                }
                for snapshot in snapshots
            ]
        }

    except Exception as e:
        logger.error("Failed to get historical metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get historical metrics")

@app.post("/alerts")
async def create_alert_definition(alert: AlertDefinitionRequest):
    """Create a new alert definition"""
    try:
        db = SessionLocal()

        alert_def = AlertDefinition(
            id=str(uuid.uuid4()),
            name=alert.name,
            description=alert.description,
            metric_name=alert.metric_name,
            condition=alert.condition,
            threshold=alert.threshold,
            severity=alert.severity,
            labels=alert.labels
        )

        db.add(alert_def)
        db.commit()
        db.close()

        return {
            "alert_id": alert_def.id,
            "message": "Alert definition created successfully"
        }

    except Exception as e:
        logger.error("Failed to create alert definition", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create alert definition")

@app.get("/alerts")
async def list_alert_definitions():
    """List all alert definitions"""
    try:
        db = SessionLocal()
        alerts = db.query(AlertDefinition).order_by(AlertDefinition.created_at.desc()).all()
        db.close()

        return {
            "alerts": [
                {
                    "id": alert.id,
                    "name": alert.name,
                    "description": alert.description,
                    "metric_name": alert.metric_name,
                    "condition": alert.condition,
                    "threshold": alert.threshold,
                    "severity": alert.severity,
                    "enabled": alert.enabled,
                    "created_at": alert.created_at.isoformat()
                }
                for alert in alerts
            ]
        }

    except Exception as e:
        logger.error("Failed to list alert definitions", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list alert definitions")

@app.get("/alerts/{alert_id}")
async def get_alert_definition(alert_id: str):
    """Get specific alert definition"""
    try:
        db = SessionLocal()
        alert = db.query(AlertDefinition).filter_by(id=alert_id).first()
        db.close()

        if not alert:
            raise HTTPException(status_code=404, detail="Alert definition not found")

        return {
            "id": alert.id,
            "name": alert.name,
            "description": alert.description,
            "metric_name": alert.metric_name,
            "condition": alert.condition,
            "threshold": alert.threshold,
            "severity": alert.severity,
            "enabled": alert.enabled,
            "labels": alert.labels,
            "created_at": alert.created_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get alert definition", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get alert definition")

@app.put("/alerts/{alert_id}")
async def update_alert_definition(alert_id: str, alert: AlertDefinitionRequest):
    """Update alert definition"""
    try:
        db = SessionLocal()
        existing_alert = db.query(AlertDefinition).filter_by(id=alert_id).first()

        if not existing_alert:
            raise HTTPException(status_code=404, detail="Alert definition not found")

        # Update fields
        existing_alert.name = alert.name
        existing_alert.description = alert.description
        existing_alert.metric_name = alert.metric_name
        existing_alert.condition = alert.condition
        existing_alert.threshold = alert.threshold
        existing_alert.severity = alert.severity
        existing_alert.labels = alert.labels

        db.commit()
        db.close()

        return {"message": "Alert definition updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update alert definition", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update alert definition")

@app.delete("/alerts/{alert_id}")
async def delete_alert_definition(alert_id: str):
    """Delete alert definition"""
    try:
        db = SessionLocal()
        alert = db.query(AlertDefinition).filter_by(id=alert_id).first()

        if not alert:
            raise HTTPException(status_code=404, detail="Alert definition not found")

        db.delete(alert)
        db.commit()
        db.close()

        return {"message": "Alert definition deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete alert definition", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete alert definition")

@app.get("/sla-compliance")
async def get_sla_compliance():
    """Get SLA compliance data"""
    try:
        db = SessionLocal()
        sla_records = db.query(SLACompliance).order_by(SLACompliance.period_end.desc()).limit(30).all()
        db.close()

        return {
            "sla_compliance": [
                {
                    "id": record.id,
                    "service_name": record.service_name,
                    "sla_metric": record.sla_metric,
                    "target_value": record.target_value,
                    "actual_value": record.actual_value,
                    "compliance_percentage": record.compliance_percentage,
                    "status": record.status,
                    "period_start": record.period_start.isoformat(),
                    "period_end": record.period_end.isoformat()
                }
                for record in sla_records
            ]
        }

    except Exception as e:
        logger.error("Failed to get SLA compliance data", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get SLA compliance data")

@app.get("/performance-baselines")
async def get_performance_baselines():
    """Get performance baselines"""
    try:
        db = SessionLocal()
        baselines = db.query(PerformanceBaseline).order_by(PerformanceBaseline.last_updated.desc()).all()
        db.close()

        return {
            "baselines": [
                {
                    "id": baseline.id,
                    "metric_name": baseline.metric_name,
                    "baseline_value": baseline.baseline_value,
                    "standard_deviation": baseline.standard_deviation,
                    "sample_size": baseline.sample_size,
                    "calculation_period_days": baseline.calculation_period_days,
                    "last_updated": baseline.last_updated.isoformat()
                }
                for baseline in baselines
            ]
        }

    except Exception as e:
        logger.error("Failed to get performance baselines", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get performance baselines")

@app.get("/dashboard", response_class=HTMLResponse)
async def metrics_dashboard():
    """Interactive metrics dashboard"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Agentic Brain Metrics Dashboard</title>
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
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                text-align: center;
            }}
            .metric-value {{
                font-size: 3em;
                font-weight: bold;
                color: #3498db;
                margin-bottom: 10px;
            }}
            .metric-label {{
                font-size: 1.1em;
                color: #666;
                margin-bottom: 5px;
            }}
            .status-healthy {{ color: #27ae60; }}
            .status-warning {{ color: #f39c12; }}
            .status-error {{ color: #e74c3c; }}
            .chart-container {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .service-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .service-card {{
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                text-align: center;
            }}
            .refresh-btn {{
                background: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px;
            }}
            .refresh-btn:hover {{
                background: #2980b9;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Agentic Brain Metrics Dashboard</h1>
            <p>Real-time monitoring and performance analytics</p>
        </div>

        <div class="container">
            <div style="text-align: center; margin: 20px 0;">
                <button class="refresh-btn" onclick="refreshMetrics()">🔄 Refresh Metrics</button>
                <button class="refresh-btn" onclick="viewAlerts()">🚨 View Alerts</button>
                <button class="refresh-btn" onclick="viewReports()">📋 View Reports</button>
            </div>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value status-healthy" id="active-agents">0</div>
                    <div class="metric-label">Active Agents</div>
                    <div id="inactive-agents" style="font-size: 0.8em; color: #666;">0 Inactive</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-healthy" id="active-workflows">0</div>
                    <div class="metric-label">Active Workflows</div>
                    <div id="workflow-success-rate" style="font-size: 0.8em; color: #666;">--% Success Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-warning" id="running-tasks">0</div>
                    <div class="metric-label">Running Tasks</div>
                    <div id="completed-tasks" style="font-size: 0.8em; color: #666;">0 Completed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-healthy" id="healthy-services">0/8</div>
                    <div class="metric-label">Healthy Services</div>
                    <div id="service-health" style="font-size: 0.8em; color: #666;">--% Uptime</div>
                </div>
            </div>

            <div class="chart-container">
                <h3>📈 Service Health Status</h3>
                <div class="service-grid" id="service-health-grid">
                    <div class="service-card">
                        <div class="metric-label">Agent Orchestrator</div>
                        <div class="metric-value status-healthy" id="health-orchestrator">✓</div>
                    </div>
                    <div class="service-card">
                        <div class="metric-label">Plugin Registry</div>
                        <div class="metric-value status-healthy" id="health-plugin">✓</div>
                    </div>
                    <div class="service-card">
                        <div class="metric-label">Workflow Engine</div>
                        <div class="metric-value status-healthy" id="health-workflow">✓</div>
                    </div>
                    <div class="service-card">
                        <div class="metric-label">Template Store</div>
                        <div class="metric-value status-healthy" id="health-template">✓</div>
                    </div>
                    <div class="service-card">
                        <div class="metric-label">Brain Factory</div>
                        <div class="metric-value status-healthy" id="health-brain">✓</div>
                    </div>
                    <div class="service-card">
                        <div class="metric-label">Deployment Pipeline</div>
                        <div class="metric-value status-healthy" id="health-deployment">✓</div>
                    </div>
                    <div class="service-card">
                        <div class="metric-label">Authentication</div>
                        <div class="metric-value status-healthy" id="health-auth">✓</div>
                    </div>
                    <div class="service-card">
                        <div class="metric-label">Audit Logging</div>
                        <div class="metric-value status-healthy" id="health-audit">✓</div>
                    </div>
                </div>
            </div>

            <div class="chart-container">
                <h3>⚡ Recent Performance</h3>
                <div id="performance-metrics">
                    <p>Loading performance metrics...</p>
                </div>
            </div>

            <div class="chart-container">
                <h3>🔧 System Information</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                    <div>
                        <strong>Collection Interval:</strong> {Config.COLLECTION_INTERVAL_SECONDS}s<br>
                        <strong>Metrics Retention:</strong> {Config.METRICS_RETENTION_DAYS} days<br>
                        <strong>Last Updated:</strong> <span id="last-updated">Just now</span>
                    </div>
                    <div>
                        <strong>Active Alerts:</strong> <span id="active-alerts">0</span><br>
                        <strong>Baseline Calculations:</strong> {'Enabled' if Config.ENABLE_PERFORMANCE_BASELINES else 'Disabled'}<br>
                        <strong>Historical Storage:</strong> {'Enabled' if Config.ENABLE_HISTORICAL_STORAGE else 'Disabled'}
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function loadMetrics() {{
                try {{
                    // Load agent metrics
                    const agentResponse = await fetch('/metrics/agent');
                    const agentData = await agentResponse.json();
                    document.getElementById('active-agents').textContent = agentData.active_agents || 0;
                    document.getElementById('inactive-agents').textContent = agentData.inactive_agents + ' Inactive';

                    // Load workflow metrics
                    const workflowResponse = await fetch('/metrics/workflow');
                    const workflowData = await workflowResponse.json();
                    document.getElementById('active-workflows').textContent = workflowData.active_workflows || 0;
                    const successRate = workflowData.success_rate ? (workflowData.success_rate * 100).toFixed(1) : '--';
                    document.getElementById('workflow-success-rate').textContent = successRate + '% Success Rate';

                    // Load task metrics
                    const taskResponse = await fetch('/metrics/task');
                    const taskData = await taskResponse.json();
                    document.getElementById('running-tasks').textContent = taskData.running_tasks || 0;
                    document.getElementById('completed-tasks').textContent = taskData.completed_tasks + ' Completed';

                    // Load service health
                    const healthResponse = await fetch('/metrics/service-health');
                    const healthData = await healthResponse.json();
                    const services = Object.values(healthData.services);
                    const healthyCount = services.filter(s => s.status === 'healthy').length;
                    document.getElementById('healthy-services').textContent = healthyCount + '/8';

                    // Update individual service status
                    const statusMap = {{
                        'agent_orchestrator': 'health-orchestrator',
                        'plugin_registry': 'health-plugin',
                        'workflow_engine': 'health-workflow',
                        'template_store': 'health-template',
                        'brain_factory': 'health-brain',
                        'deployment_pipeline': 'health-deployment',
                        'authentication_service': 'health-auth',
                        'audit_logging': 'health-audit'
                    }};

                    Object.entries(healthData.services).forEach(([service, data]) => {{
                        const elementId = statusMap[service];
                        if (elementId) {{
                            const element = document.getElementById(elementId);
                            const status = data.status === 'healthy' ? '✓' : '✗';
                            const colorClass = data.status === 'healthy' ? 'status-healthy' : 'status-error';
                            element.textContent = status;
                            element.className = `metric-value ${{colorClass}}`;
                        }}
                    }});

                    // Update timestamp
                    document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();

                }} catch (error) {{
                    console.error('Error loading metrics:', error);
                }}
            }}

            async function refreshMetrics() {{
                await loadMetrics();
                alert('Metrics refreshed successfully!');
            }}

            async function viewAlerts() {{
                window.open('/alerts', '_blank');
            }}

            async function viewReports() {{
                window.open('/sla-compliance', '_blank');
            }}

            // Load metrics on page load
            document.addEventListener('DOMContentLoaded', loadMetrics);

            // Refresh metrics every 30 seconds
            setInterval(loadMetrics, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.METRICS_SERVICE_PORT,
        reload=True,
        log_level="info"
    )
