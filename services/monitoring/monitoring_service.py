#!/usr/bin/env python3
"""
Monitoring Service for Agentic Platform

This service provides comprehensive monitoring and observability with:
- Prometheus configuration management
- Grafana dashboard creation and management
- Alerting rules and notifications
- Service health monitoring
- Performance metrics collection
- Custom monitoring dashboards
- Alert escalation and incident management
"""

import json
import os
import time
import yaml
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psycopg2
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import structlog
import uvicorn

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

# FastAPI app
app = FastAPI(
    title="Monitoring Service",
    description="Comprehensive monitoring and observability service",
    version="1.0.0"
)

# Prometheus metrics
MONITORING_OPERATIONS = Counter('monitoring_operations_total', 'Total monitoring operations', ['operation_type'])
ALERTS_TRIGGERED = Counter('alerts_triggered_total', 'Total alerts triggered', ['alert_type', 'severity'])
DASHBOARD_UPDATES = Counter('dashboard_updates_total', 'Total dashboard updates', ['dashboard_type'])
HEALTH_CHECKS = Counter('health_checks_total', 'Total health checks', ['service_name', 'status'])
METRICS_COLLECTED = Counter('metrics_collected_total', 'Total metrics collected', ['metric_type'])

# Global variables
database_connection = None
prometheus_url = "http://prometheus:9090"
grafana_url = "http://grafana:3000"

# Pydantic models
class AlertRule(BaseModel):
    """Alert rule model"""
    name: str = Field(..., description="Alert rule name")
    query: str = Field(..., description="PromQL query for the alert")
    duration: str = Field("5m", description="Duration for which the condition must be true")
    severity: str = Field("warning", description="Alert severity (info, warning, error, critical)")
    description: str = Field(..., description="Alert description")
    labels: Optional[Dict[str, str]] = Field(None, description="Additional labels")
    annotations: Optional[Dict[str, str]] = Field(None, description="Alert annotations")

class DashboardConfig(BaseModel):
    """Grafana dashboard configuration"""
    title: str = Field(..., description="Dashboard title")
    description: Optional[str] = Field(None, description="Dashboard description")
    tags: List[str] = Field([], description="Dashboard tags")
    panels: List[Dict[str, Any]] = Field(..., description="Grafana panels configuration")
    time_range: str = Field("1h", description="Default time range")

class ServiceHealthCheck(BaseModel):
    """Service health check configuration"""
    service_name: str = Field(..., description="Name of the service")
    endpoint: str = Field(..., description="Health check endpoint")
    interval_seconds: int = Field(30, description="Check interval in seconds")
    timeout_seconds: int = Field(10, description="Timeout in seconds")
    retries: int = Field(3, description="Number of retries")
    expected_status: int = Field(200, description="Expected HTTP status code")

class MonitoringManager:
    """Comprehensive monitoring manager"""

    def __init__(self):
        self.alert_rules = {}
        self.dashboards = {}
        self.health_checks = {}
        self.grafana_api_key = os.getenv("GRAFANA_API_KEY", "admin:admin")

    def setup_prometheus_config(self) -> Dict[str, Any]:
        """Setup Prometheus configuration for all services"""
        config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "rule_files": [
                "/etc/prometheus/alert_rules.yml"
            ],
            "alerting": {
                "alertmanagers": [
                    {
                        "static_configs": [
                            {"targets": ["alertmanager:9093"]}
                        ]
                    }
                ]
            },
            "scrape_configs": [
                {
                    "job_name": "prometheus",
                    "static_configs": [
                        {"targets": ["localhost:9090"]}
                    ]
                },
                {
                    "job_name": "ingestion-coordinator",
                    "static_configs": [
                        {"targets": ["ingestion-coordinator:8080"]}
                    ],
                    "scrape_interval": "10s",
                    "metrics_path": "/metrics"
                },
                {
                    "job_name": "output-coordinator",
                    "static_configs": [
                        {"targets": ["output-coordinator:8081"]}
                    ],
                    "scrape_interval": "10s",
                    "metrics_path": "/metrics"
                },
                {
                    "job_name": "data-lake-minio",
                    "static_configs": [
                        {"targets": ["data-lake-minio:8090"]}
                    ],
                    "scrape_interval": "15s",
                    "metrics_path": "/metrics"
                },
                {
                    "job_name": "message-queue",
                    "static_configs": [
                        {"targets": ["message-queue:8091"]}
                    ],
                    "scrape_interval": "10s",
                    "metrics_path": "/metrics"
                },
                {
                    "job_name": "redis-caching",
                    "static_configs": [
                        {"targets": ["redis-caching:8092"]}
                    ],
                    "scrape_interval": "10s",
                    "metrics_path": "/metrics"
                },
                {
                    "job_name": "oauth2-oidc",
                    "static_configs": [
                        {"targets": ["oauth2-oidc:8093"]}
                    ],
                    "scrape_interval": "15s",
                    "metrics_path": "/metrics"
                },
                {
                    "job_name": "data-encryption",
                    "static_configs": [
                        {"targets": ["data-encryption:8094"]}
                    ],
                    "scrape_interval": "15s",
                    "metrics_path": "/metrics"
                },
                {
                    "job_name": "rabbitmq",
                    "static_configs": [
                        {"targets": ["rabbitmq:15692"]}
                    ],
                    "scrape_interval": "30s"
                },
                {
                    "job_name": "postgresql",
                    "static_configs": [
                        {"targets": ["postgresql_ingestion:5432"]}
                    ],
                    "scrape_interval": "30s"
                },
                {
                    "job_name": "redis",
                    "static_configs": [
                        {"targets": ["redis_ingestion:6379"]}
                    ],
                    "scrape_interval": "15s"
                }
            ]
        }

        return config

    def create_alert_rules(self) -> Dict[str, Any]:
        """Create comprehensive alerting rules"""
        alert_rules = {
            "groups": [
                {
                    "name": "agentic_platform_alerts",
                    "rules": [
                        {
                            "alert": "ServiceDown",
                            "expr": "up == 0",
                            "for": "5m",
                            "labels": {
                                "severity": "critical"
                            },
                            "annotations": {
                                "summary": "Service {{ $labels.job }} is down",
                                "description": "Service {{ $labels.job }} has been down for more than 5 minutes."
                            }
                        },
                        {
                            "alert": "HighErrorRate",
                            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) > 0.1",
                            "for": "5m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "High error rate on {{ $labels.job }}",
                                "description": "Error rate is {{ $value }}% which is above 10% threshold."
                            }
                        },
                        {
                            "alert": "HighMemoryUsage",
                            "expr": "process_resident_memory_bytes / process_virtual_memory_max_bytes > 0.9",
                            "for": "10m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "High memory usage on {{ $labels.job }}",
                                "description": "Memory usage is at {{ $value }}% of available memory."
                            }
                        },
                        {
                            "alert": "QueueBacklog",
                            "expr": "rabbitmq_queue_messages > 1000",
                            "for": "5m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "Queue backlog on {{ $labels.queue }}",
                                "description": "Queue {{ $labels.queue }} has {{ $value }} messages."
                            }
                        },
                        {
                            "alert": "DatabaseConnectionIssues",
                            "expr": "pg_stat_activity_count{state=\"idle\"} > 50",
                            "for": "5m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "High database connection count",
                                "description": "Database has {{ $value }} idle connections."
                            }
                        },
                        {
                            "alert": "DataIngestionFailure",
                            "expr": "rate(data_ingestion_errors_total[5m]) > 5",
                            "for": "5m",
                            "labels": {
                                "severity": "error"
                            },
                            "annotations": {
                                "summary": "Data ingestion failures detected",
                                "description": "Data ingestion has {{ $value }} errors per minute."
                            }
                        },
                        {
                            "alert": "CacheMissRateHigh",
                            "expr": "rate(cache_misses_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) > 0.5",
                            "for": "10m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "High cache miss rate",
                                "description": "Cache miss rate is {{ $value }}% which is above 50%."
                            }
                        }
                    ]
                }
            ]
        }

        return alert_rules

    def create_grafana_dashboards(self) -> List[Dict[str, Any]]:
        """Create comprehensive Grafana dashboards"""
        dashboards = []

        # System Overview Dashboard
        system_dashboard = {
            "dashboard": {
                "title": "Agentic Platform - System Overview",
                "description": "Overall system health and performance metrics",
                "tags": ["agentic", "overview", "system"],
                "timezone": "browser",
                "panels": [
                    {
                        "title": "Service Health",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "up",
                                "legendFormat": "{{job}}"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "mappings": [
                                    {"options": {"0": {"text": "DOWN", "color": "red"}}, "type": "value"},
                                    {"options": {"1": {"text": "UP", "color": "green"}}, "type": "value"}
                                ]
                            }
                        }
                    },
                    {
                        "title": "HTTP Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total[5m])",
                                "legendFormat": "{{job}} - {{method}} {{status}}"
                            }
                        ]
                    },
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
                                "legendFormat": "{{job}} error rate"
                            }
                        ]
                    },
                    {
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "process_resident_memory_bytes / 1024 / 1024",
                                "legendFormat": "{{job}} memory (MB)"
                            }
                        ]
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            }
        }

        # Data Pipeline Dashboard
        data_pipeline_dashboard = {
            "dashboard": {
                "title": "Agentic Platform - Data Pipeline",
                "description": "Data ingestion, processing, and output metrics",
                "tags": ["agentic", "data", "pipeline"],
                "timezone": "browser",
                "panels": [
                    {
                        "title": "Data Ingestion Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(data_ingested_total[5m])",
                                "legendFormat": "{{layer}} - {{format}}"
                            }
                        ]
                    },
                    {
                        "title": "Data Transformation Success Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(data_transformed_total[5m])",
                                "legendFormat": "{{source_layer}} → {{target_layer}}"
                            }
                        ]
                    },
                    {
                        "title": "Queue Message Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(messages_published_total[5m])",
                                "legendFormat": "{{queue_name}} published"
                            },
                            {
                                "expr": "rate(messages_consumed_total[5m])",
                                "legendFormat": "{{queue_name}} consumed"
                            }
                        ]
                    },
                    {
                        "title": "Cache Performance",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) * 100",
                                "legendFormat": "{{cache_type}} hit rate %"
                            }
                        ]
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            }
        }

        # Security Dashboard
        security_dashboard = {
            "dashboard": {
                "title": "Agentic Platform - Security Monitoring",
                "description": "Security events, authentication, and encryption metrics",
                "tags": ["agentic", "security", "monitoring"],
                "timezone": "browser",
                "panels": [
                    {
                        "title": "Authentication Attempts",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(auth_requests_total[5m])",
                                "legendFormat": "{{method}} - {{status}}"
                            }
                        ]
                    },
                    {
                        "title": "Failed Authentication Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(failed_auth_attempts_total[5m])",
                                "legendFormat": "{{reason}}"
                            }
                        ]
                    },
                    {
                        "title": "Encryption Operations",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(encryption_operations_total[5m])",
                                "legendFormat": "{{operation_type}} - {{algorithm}}"
                            }
                        ]
                    },
                    {
                        "title": "TLS Connections",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "rate(tls_connections_total[5m])",
                                "legendFormat": "{{protocol_version}}"
                            }
                        ]
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            }
        }

        dashboards.extend([system_dashboard, data_pipeline_dashboard, security_dashboard])
        return dashboards

    def create_service_health_checks(self) -> List[ServiceHealthCheck]:
        """Create health check configurations for all services"""
        health_checks = [
            ServiceHealthCheck(
                service_name="ingestion-coordinator",
                endpoint="http://ingestion-coordinator:8080/health",
                interval_seconds=30,
                timeout_seconds=10,
                retries=3
            ),
            ServiceHealthCheck(
                service_name="output-coordinator",
                endpoint="http://output-coordinator:8081/health",
                interval_seconds=30,
                timeout_seconds=10,
                retries=3
            ),
            ServiceHealthCheck(
                service_name="data-lake-minio",
                endpoint="http://data-lake-minio:8090/health",
                interval_seconds=45,
                timeout_seconds=15,
                retries=3
            ),
            ServiceHealthCheck(
                service_name="message-queue",
                endpoint="http://message-queue:8091/health",
                interval_seconds=30,
                timeout_seconds=10,
                retries=3
            ),
            ServiceHealthCheck(
                service_name="redis-caching",
                endpoint="http://redis-caching:8092/health",
                interval_seconds=30,
                timeout_seconds=10,
                retries=3
            ),
            ServiceHealthCheck(
                service_name="oauth2-oidc",
                endpoint="http://oauth2-oidc:8093/health",
                interval_seconds=45,
                timeout_seconds=15,
                retries=3
            ),
            ServiceHealthCheck(
                service_name="data-encryption",
                endpoint="http://data-encryption:8094/health",
                interval_seconds=45,
                timeout_seconds=15,
                retries=3
            ),
            ServiceHealthCheck(
                service_name="rabbitmq",
                endpoint="http://rabbitmq:15672/api/aliveness-test/%2F",
                interval_seconds=60,
                timeout_seconds=20,
                retries=3
            ),
            ServiceHealthCheck(
                service_name="postgresql",
                endpoint="http://postgresql_ingestion:5432/",
                interval_seconds=60,
                timeout_seconds=20,
                retries=3
            )
        ]

        return health_checks

    def perform_health_check(self, health_check: ServiceHealthCheck) -> Dict[str, Any]:
        """Perform health check for a service"""
        try:
            import httpx
            import asyncio

            async def check():
                async with httpx.AsyncClient(timeout=health_check.timeout_seconds) as client:
                    response = await client.get(health_check.endpoint)
                    return {
                        "service_name": health_check.service_name,
                        "status": "healthy" if response.status_code == health_check.expected_status else "unhealthy",
                        "response_time": response.elapsed.total_seconds(),
                        "status_code": response.status_code,
                        "timestamp": datetime.utcnow().isoformat()
                    }

            # Run async check
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(check())
            loop.close()

            HEALTH_CHECKS.labels(
                service_name=health_check.service_name,
                status=result["status"]
            ).inc()

            return result

        except Exception as e:
            logger.error("Health check failed", service=health_check.service_name, error=str(e))

            HEALTH_CHECKS.labels(
                service_name=health_check.service_name,
                status="error"
            ).inc()

            return {
                "service_name": health_check.service_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            # Get metrics from Prometheus
            response = requests.get(f"{prometheus_url}/api/v1/query", params={
                "query": 'up'
            })
            prometheus_data = response.json() if response.status_code == 200 else {"error": "Prometheus unavailable"}

            # Get alert status
            alert_response = requests.get(f"{prometheus_url}/api/v1/alerts")
            alert_data = alert_response.json() if alert_response.status_code == 200 else {"error": "Alertmanager unavailable"}

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "prometheus_status": prometheus_data,
                "alert_status": alert_data,
                "services_monitored": len(self.create_service_health_checks()),
                "dashboards_created": len(self.create_grafana_dashboards()),
                "alert_rules_configured": len(self.create_alert_rules()["groups"][0]["rules"])
            }

        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def export_monitoring_config(self) -> Dict[str, Any]:
        """Export all monitoring configurations"""
        return {
            "prometheus_config": self.setup_prometheus_config(),
            "alert_rules": self.create_alert_rules(),
            "grafana_dashboards": self.create_grafana_dashboards(),
            "health_checks": [check.dict() for check in self.create_service_health_checks()],
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Global manager instance
monitoring_manager = MonitoringManager()

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "monitoring-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/config/prometheus")
async def get_prometheus_config():
    """Get Prometheus configuration"""
    try:
        config = monitoring_manager.setup_prometheus_config()
        return config

    except Exception as e:
        logger.error("Failed to get Prometheus config", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get Prometheus config: {str(e)}")

@app.get("/config/alerts")
async def get_alert_rules():
    """Get alerting rules configuration"""
    try:
        rules = monitoring_manager.create_alert_rules()
        return rules

    except Exception as e:
        logger.error("Failed to get alert rules", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get alert rules: {str(e)}")

@app.get("/config/dashboards")
async def get_grafana_dashboards():
    """Get Grafana dashboards configuration"""
    try:
        dashboards = monitoring_manager.create_grafana_dashboards()
        return {"dashboards": dashboards}

    except Exception as e:
        logger.error("Failed to get Grafana dashboards", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get Grafana dashboards: {str(e)}")

@app.get("/health/services")
async def get_service_health():
    """Get health status of all services"""
    try:
        health_checks = monitoring_manager.create_service_health_checks()
        results = []

        for check in health_checks:
            result = monitoring_manager.perform_health_check(check)
            results.append(result)

        return {"services": results}

    except Exception as e:
        logger.error("Failed to get service health", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get service health: {str(e)}")

@app.get("/health/service/{service_name}")
async def get_service_health_by_name(service_name: str):
    """Get health status of specific service"""
    try:
        health_checks = monitoring_manager.create_service_health_checks()
        check = next((c for c in health_checks if c.service_name == service_name), None)

        if not check:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")

        result = monitoring_manager.perform_health_check(check)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get service health", service=service_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get service health: {str(e)}")

@app.get("/metrics/system")
async def get_system_metrics():
    """Get comprehensive system metrics"""
    try:
        metrics = monitoring_manager.get_system_metrics()
        return metrics

    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")

@app.get("/config/export")
async def export_monitoring_config():
    """Export all monitoring configurations"""
    try:
        config = monitoring_manager.export_monitoring_config()
        return config

    except Exception as e:
        logger.error("Failed to export monitoring config", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to export monitoring config: {str(e)}")

@app.post("/alert/test")
async def test_alert(alert_name: str):
    """Test alert by triggering it manually"""
    try:
        # This would trigger a test alert in Prometheus/Alertmanager
        # For now, just log the test
        logger.info("Test alert triggered", alert_name=alert_name)

        ALERTS_TRIGGERED.labels(
            alert_type="test",
            severity="info"
        ).inc()

        return {
            "status": "alert_triggered",
            "alert_name": alert_name,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to test alert", alert_name=alert_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to test alert: {str(e)}")

@app.get("/stats")
async def get_monitoring_stats():
    """Get monitoring service statistics"""
    return {
        "service": "monitoring-service",
        "metrics": {
            "monitoring_operations_total": MONITORING_OPERATIONS._value.get(),
            "alerts_triggered_total": ALERTS_TRIGGERED._value.get(),
            "dashboard_updates_total": DASHBOARD_UPDATES._value.get(),
            "health_checks_total": HEALTH_CHECKS._value.get(),
            "metrics_collected_total": METRICS_COLLECTED._value.get()
        },
        "configurations": {
            "services_monitored": len(monitoring_manager.create_service_health_checks()),
            "dashboards_available": len(monitoring_manager.create_grafana_dashboards()),
            "alert_rules_configured": len(monitoring_manager.create_alert_rules()["groups"][0]["rules"])
        }
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global database_connection

    logger.info("Monitoring Service starting up...")

    # Setup database connection
    try:
        db_config = {
            "host": os.getenv("POSTGRES_HOST", "postgresql_ingestion"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agentic_ingestion"),
            "user": os.getenv("POSTGRES_USER", "agentic_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "")
        }

        if not db_config.get("password"):
            logger.error("POSTGRES_PASSWORD not configured for Monitoring Service")
            raise RuntimeError("POSTGRES_PASSWORD not configured for Monitoring Service")

        database_connection = psycopg2.connect(**db_config)
        logger.info("Database connection established")

    except Exception as e:
        logger.warning("Database connection failed", error=str(e))

    logger.info("Monitoring Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global database_connection

    logger.info("Monitoring Service shutting down...")

    # Close database connection
    if database_connection:
        database_connection.close()

    logger.info("Monitoring Service shutdown complete")

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
        "monitoring_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8095)),
        reload=False,
        log_level="info"
    )
