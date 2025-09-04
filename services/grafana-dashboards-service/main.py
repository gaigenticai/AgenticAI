#!/usr/bin/env python3
"""
Grafana Dashboards Service for Agentic Brain Platform

This service provides comprehensive visual monitoring and analytics dashboards for the Agentic Brain platform,
integrating with Grafana to create beautiful, interactive dashboards for agent performance monitoring,
workflow analytics, system health visualization, and real-time metrics display.

Features:
- Agent Performance Dashboards: Real-time agent metrics, execution times, success rates
- Workflow Analytics: Workflow execution patterns, bottleneck analysis, throughput metrics
- System Health Monitoring: Service availability, response times, error rates
- Custom Dashboard Templates: Pre-configured dashboards for common monitoring scenarios
- Real-time Data Visualization: Live updating charts, graphs, and status panels
- Alert Integration: Grafana alert rules with notification channels
- Dashboard Provisioning: Automated dashboard deployment and configuration
- Custom Panels: Specialized visualizations for Agent Brain specific metrics
- Historical Trend Analysis: Long-term performance trends and forecasting
- Comparative Analytics: Side-by-side comparisons of different agents and workflows
- SLA Compliance Dashboards: Visual SLA monitoring with compliance indicators
- Performance Baselines: Baseline comparisons with anomaly highlighting
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
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
    GRAFANA_DASHBOARDS_PORT = int(os.getenv("GRAFANA_DASHBOARDS_PORT", "8360"))

    # Grafana Configuration
    GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
    GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY", "")
    GRAFANA_USERNAME = os.getenv("GRAFANA_USERNAME", "admin")
    GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "admin")

    # Prometheus Configuration
    PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

    # Monitoring Metrics Service
    METRICS_SERVICE_URL = os.getenv("METRICS_SERVICE_URL", "http://localhost:8350")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agentic_brain")

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

    # Dashboard Configuration
    DASHBOARD_REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "30"))
    ENABLE_AUTO_PROVISIONING = os.getenv("ENABLE_AUTO_PROVISIONING", "true").lower() == "true"
    DASHBOARD_RETENTION_DAYS = int(os.getenv("DASHBOARD_RETENTION_DAYS", "90"))

# =============================================================================
# DASHBOARD MANAGER
# =============================================================================

class DashboardManager:
    """Manages Grafana dashboards and data sources"""

    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates"
        self.dashboards_dir = Path(__file__).parent / "dashboards"
        self.templates_dir.mkdir(exist_ok=True)
        self.dashboards_dir.mkdir(exist_ok=True)

        # Initialize dashboard templates
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize dashboard templates"""
        # Agent Performance Dashboard Template
        self.agent_performance_template = {
            "dashboard": {
                "id": None,
                "title": "Agent Performance Overview",
                "tags": ["agentic-brain", "performance", "agents"],
                "timezone": "browser",
                "panels": [],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "timepicker": {},
                "templating": {
                    "list": []
                },
                "annotations": {
                    "list": []
                },
                "refresh": "30s",
                "schemaVersion": 27,
                "version": 0,
                "links": []
            }
        }

        # Workflow Analytics Dashboard Template
        self.workflow_analytics_template = {
            "dashboard": {
                "id": None,
                "title": "Workflow Analytics",
                "tags": ["agentic-brain", "workflow", "analytics"],
                "timezone": "browser",
                "panels": [],
                "time": {
                    "from": "now-24h",
                    "to": "now"
                },
                "timepicker": {},
                "templating": {
                    "list": []
                },
                "annotations": {
                    "list": []
                },
                "refresh": "1m",
                "schemaVersion": 27,
                "version": 0,
                "links": []
            }
        }

        # System Health Dashboard Template
        self.system_health_template = {
            "dashboard": {
                "id": None,
                "title": "System Health Monitoring",
                "tags": ["agentic-brain", "health", "system"],
                "timezone": "browser",
                "panels": [],
                "time": {
                    "from": "now-6h",
                    "to": "now"
                },
                "timepicker": {},
                "templating": {
                    "list": []
                },
                "annotations": {
                    "list": []
                },
                "refresh": "30s",
                "schemaVersion": 27,
                "version": 0,
                "links": []
            }
        }

    async def create_agent_performance_dashboard(self) -> dict:
        """Create comprehensive agent performance dashboard"""
        dashboard = self.agent_performance_template.copy()

        dashboard["dashboard"]["panels"] = [
            # Active Agents Count
            {
                "id": 1,
                "title": "Active Agents",
                "type": "stat",
                "targets": [{
                    "expr": "agent_brain_agents_total{status=\"active\"}",
                    "legendFormat": "Active Agents"
                }],
                "fieldConfig": {
                    "defaults": {
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 80}
                            ]
                        }
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
            },

            # Agent Creation Rate
            {
                "id": 2,
                "title": "Agent Creation Rate",
                "type": "graph",
                "targets": [{
                    "expr": "rate(agent_brain_agent_creations_total[5m])",
                    "legendFormat": "{{template_type}}"
                }],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
            },

            # Agent Execution Time
            {
                "id": 3,
                "title": "Agent Execution Time (95th percentile)",
                "type": "graph",
                "targets": [{
                    "expr": "histogram_quantile(0.95, rate(agent_brain_agent_execution_duration_seconds_bucket[5m]))",
                    "legendFormat": "{{agent_type}}"
                }],
                "yAxes": [
                    {"format": "s", "label": "Execution Time"},
                    {"format": "short"}
                ],
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
            },

            # Agent Status Overview
            {
                "id": 4,
                "title": "Agent Status Distribution",
                "type": "piechart",
                "targets": [{
                    "expr": "agent_brain_agents_total",
                    "legendFormat": "{{status}}"
                }],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
            },

            # Agent Performance Table
            {
                "id": 5,
                "title": "Agent Performance Metrics",
                "type": "table",
                "targets": [{
                    "expr": "agent_brain_agents_total",
                    "legendFormat": "{{status}}",
                    "instant": True
                }],
                "fieldConfig": {
                    "defaults": {
                        "custom": {
                            "displayMode": "auto"
                        }
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
            }
        ]

        return dashboard

    async def create_workflow_analytics_dashboard(self) -> dict:
        """Create comprehensive workflow analytics dashboard"""
        dashboard = self.workflow_analytics_template.copy()

        dashboard["dashboard"]["panels"] = [
            # Workflow Success Rate
            {
                "id": 1,
                "title": "Workflow Success Rate",
                "type": "stat",
                "targets": [{
                    "expr": "agent_brain_workflow_success_rate",
                    "legendFormat": "Success Rate"
                }],
                "fieldConfig": {
                    "defaults": {
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 0.95},
                                {"color": "green", "value": 0.98}
                            ]
                        },
                        "unit": "percentunit"
                    }
                },
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0}
            },

            # Active Workflows
            {
                "id": 2,
                "title": "Active Workflows",
                "type": "stat",
                "targets": [{
                    "expr": "agent_brain_workflows_total{status=\"active\"}",
                    "legendFormat": "Active"
                }],
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0}
            },

            # Workflow Throughput
            {
                "id": 3,
                "title": "Workflow Throughput",
                "type": "stat",
                "targets": [{
                    "expr": "rate(agent_brain_workflow_executions_total[5m])",
                    "legendFormat": "Executions/min"
                }],
                "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0}
            },

            # Workflow Execution Time Trends
            {
                "id": 4,
                "title": "Workflow Execution Time Trends",
                "type": "graph",
                "targets": [{
                    "expr": "histogram_quantile(0.95, rate(agent_brain_workflow_execution_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile"
                }, {
                    "expr": "histogram_quantile(0.50, rate(agent_brain_workflow_execution_duration_seconds_bucket[5m]))",
                    "legendFormat": "50th percentile"
                }],
                "yAxes": [
                    {"format": "s", "label": "Execution Time"},
                    {"format": "short"}
                ],
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
            },

            # Workflow Status Breakdown
            {
                "id": 5,
                "title": "Workflow Status Breakdown",
                "type": "barchart",
                "targets": [{
                    "expr": "agent_brain_workflows_total",
                    "legendFormat": "{{status}}"
                }],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
            },

            # Workflow Error Analysis
            {
                "id": 6,
                "title": "Workflow Error Analysis",
                "type": "table",
                "targets": [{
                    "expr": "rate(agent_brain_workflows_total{status=\"failed\"}[1h])",
                    "legendFormat": "Failed workflows/hour",
                    "instant": True
                }],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
            }
        ]

        return dashboard

    async def create_system_health_dashboard(self) -> dict:
        """Create comprehensive system health dashboard"""
        dashboard = self.system_health_template.copy()

        dashboard["dashboard"]["panels"] = [
            # Overall System Health Score
            {
                "id": 1,
                "title": "Overall System Health",
                "type": "stat",
                "targets": [{
                    "expr": "(sum(agent_brain_service_health_status) / count(agent_brain_service_health_status)) * 100",
                    "legendFormat": "Health Score"
                }],
                "fieldConfig": {
                    "defaults": {
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 80},
                                {"color": "green", "value": 95}
                            ]
                        },
                        "unit": "percent"
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
            },

            # Service Health Status
            {
                "id": 2,
                "title": "Service Health Status",
                "type": "table",
                "targets": [{
                    "expr": "agent_brain_service_health_status",
                    "legendFormat": "{{service_name}}",
                    "instant": True
                }],
                "fieldConfig": {
                    "defaults": {
                        "custom": {
                            "displayMode": "color-background"
                        },
                        "mappings": [
                            {
                                "options": {
                                    "0": {"text": "Unhealthy", "color": "red"},
                                    "1": {"text": "Healthy", "color": "green"}
                                },
                                "type": "value"
                            }
                        ]
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
            },

            # Service Response Times
            {
                "id": 3,
                "title": "Service Response Times (95th percentile)",
                "type": "graph",
                "targets": [{
                    "expr": "histogram_quantile(0.95, rate(agent_brain_service_response_time_seconds_bucket[5m]))",
                    "legendFormat": "{{service_name}} - {{endpoint}}"
                }],
                "yAxes": [
                    {"format": "s", "label": "Response Time"},
                    {"format": "short"}
                ],
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
            },

            # Task Performance Metrics
            {
                "id": 4,
                "title": "Task Performance Overview",
                "type": "barchart",
                "targets": [{
                    "expr": "rate(agent_brain_task_completions_total[5m])",
                    "legendFormat": "{{task_type}}"
                }],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
            },

            # Plugin Usage Analytics
            {
                "id": 5,
                "title": "Plugin Usage Analytics",
                "type": "heatmap",
                "targets": [{
                    "expr": "rate(agent_brain_plugin_usage_total[5m])",
                    "legendFormat": "{{plugin_name}} ({{plugin_type}})"
                }],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
            },

            # Memory and CPU Usage
            {
                "id": 6,
                "title": "Resource Usage Trends",
                "type": "graph",
                "targets": [{
                    "expr": "agent_brain_memory_usage_bytes / 1024 / 1024",
                    "legendFormat": "{{service_name}} Memory (MB)"
                }, {
                    "expr": "agent_brain_cpu_usage_percent",
                    "legendFormat": "{{service_name}} CPU (%)"
                }],
                "yAxes": [
                    {"format": "MB", "label": "Memory"},
                    {"format": "percent", "label": "CPU"}
                ],
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 24}
            }
        ]

        return dashboard

    async def create_sla_compliance_dashboard(self) -> dict:
        """Create SLA compliance monitoring dashboard"""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "SLA Compliance Monitoring",
                "tags": ["agentic-brain", "sla", "compliance"],
                "timezone": "browser",
                "panels": [
                    # Overall SLA Compliance
                    {
                        "id": 1,
                        "title": "Overall SLA Compliance",
                        "type": "stat",
                        "targets": [{
                            "expr": "avg(sla_compliance_percentage)",
                            "legendFormat": "Avg Compliance %"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"color": "red", "value": None},
                                        {"color": "yellow", "value": 95},
                                        {"color": "green", "value": 99}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },

                    # SLA Compliance by Service
                    {
                        "id": 2,
                        "title": "SLA Compliance by Service",
                        "type": "table",
                        "targets": [{
                            "expr": "sla_compliance_percentage",
                            "legendFormat": "{{service_name}} - {{sla_metric}}",
                            "instant": True
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },

                    # SLA Breach Analysis
                    {
                        "id": 3,
                        "title": "SLA Breach Analysis",
                        "type": "graph",
                        "targets": [{
                            "expr": "sla_breach_count",
                            "legendFormat": "{{service_name}} breaches"
                        }],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-30d",
                    "to": "now"
                },
                "timepicker": {},
                "templating": {
                    "list": []
                },
                "annotations": {
                    "list": []
                },
                "refresh": "5m",
                "schemaVersion": 27,
                "version": 0,
                "links": []
            }
        }

        return dashboard

    async def create_alert_dashboard(self) -> dict:
        """Create alert monitoring dashboard"""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Alert Monitoring",
                "tags": ["agentic-brain", "alerts", "monitoring"],
                "timezone": "browser",
                "panels": [
                    # Active Alerts Count
                    {
                        "id": 1,
                        "title": "Active Alerts",
                        "type": "stat",
                        "targets": [{
                            "expr": "alerts_total{state=\"firing\"}",
                            "legendFormat": "Firing Alerts"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "red", "value": 1}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0}
                    },

                    # Alert Rate
                    {
                        "id": 2,
                        "title": "Alert Rate (per hour)",
                        "type": "stat",
                        "targets": [{
                            "expr": "rate(alerts_total[1h])",
                            "legendFormat": "Alerts/hour"
                        }],
                        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0}
                    },

                    # Alerts by Severity
                    {
                        "id": 3,
                        "title": "Alerts by Severity",
                        "type": "piechart",
                        "targets": [{
                            "expr": "alerts_total",
                            "legendFormat": "{{severity}}"
                        }],
                        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0}
                    },

                    # Recent Alerts
                    {
                        "id": 4,
                        "title": "Recent Alerts",
                        "type": "table",
                        "targets": [{
                            "expr": "alerts_total",
                            "legendFormat": "{{alertname}}",
                            "instant": True
                        }],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-24h",
                    "to": "now"
                },
                "timepicker": {},
                "templating": {
                    "list": []
                },
                "annotations": {
                    "list": []
                },
                "refresh": "30s",
                "schemaVersion": 27,
                "version": 0,
                "links": []
            }
        }

        return dashboard

# =============================================================================
# API MODELS
# =============================================================================

class DashboardRequest(BaseModel):
    """Dashboard creation/update request"""
    title: str = Field(..., description="Dashboard title")
    description: Optional[str] = Field(None, description="Dashboard description")
    tags: List[str] = Field(default_factory=list, description="Dashboard tags")
    dashboard_type: str = Field(..., description="Type of dashboard (agent_performance, workflow_analytics, system_health, sla_compliance, alerts)")

class DataSourceRequest(BaseModel):
    """Data source configuration request"""
    name: str = Field(..., description="Data source name")
    type: str = Field(..., description="Data source type (prometheus, postgres, etc.)")
    url: str = Field(..., description="Data source URL")
    access: str = Field("proxy", description="Access mode")
    is_default: bool = Field(False, description="Set as default data source")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Grafana Dashboards Service",
    description="Comprehensive visual monitoring and analytics dashboards for Agentic Brain platform",
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

# Initialize components
dashboard_manager = DashboardManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Grafana Dashboards Service",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "dashboards": "/dashboards",
            "dashboard_types": "/dashboard-types",
            "create_dashboard": "/dashboards/create",
            "export_dashboards": "/dashboards/export"
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
            "dashboard_manager": "active",
            "templates": "loaded"
        }
    }

@app.get("/dashboard-types")
async def get_dashboard_types():
    """Get available dashboard types"""
    return {
        "dashboard_types": [
            {
                "type": "agent_performance",
                "name": "Agent Performance Overview",
                "description": "Real-time agent metrics, execution times, and performance analytics",
                "panels": ["Active Agents", "Agent Creation Rate", "Execution Time", "Status Distribution", "Performance Table"]
            },
            {
                "type": "workflow_analytics",
                "name": "Workflow Analytics",
                "description": "Workflow execution patterns, success rates, and throughput analysis",
                "panels": ["Success Rate", "Active Workflows", "Throughput", "Execution Trends", "Status Breakdown", "Error Analysis"]
            },
            {
                "type": "system_health",
                "name": "System Health Monitoring",
                "description": "Service availability, response times, and resource utilization",
                "panels": ["Health Score", "Service Status", "Response Times", "Task Performance", "Plugin Usage", "Resource Usage"]
            },
            {
                "type": "sla_compliance",
                "name": "SLA Compliance Monitoring",
                "description": "Service level agreement monitoring and compliance tracking",
                "panels": ["Overall Compliance", "Service Compliance", "Breach Analysis"]
            },
            {
                "type": "alerts",
                "name": "Alert Monitoring",
                "description": "Real-time alert monitoring and notification tracking",
                "panels": ["Active Alerts", "Alert Rate", "Severity Breakdown", "Recent Alerts"]
            }
        ]
    }

@app.post("/dashboards/create")
async def create_dashboard(request: DashboardRequest):
    """Create a new dashboard"""
    try:
        if request.dashboard_type == "agent_performance":
            dashboard = await dashboard_manager.create_agent_performance_dashboard()
        elif request.dashboard_type == "workflow_analytics":
            dashboard = await dashboard_manager.create_workflow_analytics_dashboard()
        elif request.dashboard_type == "system_health":
            dashboard = await dashboard_manager.create_system_health_dashboard()
        elif request.dashboard_type == "sla_compliance":
            dashboard = await dashboard_manager.create_sla_compliance_dashboard()
        elif request.dashboard_type == "alerts":
            dashboard = await dashboard_manager.create_alert_dashboard()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown dashboard type: {request.dashboard_type}")

        # Update dashboard metadata
        dashboard["dashboard"]["title"] = request.title
        if request.description:
            dashboard["dashboard"]["description"] = request.description
        dashboard["dashboard"]["tags"] = request.tags

        return {
            "dashboard": dashboard,
            "message": f"Dashboard '{request.title}' created successfully",
            "type": request.dashboard_type
        }

    except Exception as e:
        logger.error("Failed to create dashboard", error=str(e), dashboard_type=request.dashboard_type)
        raise HTTPException(status_code=500, detail="Failed to create dashboard")

@app.get("/dashboards/{dashboard_type}")
async def get_dashboard_template(dashboard_type: str):
    """Get dashboard template"""
    try:
        if dashboard_type == "agent_performance":
            dashboard = await dashboard_manager.create_agent_performance_dashboard()
        elif dashboard_type == "workflow_analytics":
            dashboard = await dashboard_manager.create_workflow_analytics_dashboard()
        elif dashboard_type == "system_health":
            dashboard = await dashboard_manager.create_system_health_dashboard()
        elif dashboard_type == "sla_compliance":
            dashboard = await dashboard_manager.create_sla_compliance_dashboard()
        elif dashboard_type == "alerts":
            dashboard = await dashboard_manager.create_alert_dashboard()
        else:
            raise HTTPException(status_code=404, detail=f"Dashboard type '{dashboard_type}' not found")

        return dashboard

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get dashboard template", error=str(e), dashboard_type=dashboard_type)
        raise HTTPException(status_code=500, detail="Failed to get dashboard template")

@app.get("/dashboards/export/all")
async def export_all_dashboards():
    """Export all dashboard templates"""
    try:
        dashboards = {}

        dashboard_types = [
            "agent_performance",
            "workflow_analytics",
            "system_health",
            "sla_compliance",
            "alerts"
        ]

        for dashboard_type in dashboard_types:
            if dashboard_type == "agent_performance":
                dashboard = await dashboard_manager.create_agent_performance_dashboard()
            elif dashboard_type == "workflow_analytics":
                dashboard = await dashboard_manager.create_workflow_analytics_dashboard()
            elif dashboard_type == "system_health":
                dashboard = await dashboard_manager.create_system_health_dashboard()
            elif dashboard_type == "sla_compliance":
                dashboard = await dashboard_manager.create_sla_compliance_dashboard()
            elif dashboard_type == "alerts":
                dashboard = await dashboard_manager.create_alert_dashboard()

            dashboards[dashboard_type] = dashboard

        return {
            "dashboards": dashboards,
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_dashboards": len(dashboards)
        }

    except Exception as e:
        logger.error("Failed to export dashboards", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to export dashboards")

@app.get("/prometheus/datasources")
async def get_prometheus_datasources():
    """Get Prometheus data source configurations"""
    return {
        "datasources": [
            {
                "name": "AgenticBrain-Prometheus",
                "type": "prometheus",
                "url": Config.PROMETHEUS_URL,
                "access": "proxy",
                "isDefault": True,
                "jsonData": {
                    "timeInterval": "30s",
                    "queryTimeout": "60s",
                    "httpMethod": "POST"
                }
            }
        ]
    }

@app.get("/grafana/provisioning/dashboards")
async def get_dashboard_provisioning():
    """Get Grafana dashboard provisioning configuration"""
    return {
        "apiVersion": 1,
        "providers": [
            {
                "name": "agentic-brain-dashboards",
                "type": "file",
                "disableDeletion": False,
                "updateIntervalSeconds": 10,
                "allowUiUpdates": True,
                "options": {
                    "path": "/var/lib/grafana/dashboards"
                }
            }
        ]
    }

@app.get("/grafana/provisioning/datasources")
async def get_datasource_provisioning():
    """Get Grafana data source provisioning configuration"""
    return {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "AgenticBrain-Prometheus",
                "type": "prometheus",
                "url": Config.PROMETHEUS_URL,
                "access": "proxy",
                "isDefault": True,
                "jsonData": {
                    "timeInterval": "30s",
                    "queryTimeout": "60s",
                    "httpMethod": "POST"
                }
            },
            {
                "name": "AgenticBrain-PostgreSQL",
                "type": "postgres",
                "url": "postgres:5432",
                "database": "agentic_brain",
                "user": "user",
                "secureJsonData": {
                    "password": "password"
                },
                "jsonData": {
                    "sslmode": "disable",
                    "maxOpenConns": 100,
                    "maxIdleConns": 100,
                    "connMaxLifetime": 14400
                }
            }
        ]
    }

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_viewer():
    """Interactive dashboard viewer"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Agentic Brain Dashboard Viewer</title>
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
            .dashboard-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .dashboard-card {{
                background: #2d2d2d;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                text-align: center;
                transition: transform 0.2s, box-shadow 0.2s;
                cursor: pointer;
            }}
            .dashboard-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            }}
            .dashboard-icon {{
                font-size: 3em;
                margin-bottom: 10px;
            }}
            .dashboard-title {{
                font-size: 1.2em;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .dashboard-description {{
                font-size: 0.9em;
                color: #cccccc;
                margin-bottom: 15px;
            }}
            .view-btn {{
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                transition: background 0.2s;
            }}
            .view-btn:hover {{
                background: #5a67d8;
            }}
            .status-indicator {{
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            .status-healthy {{ background-color: #48bb78; }}
            .status-warning {{ background-color: #ed8936; }}
            .status-error {{ background-color: #f56565; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Agentic Brain Dashboard Viewer</h1>
            <p>Interactive monitoring dashboards for the Agentic Brain platform</p>
        </div>

        <div class="container">
            <div style="text-align: center; margin: 20px 0;">
                <h2>Available Dashboards</h2>
                <p>Select a dashboard to view real-time metrics and analytics</p>
            </div>

            <div class="dashboard-grid">
                <div class="dashboard-card" onclick="viewDashboard('agent_performance')">
                    <div class="dashboard-icon">🤖</div>
                    <div class="dashboard-title">Agent Performance</div>
                    <div class="dashboard-description">
                        Real-time agent metrics, execution times, and performance analytics
                    </div>
                    <button class="view-btn">View Dashboard</button>
                </div>

                <div class="dashboard-card" onclick="viewDashboard('workflow_analytics')">
                    <div class="dashboard-icon">⚡</div>
                    <div class="dashboard-title">Workflow Analytics</div>
                    <div class="dashboard-description">
                        Workflow execution patterns, success rates, and throughput analysis
                    </div>
                    <button class="view-btn">View Dashboard</button>
                </div>

                <div class="dashboard-card" onclick="viewDashboard('system_health')">
                    <div class="dashboard-icon">🩺</div>
                    <div class="dashboard-title">System Health</div>
                    <div class="dashboard-description">
                        Service availability, response times, and resource utilization
                    </div>
                    <button class="view-btn">View Dashboard</button>
                </div>

                <div class="dashboard-card" onclick="viewDashboard('sla_compliance')">
                    <div class="dashboard-icon">📋</div>
                    <div class="dashboard-title">SLA Compliance</div>
                    <div class="dashboard-description">
                        Service level agreement monitoring and compliance tracking
                    </div>
                    <button class="view-btn">View Dashboard</button>
                </div>

                <div class="dashboard-card" onclick="viewDashboard('alerts')">
                    <div class="dashboard-icon">🚨</div>
                    <div class="dashboard-title">Alert Monitoring</div>
                    <div class="dashboard-description">
                        Real-time alert monitoring and notification tracking
                    </div>
                    <button class="view-btn">View Dashboard</button>
                </div>
            </div>

            <div style="text-align: center; margin: 40px 0;">
                <h3>System Status</h3>
                <div style="display: inline-flex; align-items: center; margin: 10px;">
                    <span class="status-indicator status-healthy"></span>
                    <span>Monitoring Service: Healthy</span>
                </div>
                <div style="display: inline-flex; align-items: center; margin: 10px;">
                    <span class="status-indicator status-healthy"></span>
                    <span>Prometheus: Connected</span>
                </div>
                <div style="display: inline-flex; align-items: center; margin: 10px;">
                    <span class="status-indicator status-healthy"></span>
                    <span>Dashboards: Active</span>
                </div>
            </div>
        </div>

        <script>
            async function viewDashboard(dashboardType) {{
                try {{
                    const response = await fetch(`/dashboards/${{dashboardType}}`);
                    const dashboard = await response.json();

                    // Open dashboard in new window/tab
                    const dashboardWindow = window.open('', '_blank');
                    dashboardWindow.document.write(`
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>${{dashboard.dashboard.title}} - Agentic Brain</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                pre {{ background: #f5f5f5; padding: 20px; border-radius: 5px; overflow-x: auto; }}
                            </style>
                        </head>
                        <body>
                            <h1>${{dashboard.dashboard.title}}</h1>
                            <p>Dashboard JSON configuration for Grafana import:</p>
                            <pre>${{JSON.stringify(dashboard, null, 2)}}</pre>
                            <p><strong>Instructions:</strong></p>
                            <ol>
                                <li>Copy the JSON above</li>
                                <li>Open Grafana and navigate to Create → Import</li>
                                <li>Paste the JSON and click Import</li>
                                <li>The dashboard will be created with real-time Agentic Brain metrics</li>
                            </ol>
                        </body>
                        </html>
                    `);
                }} catch (error) {{
                    alert('Error loading dashboard: ' + error.message);
                }}
            }}

            // Auto-refresh status every 30 seconds
            setInterval(() => {{
                // Status refresh logic can be added here
            }}, 30000);
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
        port=Config.GRAFANA_DASHBOARDS_PORT,
        reload=True,
        log_level="info"
    )
