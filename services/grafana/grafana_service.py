#!/usr/bin/env python3
"""
Grafana Service for Agentic Platform

This service provides comprehensive dashboard management and visualization:
- Automated dashboard creation for platform metrics
- Data source configuration and management
- Custom panel and widget creation
- Alert rule management
- User and permission management
- Integration with Prometheus and other monitoring systems
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pika
import psycopg2
import requests
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest
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

# Prometheus metrics
DASHBOARD_REQUESTS = Counter('grafana_dashboard_requests_total', 'Total dashboard requests')
DATASOURCE_REQUESTS = Counter('grafana_datasource_requests_total', 'Total datasource requests')
ALERT_REQUESTS = Counter('grafana_alert_requests_total', 'Total alert requests')

# Pydantic models
class DataSourceConfig(BaseModel):
    name: str = Field(..., description="Data source name")
    type: str = Field(..., description="Data source type (prometheus, elasticsearch, etc.)")
    url: str = Field(..., description="Data source URL")
    access: str = Field("proxy", description="Access mode")
    basic_auth: Optional[bool] = Field(False, description="Basic authentication enabled")
    basic_auth_user: Optional[str] = Field(None, description="Basic auth username")
    basic_auth_password: Optional[str] = Field(None, description="Basic auth password")
    json_data: Optional[Dict[str, Any]] = Field(None, description="Additional JSON data")

class DashboardConfig(BaseModel):
    title: str = Field(..., description="Dashboard title")
    tags: Optional[List[str]] = Field(None, description="Dashboard tags")
    panels: Optional[List[Dict[str, Any]]] = Field(None, description="Dashboard panels")
    time_from: Optional[str] = Field(None, description="Time range from")
    time_to: Optional[str] = Field(None, description="Time range to")
    refresh: Optional[str] = Field(None, description="Refresh interval")

class AlertRule(BaseModel):
    name: str = Field(..., description="Alert rule name")
    query: str = Field(..., description="Prometheus query")
    duration: str = Field("5m", description="For duration")
    labels: Optional[Dict[str, str]] = Field(None, description="Alert labels")
    annotations: Optional[Dict[str, str]] = Field(None, description="Alert annotations")

class GrafanaService:
    """Main Grafana service class"""

    def __init__(self):
        self.app = FastAPI(
            title="Grafana Service",
            description="Dashboard and visualization management service for Agentic Platform",
            version="1.0.0"
        )
        self.grafana_url = None
        self.grafana_api_key = None
        self.rabbitmq_connection = None
        self.setup_clients()
        self.setup_routes()
        self.setup_message_consumer()

    def setup_clients(self):
        """Initialize Grafana and RabbitMQ clients"""
        try:
            # Grafana configuration
            grafana_host = os.getenv('GRAFANA_HOST', 'grafana')
            grafana_port = os.getenv('GRAFANA_PORT', '3000')
            self.grafana_url = f'http://{grafana_host}:{grafana_port}'
            self.grafana_username = os.getenv('GRAFANA_USER', 'admin')
            self.grafana_password = os.getenv('GRAFANA_PASSWORD', '')

            # Wait for Grafana to be ready
            self.wait_for_grafana()

            logger.info("Grafana client initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize Grafana client", error=str(e))

        try:
            # RabbitMQ client
            rabbitmq_host = os.getenv('RABBITMQ_HOST', 'rabbitmq')
            rabbitmq_user = os.getenv('RABBITMQ_USER', 'agentic_user')
            rabbitmq_password = os.getenv('RABBITMQ_PASSWORD', '')
            if not rabbitmq_password:
                logger.error('RABBITMQ_PASSWORD not configured for Grafana Service')
                raise RuntimeError('RABBITMQ_PASSWORD not configured for Grafana Service')

            credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
            parameters = pika.ConnectionParameters(
                host=rabbitmq_host,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            self.rabbitmq_connection = pika.BlockingConnection(parameters)

            logger.info("RabbitMQ client initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize RabbitMQ client", error=str(e))
            self.rabbitmq_connection = None

    def wait_for_grafana(self):
        """Wait for Grafana to be ready"""
        max_retries = 30
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.grafana_url}/api/health", timeout=5)
                if response.status_code == 200:
                    logger.info("Grafana is ready")
                    return
            except Exception as e:
                logger.warning(f"Waiting for Grafana (attempt {attempt + 1}/{max_retries})", error=str(e))

            time.sleep(2)

        raise Exception("Grafana is not ready after maximum retries")

    def get_auth_headers(self):
        """Get authentication headers for Grafana API"""
        if self.grafana_api_key:
            return {"Authorization": f"Bearer {self.grafana_api_key}"}
        else:
            import base64
            auth_string = f"{self.grafana_username}:{self.grafana_password}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            return {"Authorization": f"Basic {encoded_auth}"}

    def setup_message_consumer(self):
        """Setup RabbitMQ message consumer"""
        if not self.rabbitmq_connection:
            return

        try:
            channel = self.rabbitmq_connection.channel()
            channel.exchange_declare(exchange='grafana', exchange_type='direct', durable=True)
            channel.queue_declare(queue='grafana_dashboards', durable=True)
            channel.queue_bind(exchange='grafana', queue='grafana_dashboards')

            channel.basic_consume(
                queue='grafana_dashboards',
                on_message_callback=self.handle_dashboard_message,
                auto_ack=False
            )

            # Start consuming in a separate thread
            import threading
            consumer_thread = threading.Thread(target=channel.start_consuming)
            consumer_thread.daemon = True
            consumer_thread.start()

            logger.info("Message consumer started")

        except Exception as e:
            logger.error("Failed to setup message consumer", error=str(e))

    def handle_dashboard_message(self, ch, method, properties, body):
        """Handle incoming dashboard messages"""
        try:
            message = json.loads(body)
            logger.info("Received dashboard message", message_type=message.get('type'))

            if message.get('type') == 'create_dashboard':
                self.create_dashboard(message['data'])
            elif message.get('type') == 'update_dashboard':
                self.update_dashboard(message['data'])
            elif message.get('type') == 'create_datasource':
                self.create_datasource(message['data'])

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error("Failed to process dashboard message", error=str(e))
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def create_datasource(self, config: Dict[str, Any]):
        """Create a data source in Grafana"""
        try:
            DATASOURCE_REQUESTS.inc()

            payload = {
                "name": config["name"],
                "type": config["type"],
                "url": config["url"],
                "access": config.get("access", "proxy"),
                "basicAuth": config.get("basic_auth", False),
                "basicAuthUser": config.get("basic_auth_user"),
                "basicAuthPassword": config.get("basic_auth_password"),
                "jsonData": config.get("json_data", {})
            }

            response = requests.post(
                f"{self.grafana_url}/api/datasources",
                json=payload,
                headers=self.get_auth_headers(),
                timeout=10
            )

            if response.status_code in [200, 201]:
                logger.info("Data source created successfully", name=config["name"])
                return response.json()
            else:
                logger.error("Failed to create data source",
                            status_code=response.status_code,
                            response=response.text)
                raise Exception(f"Failed to create data source: {response.text}")

        except Exception as e:
            logger.error("Failed to create data source", error=str(e), config=config)

    def create_dashboard(self, config: Dict[str, Any]):
        """Create a dashboard in Grafana"""
        try:
            DASHBOARD_REQUESTS.inc()

            dashboard_payload = {
                "dashboard": {
                    "title": config["title"],
                    "tags": config.get("tags", []),
                    "timezone": "browser",
                    "panels": config.get("panels", []),
                    "time": {
                        "from": config.get("time_from", "now-1h"),
                        "to": config.get("time_to", "now")
                    },
                    "refresh": config.get("refresh", "5s")
                },
                "overwrite": True
            }

            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                json=dashboard_payload,
                headers=self.get_auth_headers(),
                timeout=10
            )

            if response.status_code in [200, 201]:
                logger.info("Dashboard created successfully", title=config["title"])
                return response.json()
            else:
                logger.error("Failed to create dashboard",
                            status_code=response.status_code,
                            response=response.text)
                raise Exception(f"Failed to create dashboard: {response.text}")

        except Exception as e:
            logger.error("Failed to create dashboard", error=str(e), config=config)

    def update_dashboard(self, config: Dict[str, Any]):
        """Update an existing dashboard"""
        try:
            # Get current dashboard
            response = requests.get(
                f"{self.grafana_url}/api/dashboards/db/{config['title'].lower().replace(' ', '-')}",
                headers=self.get_auth_headers(),
                timeout=10
            )

            if response.status_code == 200:
                current_dashboard = response.json()
                config["id"] = current_dashboard["dashboard"]["id"]
                config["uid"] = current_dashboard["dashboard"]["uid"]

            return self.create_dashboard(config)

        except Exception as e:
            logger.error("Failed to update dashboard", error=str(e), config=config)

    def create_default_dashboards(self):
        """Create default dashboards for the platform"""
        try:
            # Create Prometheus data source
            prometheus_config = {
                "name": "Prometheus",
                "type": "prometheus",
                "url": "http://prometheus:9090",
                "access": "proxy"
            }
            self.create_datasource(prometheus_config)

            # Create system metrics dashboard
            system_dashboard = {
                "title": "System Metrics",
                "tags": ["system", "metrics"],
                "panels": [
                    {
                        "title": "CPU Usage",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(process_cpu_user_seconds_total[5m])",
                            "legendFormat": "CPU Usage"
                        }]
                    },
                    {
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [{
                            "expr": "process_resident_memory_bytes",
                            "legendFormat": "Memory Usage"
                        }]
                    }
                ]
            }
            self.create_dashboard(system_dashboard)

            logger.info("Default dashboards created successfully")

        except Exception as e:
            logger.error("Failed to create default dashboards", error=str(e))

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            health_status = {
                "status": "healthy",
                "service": "grafana",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "grafana": {
                    "connected": False,
                    "url": self.grafana_url
                },
                "rabbitmq": {
                    "connected": self.rabbitmq_connection is not None and not self.rabbitmq_connection.is_closed
                }
            }

            # Check Grafana health
            try:
                response = requests.get(f"{self.grafana_url}/api/health",
                                      headers=self.get_auth_headers(), timeout=5)
                health_status["grafana"]["connected"] = response.status_code == 200
            except Exception as e:
                health_status["grafana"]["error"] = str(e)
                health_status["status"] = "degraded"

            # Check RabbitMQ connection
            if not health_status["rabbitmq"]["connected"]:
                health_status["status"] = "degraded"

            return JSONResponse(
                content=health_status,
                status_code=200 if health_status["status"] == "healthy" else 503
            )

        @self.app.post("/datasources")
        async def create_datasource_endpoint(config: DataSourceConfig):
            """Create a data source"""
            try:
                result = self.create_datasource(config.dict())
                return {"status": "created", "datasource": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to create datasource: {str(e)}")

        @self.app.post("/dashboards")
        async def create_dashboard_endpoint(config: DashboardConfig):
            """Create a dashboard"""
            try:
                result = self.create_dashboard(config.dict())
                return {"status": "created", "dashboard": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to create dashboard: {str(e)}")

        @self.app.get("/dashboards")
        async def list_dashboards():
            """List all dashboards"""
            try:
                response = requests.get(
                    f"{self.grafana_url}/api/search?type=dash-db",
                    headers=self.get_auth_headers(),
                    timeout=10
                )
                return response.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to list dashboards: {str(e)}")

        @self.app.post("/setup")
        async def setup_default():
            """Setup default dashboards and data sources"""
            try:
                self.create_default_dashboards()
                return {"status": "setup completed"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Setup failed: {str(e)}")

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()

# Main application
service = GrafanaService()

if __name__ == "__main__":
    port = int(os.getenv('SERVICE_PORT', '8120'))
    logger.info("Starting Grafana Service", port=port)

    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
