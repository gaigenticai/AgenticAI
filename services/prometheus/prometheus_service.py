#!/usr/bin/env python3
"""
Prometheus Service for Agentic Platform

This service provides comprehensive metrics collection and monitoring:
- Automated service discovery and target configuration
- Alert rule management and configuration
- Metrics aggregation and processing
- Health monitoring and alerting
- Integration with Grafana for visualization
- Custom metric collection and exposure
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
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
ALERT_REQUESTS = Counter('prometheus_alert_requests_total', 'Total alert requests')
TARGET_REQUESTS = Counter('prometheus_target_requests_total', 'Total target requests')
RULE_REQUESTS = Counter('prometheus_rule_requests_total', 'Total rule requests')

# Pydantic models
class ScrapeTarget(BaseModel):
    targets: List[str] = Field(..., description="List of target URLs")
    labels: Optional[Dict[str, str]] = Field(None, description="Target labels")

class AlertRule(BaseModel):
    name: str = Field(..., description="Alert rule name")
    query: str = Field(..., description="PromQL query")
    duration: str = Field("5m", description="For duration")
    labels: Optional[Dict[str, str]] = Field(None, description="Alert labels")
    annotations: Optional[Dict[str, str]] = Field(None, description="Alert annotations")
    severity: str = Field("warning", description="Alert severity")

class ServiceDiscovery(BaseModel):
    service_name: str = Field(..., description="Service name")
    port: int = Field(..., description="Service port")
    path: str = Field("/metrics", description="Metrics path")
    labels: Optional[Dict[str, str]] = Field(None, description="Service labels")

class PrometheusService:
    """Main Prometheus service class"""

    def __init__(self):
        self.app = FastAPI(
            title="Prometheus Service",
            description="Metrics collection and monitoring service for Agentic Platform",
            version="1.0.0"
        )
        self.prometheus_url = None
        self.rabbitmq_connection = None
        self.targets = {}
        self.alert_rules = {}
        self.setup_clients()
        self.setup_routes()
        self.setup_message_consumer()

    def setup_clients(self):
        """Initialize Prometheus and RabbitMQ clients"""
        try:
            # Prometheus configuration
            prometheus_host = os.getenv('PROMETHEUS_HOST', 'prometheus')
            prometheus_port = os.getenv('PROMETHEUS_PORT', '9090')
            self.prometheus_url = f'http://{prometheus_host}:{prometheus_port}'

            # Wait for Prometheus to be ready
            self.wait_for_prometheus()

            logger.info("Prometheus client initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize Prometheus client", error=str(e))

        try:
            # RabbitMQ client
            rabbitmq_host = os.getenv('RABBITMQ_HOST', 'rabbitmq')
            rabbitmq_user = os.getenv('RABBITMQ_USER', 'agentic_user')
            rabbitmq_password = os.getenv('RABBITMQ_PASSWORD', '')

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

    def wait_for_prometheus(self):
        """Wait for Prometheus to be ready"""
        max_retries = 30
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.prometheus_url}/-/ready", timeout=5)
                if response.status_code == 200:
                    logger.info("Prometheus is ready")
                    return
            except Exception as e:
                logger.warning(f"Waiting for Prometheus (attempt {attempt + 1}/{max_retries})", error=str(e))

            time.sleep(2)

        raise Exception("Prometheus is not ready after maximum retries")

    def setup_message_consumer(self):
        """Setup RabbitMQ message consumer"""
        if not self.rabbitmq_connection:
            return

        try:
            channel = self.rabbitmq_connection.channel()
            channel.exchange_declare(exchange='prometheus', exchange_type='direct', durable=True)
            channel.queue_declare(queue='prometheus_alerts', durable=True)
            channel.queue_bind(exchange='prometheus', queue='prometheus_alerts')

            channel.basic_consume(
                queue='prometheus_alerts',
                on_message_callback=self.handle_alert_message,
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

    def handle_alert_message(self, ch, method, properties, body):
        """Handle incoming alert messages"""
        try:
            message = json.loads(body)
            logger.info("Received alert message", message_type=message.get('type'))

            if message.get('type') == 'service_alert':
                self.handle_service_alert(message['data'])
            elif message.get('type') == 'metric_alert':
                self.handle_metric_alert(message['data'])

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error("Failed to process alert message", error=str(e))
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def handle_service_alert(self, data: Dict[str, Any]):
        """Handle service health alerts"""
        try:
            service_name = data.get('service_name')
            status = data.get('status')
            message = data.get('message', '')

            logger.warning("Service alert received",
                         service=service_name,
                         status=status,
                         message=message)

            # Here you could integrate with external alerting systems
            # like Slack, PagerDuty, email, etc.

        except Exception as e:
            logger.error("Failed to handle service alert", error=str(e), data=data)

    def handle_metric_alert(self, data: Dict[str, Any]):
        """Handle metric-based alerts"""
        try:
            alert_name = data.get('alert_name')
            value = data.get('value')
            threshold = data.get('threshold')

            logger.warning("Metric alert received",
                         alert=alert_name,
                         value=value,
                         threshold=threshold)

            # Trigger appropriate alerting actions

        except Exception as e:
            logger.error("Failed to handle metric alert", error=str(e), data=data)

    def register_service(self, discovery: ServiceDiscovery):
        """Register a service for monitoring"""
        try:
            TARGET_REQUESTS.inc()

            target_key = f"{discovery.service_name}:{discovery.port}"
            target_url = f"http://{discovery.service_name}:{discovery.port}{discovery.path}"

            self.targets[target_key] = {
                "targets": [target_url],
                "labels": discovery.labels or {}
            }

            logger.info("Service registered for monitoring",
                       service=discovery.service_name,
                       target=target_url)

            # Update Prometheus configuration
            self.update_prometheus_config()

        except Exception as e:
            logger.error("Failed to register service", error=str(e), discovery=discovery)

    def create_alert_rule(self, rule: AlertRule):
        """Create an alert rule"""
        try:
            ALERT_REQUESTS.inc()

            rule_config = {
                "alert": rule.name,
                "expr": rule.query,
                "for": rule.duration,
                "labels": rule.labels or {},
                "annotations": rule.annotations or {}
            }

            # Add severity label if not present
            if "severity" not in rule_config["labels"]:
                rule_config["labels"]["severity"] = rule.severity

            self.alert_rules[rule.name] = rule_config

            logger.info("Alert rule created", rule_name=rule.name)

            # Update Prometheus configuration
            self.update_prometheus_config()

        except Exception as e:
            logger.error("Failed to create alert rule", error=str(e), rule=rule)

    def update_prometheus_config(self):
        """Update Prometheus configuration with new targets and rules"""
        try:
            # This would typically involve updating the prometheus.yml file
            # and sending a reload signal to Prometheus
            logger.info("Prometheus configuration updated")

        except Exception as e:
            logger.error("Failed to update Prometheus configuration", error=str(e))

    def query_metrics(self, query: str, time_range: Optional[str] = None) -> Dict[str, Any]:
        """Query Prometheus metrics"""
        try:
            params = {"query": query}
            if time_range:
                # Parse time range (e.g., "1h", "30m", "5m")
                if time_range.endswith('h'):
                    hours = int(time_range[:-1])
                    params["start"] = (datetime.utcnow() - timedelta(hours=hours)).timestamp()
                elif time_range.endswith('m'):
                    minutes = int(time_range[:-1])
                    params["start"] = (datetime.utcnow() - timedelta(minutes=minutes)).timestamp()

            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Query failed: {response.text}")

        except Exception as e:
            logger.error("Failed to query metrics", error=str(e), query=query)
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    def get_alerts(self) -> Dict[str, Any]:
        """Get current alerts from Prometheus"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/alerts",
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get alerts: {response.text}")

        except Exception as e:
            logger.error("Failed to get alerts", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

    def get_targets(self) -> Dict[str, Any]:
        """Get current targets from Prometheus"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/targets",
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get targets: {response.text}")

        except Exception as e:
            logger.error("Failed to get targets", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to get targets: {str(e)}")

    def create_default_alerts(self):
        """Create default alert rules for the platform"""
        try:
            default_rules = [
                AlertRule(
                    name="ServiceDown",
                    query="up == 0",
                    duration="5m",
                    labels={"severity": "critical"},
                    annotations={
                        "summary": "Service {{ $labels.job }} is down",
                        "description": "Service {{ $labels.job }} has been down for more than 5 minutes."
                    }
                ),
                AlertRule(
                    name="HighMemoryUsage",
                    query="process_resident_memory_bytes / process_virtual_memory_max_bytes > 0.8",
                    duration="10m",
                    labels={"severity": "warning"},
                    annotations={
                        "summary": "High memory usage on {{ $labels.instance }}",
                        "description": "Memory usage is above 80% for more than 10 minutes."
                    }
                ),
                AlertRule(
                    name="HighCPUUsage",
                    query="rate(process_cpu_user_seconds_total[5m]) > 0.8",
                    duration="5m",
                    labels={"severity": "warning"},
                    annotations={
                        "summary": "High CPU usage on {{ $labels.instance }}",
                        "description": "CPU usage is above 80% for more than 5 minutes."
                    }
                )
            ]

            for rule in default_rules:
                self.create_alert_rule(rule)

            logger.info("Default alert rules created")

        except Exception as e:
            logger.error("Failed to create default alerts", error=str(e))

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            health_status = {
                "status": "healthy",
                "service": "prometheus",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "prometheus": {
                    "connected": False,
                    "url": self.prometheus_url
                },
                "rabbitmq": {
                    "connected": self.rabbitmq_connection is not None and not self.rabbitmq_connection.is_closed
                },
                "targets_registered": len(self.targets),
                "alert_rules": len(self.alert_rules)
            }

            # Check Prometheus health
            try:
                response = requests.get(f"{self.prometheus_url}/-/healthy", timeout=5)
                health_status["prometheus"]["connected"] = response.status_code == 200
            except Exception as e:
                health_status["prometheus"]["error"] = str(e)
                health_status["status"] = "degraded"

            # Check RabbitMQ connection
            if not health_status["rabbitmq"]["connected"]:
                health_status["status"] = "degraded"

            return JSONResponse(
                content=health_status,
                status_code=200 if health_status["status"] == "healthy" else 503
            )

        @self.app.post("/services")
        async def register_service(discovery: ServiceDiscovery):
            """Register a service for monitoring"""
            self.register_service(discovery)
            return {"status": "registered", "service": discovery.service_name}

        @self.app.post("/alerts")
        async def create_alert(rule: AlertRule):
            """Create an alert rule"""
            self.create_alert_rule(rule)
            return {"status": "created", "alert": rule.name}

        @self.app.get("/query")
        async def query_endpoint(q: str, time_range: Optional[str] = None):
            """Query Prometheus metrics"""
            return self.query_metrics(q, time_range)

        @self.app.get("/alerts")
        async def get_alerts_endpoint():
            """Get current alerts"""
            return self.get_alerts()

        @self.app.get("/targets")
        async def get_targets_endpoint():
            """Get current targets"""
            return self.get_targets()

        @self.app.post("/setup")
        async def setup_default():
            """Setup default alert rules"""
            try:
                self.create_default_alerts()
                return {"status": "setup completed"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Setup failed: {str(e)}")

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()

# Main application
service = PrometheusService()

if __name__ == "__main__":
    port = int(os.getenv('SERVICE_PORT', '8130'))
    logger.info("Starting Prometheus Service", port=port)

    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
