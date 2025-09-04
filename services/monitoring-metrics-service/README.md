# Monitoring Metrics Service

A comprehensive monitoring and metrics collection service for the Agentic Brain platform that provides real-time performance monitoring, Prometheus integration, alert management, and analytics for all Agent Brain services including agent performance, workflow analytics, task completion rates, and plugin usage statistics.

## 🎯 Features

### Core Monitoring Capabilities
- **Real-time Metrics Collection**: Continuous monitoring of all Agent Brain services
- **Prometheus Integration**: Full Prometheus metrics export and integration
- **Agent Performance Monitoring**: Detailed agent execution metrics and performance tracking
- **Workflow Analytics**: Comprehensive workflow execution analytics and success rates
- **Task Monitoring**: Task completion rates, error tracking, and performance metrics
- **Plugin Usage Analytics**: Plugin utilization statistics and performance monitoring
- **Service Health Monitoring**: Real-time service availability and response time tracking

### Advanced Analytics
- **Historical Metrics Storage**: Long-term metrics storage and trend analysis
- **Performance Baselines**: Automated baseline calculation for anomaly detection
- **Custom Alert Rules**: Configurable alert thresholds and notification rules
- **SLA Compliance Tracking**: Service level agreement monitoring and reporting
- **Interactive Dashboard**: Real-time metrics dashboard with visual analytics
- **Metrics Aggregation**: Intelligent aggregation and correlation of metrics data
- **Performance Bottleneck Detection**: Automated identification of performance issues

### Alert & Notification System
- **Real-time Alerting**: Immediate notifications for critical events and thresholds
- **Multi-channel Notifications**: Email, webhook, and custom notification channels
- **Alert Escalation**: Configurable alert escalation policies and procedures
- **Alert Correlation**: Intelligent correlation of related alerts and events
- **Alert History**: Complete audit trail of all alerts and responses
- **Alert Suppression**: Smart alert suppression to reduce noise

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- PostgreSQL database
- Redis instance
- Prometheus (optional, for external monitoring)

### Environment Setup
```bash
# Copy environment file
cp .env.example .env

# Configure environment variables
nano .env
```

### Docker Deployment
```bash
# Build and start the service
docker-compose up -d monitoring-metrics-service

# Check service health
curl http://localhost:8350/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

## 📡 API Endpoints

### Core Metrics Endpoints
```http
GET /metrics/agent              # Get agent performance metrics
GET /metrics/workflow           # Get workflow execution metrics
GET /metrics/task               # Get task completion metrics
GET /metrics/plugin             # Get plugin usage metrics
GET /metrics/service-health     # Get service health status
GET /metrics/performance        # Get performance metrics
GET /metrics/business           # Get business metrics
```

### Prometheus Integration
```http
GET /metrics                    # Prometheus metrics export endpoint
```

### Analytics & Reporting
```http
GET /metrics/historical?metric_name=agent_count&days=7
                                  # Get historical metrics data
GET /analytics/summary?days=30    # Get comprehensive analytics summary
```

### Alert Management
```http
POST /alerts                     # Create alert definition
GET /alerts                      # List all alert definitions
GET /alerts/{alert_id}           # Get specific alert definition
PUT /alerts/{alert_id}           # Update alert definition
DELETE /alerts/{alert_id}         # Delete alert definition
```

### SLA & Compliance
```http
GET /sla-compliance              # Get SLA compliance data
GET /performance-baselines       # Get performance baselines
```

### Dashboard & Visualization
```http
GET /dashboard                   # Interactive metrics dashboard
GET /health                      # Service health check
```

## ⚙️ Configuration

### Environment Variables
```bash
# Service Configuration
METRICS_SERVICE_PORT=8350
METRICS_SERVICE_HOST=0.0.0.0

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/agentic_brain

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Agent Brain Service URLs
AGENT_ORCHESTRATOR_URL=http://localhost:8200
PLUGIN_REGISTRY_URL=http://localhost:8201
WORKFLOW_ENGINE_URL=http://localhost:8202
TEMPLATE_STORE_URL=http://localhost:8203
BRAIN_FACTORY_URL=http://localhost:8301
DEPLOYMENT_PIPELINE_URL=http://localhost:8303
AUTHENTICATION_SERVICE_URL=http://localhost:8330
AUDIT_LOGGING_URL=http://localhost:8340

# Metrics Collection
COLLECTION_INTERVAL_SECONDS=30
METRICS_RETENTION_DAYS=90
ENABLE_HISTORICAL_STORAGE=true

# Alert Configuration
ALERT_ENABLED=true
ALERT_EMAIL_ENABLED=false
ALERT_WEBHOOK_ENABLED=false

# Performance Monitoring
ENABLE_PERFORMANCE_BASELINES=true
BASELINE_CALCULATION_DAYS=30

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8005
```

### Alert Rule Configuration
```json
{
  "name": "High Agent Failure Rate",
  "description": "Alert when agent failure rate exceeds threshold",
  "metric_name": "agent_failure_rate",
  "condition": "gt",
  "threshold": 0.05,
  "severity": "warning",
  "notification_channels": ["email", "webhook"]
}
```

### SLA Configuration
```json
{
  "service_name": "agent_orchestrator",
  "sla_metric": "response_time_p95",
  "target_value": 500,
  "period_days": 30
}
```

## 📊 Metrics Collection

### Agent Metrics
```python
# Agent count by status
agent_count{status="active"} 5
agent_count{status="inactive"} 2

# Agent creation rate by template type
agent_creation_rate{template_type="underwriting"} 3
agent_creation_rate{template_type="claims"} 2

# Agent execution time percentiles
agent_execution_time{agent_type="underwriting", percentile="p50"} 2.5
agent_execution_time{agent_type="underwriting", percentile="p95"} 5.0
agent_execution_time{agent_type="underwriting", percentile="p99"} 8.0
```

### Workflow Metrics
```python
# Workflow count by status
workflow_count{status="active"} 12
workflow_count{status="completed"} 45
workflow_count{status="failed"} 3

# Workflow success rate
workflow_success_rate 0.94

# Workflow execution time
workflow_execution_time{workflow_type="complex", percentile="p50"} 15.2
```

### Task Metrics
```python
# Task count by status
task_count{status="running"} 8
task_count{status="completed"} 156
task_count{status="failed"} 12

# Task completion rate by type
task_completion_rate{task_type="data_processing"} 98
task_completion_rate{task_type="ai_inference"} 95

# Task error rate by type
task_error_rate{error_type="timeout"} 2
task_error_rate{error_type="validation_error"} 1
```

### Plugin Metrics
```python
# Plugin usage count
plugin_usage_count{plugin_name="risk_calculator", plugin_type="domain"} 45
plugin_usage_count{plugin_name="data_retriever", plugin_type="generic"} 78

# Plugin execution time
plugin_execution_time{plugin_name="risk_calculator", percentile="p50"} 0.8
plugin_execution_time{plugin_name="risk_calculator", percentile="p95"} 2.1

# Plugin error rate
plugin_error_rate{plugin_name="risk_calculator"} 0.02
```

### Service Health Metrics
```python
# Service health status
service_health_status{service_name="agent_orchestrator"} 1
service_health_status{service_name="plugin_registry"} 1

# Service response time
service_response_time{service_name="agent_orchestrator", endpoint="health", percentile="p50"} 0.05
service_response_time{service_name="brain_factory", endpoint="generate-agent", percentile="p95"} 2.5
```

### Performance Metrics
```python
# Memory usage
memory_usage{service_name="agent_orchestrator"} 134217728

# CPU usage percentage
cpu_usage{service_name="agent_orchestrator"} 15.2

# Active connections
active_connections{service_name="agent_orchestrator"} 23
```

## 🚨 Alert Management

### Alert Rule Types
- **Threshold Alerts**: Trigger when metrics exceed predefined thresholds
- **Rate-based Alerts**: Trigger based on rate of change in metrics
- **Anomaly Alerts**: Trigger when metrics deviate from baseline patterns
- **Composite Alerts**: Trigger based on combination of multiple metrics
- **Time-based Alerts**: Trigger during specific time windows or schedules

### Alert Severity Levels
- **info**: Informational alerts for awareness
- **warning**: Warning alerts requiring attention
- **error**: Error alerts indicating problems
- **critical**: Critical alerts requiring immediate action

### Notification Channels
- **Email**: Send alerts via email with detailed information
- **Webhook**: Send alerts to external systems via HTTP webhooks
- **SMS**: Send critical alerts via SMS (premium feature)
- **Slack**: Integration with Slack channels for team notifications
- **PagerDuty**: Integration with PagerDuty for incident management

## 📈 Analytics & Reporting

### Real-time Analytics
```python
# Get comprehensive analytics summary
@app.get("/analytics/summary")
async def get_analytics_summary(days: int = 30):
    return {
        "period_days": days,
        "total_events": 15420,
        "event_types": {
            "authentication": 2340,
            "operation": 8950,
            "security": 1230,
            "compliance": 2900
        },
        "severities": {
            "info": 11230,
            "warning": 3240,
            "error": 780,
            "critical": 170
        },
        "services": {
            "agent_orchestrator": 4520,
            "plugin_registry": 3210,
            "workflow_engine": 3890,
            "brain_factory": 2340
        },
        "risk_analysis": {
            "low_risk": 12450,
            "medium_risk": 2340,
            "high_risk": 580,
            "critical_risk": 50
        },
        "user_activity": {
            "active_users": 45,
            "total_sessions": 892,
            "avg_session_duration": 1450
        },
        "compliance_metrics": {
            "gdpr_compliant": 0.96,
            "sox_compliant": 0.98,
            "hipaa_compliant": 0.99,
            "pci_compliant": 0.97
        }
    }
```

### Historical Metrics Analysis
```python
# Get historical metrics data
@app.get("/metrics/historical")
async def get_historical_metrics(metric_name: str, days: int = 7):
    return {
        "metric_name": "agent_count_active",
        "period_days": 7,
        "data_points": [
            {"timestamp": "2024-01-01T00:00:00Z", "value": 5},
            {"timestamp": "2024-01-02T00:00:00Z", "value": 7},
            {"timestamp": "2024-01-03T00:00:00Z", "value": 6},
            # ... more data points
        ]
    }
```

### Performance Baseline Calculation
```python
# Calculate performance baselines
async def _update_performance_baselines(self):
    for metric_name, values in metric_values.items():
        if len(values) >= 10:  # Minimum sample size
            mean_value = statistics.mean(values)
            std_dev = statistics.stdev(values)

            baseline = {
                "metric_name": metric_name,
                "baseline_value": mean_value,
                "standard_deviation": std_dev,
                "sample_size": len(values),
                "calculation_period_days": 30,
                "last_updated": datetime.utcnow()
            }

            await self._store_performance_baseline(baseline)
```

## 🎛️ Dashboard & Visualization

### Interactive Dashboard Features
- **Real-time Metrics Display**: Live updating metrics with auto-refresh
- **Service Health Overview**: Visual status indicators for all services
- **Performance Charts**: Historical performance trends and comparisons
- **Alert Status Panel**: Current alert status and recent alerts
- **Customizable Views**: User-configurable dashboard layouts
- **Export Capabilities**: Export metrics data in various formats
- **Mobile Responsive**: Optimized for mobile and tablet devices

### Dashboard Components
```html
<!-- Real-time Metrics Cards -->
<div class="metric-card">
    <div class="metric-value">15</div>
    <div class="metric-label">Active Agents</div>
    <div class="metric-trend">+12% from yesterday</div>
</div>

<!-- Service Health Grid -->
<div class="service-grid">
    <div class="service-card healthy">
        <div class="service-name">Agent Orchestrator</div>
        <div class="service-status">✓ Healthy</div>
        <div class="service-response-time">45ms</div>
    </div>
    <!-- More service cards... -->
</div>

<!-- Performance Charts -->
<div class="chart-container">
    <canvas id="performance-chart"></canvas>
</div>

<!-- Alert Panel -->
<div class="alert-panel">
    <div class="alert-item warning">
        <div class="alert-message">High memory usage detected</div>
        <div class="alert-time">2 minutes ago</div>
    </div>
</div>
```

## 🔧 Integration Examples

### Prometheus Integration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'agentic-brain-metrics'
    static_configs:
      - targets: ['monitoring-metrics-service:8350']
    scrape_interval: 30s
    metrics_path: '/metrics'
```

### Grafana Dashboard Integration
```json
{
  "dashboard": {
    "title": "Agentic Brain Metrics",
    "panels": [
      {
        "title": "Active Agents",
        "type": "stat",
        "targets": [
          {
            "expr": "agent_brain_agents_total{status=\"active\"}",
            "legendFormat": "Active Agents"
          }
        ]
      },
      {
        "title": "Workflow Success Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "agent_brain_workflow_success_rate",
            "legendFormat": "Success Rate"
          }
        ]
      }
    ]
  }
}
```

### Alertmanager Integration
```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'email'
  routes:
  - match:
      severity: critical
    receiver: 'pagerduty'

receivers:
- name: 'email'
  email_configs:
  - to: 'alerts@agenticbrain.com'
    from: 'alertmanager@agenticbrain.com'
    smarthost: 'smtp.gmail.com:587'
    auth_username: 'alertmanager@agenticbrain.com'
    auth_password: 'password'
```

### Service Integration Pattern
```python
# In any Agent Brain service
from prometheus_client import Counter, Histogram, Gauge
import time

# Define service metrics
REQUEST_COUNT = Counter('service_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('service_request_latency_seconds', 'Request latency', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('service_active_connections', 'Active connections')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()

    # Increment request count
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()

    # Track active connections
    ACTIVE_CONNECTIONS.inc()

    response = await call_next(request)

    # Record request latency
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(time.time() - start_time)

    # Decrement active connections
    ACTIVE_CONNECTIONS.dec()

    return response

# Custom business metrics
AGENT_EXECUTION_COUNT = Counter('agent_executions_total', 'Total agent executions', ['agent_type'])
TASK_COMPLETION_COUNT = Counter('task_completions_total', 'Total task completions', ['task_type'])

def record_agent_execution(agent_type: str, execution_time: float, success: bool):
    """Record agent execution metrics"""
    AGENT_EXECUTION_COUNT.labels(agent_type=agent_type).inc()

    if success:
        # Record successful execution time
        AGENT_EXECUTION_TIME.labels(agent_type=agent_type).observe(execution_time)
    else:
        # Record failed execution
        AGENT_EXECUTION_FAILURES.labels(agent_type=agent_type).inc()
```

## 📚 API Documentation

### Complete API Reference
- [Metrics Collection API](./docs/api/metrics-collection.md)
- [Alert Management API](./docs/api/alert-management.md)
- [Analytics API](./docs/api/analytics.md)
- [Dashboard API](./docs/api/dashboard.md)
- [Prometheus Integration](./docs/api/prometheus-integration.md)

### SDKs and Libraries
- **Python SDK**: `pip install agentic-brain-monitoring`
- **REST API Client**: Comprehensive HTTP client libraries
- **Grafana Plugins**: Custom Grafana panels for Agentic Brain metrics
- **Alertmanager Integration**: Pre-built integrations for popular alerting systems

## 🤝 Contributing

### Development Setup
```bash
# Clone the repository
git clone https://github.com/agentic-brain/monitoring-metrics-service.git
cd monitoring-metrics-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v --asyncio-mode=auto

# Start development server
python main.py --reload --debug
```

### Code Standards
- Follow PEP 8 style guidelines for Python code
- Write comprehensive docstrings for all functions
- Include type hints for function parameters and return values
- Add unit tests for all new functionality
- Update documentation for API changes

### Pull Request Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/custom-alerts`)
3. Make your changes with tests
4. Ensure all tests pass (`pytest tests/ -v`)
5. Update documentation if needed
6. Submit a pull request with detailed description

### Performance Considerations
- Implement efficient metrics storage and retrieval
- Use appropriate data structures for high-throughput metrics
- Implement metrics aggregation to reduce storage requirements
- Consider horizontal scaling for high-volume deployments
- Implement proper caching strategies for frequently accessed metrics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: [docs.agenticbrain.com/monitoring](https://docs.agenticbrain.com/monitoring)
- **Community Forum**: [community.agenticbrain.com](https://community.agenticbrain.com)
- **Issue Tracker**: [github.com/agentic-brain/monitoring-metrics-service/issues](https://github.com/agentic-brain/monitoring-metrics-service/issues)
- **Email Support**: support@agenticbrain.com

### Service Level Agreements
- **Metrics Collection**: < 5 seconds average collection time
- **Alert Notification**: < 10 seconds for critical alerts
- **Dashboard Load Time**: < 2 seconds for dashboard rendering
- **Data Retention**: 90 days of historical metrics data
- **Uptime**: 99.9% service availability
- **API Response Time**: < 100ms for standard queries

---

**Built with ❤️ for the Agentic Brain Platform**

*Comprehensive monitoring and metrics collection for enterprise-grade AI automation*
