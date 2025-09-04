# Grafana Dashboards Service

A comprehensive visual monitoring and analytics service that provides pre-configured Grafana dashboards for the Agentic Brain platform, featuring real-time metrics visualization, interactive dashboards, and automated dashboard provisioning for agent performance monitoring, workflow analytics, system health visualization, SLA compliance tracking, and alert monitoring.

## 🎯 Features

### Core Dashboard Capabilities
- **Agent Performance Dashboards**: Real-time agent metrics, execution times, success rates, and performance analytics with interactive visualizations
- **Workflow Analytics**: Comprehensive workflow execution patterns, bottleneck analysis, throughput metrics, and success rate tracking
- **System Health Monitoring**: Service availability dashboards, response time tracking, resource utilization graphs, and health score calculations
- **SLA Compliance Dashboards**: Service level agreement monitoring with compliance indicators, breach analysis, and historical trends
- **Alert Monitoring**: Real-time alert visualization with severity breakdown, alert rates, and notification tracking

### Advanced Visualization Features
- **Interactive Dashboard Viewer**: Web-based dashboard browser with live previews and configuration options
- **Real-time Data Updates**: Auto-refreshing dashboards with configurable update intervals (30 seconds to 5 minutes)
- **Custom Panel Types**: Specialized visualizations including stat panels, graphs, tables, heatmaps, and pie charts
- **Responsive Design**: Mobile-friendly dashboards that adapt to different screen sizes and devices
- **Drill-down Capabilities**: Click-through navigation between related dashboards and detailed views
- **Time Range Selection**: Flexible time range controls for historical analysis and trend viewing

### Dashboard Management
- **Pre-configured Templates**: Ready-to-use dashboard templates for common monitoring scenarios
- **Dashboard Provisioning**: Automated deployment and configuration of dashboards in Grafana
- **Version Control**: Dashboard versioning and change tracking with rollback capabilities
- **Export/Import**: Dashboard export functionality for backup and sharing across environments
- **Template Customization**: Easy customization of dashboard templates for specific use cases
- **Multi-environment Support**: Environment-specific dashboard configurations and data sources

### Integration & Automation
- **Grafana API Integration**: Direct integration with Grafana API for dashboard creation and management
- **Prometheus Data Sources**: Automated configuration of Prometheus data sources for metrics collection
- **Alert Integration**: Grafana alert rules integration with notification channels
- **CI/CD Pipeline Integration**: Automated dashboard deployment as part of CI/CD pipelines
- **Multi-tenant Support**: Isolated dashboard environments for different teams and projects
- **API-driven Management**: RESTful APIs for programmatic dashboard creation and management

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Grafana instance (recommended: version 9.0+)
- Prometheus instance with Agentic Brain metrics
- Agentic Brain platform running

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
docker-compose up -d grafana-dashboards-service

# Check service health
curl http://localhost:8360/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

### Grafana Setup
```bash
# Access Grafana web interface
open http://localhost:3000

# Default credentials: admin/admin
# Change password on first login

# Import dashboards from the service
curl http://localhost:8360/dashboards/export/all | jq .dashboards
```

## 📡 API Endpoints

### Dashboard Management
```http
GET /dashboard-types                    # Get available dashboard types
POST /dashboards/create                 # Create new dashboard
GET /dashboards/{type}                  # Get dashboard template
GET /dashboards/export/all              # Export all dashboard templates
```

### Grafana Integration
```http
GET /prometheus/datasources             # Get Prometheus data source configs
GET /grafana/provisioning/dashboards    # Get dashboard provisioning configs
GET /grafana/provisioning/datasources   # Get data source provisioning configs
```

### Dashboard Viewer
```http
GET /dashboard                          # Interactive dashboard viewer
GET /health                             # Service health check
```

## 🎛️ Dashboard Types

### 1. Agent Performance Dashboard
**Purpose**: Monitor real-time agent performance metrics and analytics

**Key Panels**:
- Active Agents Count (Stat Panel)
- Agent Creation Rate (Graph)
- Agent Execution Time (95th percentile) (Graph)
- Agent Status Distribution (Pie Chart)
- Agent Performance Metrics Table (Table)

**Metrics**:
```prometheus
agent_brain_agents_total{status="active"}
rate(agent_brain_agent_creations_total[5m])
histogram_quantile(0.95, rate(agent_brain_agent_execution_duration_seconds_bucket[5m]))
```

### 2. Workflow Analytics Dashboard
**Purpose**: Analyze workflow execution patterns and performance

**Key Panels**:
- Workflow Success Rate (Stat Panel)
- Active Workflows Count (Stat Panel)
- Workflow Throughput (Stat Panel)
- Workflow Execution Time Trends (Graph)
- Workflow Status Breakdown (Bar Chart)
- Workflow Error Analysis (Table)

**Metrics**:
```prometheus
agent_brain_workflow_success_rate
agent_brain_workflows_total{status="active"}
rate(agent_brain_workflow_executions_total[5m])
histogram_quantile(0.95, rate(agent_brain_workflow_execution_duration_seconds_bucket[5m]))
```

### 3. System Health Dashboard
**Purpose**: Monitor overall system health and service availability

**Key Panels**:
- Overall System Health Score (Stat Panel)
- Service Health Status Table (Table)
- Service Response Times (95th percentile) (Graph)
- Task Performance Overview (Bar Chart)
- Plugin Usage Analytics (Heatmap)
- Resource Usage Trends (Graph)

**Metrics**:
```prometheus
(sum(agent_brain_service_health_status) / count(agent_brain_service_health_status)) * 100
agent_brain_service_health_status
histogram_quantile(0.95, rate(agent_brain_service_response_time_seconds_bucket[5m]))
rate(agent_brain_task_completions_total[5m])
rate(agent_brain_plugin_usage_total[5m])
```

### 4. SLA Compliance Dashboard
**Purpose**: Track service level agreement compliance and breaches

**Key Panels**:
- Overall SLA Compliance (Stat Panel)
- SLA Compliance by Service (Table)
- SLA Breach Analysis (Graph)

**Metrics**:
```prometheus
avg(sla_compliance_percentage)
sla_compliance_percentage
sla_breach_count
```

### 5. Alert Monitoring Dashboard
**Purpose**: Monitor real-time alerts and notification tracking

**Key Panels**:
- Active Alerts Count (Stat Panel)
- Alert Rate (per hour) (Stat Panel)
- Alerts by Severity (Pie Chart)
- Recent Alerts (Table)

**Metrics**:
```prometheus
alerts_total{state="firing"}
rate(alerts_total[1h])
alerts_total
```

## ⚙️ Configuration

### Environment Variables
```bash
# Service Configuration
GRAFANA_DASHBOARDS_PORT=8360
GRAFANA_DASHBOARDS_HOST=0.0.0.0

# Grafana Configuration
GRAFANA_URL=http://localhost:3000
GRAFANA_API_KEY=your_grafana_api_key
GRAFANA_USERNAME=admin
GRAFANA_PASSWORD=admin

# Prometheus Configuration
PROMETHEUS_URL=http://localhost:9090

# Agentic Brain Services
METRICS_SERVICE_URL=http://localhost:8350
AGENT_ORCHESTRATOR_URL=http://localhost:8200
WORKFLOW_ENGINE_URL=http://localhost:8202

# Dashboard Configuration
DASHBOARD_REFRESH_INTERVAL=30
ENABLE_AUTO_PROVISIONING=true
DASHBOARD_RETENTION_DAYS=90
```

### Grafana Data Source Configuration
```json
{
  "name": "AgenticBrain-Prometheus",
  "type": "prometheus",
  "url": "http://prometheus:9090",
  "access": "proxy",
  "isDefault": true,
  "jsonData": {
    "timeInterval": "30s",
    "queryTimeout": "60s",
    "httpMethod": "POST"
  }
}
```

### Dashboard Provisioning Configuration
```json
{
  "apiVersion": 1,
  "providers": [
    {
      "name": "agentic-brain-dashboards",
      "type": "file",
      "disableDeletion": false,
      "updateIntervalSeconds": 10,
      "allowUiUpdates": true,
      "options": {
        "path": "/var/lib/grafana/dashboards"
      }
    }
  ]
}
```

## 📊 Dashboard Templates

### Agent Performance Template
```json
{
  "dashboard": {
    "title": "Agent Performance Overview",
    "tags": ["agentic-brain", "performance", "agents"],
    "timezone": "browser",
    "panels": [
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
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "green", "value": null},
                {"color": "red", "value": 80}
              ]
            }
          }
        }
      }
    ]
  }
}
```

### System Health Template
```json
{
  "dashboard": {
    "title": "System Health Monitoring",
    "tags": ["agentic-brain", "health", "system"],
    "timezone": "browser",
    "panels": [
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
            "unit": "percent",
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "red", "value": null},
                {"color": "yellow", "value": 80},
                {"color": "green", "value": 95}
              ]
            }
          }
        }
      }
    ]
  }
}
```

## 🔧 Integration Examples

### Importing Dashboards into Grafana
```bash
# Get dashboard JSON from the service
curl http://localhost:8360/dashboards/agent_performance > agent_performance.json

# Import into Grafana via API
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @agent_performance.json \
  http://localhost:3000/api/dashboards/import
```

### Automated Dashboard Provisioning
```bash
# Export all dashboards
curl http://localhost:8360/dashboards/export/all > all_dashboards.json

# Create provisioning directory
mkdir -p /var/lib/grafana/dashboards/agentic-brain

# Save dashboard files
jq -r '.dashboards.agent_performance | tostring' all_dashboards.json > /var/lib/grafana/dashboards/agentic-brain/agent_performance.json
jq -r '.dashboards.workflow_analytics | tostring' all_dashboards.json > /var/lib/grafana/dashboards/agentic-brain/workflow_analytics.json
jq -r '.dashboards.system_health | tostring' all_dashboards.json > /var/lib/grafana/dashboards/agentic-brain/system_health.json
```

### Custom Dashboard Creation
```python
from grafana_api.grafana_face import GrafanaFace

# Initialize Grafana API client
grafana = GrafanaFace(
    auth=('admin', 'admin'),
    host='localhost',
    port=3000
)

# Get dashboard template
response = requests.get('http://localhost:8360/dashboards/agent_performance')
dashboard_config = response.json()

# Create dashboard
result = grafana.dashboard.update_dashboard(
    dashboard=dashboard_config['dashboard'],
    dashboard_id=None
)

print(f"Dashboard created: {result['url']}")
```

### Alert Rule Configuration
```yaml
# Grafana alert rules
groups:
  - name: agentic-brain-alerts
    rules:
      - alert: HighAgentFailureRate
        expr: rate(agent_brain_agents_total{status="failed"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High agent failure rate detected"
          description: "Agent failure rate is {{ $value }} over the last 5 minutes"

      - alert: ServiceUnavailable
        expr: agent_brain_service_health_status == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service is unavailable"
          description: "Service {{ $labels.service_name }} is not responding"
```

## 📈 Advanced Analytics

### Performance Baseline Comparison
```python
# Compare current performance with baseline
baseline_query = """
SELECT
    metric_name,
    baseline_value,
    current_value,
    CASE
        WHEN ABS(current_value - baseline_value) > (2 * standard_deviation)
        THEN 'anomaly'
        WHEN ABS(current_value - baseline_value) > standard_deviation
        THEN 'warning'
        ELSE 'normal'
    END as status
FROM performance_baselines pb
JOIN current_metrics cm ON pb.metric_name = cm.metric_name
"""

# Visualize in dashboard with conditional coloring
```

### Trend Analysis
```python
# Calculate performance trends over time
trend_query = """
SELECT
    metric_name,
    date,
    value,
    LAG(value) OVER (ORDER BY date) as previous_value,
    ((value - LAG(value) OVER (ORDER BY date)) / LAG(value) OVER (ORDER BY date)) * 100 as percent_change
FROM metrics_history
WHERE date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY metric_name, date
"""

# Create trend visualization with percentage changes
```

### Correlation Analysis
```python
# Analyze correlations between different metrics
correlation_query = """
SELECT
    m1.metric_name as metric1,
    m2.metric_name as metric2,
    CORR(m1.value, m2.value) as correlation_coefficient
FROM metrics m1
JOIN metrics m2 ON m1.timestamp = m2.timestamp
WHERE m1.metric_name < m2.metric_name
GROUP BY m1.metric_name, m2.metric_name
HAVING CORR(m1.value, m2.value) > 0.7
"""

# Visualize correlations as a heatmap
```

## 🎨 Custom Panel Development

### Creating Custom Panels
```javascript
// Custom Agent Performance Panel
export class AgentPerformancePanel {
  constructor() {
    this.panel = {
      type: 'agent-performance',
      name: 'Agent Performance',
      module: 'public/plugins/agentic-brain/agent-performance/module.js'
    };
  }

  render(data) {
    // Custom rendering logic
    const activeAgents = data.find(d => d.target === 'Active Agents');
    const executionTime = data.find(d => d.target === 'Execution Time');

    return `
      <div class="agent-performance-panel">
        <div class="metric active-agents">
          <span class="value">${activeAgents.datapoints[activeAgents.datapoints.length - 1][0]}</span>
          <span class="label">Active Agents</span>
        </div>
        <div class="metric execution-time">
          <span class="value">${executionTime.datapoints[executionTime.datapoints.length - 1][0].toFixed(2)}s</span>
          <span class="label">Avg Execution Time</span>
        </div>
      </div>
    `;
  }
}
```

### Panel Registration
```javascript
// Register custom panel
import { AgentPerformancePanel } from './agent-performance-panel';

export {
  AgentPerformancePanel as PanelComponent
};
```

## 🤝 API Integration

### RESTful API Usage
```python
import requests

# Get available dashboard types
response = requests.get('http://localhost:8360/dashboard-types')
dashboard_types = response.json()

# Create new dashboard
dashboard_request = {
    "title": "Custom Agent Dashboard",
    "description": "Custom dashboard for agent monitoring",
    "tags": ["custom", "agents"],
    "dashboard_type": "agent_performance"
}

response = requests.post('http://localhost:8360/dashboards/create', json=dashboard_request)
dashboard = response.json()

# Export dashboard for Grafana import
response = requests.get('http://localhost:8360/dashboards/export/all')
exported_dashboards = response.json()
```

### Grafana API Integration
```python
from grafana_api.grafana_face import GrafanaFace

# Initialize Grafana client
grafana = GrafanaFace(
    auth=('admin', 'your_password'),
    host='localhost',
    port=3000
)

# Create data source
datasource_config = {
    "name": "AgenticBrain-Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy"
}

grafana.datasource.create_datasource(datasource_config)

# Import dashboard
dashboard_config = requests.get('http://localhost:8360/dashboards/agent_performance').json()
grafana.dashboard.update_dashboard(dashboard_config['dashboard'])
```

## 📚 SDKs and Libraries

### Python SDK
```python
from agentic_brain_grafana import GrafanaDashboardsClient

# Initialize client
client = GrafanaDashboardsClient(
    base_url='http://localhost:8360',
    grafana_url='http://localhost:3000',
    api_key='your_api_key'
)

# Create and deploy dashboard
dashboard = client.create_dashboard(
    title="Production Agents",
    dashboard_type="agent_performance",
    tags=["production", "agents"]
)

client.deploy_to_grafana(dashboard)
```

### JavaScript SDK
```javascript
import { AgenticBrainGrafana } from 'agentic-brain-grafana-sdk';

const client = new AgenticBrainGrafana({
  baseURL: 'http://localhost:8360',
  grafanaURL: 'http://localhost:3000',
  apiKey: 'your_api_key'
});

// Create dashboard with custom configuration
const dashboard = await client.createDashboard({
  title: 'Custom Analytics',
  type: 'workflow_analytics',
  refresh: '10s',
  timeRange: { from: 'now-1h', to: 'now' }
});

console.log('Dashboard created:', dashboard.url);
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: [docs.agenticbrain.com/grafana-dashboards](https://docs.agenticbrain.com/grafana-dashboards)
- **Community Forum**: [community.agenticbrain.com/grafana](https://community.agenticbrain.com/grafana)
- **Issue Tracker**: [github.com/agentic-brain/grafana-dashboards-service/issues](https://github.com/agentic-brain/grafana-dashboards-service/issues)
- **Email Support**: grafana-support@agenticbrain.com

### Service Level Agreements
- **Dashboard Generation**: < 5 seconds for standard dashboard creation
- **Grafana Import**: < 10 seconds for dashboard import and provisioning
- **Real-time Updates**: < 30 seconds for metric updates in dashboards
- **API Response Time**: < 100ms for dashboard template requests
- **Uptime**: 99.9% service availability
- **Data Retention**: 90 days of dashboard configuration history

---

**Built with ❤️ for the Agentic Brain Platform**

*Enterprise-grade visual monitoring and analytics for AI automation*
