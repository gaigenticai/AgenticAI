# Error Handling Service

A comprehensive error management, logging, and recovery service for the Agentic Brain platform that provides enterprise-grade error handling patterns across all Agent Brain services with intelligent error classification, automated recovery strategies, and comprehensive error monitoring and analytics.

## 🎯 Features

### Core Error Management
- **Global Error Classification**: Intelligent error categorization based on message analysis and pattern matching
- **Automated Error Recovery**: Configurable recovery strategies including retry, failover, restart, and rollback
- **Error Pattern Recognition**: Machine learning-based error pattern detection and classification
- **Recovery Strategy Orchestration**: Automated execution of recovery workflows with monitoring
- **Error Correlation Analysis**: Advanced correlation detection between different error types
- **Root Cause Analysis**: Automated root cause identification and impact assessment

### Advanced Error Handling
- **Service-Specific Error Patterns**: Custom error handling rules for different microservices
- **Multi-Level Error Severity**: Critical, High, Medium, and Low severity classification
- **Error Aggregation and Analysis**: Real-time error aggregation with trend analysis
- **Performance Impact Assessment**: Automatic assessment of error impact on system performance
- **Error Prevention Recommendations**: AI-powered recommendations for error prevention
- **Historical Error Analysis**: Long-term error trend analysis and forecasting

### Recovery & Remediation
- **Intelligent Retry Logic**: Smart retry mechanisms with exponential backoff
- **Failover Management**: Automatic failover to backup services and systems
- **Graceful Degradation**: Controlled service degradation during error conditions
- **Rollback Capabilities**: Automatic rollback to previous stable states
- **Circuit Breaker Pattern**: Implementation of circuit breaker for external service calls
- **Self-Healing Systems**: Autonomous error recovery and system stabilization

### Monitoring & Analytics
- **Real-time Error Dashboard**: Interactive dashboard with live error monitoring
- **Error Trend Visualization**: Charts and graphs showing error patterns over time
- **Service Health Monitoring**: Comprehensive service health status tracking
- **Alert Integration**: Integration with alerting systems for critical error notifications
- **Error Rate Monitoring**: Real-time error rate tracking and alerting
- **MTTR (Mean Time To Resolution) Tracking**: Automated calculation of error resolution times

### Integration & Automation
- **API Integration**: RESTful APIs for error reporting and management
- **Webhook Support**: Webhook notifications for external system integration
- **Audit Logging**: Complete audit trail of all error handling activities
- **Metrics Export**: Prometheus metrics export for error monitoring
- **CI/CD Integration**: Automated error handling in deployment pipelines
- **Multi-environment Support**: Environment-specific error handling configurations

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- PostgreSQL database
- Redis instance
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
docker-compose up -d error-handling-service

# Check service health
curl http://localhost:8370/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

### Error Reporting Example
```bash
# Report an error
curl -X POST http://localhost:8370/errors/report \
  -H "Content-Type: application/json" \
  -d '{
    "service_name": "agent_orchestrator",
    "error_message": "Connection timeout to workflow engine",
    "error_type": "network_error",
    "user_id": "user123",
    "session_id": "session456"
  }'
```

## 📡 API Endpoints

### Error Management
```http
POST /errors/report                    # Report a new error
GET /errors                           # Get errors with filtering
GET /errors/{error_id}                # Get specific error details
POST /recovery/{error_id}             # Execute recovery for error
```

### Analytics & Monitoring
```http
GET /analytics/errors                 # Get error analytics
GET /analytics/correlations           # Get error correlations
GET /analytics/report                 # Get comprehensive error report
```

### Pattern Management
```http
GET /patterns                        # Get all error patterns
POST /patterns                       # Create new error pattern
```

### Dashboard & Visualization
```http
GET /dashboard                       # Interactive error dashboard
GET /health                          # Service health check
```

## 🎛️ Error Classification System

### Error Categories
- **NETWORK**: Network connectivity and communication errors
- **DATABASE**: Database connection, query, and transaction errors
- **AUTHENTICATION**: User authentication and authorization failures
- **AUTHORIZATION**: Permission and access control errors
- **VALIDATION**: Input validation and data integrity errors
- **BUSINESS_LOGIC**: Application business logic errors
- **EXTERNAL_SERVICE**: Third-party service integration errors
- **RESOURCE_LIMIT**: Resource exhaustion and limit errors
- **CONFIGURATION**: Configuration and setup errors
- **SECURITY**: Security violations and breach attempts
- **PERFORMANCE**: Performance degradation and timeout errors

### Error Severity Levels
- **CRITICAL**: System stability threatened, immediate action required
- **HIGH**: Major functionality impacted, urgent attention needed
- **MEDIUM**: Partial functionality affected, attention needed
- **LOW**: Minor issues, can be addressed in regular maintenance

### Recovery Strategies
- **RETRY**: Retry the operation with configurable backoff
- **FAILOVER**: Switch to backup service or system
- **RESTART**: Restart the affected service or component
- **ROLLBACK**: Rollback to previous stable state
- **NOTIFICATION**: Send notifications for manual intervention
- **MANUAL_INTERVENTION**: Requires human intervention
- **IGNORE**: Log but don't take action (for low-severity errors)

## 🔧 Configuration

### Environment Variables
```bash
# Service Configuration
ERROR_HANDLING_PORT=8370
ERROR_HANDLING_HOST=0.0.0.0

# Error Handling Configuration
ENABLE_AUTO_RECOVERY=true
ERROR_RETENTION_DAYS=90
MAX_RECOVERY_ATTEMPTS=3
RECOVERY_TIMEOUT_SECONDS=300

# Alert Configuration
ENABLE_ERROR_ALERTS=true
CRITICAL_ERROR_THRESHOLD=10
ERROR_RATE_ALERT_THRESHOLD=0.1

# Agent Brain Service URLs
AGENT_ORCHESTRATOR_URL=http://localhost:8200
PLUGIN_REGISTRY_URL=http://localhost:8201
WORKFLOW_ENGINE_URL=http://localhost:8202
BRAIN_FACTORY_URL=http://localhost:8301
DEPLOYMENT_PIPELINE_URL=http://localhost:8303
AUDIT_LOGGING_URL=http://localhost:8340
MONITORING_SERVICE_URL=http://localhost:8350

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/agentic_brain

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### Error Pattern Configuration
```python
# Custom error pattern definition
error_pattern = {
    "name": "Database Connection Timeout",
    "description": "Database connection timeout errors",
    "error_keywords": ["timeout", "connection", "database", "pool"],
    "category": "database",
    "severity": "high",
    "recovery_strategy": "retry",
    "max_retries": 3,
    "retry_delay_seconds": 10,
    "alert_required": True
}
```

### Recovery Strategy Configuration
```python
# Recovery strategy configuration
recovery_config = {
    "strategy": "retry",
    "max_attempts": 3,
    "backoff_multiplier": 2.0,
    "initial_delay": 5,
    "max_delay": 300,
    "jitter": True
}
```

## 📊 Error Pattern Engine

### Intelligent Classification
```python
# Error classification example
error_instance = error_pattern_engine.classify_error(
    error_message="Connection timeout to PostgreSQL database",
    stack_trace="psycopg2.OperationalError: connection timeout"
)

print(f"Category: {error_instance.category}")        # database
print(f"Severity: {error_instance.severity}")        # high
print(f"Recovery: {error_instance.recovery_strategy}") # retry
```

### Pattern Matching Algorithm
```python
# Pattern matching with keyword analysis
def classify_error(self, error_message: str) -> ErrorInstance:
    keywords = ["timeout", "connection", "database"]
    matches = sum(1 for keyword in keywords if keyword in error_message.lower())

    if matches >= 2:
        return ErrorInstance(
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY
        )
```

### Custom Pattern Definition
```python
# Define custom error pattern
custom_pattern = ErrorPattern(
    pattern_id="custom_network_error",
    name="Custom Network Error",
    description="Custom network connectivity issues",
    error_keywords=["network", "connectivity", "unreachable", "dns"],
    category=ErrorCategory.NETWORK,
    severity=ErrorSeverity.MEDIUM,
    recovery_strategy=RecoveryStrategy.FAILOVER,
    max_retries=2,
    alert_required=True
)
```

## 🔄 Recovery Engine

### Recovery Strategy Execution
```python
# Execute recovery strategy
recovery_result = await recovery_engine.execute_recovery(error_instance)

if recovery_result["success"]:
    print(f"Recovery successful: {recovery_result['strategy']}")
    # Update error status to resolved
else:
    print(f"Recovery failed: {recovery_result['error']}")
    # Escalate to manual intervention
```

### Retry Strategy Implementation
```python
async def _execute_retry_strategy(self, error_instance: ErrorInstance):
    if error_instance.recovery_attempts >= error_instance.max_recovery_attempts:
        return {"success": False, "error": "Max retries exceeded"}

    # Implement exponential backoff
    delay = min(2 ** error_instance.recovery_attempts, 300)  # Max 5 minutes
    await asyncio.sleep(delay)

    # Attempt recovery
    success = await self._attempt_service_recovery(error_instance)

    return {
        "success": success,
        "attempts": error_instance.recovery_attempts + 1,
        "next_retry_in": delay * 2
    }
```

### Failover Strategy Implementation
```python
async def _execute_failover_strategy(self, error_instance: ErrorInstance):
    # Identify backup services
    backup_services = await self._find_backup_services(error_instance.service_name)

    for backup in backup_services:
        success = await self._switch_to_backup_service(backup)
        if success:
            return {
                "success": True,
                "failover_service": backup["name"],
                "estimated_recovery_time": 60
            }

    return {"success": False, "error": "No suitable failover target found"}
```

## 📈 Error Analytics Engine

### Error Trend Analysis
```python
# Analyze error trends over time
analytics = error_analytics.analyze_error_patterns(hours=24)

print(f"Total errors: {analytics['total_errors']}")
print(f"Error rate: {analytics['error_rate_per_hour']} per hour")
print(f"Top categories: {analytics['top_error_categories']}")
```

### Correlation Detection
```python
# Detect error correlations
correlations = error_analytics.detect_error_correlations()

for correlation in correlations["correlations"]:
    print(f"Correlation: {correlation['error_a']} ↔ {correlation['error_b']}")
    print(f"Strength: {correlation['correlation_strength']}")
```

### Comprehensive Reporting
```python
# Generate comprehensive error report
report = error_analytics.generate_error_report(days=7)

print(f"Resolution rate: {report['summary']['resolution_rate']}%")
print(f"Average MTTR: {report['service_breakdown'][0]['mttr_hours']} hours")
```

## 🎨 Interactive Dashboard

### Dashboard Features
- **Real-time Error Monitoring**: Live updates with error statistics
- **Interactive Filtering**: Filter by service, severity, category, and time range
- **Error Trend Charts**: Visual representation of error patterns over time
- **Recovery Status Tracking**: Monitor recovery execution and success rates
- **Alert Management**: View and manage active alerts
- **Performance Metrics**: Error impact on system performance

### Dashboard Components
```html
<!-- Error Statistics Cards -->
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value" id="total-errors">0</div>
        <div class="stat-label">Total Errors (24h)</div>
    </div>
    <div class="stat-card">
        <div class="stat-value" id="critical-errors">0</div>
        <div class="stat-label">Critical Errors</div>
    </div>
</div>

<!-- Error Trend Chart -->
<div class="chart-container">
    <h3>Error Trends (Last 24 Hours)</h3>
    <canvas id="error-trends-chart"></canvas>
</div>
```

## 🔗 Integration Examples

### Service Integration
```python
# In any Agent Brain service
from error_handling_client import ErrorHandler

error_handler = ErrorHandler(
    service_url="http://localhost:8370",
    service_name="agent_orchestrator"
)

try:
    # Your service logic here
    result = await process_workflow(workflow_data)
except Exception as e:
    # Report error to error handling service
    await error_handler.report_error(
        error_message=str(e),
        error_type="workflow_processing_error",
        user_id=request.user_id,
        session_id=request.session_id,
        request_id=request.request_id,
        stack_trace=traceback.format_exc()
    )
    raise
```

### Prometheus Integration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'error-handling-service'
    static_configs:
      - targets: ['error-handling-service:8370']
    scrape_interval: 30s
    metrics_path: '/metrics'
```

### Alertmanager Integration
```yaml
# alertmanager.yml
route:
  group_by: ['service_name', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'error-handling-webhook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-errors'

receivers:
- name: 'error-handling-webhook'
  webhook_configs:
  - url: 'http://error-handling-service:8370/webhooks/alert'
    send_resolved: true
```

## 🤝 API Integration

### RESTful API Usage
```python
import requests

# Report an error
error_data = {
    "service_name": "agent_orchestrator",
    "error_message": "Failed to connect to workflow engine",
    "error_type": "connection_error",
    "user_id": "user123",
    "session_id": "session456",
    "metadata": {
        "workflow_id": "wf_789",
        "attempt_count": 3
    }
}

response = requests.post(
    'http://localhost:8370/errors/report',
    json=error_data
)

error_id = response.json()['error_id']

# Execute recovery
recovery_response = requests.post(
    f'http://localhost:8370/recovery/{error_id}'
)
```

### Python SDK
```python
from agentic_brain_error_handling import ErrorHandlingClient

client = ErrorHandlingClient(
    base_url='http://localhost:8370',
    service_name='my_service'
)

# Report error with context
error_id = await client.report_error(
    error=Exception("Database connection failed"),
    user_id="user123",
    session_id="session456",
    metadata={"database": "postgres", "operation": "select"}
)

# Execute automatic recovery
recovery_result = await client.execute_recovery(error_id)

# Get error analytics
analytics = await client.get_error_analytics(hours=24)
```

### JavaScript SDK
```javascript
import { AgenticBrainErrorHandler } from 'agentic-brain-error-handling-sdk';

const errorHandler = new AgenticBrainErrorHandler({
  baseURL: 'http://localhost:8370',
  serviceName: 'frontend-service'
});

// Report error with context
const errorId = await errorHandler.reportError({
  message: 'API call failed',
  type: 'network_error',
  userId: 'user123',
  sessionId: 'session456',
  metadata: {
    endpoint: '/api/workflows',
    method: 'POST',
    statusCode: 500
  }
});

// Monitor recovery
const recoveryStatus = await errorHandler.getRecoveryStatus(errorId);
```

## 📚 SDKs and Libraries

### Python SDK
```python
from agentic_brain_error_handling import ErrorHandler, RecoveryManager

# Initialize error handler
handler = ErrorHandler(
    service_url="http://localhost:8370",
    service_name="my_microservice",
    enable_auto_recovery=True
)

# Context manager for automatic error handling
async with handler.error_context(user_id="user123"):
    # Your business logic here
    await process_data(data)

# Manual error reporting
await handler.report_error(
    error=ValueError("Invalid input data"),
    severity="medium",
    recovery_strategy="retry"
)
```

### Go SDK
```go
package main

import (
    "context"
    "github.com/agentic-brain/error-handling-go"
)

func main() {
    handler := errorhandling.NewHandler(&errorhandling.Config{
        ServiceURL:    "http://localhost:8370",
        ServiceName:   "my-service",
        EnableAutoRecovery: true,
    })

    // Report error
    err := handler.ReportError(context.Background(), &errorhandling.ErrorReport{
        Message:   "Database connection failed",
        Type:      "database_error",
        Severity:  "high",
        UserID:    "user123",
        SessionID: "session456",
    })

    // Execute recovery
    recoveryResult := handler.ExecuteRecovery(context.Background(), errorID)
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: [docs.agenticbrain.com/error-handling](https://docs.agenticbrain.com/error-handling)
- **Community Forum**: [community.agenticbrain.com/error-handling](https://community.agenticbrain.com/error-handling)
- **Issue Tracker**: [github.com/agentic-brain/error-handling-service/issues](https://github.com/agentic-brain/error-handling-service/issues)
- **Email Support**: error-handling-support@agenticbrain.com

### Service Level Agreements
- **Error Classification**: < 1 second for error pattern matching and classification
- **Recovery Execution**: < 30 seconds for automated recovery strategy execution
- **Analytics Generation**: < 5 seconds for error analytics and reporting
- **Dashboard Load Time**: < 2 seconds for dashboard rendering
- **API Response Time**: < 100ms for standard error reporting operations
- **Uptime**: 99.9% service availability
- **Data Retention**: 90 days of error history and recovery actions

---

**Built with ❤️ for the Agentic Brain Platform**

*Enterprise-grade error management and recovery for AI automation*
