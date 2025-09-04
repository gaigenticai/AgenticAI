# Audit Logging Service

A comprehensive audit logging and compliance monitoring service for the Agentic Brain platform that provides enterprise-grade audit trails, compliance reporting, security monitoring, and data retention capabilities for regulatory compliance and security analysis.

## 🎯 Features

### Core Audit Logging
- **Comprehensive Event Logging**: Complete audit trail for all operations across the platform
- **Real-time Event Processing**: Asynchronous event ingestion and processing with queuing
- **Structured Event Data**: Rich, structured audit events with metadata and context
- **Multi-level Event Classification**: Authentication, authorization, operations, security, and compliance events
- **Event Enrichment**: Automatic event enrichment with compliance flags and risk scoring

### Compliance & Regulatory
- **GDPR Compliance**: Data protection and privacy compliance monitoring
- **SOX Compliance**: Financial and operational compliance tracking
- **HIPAA Compliance**: Healthcare data protection and audit requirements
- **PCI DSS Compliance**: Payment card industry security standards
- **Automated Compliance Reports**: Scheduled and on-demand compliance reporting
- **Data Retention Policies**: Configurable data retention and archiving policies

### Security Monitoring
- **Real-time Security Alerts**: Configurable alert rules and thresholds
- **Risk-based Event Scoring**: Automated risk assessment and prioritization
- **Anomaly Detection**: Statistical analysis for unusual activity patterns
- **Incident Response**: Automated incident detection and notification
- **Security Dashboard**: Real-time security monitoring and analytics

### Advanced Analytics
- **Search & Filtering**: Advanced search capabilities with complex filters
- **Elasticsearch Integration**: Full-text search and analytics capabilities
- **Trend Analysis**: Historical trends and pattern analysis
- **Performance Monitoring**: Audit system performance and latency tracking
- **Custom Dashboards**: Configurable dashboards for different user roles

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- PostgreSQL database
- Redis instance
- Elasticsearch (optional, for advanced search)
- SMTP server (for email alerts)

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
docker-compose up -d audit-logging-service

# Check service health
curl http://localhost:8340/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

## 📡 API Endpoints

### Audit Event Logging
```http
POST /audit/events
Content-Type: application/json

{
  "event_type": "operation",
  "severity": "info",
  "user_id": "user_123",
  "service_name": "agent-orchestrator",
  "operation": "agent_creation",
  "resource_type": "agent",
  "resource_id": "agent_456",
  "success": true,
  "execution_time_ms": 150.5,
  "tags": ["creation", "agent"],
  "metadata": {
    "agent_type": "underwriting",
    "template_used": "underwriting_v1"
  }
}
```

```http
POST /audit/events/batch
Content-Type: application/json

[
  {
    "event_type": "authentication",
    "severity": "info",
    "user_id": "user_123",
    "service_name": "authentication-service",
    "operation": "login",
    "resource_type": "user",
    "resource_id": "user_123",
    "success": true,
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0..."
  }
]
```

### Event Search & Retrieval
```http
GET /audit/events?event_type=operation&user_id=user_123&limit=50

# Advanced search with date range
GET /audit/events?start_date=2024-01-01T00:00:00Z&end_date=2024-01-31T23:59:59Z&severity=error

# Search by tags
GET /audit/events?tags=security,alert
```

```http
# Get specific event
GET /audit/events/{event_id}
```

### Compliance Reporting
```http
POST /compliance/reports
Content-Type: application/json

{
  "report_type": "gdpr",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T23:59:59Z"
}
```

```http
# List compliance reports
GET /compliance/reports?limit=20&offset=0
```

### Analytics & Monitoring
```http
# Get analytics summary
GET /analytics/summary?days=30

# Dashboard view
GET /dashboard
```

### Administrative Operations
```http
# Archive old events
POST /admin/archive?days=90

# Health check
GET /health

# Metrics
GET /metrics
```

## ⚙️ Configuration

### Environment Variables
```bash
# Service Configuration
AUDIT_SERVICE_PORT=8340
AUDIT_SERVICE_HOST=0.0.0.0

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/agentic_brain

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Elasticsearch (Optional)
ELASTICSEARCH_ENABLED=true
ELASTICSEARCH_HOST=elasticsearch
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX=audit_events

# Audit Configuration
MAX_EVENTS_PER_REQUEST=100
AUDIT_QUEUE_SIZE=10000
BATCH_SIZE=50

# Retention Policies
DEFAULT_RETENTION_DAYS=365
ARCHIVE_AFTER_DAYS=90
DELETE_AFTER_DAYS=2555

# Compliance Settings
GDPR_ENABLED=true
SOX_ENABLED=true
HIPAA_ENABLED=false
PCI_ENABLED=false

# Alert Configuration
ALERT_ENABLED=true
ALERT_EMAIL_ENABLED=true
ALERT_WEBHOOK_ENABLED=false

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=audit@agenticbrain.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=audit@agenticbrain.com

# Security
API_KEY_REQUIRED=true
RATE_LIMIT_ENABLED=true
MAX_REQUESTS_PER_MINUTE=1000

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8004
```

### Event Types & Severities

#### Event Types
- **authentication**: User login, logout, MFA operations
- **authorization**: Permission checks, role changes, access control
- **operation**: Business operations, data processing, agent execution
- **security**: Security events, failed authentications, suspicious activity
- **compliance**: Compliance-related operations, data exports, audits

#### Severity Levels
- **info**: Normal operations and successful activities
- **warning**: Unusual but not critical events
- **error**: Failed operations and error conditions
- **critical**: Critical security events and system failures

## 🔐 Integration Examples

### FastAPI Service Integration
```python
from fastapi import Request, BackgroundTasks
import httpx

class AuditLogger:
    def __init__(self, audit_service_url: str, service_name: str):
        self.audit_url = audit_service_url
        self.service_name = service_name

    async def log_event(self, event_data: dict, request: Request = None):
        """Log audit event"""
        enriched_data = event_data.copy()
        enriched_data.update({
            'service_name': self.service_name,
            'ip_address': request.client.host if request else None,
            'user_agent': request.headers.get('user-agent') if request else None
        })

        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{self.audit_url}/audit/events",
                    json=enriched_data,
                    headers={'X-API-Key': 'your-audit-api-key'}
                )
            except Exception as e:
                # Log locally if audit service is unavailable
                print(f"Audit logging failed: {e}")

# Initialize audit logger
audit_logger = AuditLogger("http://localhost:8340", "agent-orchestrator")

# Use in your service
@app.post("/orchestrator/execute-task")
async def execute_task(request: Request, background_tasks: BackgroundTasks):
    try:
        # Your business logic here
        result = await process_task()

        # Log successful operation
        await audit_logger.log_event({
            'event_type': 'operation',
            'severity': 'info',
            'operation': 'task_execution',
            'resource_type': 'task',
            'resource_id': task_id,
            'success': True,
            'execution_time_ms': execution_time,
            'user_id': current_user_id,
            'metadata': {'task_type': 'underwriting'}
        }, request)

        return result

    except Exception as e:
        # Log failed operation
        await audit_logger.log_event({
            'event_type': 'operation',
            'severity': 'error',
            'operation': 'task_execution',
            'resource_type': 'task',
            'resource_id': task_id,
            'success': False,
            'error_message': str(e),
            'user_id': current_user_id
        }, request)

        raise
```

### Agent Brain Service Integration
```python
from audit_logger import AuditLogger

class AgentOrchestrator:
    def __init__(self):
        self.audit_logger = AuditLogger("http://localhost:8340", "agent-orchestrator")

    async def create_agent(self, agent_config: dict, user_id: str, request: Request):
        """Create new agent with audit logging"""

        start_time = time.time()

        try:
            # Create agent logic
            agent_id = await self._create_agent_logic(agent_config)
            execution_time = (time.time() - start_time) * 1000

            # Log successful creation
            await self.audit_logger.log_event({
                'event_type': 'operation',
                'severity': 'info',
                'operation': 'agent_creation',
                'resource_type': 'agent',
                'resource_id': agent_id,
                'success': True,
                'execution_time_ms': execution_time,
                'user_id': user_id,
                'metadata': {
                    'agent_type': agent_config.get('type'),
                    'template_used': agent_config.get('template')
                }
            }, request)

            return agent_id

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            # Log failed creation
            await self.audit_logger.log_event({
                'event_type': 'operation',
                'severity': 'error',
                'operation': 'agent_creation',
                'resource_type': 'agent',
                'success': False,
                'execution_time_ms': execution_time,
                'user_id': user_id,
                'error_message': str(e),
                'metadata': {'agent_config': agent_config}
            }, request)

            raise

    async def execute_task(self, task_data: dict, user_id: str, request: Request):
        """Execute task with comprehensive audit logging"""

        start_time = time.time()
        task_id = task_data.get('task_id')

        try:
            # Execute task
            result = await self._execute_task_logic(task_data)
            execution_time = (time.time() - start_time) * 1000

            # Log successful execution
            await self.audit_logger.log_event({
                'event_type': 'operation',
                'severity': 'info',
                'operation': 'task_execution',
                'resource_type': 'task',
                'resource_id': task_id,
                'success': True,
                'execution_time_ms': execution_time,
                'user_id': user_id,
                'metadata': {
                    'task_type': task_data.get('type'),
                    'agent_id': task_data.get('agent_id'),
                    'result_summary': self._summarize_result(result)
                }
            }, request)

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            # Log failed execution
            await self.audit_logger.log_event({
                'event_type': 'operation',
                'severity': 'error',
                'operation': 'task_execution',
                'resource_type': 'task',
                'resource_id': task_id,
                'success': False,
                'execution_time_ms': execution_time,
                'user_id': user_id,
                'error_message': str(e),
                'metadata': {'task_data': task_data}
            }, request)

            raise
```

### Authentication Service Integration
```python
# In authentication service
async def authenticate_user(self, username: str, password: str, request: Request):
    """Authenticate user with audit logging"""

    start_time = time.time()

    try:
        user = await self._validate_credentials(username, password)
        execution_time = (time.time() - start_time) * 1000

        # Log successful authentication
        await self.audit_logger.log_event({
            'event_type': 'authentication',
            'severity': 'info',
            'operation': 'login',
            'resource_type': 'user',
            'resource_id': user.id,
            'success': True,
            'execution_time_ms': execution_time,
            'user_id': user.id,
            'metadata': {
                'login_method': 'password',
                'ip_address': request.client.host,
                'user_agent': request.headers.get('user-agent')
            }
        }, request)

        return user

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000

        # Log failed authentication
        await self.audit_logger.log_event({
            'event_type': 'authentication',
            'severity': 'warning',
            'operation': 'login',
            'resource_type': 'user',
            'success': False,
            'execution_time_ms': execution_time,
            'error_message': str(e),
            'metadata': {
                'username': username,
                'ip_address': request.client.host,
                'user_agent': request.headers.get('user-agent')
            }
        }, request)

        raise
```

## 📊 Compliance & Reporting

### GDPR Compliance Monitoring
```python
# GDPR-specific audit events
async def log_gdpr_event(self, operation: str, user_id: str, data_details: dict):
    """Log GDPR-related operations"""

    await self.audit_logger.log_event({
        'event_type': 'compliance',
        'severity': 'info',
        'operation': operation,
        'resource_type': 'user_data',
        'resource_id': user_id,
        'success': True,
        'compliance_flags': {'gdpr': True},
        'user_id': user_id,
        'metadata': {
            'gdpr_operation': operation,
            'data_categories': data_details.get('categories', []),
            'legal_basis': data_details.get('legal_basis'),
            'retention_period': data_details.get('retention_days')
        }
    })
```

### SOX Compliance Tracking
```python
# SOX-specific audit events
async def log_sox_event(self, operation: str, user_id: str, financial_data: dict):
    """Log SOX-related financial operations"""

    await self.audit_logger.log_event({
        'event_type': 'compliance',
        'severity': 'info',
        'operation': operation,
        'resource_type': 'financial_record',
        'success': True,
        'compliance_flags': {'sox': True},
        'user_id': user_id,
        'risk_score': 6.0,  # Financial operations are higher risk
        'metadata': {
            'sox_control': financial_data.get('control_id'),
            'transaction_amount': financial_data.get('amount'),
            'approval_required': financial_data.get('requires_approval')
        }
    })
```

### Automated Compliance Reports
```python
# Generate compliance report
async def generate_compliance_report(self, report_type: str, start_date: datetime, end_date: datetime):
    """Generate automated compliance report"""

    # Query audit events for the period
    events = await self._query_events_by_date_range(start_date, end_date)

    # Analyze compliance
    compliance_analysis = await self._analyze_compliance(events, report_type)

    # Generate findings and recommendations
    findings = []
    recommendations = []

    if compliance_analysis['violation_rate'] > 0.05:  # 5% threshold
        findings.append("Compliance violation rate exceeds threshold")
        recommendations.append("Review access controls and audit policies")

    # Store report
    report_id = await self._store_compliance_report({
        'report_type': report_type,
        'period': f"{start_date.date()} to {end_date.date()}",
        'total_events': len(events),
        'compliant_events': compliance_analysis['compliant_count'],
        'violations': compliance_analysis['violation_count'],
        'findings': findings,
        'recommendations': recommendations
    })

    return report_id
```

## 🚨 Security Alerts & Monitoring

### Alert Rule Configuration
```python
# Define security alert rules
alert_rules = [
    {
        'name': 'Multiple Failed Logins',
        'event_type': 'authentication',
        'conditions': {'success': False, 'operation': 'login'},
        'threshold': 5,
        'time_window_minutes': 10,
        'severity': 'warning',
        'notification_channels': ['email', 'webhook']
    },
    {
        'name': 'High Risk Operations',
        'event_type': 'operation',
        'conditions': {'risk_score': {'gt': 8.0}},
        'threshold': 1,
        'time_window_minutes': 5,
        'severity': 'critical',
        'notification_channels': ['email', 'sms']
    },
    {
        'name': 'Unauthorized Access Attempts',
        'event_type': 'authorization',
        'conditions': {'success': False, 'operation': 'access_check'},
        'threshold': 3,
        'time_window_minutes': 15,
        'severity': 'warning',
        'notification_channels': ['email']
    }
]
```

### Real-time Alert Processing
```python
async def process_security_alerts(self, events: List[dict]):
    """Process events for security alerts"""

    for rule in self.alert_rules:
        # Check if events match rule conditions
        matching_events = [
            event for event in events
            if self._matches_rule_conditions(event, rule['conditions'])
        ]

        # Check threshold
        if len(matching_events) >= rule['threshold']:
            await self._trigger_alert(rule, matching_events)
```

### Alert Notification System
```python
async def send_alert_notification(self, alert: dict, channels: List[str]):
    """Send alert notifications via configured channels"""

    alert_message = self._format_alert_message(alert)

    for channel in channels:
        if channel == 'email':
            await self._send_email_alert(alert_message, alert)
        elif channel == 'webhook':
            await self._send_webhook_alert(alert_message, alert)
        elif channel == 'sms':
            await self._send_sms_alert(alert_message, alert)
```

## 📈 Analytics & Dashboards

### Real-time Analytics
```python
# Get analytics summary
@app.get("/analytics/summary")
async def get_analytics_summary(days: int = 30):
    """Get comprehensive analytics summary"""

    # Event type distribution
    event_types = await self._get_event_type_distribution(days)

    # Severity distribution
    severities = await self._get_severity_distribution(days)

    # Risk score analysis
    risk_analysis = await self._get_risk_score_analysis(days)

    # User activity metrics
    user_activity = await self._get_user_activity_metrics(days)

    # Compliance metrics
    compliance_metrics = await self._get_compliance_metrics(days)

    return {
        'period_days': days,
        'total_events': sum(event_types.values()),
        'event_types': event_types,
        'severities': severities,
        'risk_analysis': risk_analysis,
        'user_activity': user_activity,
        'compliance_metrics': compliance_metrics
    }
```

### Custom Dashboard Creation
```python
# Create custom dashboard
@app.post("/dashboards")
async def create_dashboard(dashboard_config: dict):
    """Create custom audit dashboard"""

    dashboard = {
        'id': str(uuid.uuid4()),
        'name': dashboard_config['name'],
        'description': dashboard_config['description'],
        'widgets': dashboard_config['widgets'],
        'filters': dashboard_config['filters'],
        'refresh_interval': dashboard_config.get('refresh_interval', 300),
        'permissions': dashboard_config.get('permissions', ['admin'])
    }

    # Store dashboard configuration
    await self._store_dashboard(dashboard)

    return {'dashboard_id': dashboard['id']}
```

## 🔧 Advanced Features

### Data Archiving & Retention
```python
async def archive_old_events(self, days_to_archive: int = 90):
    """Archive events older than specified days"""

    cutoff_date = datetime.utcnow() - timedelta(days=days_to_archive)

    # Find events to archive
    events_to_archive = await self._find_events_older_than(cutoff_date)

    if not events_to_archive:
        return {'archived_count': 0}

    # Create archive file
    archive_id = str(uuid.uuid4())
    archive_filename = f"audit_archive_{archive_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json.gz"

    # Compress and store events
    await self._create_compressed_archive(events_to_archive, archive_filename)

    # Update database records
    await self._mark_events_as_archived(events_to_archive, archive_id)

    # Clean up old archives if needed
    await self._cleanup_old_archives()

    return {
        'archived_count': len(events_to_archive),
        'archive_id': archive_id,
        'archive_file': archive_filename
    }
```

### Advanced Search Capabilities
```python
async def advanced_search(self, search_query: dict):
    """Perform advanced audit event search"""

    # Build complex query
    query_builder = AuditQueryBuilder()

    # Add filters
    if 'event_type' in search_query:
        query_builder.add_filter('event_type', search_query['event_type'])

    if 'date_range' in search_query:
        query_builder.add_date_range(
            search_query['date_range']['start'],
            search_query['date_range']['end']
        )

    if 'user_id' in search_query:
        query_builder.add_filter('user_id', search_query['user_id'])

    if 'risk_score_range' in search_query:
        query_builder.add_range_filter(
            'risk_score',
            search_query['risk_score_range']['min'],
            search_query['risk_score_range']['max']
        )

    # Add text search
    if 'query_string' in search_query:
        query_builder.add_text_search(search_query['query_string'])

    # Execute search
    if self.elasticsearch_enabled:
        results = await self._elasticsearch_search(query_builder)
    else:
        results = await self._database_search(query_builder)

    return results
```

## 📚 API Documentation

### Complete API Reference
- [Audit Events API](./docs/api/audit-events.md)
- [Compliance API](./docs/api/compliance.md)
- [Analytics API](./docs/api/analytics.md)
- [Alert Management API](./docs/api/alerts.md)
- [Dashboard API](./docs/api/dashboards.md)

### SDKs and Libraries
- **Python SDK**: `pip install agentic-brain-audit`
- **REST API Client**: Comprehensive HTTP client libraries
- **Integration Libraries**: Pre-built integrations for popular frameworks

## 🤝 Contributing

### Development Setup
```bash
# Clone the repository
git clone https://github.com/agentic-brain/audit-logging-service.git
cd audit-logging-service

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
2. Create a feature branch (`git checkout -b feature/gdpr-compliance`)
3. Make your changes with tests
4. Ensure all tests pass (`pytest tests/ -v`)
5. Update documentation if needed
6. Submit a pull request with detailed description

### Security Considerations
- Never commit sensitive audit data
- Use encryption for sensitive event data
- Implement proper access controls for audit APIs
- Follow data retention and privacy regulations
- Regular security code reviews for audit functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: [docs.agenticbrain.com/audit](https://docs.agenticbrain.com/audit)
- **Community Forum**: [community.agenticbrain.com](https://community.agenticbrain.com)
- **Issue Tracker**: [github.com/agentic-brain/audit-logging-service/issues](https://github.com/agentic-brain/audit-logging-service/issues)
- **Email Support**: support@agenticbrain.com

### Service Level Agreements
- **Response Time**: < 4 hours for critical audit issues
- **Resolution Time**: < 24 hours for standard issues
- **Data Retention**: 7 years for compliance-related audit data
- **Uptime**: 99.9% service availability
- **Event Processing**: < 5 seconds for 95% of audit events
- **Search Response**: < 2 seconds for standard searches

---

**Built with ❤️ for the Agentic Brain Platform**

*Ensuring compliance, security, and transparency through comprehensive audit logging*
