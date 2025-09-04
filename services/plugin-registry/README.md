# Plugin Registry Service

The Plugin Registry Service is a core component of the Agentic Brain Platform that manages the registration, discovery, loading, and execution of plugins. It provides a centralized hub for both domain-specific plugins (underwriting, claims, fraud detection) and generic plugins (data processing, validation).

## Features

- **Plugin Management**: Register, discover, and manage plugins
- **Plugin Execution**: Execute plugins with input data and configuration
- **Built-in Plugins**: Includes pre-built domain and generic plugins
- **Plugin Interfaces**: Standardized interfaces for plugin development
- **Security Validation**: Plugin security scanning and validation
- **Usage Analytics**: Track plugin usage and performance metrics
- **RESTful API**: Complete API for plugin operations
- **Authentication Integration**: Support for JWT-based authentication
- **Monitoring & Metrics**: Prometheus metrics for observability
- **Database Persistence**: Store plugin metadata and execution history

## Built-in Plugins

### Domain-Specific Plugins

#### Risk Calculator (`riskCalculator`)
- **Domain**: Underwriting
- **Purpose**: Advanced risk calculation for loan underwriting decisions
- **Features**:
  - Credit score analysis
  - Debt-to-income ratio evaluation
  - Loan-to-income ratio assessment
  - Risk category determination
  - Approval probability calculation

#### Fraud Detector (`fraudDetector`)
- **Domain**: Claims
- **Purpose**: Machine learning-based fraud detection for claims processing
- **Features**:
  - Claim pattern analysis
  - Time-based anomaly detection
  - Frequency analysis
  - Risk scoring
  - Investigation recommendations

#### Regulatory Checker (`regulatoryChecker`)
- **Domain**: Compliance
- **Purpose**: Automated regulatory compliance verification
- **Features**:
  - Compliance rule validation
  - Regulatory requirement checking
  - Audit trail generation
  - Compliance reporting

### Generic Plugins

#### Data Retriever (`dataRetriever`)
- **Purpose**: Generic data retrieval and transformation
- **Features**:
  - Multi-source data retrieval
  - Data transformation
  - Query execution
  - Result formatting

#### Validator (`validator`)
- **Purpose**: Comprehensive data validation and quality assurance
- **Features**:
  - Required field validation
  - Data type checking
  - Format validation
  - Quality scoring
  - Error reporting

## API Endpoints

### Plugin Management
- `GET /plugins` - List all available plugins
- `GET /plugins/{plugin_id}` - Get detailed plugin information
- `POST /plugins/execute` - Execute a plugin with input data
- `GET /plugins/types` - Get available plugin types and domains
- `GET /plugins/stats` - Get plugin usage statistics

### Monitoring
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics endpoint

## Configuration

The service supports the following environment variables:

### Database Configuration
- `POSTGRES_HOST` - PostgreSQL host (default: postgresql_ingestion)
- `POSTGRES_PORT` - PostgreSQL port (default: 5432)
- `POSTGRES_DB` - Database name (default: agentic_ingestion)
- `POSTGRES_USER` - Database user (default: agentic_user)
- `POSTGRES_PASSWORD` - Database password

### Redis Configuration
- `REDIS_HOST` - Redis host (default: redis_ingestion)
- `REDIS_PORT` - Redis port (default: 6379)

### Service Configuration
- `PLUGIN_REGISTRY_HOST` - Service host (default: 0.0.0.0)
- `PLUGIN_REGISTRY_PORT` - Service port (default: 8201)
- `REQUIRE_AUTH` - Enable authentication (default: false)
- `JWT_SECRET` - JWT secret key for authentication

### Plugin Configuration
- `PLUGIN_LOAD_TIMEOUT_SECONDS` - Plugin load timeout (default: 30)
- `PLUGIN_EXECUTION_TIMEOUT_SECONDS` - Plugin execution timeout (default: 300)
- `PLUGIN_CACHE_ENABLED` - Enable plugin caching (default: true)
- `PLUGIN_AUTO_UPDATE_ENABLED` - Enable automatic plugin updates (default: false)
- `ALLOW_PLUGIN_UPLOAD` - Allow plugin uploads (default: true)
- `PLUGIN_SANDBOX_ENABLED` - Enable plugin sandboxing (default: true)

## Usage Examples

### Execute Risk Calculator Plugin

```json
POST /plugins/execute
{
  "plugin_id": "riskCalculator",
  "input_data": {
    "loan_amount": 300000,
    "credit_score": 720,
    "income": 85000,
    "debt_ratio": 0.32
  },
  "execution_config": {},
  "timeout_seconds": 60
}
```

**Response:**
```json
{
  "risk_score": 25,
  "risk_category": "MEDIUM",
  "approval_probability": 0.7,
  "recommendation": "APPROVE",
  "calculated_at": "2024-01-15T10:30:00Z"
}
```

### Execute Fraud Detector Plugin

```json
POST /plugins/execute
{
  "plugin_id": "fraudDetector",
  "input_data": {
    "claim_amount": 25000,
    "incident_date": "2024-01-10T00:00:00Z",
    "policy_start_date": "2024-01-01T00:00:00Z",
    "claim_history": [
      {"amount": 5000, "date": "2023-06-15"},
      {"amount": 8000, "date": "2023-09-22"}
    ]
  }
}
```

**Response:**
```json
{
  "fraud_score": 35,
  "fraud_risk": "MEDIUM",
  "investigation_required": true,
  "flags": ["High claim frequency"],
  "recommendation": "INVESTIGATE",
  "detected_at": "2024-01-15T10:30:00Z"
}
```

### Execute Data Validator Plugin

```json
POST /plugins/execute
{
  "plugin_id": "validator",
  "input_data": {
    "data": [
      {"id": 1, "name": "John Doe", "email": "john@example.com", "age": 30},
      {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "age": "thirty"}
    ],
    "validation_rules": {
      "required_fields": ["id", "name", "email"],
      "field_types": {
        "id": "number",
        "name": "string",
        "email": "string",
        "age": "number"
      }
    }
  }
}
```

**Response:**
```json
{
  "total_records": 2,
  "valid_records": 1,
  "invalid_records": 1,
  "validation_score": 0.5,
  "errors": ["Field age should be number"],
  "validated_at": "2024-01-15T10:30:00Z"
}
```

## Plugin Development

### Creating a Custom Plugin

To create a custom plugin, implement the appropriate interface:

```python
from plugin_interfaces import DomainPlugin

class CustomRiskPlugin(DomainPlugin):
    @property
    def plugin_id(self) -> str:
        return "customRiskPlugin"

    @property
    def name(self) -> str:
        return "Custom Risk Assessment"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Custom risk assessment logic"

    @property
    def domain(self) -> str:
        return "underwriting"

    def validate_domain_data(self, data: Dict[str, Any]) -> bool:
        # Validate input data
        return True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Implement plugin logic
        return {"result": "processed"}
```

### Plugin Registration

Register your plugin through the API:

```json
POST /plugins/register
{
  "plugin_id": "customRiskPlugin",
  "name": "Custom Risk Assessment",
  "plugin_type": "domain",
  "domain": "underwriting",
  "description": "Custom risk assessment logic",
  "version": "1.0.0",
  "author": "Your Name",
  "entry_point": "custom_plugins.risk_assessment",
  "dependencies": [],
  "tags": ["risk", "assessment", "custom"]
}
```

## Security Considerations

- **Plugin Sandboxing**: Plugins run in isolated environments
- **Input Validation**: All plugin inputs are validated
- **Execution Timeouts**: Prevents runaway plugin execution
- **Security Scanning**: Automatic security checks on plugin code
- **Access Control**: Role-based access to plugin execution
- **Audit Logging**: Complete audit trail of plugin operations

## Monitoring

The service exposes comprehensive metrics via Prometheus:

- `plugin_registry_active_plugins`: Number of active plugins
- `plugin_registry_executions_total`: Total plugin executions by plugin and status
- `plugin_registry_execution_duration_seconds`: Plugin execution duration
- `plugin_registry_load_duration_seconds`: Plugin load duration
- `plugin_registry_requests_total`: Total API requests
- `plugin_registry_errors_total`: Error count by type

## Performance Optimization

- **Plugin Caching**: Frequently used plugins are cached in memory
- **Connection Pooling**: Database connections are pooled for efficiency
- **Async Execution**: Plugin execution uses async patterns for scalability
- **Result Caching**: Plugin results can be cached based on input hash
- **Load Balancing**: Multiple plugin instances can be load balanced

## Architecture

The Plugin Registry follows a modular architecture:

1. **PluginManager**: Core plugin management and execution
2. **PluginLoader**: Dynamic plugin loading and validation
3. **SecurityValidator**: Plugin security scanning and validation
4. **MetricsCollector**: Performance monitoring and metrics
5. **Database Layer**: Plugin metadata and execution history
6. **Cache Layer**: Fast access to frequently used plugins

## Development

To run the service locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PASSWORD=your_password
# ... other variables

# Run the service
python main.py
```

## Production Deployment

The service is designed to run in Docker containers and integrates with the broader Agentic Platform. It automatically discovers and communicates with other platform services through the shared Docker network.

## Integration Points

The Plugin Registry integrates with:

- **Agent Orchestrator**: Provides plugins for agent execution
- **Brain Factory**: Supplies plugins for agent construction
- **Workflow Engine**: Uses plugins in workflow execution
- **Agent Builder UI**: Lists available plugins for workflow creation
- **Deployment Pipeline**: Validates plugin availability during deployment
