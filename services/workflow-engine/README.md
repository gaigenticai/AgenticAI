# Workflow Engine Service

The Workflow Engine Service is a core component of the Agentic Brain Platform that executes agent workflows composed of drag-and-drop components. It handles the orchestration, dependency management, parallel execution, and state tracking of complex multi-component workflows.

## Features

- **Workflow Execution**: Execute complex workflows with multiple component types
- **Dependency Management**: Automatic resolution of component dependencies
- **Parallel Execution**: Support for parallel component execution where possible
- **Component Types**: Support for various component types (Data Input, LLM Processor, Rule Engine, etc.)
- **State Tracking**: Real-time monitoring of workflow execution progress
- **Error Handling**: Comprehensive error handling and recovery mechanisms
- **RESTful API**: Complete API for workflow operations and monitoring
- **Authentication Integration**: Support for JWT-based authentication
- **Monitoring & Metrics**: Prometheus metrics for observability
- **Database Persistence**: Store workflow definitions and execution history

## Supported Component Types

### Data Input Components
- **CSV Input**: Import data from CSV files
- **API Input**: Fetch data from REST API endpoints
- **Database Input**: Query data from database tables
- **File Input**: Read data from various file formats

### Processing Components
- **LLM Processor**: Process data using Large Language Models
- **Rule Engine**: Apply business rules and logic
- **Plugin Executor**: Execute registered plugins
- **Data Transformer**: Transform and clean data

### Logic Components
- **Decision Node**: Conditional branching based on data values
- **Switch Node**: Multi-way branching logic
- **Merge Node**: Combine multiple data streams
- **Filter Node**: Filter data based on conditions

### Output Components
- **Database Output**: Write results to database tables
- **Email Output**: Send results via email notifications
- **File Output**: Export results to files
- **API Output**: Send results to external APIs

## Architecture

The Workflow Engine uses a component-based architecture:

1. **WorkflowExecutor**: Main orchestration engine
2. **ComponentFactory**: Creates component instances from configurations
3. **DependencyResolver**: Manages component dependencies and execution order
4. **StateManager**: Tracks execution state and progress
5. **MetricsCollector**: Performance monitoring and metrics
6. **ErrorHandler**: Error detection and recovery

## Workflow Execution Flow

1. **Validation**: Validate workflow definition and component configurations
2. **Planning**: Create execution plan with dependency resolution
3. **Execution**: Execute components in dependency order
4. **Monitoring**: Track progress and handle errors
5. **Completion**: Aggregate results and update status

## API Endpoints

### Workflow Management
- `POST /workflows/execute` - Execute a workflow
- `GET /workflows/executions/{execution_id}` - Get execution status
- `POST /workflows/validate` - Validate workflow definition
- `GET /workflows/components` - List available component types

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
- `WORKFLOW_ENGINE_HOST` - Service host (default: 0.0.0.0)
- `WORKFLOW_ENGINE_PORT` - Service port (default: 8202)
- `REQUIRE_AUTH` - Enable authentication (default: false)
- `JWT_SECRET` - JWT secret key for authentication

### Workflow Configuration
- `WORKFLOW_MAX_COMPONENTS` - Maximum components per workflow (default: 50)
- `WORKFLOW_EXECUTION_TIMEOUT` - Maximum execution time (default: 1800)
- `WORKFLOW_PARALLEL_EXECUTION` - Enable parallel execution (default: true)
- `WORKFLOW_ERROR_RECOVERY_ENABLED` - Enable error recovery (default: true)

## Usage Examples

### Execute a Simple Workflow

```json
POST /workflows/execute
{
  "workflow_id": "simple_workflow_001",
  "agent_id": "agent_123",
  "input_data": {
    "source_file": "data.csv",
    "query": "SELECT * FROM policies WHERE status = 'active'"
  },
  "execution_config": {
    "parallel_execution": true,
    "timeout_seconds": 600
  }
}
```

**Response:**
```json
{
  "execution_id": "exec_12345678-1234-1234-1234-123456789abc",
  "status": "started",
  "message": "Workflow execution started successfully"
}
```

### Get Execution Status

```json
GET /workflows/executions/exec_12345678-1234-1234-1234-123456789abc
```

**Response:**
```json
{
  "execution_id": "exec_12345678-1234-1234-1234-123456789abc",
  "status": "running",
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": null,
  "duration_seconds": null,
  "total_components": 4,
  "completed_components": 2,
  "failed_components": 0,
  "progress_percentage": 50.0,
  "current_component": "llm_processor_1",
  "error_message": null
}
```

### Validate Workflow Definition

```json
POST /workflows/validate
{
  "workflow_id": "test_workflow",
  "agent_id": "agent_123",
  "name": "Test Workflow",
  "description": "A test workflow",
  "components": [
    {
      "component_id": "data_input_1",
      "component_type": "data_input",
      "config": {
        "source_type": "csv",
        "file_path": "data.csv"
      }
    },
    {
      "component_id": "llm_processor_1",
      "component_type": "llm_processor",
      "config": {
        "model": "gpt-4",
        "prompt_template": "Analyze the following data: {data}",
        "temperature": 0.7
      }
    }
  ],
  "connections": [
    {
      "from": "data_input_1",
      "to": "llm_processor_1"
    }
  ]
}
```

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "component_count": 2,
  "connection_count": 1
}
```

## Component Configuration Examples

### Data Input Component

```json
{
  "component_id": "csv_input_1",
  "component_type": "data_input",
  "config": {
    "source_type": "csv",
    "file_path": "/data/input.csv",
    "delimiter": ",",
    "has_header": true,
    "encoding": "utf-8"
  }
}
```

### LLM Processor Component

```json
{
  "component_id": "llm_processor_1",
  "component_type": "llm_processor",
  "config": {
    "model": "gpt-4",
    "prompt_template": "Analyze the following insurance claim data and determine if it's fraudulent: {claim_data}",
    "temperature": 0.3,
    "max_tokens": 1000,
    "system_prompt": "You are an expert insurance fraud analyst."
  }
}
```

### Rule Engine Component

```json
{
  "component_id": "fraud_rules_1",
  "component_type": "rule_engine",
  "config": {
    "rule_set": "fraud_detection",
    "rules": [
      {
        "name": "high_amount_check",
        "condition": "claim_amount > 50000",
        "action": "flag_for_review"
      },
      {
        "name": "recent_policy_check",
        "condition": "policy_age_days < 30",
        "action": "escalate"
      }
    ]
  }
}
```

### Decision Node Component

```json
{
  "component_id": "decision_1",
  "component_type": "decision_node",
  "config": {
    "condition_field": "fraud_score",
    "operator": "greater_than",
    "threshold": 0.7,
    "true_branch": "investigate_fraud",
    "false_branch": "process_claim"
  }
}
```

### Database Output Component

```json
{
  "component_id": "db_output_1",
  "component_type": "database_output",
  "config": {
    "table_name": "processed_claims",
    "connection_config": {
      "host": "postgresql_output",
      "port": 5432,
      "database": "claims_db"
    },
    "operation": "INSERT",
    "on_conflict": "UPDATE"
  }
}
```

## Execution Planning

The Workflow Engine uses sophisticated execution planning:

1. **Dependency Analysis**: Analyzes component dependencies using graph algorithms
2. **Topological Sorting**: Determines optimal execution order
3. **Parallel Execution**: Groups independent components for parallel processing
4. **Resource Allocation**: Manages resource usage across components
5. **Failure Handling**: Implements retry logic and fallback strategies

## Monitoring and Metrics

The service exposes comprehensive metrics via Prometheus:

- `workflow_engine_active_workflows`: Number of active workflow executions
- `workflow_engine_executions_total`: Total workflow executions by status
- `workflow_engine_execution_duration_seconds`: Workflow execution duration
- `workflow_engine_component_executions_total`: Component executions by type and status
- `workflow_engine_component_duration_seconds`: Component execution duration
- `workflow_engine_requests_total`: Total API requests
- `workflow_engine_errors_total`: Error count by type

## Error Handling

The Workflow Engine implements robust error handling:

- **Component-Level Errors**: Isolated failure handling per component
- **Workflow-Level Recovery**: Continue execution when possible
- **Retry Logic**: Automatic retry with exponential backoff
- **Fallback Strategies**: Alternative execution paths
- **Error Aggregation**: Comprehensive error reporting
- **State Persistence**: Maintain execution state across failures

## Performance Optimization

- **Async Execution**: Non-blocking component execution
- **Connection Pooling**: Efficient database and service connections
- **Caching**: Component results and configuration caching
- **Resource Limits**: Configurable resource usage limits
- **Load Balancing**: Distribute execution across multiple instances
- **Monitoring**: Real-time performance tracking

## Security Considerations

- **Input Validation**: Comprehensive validation of workflow definitions
- **Access Control**: Component-level permission checking
- **Execution Isolation**: Sandboxed component execution
- **Audit Logging**: Complete audit trail of all operations
- **Secure Communication**: Encrypted inter-service communication
- **Resource Limits**: Prevent resource exhaustion attacks

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

The Workflow Engine integrates with:

- **Agent Orchestrator**: Receives workflow execution requests
- **Plugin Registry**: Executes plugin components
- **Rule Engine**: Processes rule-based components
- **Memory Manager**: Manages execution state and caching
- **Brain Factory**: Provides workflow definitions
- **Database Services**: Persistent storage for execution data
