# Agent Orchestrator Service

The Agent Orchestrator Service is a core component of the Agentic Brain Platform that manages the lifecycle and coordination of multiple AI agents. It provides a centralized control plane for agent registration, task routing, performance monitoring, and multi-agent orchestration.

## Features

- **Agent Lifecycle Management**: Register, start, stop, and monitor agent instances
- **Task Orchestration**: Route tasks to appropriate agent instances with load balancing
- **Multi-Agent Coordination**: Enable communication and coordination between agents
- **Performance Monitoring**: Track agent metrics, session management, and success rates
- **Health Monitoring**: Real-time health checks and status reporting
- **Authentication Integration**: Support for JWT-based authentication
- **Metrics & Monitoring**: Prometheus metrics for observability
- **Database Persistence**: Store agent metadata and session history
- **Redis Caching**: Fast access to agent states and session data

## API Endpoints

### Agent Management
- `POST /orchestrator/register` - Register a new agent instance
- `POST /orchestrator/unregister` - Unregister an agent instance
- `GET /orchestrator/agents` - List all registered agents
- `GET /orchestrator/agents/{agent_id}` - Get status of specific agent
- `POST /orchestrator/start` - Start an agent instance
- `POST /orchestrator/stop` - Stop an agent instance

### Task Execution
- `POST /orchestrator/execute-task` - Execute a task on specified agent
- `GET /orchestrator/tasks/{task_id}` - Get status of task execution

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

### Message Queue
- `RABBITMQ_HOST` - RabbitMQ host (default: rabbitmq)
- `RABBITMQ_PORT` - RabbitMQ port (default: 5672)
- `RABBITMQ_USER` - RabbitMQ user (default: agentic_user)
- `RABBITMQ_PASSWORD` - RabbitMQ password

### Service Configuration
- `AGENT_ORCHESTRATOR_HOST` - Service host (default: 0.0.0.0)
- `AGENT_ORCHESTRATOR_PORT` - Service port (default: 8200)
- `REQUIRE_AUTH` - Enable authentication (default: false)
- `JWT_SECRET` - JWT secret key for authentication

### Agent Configuration
- `MAX_CONCURRENT_AGENT_SESSIONS` - Maximum concurrent sessions (default: 10)
- `AGENT_SESSION_TIMEOUT_MINUTES` - Session timeout (default: 60)
- `AGENT_METRICS_ENABLED` - Enable metrics collection (default: true)

## Dependencies

The service integrates with:
- **PostgreSQL**: Agent metadata and session storage
- **Redis**: Session caching and state management
- **RabbitMQ**: Task queuing and distribution
- **Brain Factory**: Agent instantiation and configuration
- **Deployment Pipeline**: Agent deployment management

## Usage

### Registering an Agent

```json
POST /orchestrator/register
{
  "agent_id": "underwriting_agent_01",
  "agent_name": "Underwriting Assistant",
  "domain": "underwriting",
  "brain_config": {
    "persona": {"role": "Underwriting Analyst", "expertise": ["risk"], "personality": "balanced"},
    "reasoningPattern": "react",
    "components": [...],
    "connections": [...]
  },
  "deployment_id": "deploy_001"
}
```

### Executing a Task

```json
POST /orchestrator/execute-task
{
  "agent_id": "underwriting_agent_01",
  "task_type": "risk_assessment",
  "task_data": {
    "applicant_id": "APP001",
    "loan_amount": 500000,
    "credit_score": 720
  },
  "priority": 5,
  "timeout_seconds": 300
}
```

## Architecture

The Agent Orchestrator follows a modular architecture:

1. **AgentManager**: Handles agent lifecycle and registration
2. **TaskOrchestrator**: Manages task execution and routing
3. **MetricsCollector**: Collects performance metrics
4. **Database Layer**: Persists agent and session data
5. **Cache Layer**: Provides fast access to frequently used data

## Monitoring

The service exposes comprehensive metrics via Prometheus:

- `agent_orchestrator_active_agents`: Number of active agents
- `agent_orchestrator_total_sessions`: Total session count
- `agent_orchestrator_session_duration_seconds`: Session duration histogram
- `agent_orchestrator_task_success_rate`: Task success rate
- `agent_orchestrator_requests_total`: Total API requests
- `agent_orchestrator_errors_total`: Error count by type

## Security

- JWT-based authentication when REQUIRE_AUTH=true
- Input validation using Pydantic models
- SQL injection prevention via SQLAlchemy
- Secure configuration management
- Audit logging for all operations

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

The service is designed to run in Docker containers and integrates with the broader Agentic Platform orchestration system. It automatically discovers and communicates with other platform services through the shared Docker network.
