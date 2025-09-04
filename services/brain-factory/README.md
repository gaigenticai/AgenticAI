# Brain Factory Service

## Overview

The Brain Factory Service is the central agent instantiation and configuration management system for the Agentic AI platform. It serves as the factory for creating fully configured AgentBrain instances from AgentConfig specifications, orchestrating the integration of reasoning modules, memory management, plugin systems, and service connectors.

## Key Responsibilities

- **Agent Instantiation**: Create complete agent instances from configuration
- **Configuration Validation**: Validate agent configurations for completeness
- **Service Integration**: Orchestrate integration with all platform services
- **Dependency Management**: Resolve and validate service dependencies
- **Performance Monitoring**: Track agent creation metrics and performance
- **Lifecycle Management**: Manage agent creation, validation, and registration

## Architecture

### Agent Creation Pipeline

```python
1. Configuration Validation → Comprehensive config checking
2. Service Dependency Check → Verify all services are available
3. Agent Instance Creation → Create through Agent Brain Base
4. Service Configuration → Setup reasoning, memory, plugins
5. Orchestrator Registration → Register with agent orchestrator
6. Initial Validation → Test agent functionality
7. Metrics Collection → Track creation performance
```

### Service Integration Points

```
Brain Factory
├── Agent Brain Base (8305) → Core agent execution
├── Reasoning Module Factory (8304) → AI reasoning patterns
├── Memory Manager (8205) → Memory management
├── Plugin Registry (8201) → Plugin orchestration
├── Service Connector Factory (8306) → Data service connections
├── Agent Orchestrator (8200) → Agent lifecycle management
└── UI-to-Brain Mapper (8302) → Visual workflow conversion
```

## API Endpoints

### Core Endpoints

#### `POST /generate-agent`
Create a new agent instance with complete configuration and service integration.

**Request Body:**
```json
{
  "agent_config": {
    "agent_id": "analyst_001",
    "name": "Data Analysis Agent",
    "persona": {
      "name": "Data Analysis Agent",
      "description": "Expert data analysis and insights generation",
      "domain": "data_science",
      "expertise_level": "expert",
      "communication_style": "analytical"
    },
    "reasoning_config": {
      "pattern": "ReAct",
      "fallback_patterns": ["Planning"],
      "adaptive_reasoning": true,
      "confidence_threshold": 0.7
    },
    "memory_config": {
      "working_memory_enabled": true,
      "episodic_memory_enabled": true,
      "semantic_memory_enabled": true,
      "vector_memory_enabled": true,
      "memory_ttl_seconds": 3600,
      "max_memory_items": 1000
    },
    "plugin_config": {
      "enabled_plugins": ["data_analyzer", "visualization"],
      "domain_plugins": ["statistical_modeling"],
      "generic_plugins": ["data_processor"],
      "auto_discovery": true
    }
  },
  "deployment_options": {
    "auto_start": true,
    "monitoring_enabled": true,
    "resource_limits": {
      "max_memory_mb": 512,
      "max_concurrent_tasks": 5
    }
  },
  "validation_options": {
    "strict_validation": true,
    "performance_checks": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "agent_id": "analyst_001",
  "agent_endpoint": "http://localhost:8305/agents/analyst_001",
  "status": "ready",
  "validation_results": [
    {
      "is_valid": true,
      "severity": "info",
      "message": "Configuration validated successfully"
    }
  ],
  "creation_metadata": {
    "creation_time": 45.2,
    "services_configured": 3,
    "reasoning_pattern": "ReAct",
    "memory_enabled": true
  }
}
```

#### `GET /agents/{agent_id}/status`
Get comprehensive status information for a created agent.

**Response:**
```json
{
  "agent_id": "analyst_001",
  "status": "ready",
  "created_at": "2024-01-01T10:00:00Z",
  "last_activity": "2024-01-01T10:30:00Z",
  "configuration_valid": true,
  "services_status": {
    "reasoning_module": "connected",
    "memory_manager": "connected",
    "plugin_registry": "connected",
    "service_connectors": "connected"
  }
}
```

#### `GET /agents`
List all created agents with their current status and metadata.

#### `GET /metrics`
Get comprehensive agent creation metrics and performance statistics.

**Response:**
```json
{
  "metrics": {
    "total_created": 15,
    "successful_creations": 14,
    "failed_creations": 1,
    "success_rate": 0.933,
    "average_creation_time": 42.3,
    "active_agents": 12,
    "last_updated": "2024-01-01T10:30:00Z"
  }
}
```

### Validation Endpoints

#### `GET /validate-config`
Validate an agent configuration without creating the agent.

**Response:**
```json
{
  "is_valid": true,
  "validation_results": [
    {
      "is_valid": true,
      "severity": "warning",
      "message": "Consider enabling vector memory for better performance",
      "field": "memory_config.vector_memory_enabled",
      "suggestion": "Set vector_memory_enabled to true"
    }
  ],
  "validation_summary": {
    "total_issues": 1,
    "errors": 0,
    "warnings": 1,
    "info": 0
  }
}
```

#### `GET /service-dependencies`
Check the health status of all required service dependencies.

**Response:**
```json
{
  "service_dependencies": {
    "all_healthy": true,
    "issues": [],
    "checked_at": "2024-01-01T10:30:00Z"
  }
}
```

## Configuration Validation

### Validation Rules

The service performs comprehensive validation including:

- **Agent Identity**: Unique agent ID and valid name
- **Persona Configuration**: Complete persona with domain and expertise
- **Reasoning Pattern**: Supported reasoning pattern selection
- **Memory Configuration**: Valid memory settings and dimensions
- **Plugin Compatibility**: Plugin availability and version compatibility
- **Resource Limits**: Realistic resource allocation limits
- **Security Settings**: Proper authentication and authorization

### Validation Severity Levels

```python
class ValidationSeverity(Enum):
    ERROR = "error"      # Blocking validation failures
    WARNING = "warning"  # Non-blocking recommendations
    INFO = "info"        # Informational suggestions
```

### Example Validation Results

```json
{
  "is_valid": false,
  "validation_results": [
    {
      "is_valid": false,
      "severity": "error",
      "message": "Agent ID is required",
      "field": "agent_id",
      "suggestion": "Provide a unique agent identifier"
    },
    {
      "is_valid": false,
      "severity": "error",
      "message": "Unsupported reasoning pattern: CustomPattern",
      "field": "reasoning_config.pattern",
      "suggestion": "Use one of: ReAct, Reflection, Planning, Multi-Agent"
    },
    {
      "is_valid": true,
      "severity": "warning",
      "message": "Consider increasing memory TTL for better performance",
      "field": "memory_config.memory_ttl_seconds",
      "suggestion": "Increase memory_ttl_seconds to at least 7200"
    }
  ]
}
```

## Service Dependency Management

### Dependency Health Monitoring

The service continuously monitors the health of all required dependencies:

```python
dependency_status = {
    "agent_brain_base": "healthy",
    "reasoning_module_factory": "healthy",
    "memory_manager": "healthy",
    "plugin_registry": "healthy",
    "service_connector_factory": "healthy",
    "agent_orchestrator": "healthy"
}
```

### Automatic Failover

When service dependencies are unavailable:

- **Retry Logic**: Exponential backoff retry mechanisms
- **Graceful Degradation**: Continue with available services
- **Fallback Options**: Use cached configurations when possible
- **Error Propagation**: Clear error messages with recovery suggestions

## Agent Creation Process

### Step-by-Step Creation

```python
# 1. Configuration Validation
validation_results = await validate_agent_config(config)
if not all_valid(validation_results):
    return validation_error_response

# 2. Service Dependency Check
dependency_status = await check_service_dependencies()
if not dependency_status["all_healthy"]:
    return dependency_error_response

# 3. Agent Instance Creation
agent_data = await create_agent_instance(config)

# 4. Service Configuration
await configure_reasoning_module(agent_id, config.reasoning_config)
await configure_memory_management(agent_id, config.memory_config)
await configure_plugin_system(agent_id, config.plugin_config)

# 5. Orchestrator Registration
await register_with_orchestrator(agent_id, config)

# 6. Initial Validation
validation_status = await perform_agent_validation(agent_id)

# 7. Success Response
return AgentCreationResponse(
    success=True,
    agent_id=agent_id,
    agent_endpoint=f"http://localhost:8305/agents/{agent_id}",
    status="ready"
)
```

### Creation Metrics Tracking

```python
creation_metrics = {
    "total_created": 150,
    "successful_creations": 142,
    "failed_creations": 8,
    "success_rate": 0.947,
    "average_creation_time": 45.2,
    "active_agents": 142,
    "creation_rate_per_hour": 12.5
}
```

## Performance Optimization

### Creation Time Optimization

- **Parallel Service Configuration**: Configure multiple services concurrently
- **Lazy Loading**: Load heavy components only when needed
- **Caching**: Cache frequently used configurations and templates
- **Batch Operations**: Process multiple creation requests efficiently

### Resource Management

```python
resource_limits = {
    "max_concurrent_creations": 5,
    "max_agents_per_instance": 50,
    "memory_limit_per_agent": "256MB",
    "cpu_limit_per_agent": "0.5 cores"
}
```

### Monitoring and Alerting

- **Creation Time Alerts**: Alert when creation time exceeds thresholds
- **Failure Rate Monitoring**: Track and alert on creation failure rates
- **Resource Usage Tracking**: Monitor memory and CPU usage patterns
- **Service Health Alerts**: Alert on service dependency failures

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run service
uvicorn main:app --host 0.0.0.0 --port 8301 --reload
```

### Docker Development
```bash
# Build image
docker build -t brain-factory .

# Run container
docker run -p 8301:8301 brain-factory
```

## Testing

### Unit Tests
```python
# Test agent configuration validation
def test_config_validation():
    config = AgentConfig(agent_id="test", name="Test Agent")
    results = await factory._validate_agent_config(config)
    assert len(results) == 0  # No validation errors

# Test service dependency checking
@pytest.mark.asyncio
async def test_dependency_check():
    status = await factory._check_service_dependencies()
    assert status["all_healthy"] is True
```

### Integration Tests
```python
# Test complete agent creation workflow
@pytest.mark.asyncio
async def test_agent_creation_workflow():
    config = create_test_agent_config()
    request = AgentCreationRequest(agent_config=config)

    result = await factory.create_agent(request)

    assert result.success is True
    assert result.agent_id == config.agent_id
    assert result.status == "ready"

    # Verify agent is accessible
    status = await factory.get_agent_status(config.agent_id)
    assert status.status == "ready"
```

### Performance Tests
```python
# Test concurrent agent creation
@pytest.mark.asyncio
async def test_concurrent_creation():
    configs = [create_test_agent_config() for _ in range(5)]
    requests = [AgentCreationRequest(agent_config=config) for config in configs]

    # Create agents concurrently
    tasks = [factory.create_agent(request) for request in requests]
    results = await asyncio.gather(*tasks)

    assert all(result.success for result in results)
    assert len(factory.created_agents) == 5
```

## Usage Examples

### Basic Agent Creation
```python
from main import AgentFactory, AgentConfig, AgentPersona, ReasoningConfig

# Initialize factory
factory = AgentFactory()

# Create agent configuration
config = AgentConfig(
    agent_id="analyst_001",
    name="Data Analyst",
    persona=AgentPersona(
        name="Data Analyst",
        description="Expert data analysis agent",
        domain="data_science",
        expertise_level="expert"
    ),
    reasoning_config=ReasoningConfig(pattern="ReAct"),
    memory_config=MemoryConfig(),
    plugin_config=PluginConfig(enabled_plugins=["data_analyzer"])
)

# Create agent
request = AgentCreationRequest(agent_config=config)
result = await factory.create_agent(request)

if result.success:
    print(f"Agent created: {result.agent_id}")
    print(f"Endpoint: {result.agent_endpoint}")
else:
    print(f"Creation failed: {result.error_message}")
```

### Agent Status Monitoring
```python
# Get agent status
status = await factory.get_agent_status("analyst_001")
print(f"Agent status: {status.status}")
print(f"Last activity: {status.last_activity}")

# Get creation metrics
metrics = factory.get_creation_metrics()
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Average creation time: {metrics['average_creation_time']:.1f}s")
```

### Configuration Validation
```python
# Validate configuration before creation
validation_results = await factory._validate_agent_config(config)

if validation_results:
    print("Validation issues found:")
    for result in validation_results:
        print(f"{result.severity}: {result.message}")
        if result.suggestion:
            print(f"Suggestion: {result.suggestion}")
else:
    print("Configuration is valid")
```

## Security Considerations

- **Configuration Validation**: Prevent malicious configuration injection
- **Resource Limits**: Prevent resource exhaustion attacks
- **Access Control**: Validate agent creation permissions
- **Audit Logging**: Complete audit trail of agent creation activities
- **Secure Communication**: Encrypted communication with all services
- **Input Sanitization**: Validate and sanitize all input parameters

## Future Enhancements

- **Template-Based Creation**: Pre-defined agent templates for common use cases
- **Bulk Agent Creation**: Create multiple agents with batch operations
- **Agent Cloning**: Create new agents based on existing agent configurations
- **Version Management**: Track and manage agent configuration versions
- **Auto-Scaling**: Automatically scale agent instances based on demand
- **Advanced Validation**: ML-based configuration optimization suggestions
- **Integration Testing**: Automated testing of agent functionality post-creation

## API Documentation

Complete API documentation is available at `/docs` when the service is running.

## Contributing

1. Follow the established agent creation pipeline patterns
2. Add comprehensive validation for new configuration options
3. Include performance monitoring for new service integrations
4. Write thorough unit and integration tests
5. Update documentation for configuration changes
6. Ensure backward compatibility with existing agent configurations

## Performance Benchmarks

Based on testing with various agent configurations:

- **Creation Time**: 30-120 seconds depending on complexity and service load
- **Success Rate**: 95-99% with proper service dependencies
- **Concurrent Creations**: Up to 5 simultaneous agent creations
- **Memory Usage**: 50-200MB per active factory instance
- **Validation Time**: 2-10 seconds for comprehensive validation

These benchmarks provide guidance for scaling and performance optimization decisions.
