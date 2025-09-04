# UI-to-Brain Mapper Service

## Overview

The UI-to-Brain Mapper Service is a critical component of the AgenticAI platform that converts visual workflow configurations from the Agent Builder UI into structured agent configurations that can be executed by backend services.

## Key Responsibilities

- **Component Mapping**: Converts visual UI components (data_input_csv, llm_processor, etc.) to service configurations
- **Connection Mapping**: Transforms visual connections into data flow definitions
- **Service Discovery**: Resolves component types to appropriate service endpoints
- **Configuration Generation**: Creates complete AgentConfig JSON with all required fields
- **Validation**: Ensures workflow integrity and component compatibility
- **Reasoning Pattern Selection**: Automatically determines optimal reasoning patterns

## API Endpoints

### Core Endpoints

#### `POST /map-workflow`
Converts visual workflow to agent configuration.

**Request Body:**
```json
{
  "visual_workflow": {
    "workflow_id": "wf_123",
    "name": "Sample Agent Workflow",
    "components": [...],
    "connections": [...]
  },
  "agent_metadata": {
    "name": "My Custom Agent",
    "domain": "insurance",
    "persona": "Risk assessment specialist"
  }
}
```

**Response:**
```json
{
  "success": true,
  "agent_config": {
    "agent_id": "agent_abc123",
    "name": "My Custom Agent",
    "domain": "insurance",
    "reasoning_pattern": "ReAct",
    "components": [...],
    "connections": [...],
    "memory_config": {...},
    "plugin_config": {...}
  },
  "mapping_report": {...}
}
```

#### `GET /supported-components`
Returns list of supported component types and configurations.

#### `GET /health`
Service health check endpoint.

## Supported Components

### Data Input Components
- `data_input_csv`: CSV file ingestion
- `data_input_api`: REST API data ingestion
- `data_input_pdf`: PDF document processing

### Processing Components
- `llm_processor`: Large Language Model processing
- `rule_engine`: Business rule evaluation
- `decision_node`: Conditional logic routing

### Output Components
- `database_output`: Database storage (PostgreSQL)
- `email_output`: Email notifications
- `pdf_report_output`: PDF report generation

## Reasoning Patterns

The service automatically selects the most appropriate reasoning pattern:

- **ReAct**: Default pattern for general-purpose agents
- **Reflection**: For agents requiring self-assessment and improvement
- **Planning**: For complex multi-step workflows
- **Multi-Agent**: For workflows with multiple LLM processors

## Configuration Features

### Memory Configuration
- **Working Memory**: Short-term context with TTL
- **Episodic Memory**: Event-based memory with retention
- **Semantic Memory**: Knowledge base with vector similarity
- **Vector Memory**: High-dimensional vector storage

### Plugin Configuration
- **Domain Plugins**: Industry-specific capabilities (fraud detection, risk calculation)
- **Generic Plugins**: Common utilities (data retrieval, validation)
- **Plugin Settings**: Timeout, retry, and caching configurations

## Service Integration

The mapper integrates with multiple backend services:

- **Plugin Registry** (port 8201): Plugin discovery and configuration
- **Workflow Engine** (port 8202): Workflow execution coordination
- **Rule Engine** (port 8204): Business rule processing
- **Memory Manager** (port 8205): Memory management services
- **Template Store** (port 8203): Template management

## Error Handling

The service provides comprehensive error handling:

- **Validation Errors**: Component and connection validation failures
- **Mapping Warnings**: Non-critical issues that may affect performance
- **Service Unavailability**: Graceful degradation when services are unavailable
- **Configuration Errors**: Invalid component configurations

## Performance Optimization

- **Async Processing**: Non-blocking request handling
- **Connection Pooling**: Efficient service communication
- **Caching**: Component and service endpoint caching
- **Batch Processing**: Optimized for multiple component mappings

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run service
uvicorn main:app --host 0.0.0.0 --port 8302 --reload
```

### Docker Development
```bash
# Build image
docker build -t ui-to-brain-mapper .

# Run container
docker run -p 8302:8302 ui-to-brain-mapper
```

## Configuration

Environment variables:

- `UI_TO_BRAIN_MAPPER_PORT`: Service port (default: 8302)
- `SERVICE_HOST`: Backend service host (default: localhost)
- `DEFAULT_LLM_MODEL`: Default LLM model
- `DEFAULT_MEMORY_TTL`: Default memory TTL in seconds

## Testing

The service includes comprehensive test coverage:

- **Unit Tests**: Component mapping and validation logic
- **Integration Tests**: End-to-end workflow conversion
- **API Tests**: REST endpoint validation
- **Performance Tests**: Load and stress testing

## Monitoring

The service provides detailed monitoring:

- **Health Checks**: Service availability monitoring
- **Metrics**: Request latency, error rates, throughput
- **Logs**: Structured logging with correlation IDs
- **Tracing**: Distributed tracing support

## Security

- **Input Validation**: Comprehensive request validation
- **Error Sanitization**: Secure error message handling
- **Rate Limiting**: Protection against abuse
- **CORS Support**: Cross-origin request handling

## Future Enhancements

- **Advanced Data Mapping**: Field-level data transformation
- **Workflow Optimization**: Automatic performance tuning
- **Template Auto-Generation**: Learning-based template creation
- **Multi-Modal Support**: Support for various input/output formats
- **Real-time Validation**: Live workflow validation feedback

## API Documentation

Complete API documentation is available at `/docs` when the service is running.

## Contributing

1. Follow the established code patterns and error handling
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility for existing workflows
