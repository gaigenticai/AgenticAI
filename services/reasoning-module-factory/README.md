# Reasoning Module Factory Service

## Overview

The Reasoning Module Factory Service is a core component of the Agentic Brain platform that implements multiple AI reasoning patterns using a factory pattern with dependency injection. It provides four distinct reasoning approaches:

- **ReAct**: Reasoning + Acting pattern for tool-using agents
- **Reflection**: Self-assessment and iterative improvement pattern
- **Planning**: Multi-step task decomposition and structured execution
- **Multi-Agent**: Coordination and collaboration between specialized agents

## Key Responsibilities

- **Pattern Factory**: Create and configure reasoning pattern instances
- **Dependency Injection**: Manage service dependencies (LLM, memory, plugins, rules)
- **Reasoning Execution**: Execute reasoning patterns with proper error handling
- **Result Synthesis**: Process and format reasoning results
- **Performance Monitoring**: Track execution time and success rates
- **Capability Discovery**: Provide pattern capabilities and recommendations

## Architecture

### Factory Pattern Implementation
```python
# Create reasoning module
config = ReasoningConfig(pattern="ReAct", model="gpt-4")
reasoning_module = await factory.create_reasoning_module(config)
result = await reasoning_module.reason(context)
```

### Dependency Injection
```python
# Register service dependencies
factory.register_dependency("llm_service", llm_client)
factory.register_dependency("memory_manager", memory_client)
factory.register_dependency("plugin_registry", plugin_client)
```

## Reasoning Patterns

### ReAct Pattern
**Description**: Alternates between reasoning about what to do and taking actions
```python
# ReAct execution cycle
1. Observe current state
2. Reason about next action
3. Take the action
4. Observe the result
5. Repeat until goal achieved
```

**Best For**:
- Tasks requiring tool usage
- Information gathering
- Multi-step decision making

### Reflection Pattern
**Description**: Self-assessment and iterative improvement through metacognition
```python
# Reflection execution cycle
1. Generate initial solution
2. Reflect on solution quality
3. Identify improvements
4. Generate improved solution
5. Repeat for multiple iterations
```

**Best For**:
- Complex problem-solving
- Creative tasks needing iteration
- Quality-critical applications

### Planning Pattern
**Description**: Multi-step task decomposition and structured execution
```python
# Planning execution cycle
1. Analyze task and break it down
2. Create structured execution plan
3. Execute plan step by step
4. Monitor progress and adapt
5. Provide final result
```

**Best For**:
- Complex projects with dependencies
- Tasks requiring systematic execution
- Problems needing careful planning

### Multi-Agent Pattern
**Description**: Coordination and collaboration between multiple specialized agents
```python
# Multi-Agent execution cycle
1. Analyze task and determine agent roles
2. Create or assign specialized agents
3. Coordinate agent activities
4. Synthesize results from multiple agents
5. Provide unified final answer
```

**Best For**:
- Complex problems needing diverse expertise
- Tasks benefiting from multiple perspectives
- Problems requiring specialization

## API Endpoints

### Core Endpoints

#### `POST /reason`
Execute reasoning with specified pattern and context.

**Request Body:**
```json
{
  "context": {
    "task_description": "Analyze customer churn data and provide insights",
    "agent_id": "analyst_001",
    "agent_name": "Data Analyst Agent",
    "domain": "customer_analytics",
    "persona": "Expert data analyst with statistical expertise",
    "available_tools": [
      {"name": "data_analyzer", "description": "Analyze datasets"},
      {"name": "visualization", "description": "Create charts and graphs"}
    ],
    "previous_actions": [],
    "constraints": {"time_limit": "5 minutes"}
  },
  "config": {
    "pattern": "ReAct",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 4096,
    "pattern_specific_config": {
      "max_steps": 10
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "reasoning_id": "reasoning_abc123",
    "pattern_used": "ReAct",
    "final_answer": "Based on the analysis...",
    "reasoning_steps": [
      {
        "step": 1,
        "thought": "I need to examine the customer data...",
        "action": "data_analyzer"
      }
    ],
    "actions_taken": [
      {
        "step": 1,
        "action": "data_analyzer",
        "result": "Analysis completed..."
      }
    ],
    "confidence_score": 0.85,
    "execution_time": 2.34,
    "success": true
  }
}
```

#### `GET /patterns`
List all supported reasoning patterns.

#### `GET /patterns/{pattern}/capabilities`
Get detailed capabilities of a specific pattern.

#### `GET /health`
Service health check.

## Configuration

### Environment Variables

```bash
# Service Configuration
REASONING_MODULE_FACTORY_PORT=8304
SERVICE_HOST=localhost

# LLM Configuration
DEFAULT_LLM_MODEL=gpt-4
DEFAULT_LLM_TEMPERATURE=0.7
MAX_TOKENS=4096

# Pattern-specific Configuration
REACT_MAX_STEPS=10
REFLECTION_MAX_ITERATIONS=5
PLANNING_MAX_DEPTH=5
MULTI_AGENT_MAX_AGENTS=5
```

### Pattern-Specific Configuration

#### ReAct Configuration
```json
{
  "max_steps": 10,
  "action_timeout": 30,
  "observation_limit": 1000
}
```

#### Reflection Configuration
```json
{
  "max_iterations": 5,
  "improvement_threshold": 0.8,
  "reflection_depth": 3
}
```

#### Planning Configuration
```json
{
  "max_depth": 5,
  "step_timeout": 60,
  "adaptation_enabled": true
}
```

#### Multi-Agent Configuration
```json
{
  "max_agents": 5,
  "coordination_timeout": 120,
  "consensus_threshold": 0.7
}
```

## Service Dependencies

The factory integrates with multiple backend services:

- **LLM Processor** (port 8005): Language model generation and reasoning
- **Memory Manager** (port 8205): Working memory, episodic memory, vector memory
- **Plugin Registry** (port 8201): Domain and generic plugin management
- **Rule Engine** (port 8204): Business rule evaluation and processing

## Error Handling

The service provides comprehensive error handling:

- **Pattern Validation**: Ensures requested pattern is supported
- **Dependency Checks**: Validates required services are available
- **Timeout Handling**: Prevents runaway reasoning processes
- **Result Validation**: Ensures reasoning results are valid and complete
- **Graceful Degradation**: Fallback behavior when dependencies fail

## Performance Optimization

- **Async Processing**: Non-blocking reasoning execution
- **Connection Pooling**: Efficient service communication
- **Caching**: Pattern instances and configuration caching
- **Resource Limits**: Configurable limits on steps, iterations, and agents
- **Monitoring**: Execution time and success rate tracking

## Monitoring & Observability

The service provides detailed monitoring:

- **Execution Metrics**: Reasoning time, success rates, pattern usage
- **Error Tracking**: Failure rates and error types
- **Performance Profiling**: Bottleneck identification and optimization
- **Health Checks**: Automated service availability monitoring
- **Structured Logging**: Comprehensive logging with correlation IDs

## Testing

### Unit Tests
- Pattern implementation correctness
- Dependency injection functionality
- Configuration validation
- Error handling scenarios

### Integration Tests
- End-to-end reasoning execution
- Service dependency integration
- Performance benchmarking
- Load testing with multiple patterns

### API Tests
- REST endpoint validation
- Request/response format verification
- Error response handling
- Authentication and authorization

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run service
uvicorn main:app --host 0.0.0.0 --port 8304 --reload
```

### Docker Development
```bash
# Build image
docker build -t reasoning-module-factory .

# Run container
docker run -p 8304:8304 reasoning-module-factory
```

## Usage Examples

### Simple ReAct Reasoning
```python
from main import ReasoningModuleFactory, ReasoningContext, ReasoningConfig

factory = ReasoningModuleFactory()
config = ReasoningConfig(pattern="ReAct", model="gpt-4")
context = ReasoningContext(
    task_description="Solve this math problem: 2x + 3 = 7",
    agent_name="Math Tutor",
    domain="mathematics"
)

result = await factory.reason_with_pattern("ReAct", context, config)
print(result.final_answer)
```

### Complex Multi-Agent Reasoning
```python
config = ReasoningConfig(
    pattern="Multi-Agent",
    pattern_specific_config={"max_agents": 3}
)
context = ReasoningContext(
    task_description="Design a comprehensive business strategy",
    agent_name="Strategy Consultant",
    domain="business_strategy"
)

result = await factory.reason_with_pattern("Multi-Agent", context, config)
print(result.final_answer)
```

## Future Enhancements

- **Custom Pattern Development**: Framework for creating custom reasoning patterns
- **Pattern Composition**: Combining multiple patterns in sequence
- **Learning and Adaptation**: Pattern performance learning and automatic selection
- **Distributed Reasoning**: Multi-node reasoning coordination
- **Real-time Collaboration**: Live multi-agent collaboration interfaces
- **Pattern Marketplace**: Community-contributed reasoning patterns

## API Documentation

Complete API documentation is available at `/docs` when the service is running.

## Contributing

1. Follow the existing pattern implementation structure
2. Add comprehensive tests for new patterns
3. Update documentation for configuration changes
4. Ensure backward compatibility with existing patterns
5. Follow the dependency injection pattern for new services

## Security Considerations

- **Input Validation**: Comprehensive request validation
- **Resource Limits**: Configurable limits to prevent abuse
- **Error Information**: Secure error messages without sensitive data
- **Service Authentication**: Integration with authentication systems
- **Rate Limiting**: Protection against excessive API usage

## Performance Benchmarks

Based on testing with various reasoning patterns:

- **ReAct**: 2-5 seconds for typical tasks
- **Reflection**: 5-15 seconds for iterative improvement
- **Planning**: 3-8 seconds for structured tasks
- **Multi-Agent**: 10-30 seconds for complex coordination

Performance varies based on task complexity, available tools, and LLM model used.
