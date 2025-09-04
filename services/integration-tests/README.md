# Integration Tests Service

A comprehensive end-to-end integration testing service for the Agentic Brain platform that validates the complete system workflow from template instantiation to agent deployment and task execution.

## 🎯 Features

### Core Testing Capabilities
- **End-to-End Workflow Testing**: Complete agent lifecycle validation from creation to execution
- **Service Integration Testing**: Cross-service communication and data flow validation
- **Template Loading & Instantiation**: Pre-built template functionality verification
- **Agent Creation & Configuration**: Agent generation and setup validation
- **Deployment Pipeline Testing**: Automated deployment and rollback validation
- **Task Execution Validation**: Agent task processing and result verification
- **Performance & Scalability**: Load testing and performance benchmarking
- **Error Handling & Recovery**: Failure scenario testing and recovery validation

### Advanced Features
- **Real-time Test Monitoring**: Live test execution tracking and progress reporting
- **Automated Test Reporting**: Comprehensive test results and analytics
- **Parallel Test Execution**: Concurrent test suite execution for faster feedback
- **Retry Logic**: Automatic retry for transient failures
- **Test Data Management**: Isolated test data and cleanup
- **Custom Test Scenarios**: Extensible test framework for specific use cases
- **Performance Benchmarking**: Automated performance regression detection
- **Service Health Monitoring**: Real-time service availability and performance tracking

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- All Agent Brain services running
- PostgreSQL database
- Redis instance

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
docker-compose up -d integration-tests

# Check service health
curl http://localhost:8320/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

## 📡 API Endpoints

### Test Suite Management
```http
POST /test-suites
Content-Type: application/json

{
  "name": "E2E Test Suite",
  "description": "Complete end-to-end workflow testing",
  "category": "e2e",
  "metadata": {
    "environment": "staging",
    "priority": "high"
  }
}
```

### Test Monitoring
```http
# Get test suite results
GET /test-suites/{suite_id}

# List all test suites
GET /test-suites?limit=50&offset=0&category=e2e

# Get test summary
GET /test-results/summary?days=7
```

### Dashboard & Reporting
```http
# Web-based dashboard
GET /dashboard

# Prometheus metrics
GET /metrics

# Retry failed tests
POST /test-suites/{suite_id}/retry
```

## 🧪 Test Categories

### 1. End-to-End Tests (E2E)
Complete workflow validation from template to execution:

#### Underwriting Agent E2E Test
```json
{
  "test_name": "underwriting_agent_e2e",
  "description": "Complete underwriting workflow validation",
  "steps": [
    {
      "step": "get_template",
      "description": "Retrieve underwriting template"
    },
    {
      "step": "instantiate_template",
      "description": "Create agent from template"
    },
    {
      "step": "generate_agent",
      "description": "Generate agent configuration"
    },
    {
      "step": "register_agent",
      "description": "Register agent with orchestrator"
    },
    {
      "step": "deploy_agent",
      "description": "Deploy agent to production"
    },
    {
      "step": "execute_task",
      "description": "Execute risk assessment task"
    }
  ]
}
```

#### Claims Processing E2E Test
```json
{
  "test_name": "claims_processing_e2e",
  "description": "Complete claims processing workflow",
  "steps": [
    {
      "step": "get_claims_template",
      "description": "Retrieve claims processing template"
    },
    {
      "step": "instantiate_with_parameters",
      "description": "Configure with specific parameters"
    },
    {
      "step": "validate_fraud_detection",
      "description": "Test fraud detection integration"
    },
    {
      "step": "execute_claim_processing",
      "description": "Process sample claim"
    }
  ]
}
```

### 2. Service Integration Tests
Cross-service communication and data flow validation:

#### Plugin Registry Integration
```json
{
  "test_name": "plugin_registry_integration",
  "description": "Plugin registry service integration",
  "validations": [
    {
      "service": "plugin_registry",
      "endpoint": "/health",
      "expected_status": 200
    },
    {
      "service": "plugin_registry",
      "endpoint": "/plugins",
      "validate_response": true
    },
    {
      "service": "plugin_registry",
      "endpoint": "/plugins/{plugin_id}/execute",
      "test_data": {"input": "test"}
    }
  ]
}
```

#### Data Flow Integration
```json
{
  "test_name": "data_flow_integration",
  "description": "Cross-service data flow validation",
  "data_flows": [
    {
      "source": "template_store",
      "target": "brain_factory",
      "transformation": "template_to_config"
    },
    {
      "source": "brain_factory",
      "target": "agent_orchestrator",
      "transformation": "config_to_agent"
    },
    {
      "source": "agent_orchestrator",
      "target": "deployment_pipeline",
      "transformation": "agent_to_deployment"
    }
  ]
}
```

### 3. Performance Tests
Load testing and performance validation:

#### Concurrent Agent Creation
```json
{
  "test_name": "concurrent_agent_creation",
  "description": "Test concurrent agent creation performance",
  "parameters": {
    "num_agents": 5,
    "max_concurrent": 3,
    "timeout_seconds": 300
  },
  "metrics": [
    "total_time_seconds",
    "successful_creations",
    "failed_creations",
    "success_rate",
    "average_time_per_agent"
  ]
}
```

#### Service Response Times
```json
{
  "test_name": "service_response_times",
  "description": "Validate service response times",
  "services": [
    "agent_orchestrator",
    "plugin_registry",
    "template_store",
    "brain_factory",
    "deployment_pipeline"
  ],
  "thresholds": {
    "max_response_time_ms": 5000,
    "min_success_rate": 0.95
  }
}
```

## ⚙️ Configuration

### Environment Variables
```bash
# Service Configuration
INTEGRATION_TESTS_PORT=8320
DEFAULT_TEST_TIMEOUT=300
MAX_CONCURRENT_TESTS=5
RETRY_FAILED_TESTS=true
MAX_RETRIES=3

# Service URLs
AGENT_ORCHESTRATOR_URL=http://localhost:8200
PLUGIN_REGISTRY_URL=http://localhost:8201
TEMPLATE_STORE_URL=http://localhost:8202
BRAIN_FACTORY_URL=http://localhost:8301
UI_TO_BRAIN_MAPPER_URL=http://localhost:8302
DEPLOYMENT_PIPELINE_URL=http://localhost:8303

# Database & Redis
DATABASE_URL=postgresql://user:password@postgres:5432/agentic_brain
REDIS_HOST=redis
REDIS_PORT=6379

# Performance Thresholds
MAX_RESPONSE_TIME_MS=5000
MIN_SUCCESS_RATE=0.95

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8002
```

### Test Configuration Files
```json
// test_scenarios.json
{
  "e2e_scenarios": {
    "underwriting_complete": {
      "name": "Complete Underwriting Workflow",
      "template": "underwriting_template",
      "test_data": {
        "loan_amount": 250000,
        "credit_score": 720,
        "income": 85000
      },
      "expected_results": {
        "risk_category": "LOW",
        "approval_probability": 0.9
      }
    }
  },
  "integration_scenarios": {
    "service_health_check": {
      "services": ["orchestrator", "brain_factory", "deployment"],
      "checks": ["health", "readiness", "metrics"]
    }
  }
}
```

## 📊 Test Reporting

### Real-time Dashboard
Access the integration tests dashboard at `http://localhost:8320/dashboard` for:
- **Live Test Monitoring**: Real-time test execution status
- **Performance Metrics**: Response times and success rates
- **Test History**: Complete test suite execution history
- **Failure Analysis**: Detailed error reporting and diagnostics
- **Trend Analysis**: Performance trends and regression detection

### Test Reports
```bash
# Generate comprehensive test report
curl -X GET "http://localhost:8320/test-suites/{suite_id}" \
     -H "Accept: application/json" > test_results.json

# Get test summary for last 7 days
curl -X GET "http://localhost:8320/test-results/summary?days=7" \
     -H "Accept: application/json" > test_summary.json

# Export test results as HTML
curl -X GET "http://localhost:8320/dashboard" \
     -H "Accept: text/html" > dashboard.html
```

### Test Metrics
```json
{
  "period_days": 7,
  "summary": {
    "total_suites": 15,
    "completed_suites": 14,
    "failed_suites": 1,
    "suite_success_rate": 93.3,
    "total_tests": 89,
    "passed_tests": 83,
    "failed_tests": 6,
    "test_success_rate": 93.3
  },
  "categories": {
    "e2e": 8,
    "integration": 4,
    "performance": 3
  }
}
```

## 🔧 Advanced Usage

### Custom Test Scenarios
```python
from integration_tests import IntegrationTestManager, EndToEndTestSuite

# Create custom E2E test
async def custom_underwriting_test():
    test_manager = IntegrationTestManager()

    # Create custom test suite
    suite_id = test_manager.create_test_suite(
        "Custom Underwriting Test",
        "Specialized underwriting workflow test",
        "e2e",
        {"custom_parameters": {"loan_amount": 500000}}
    )

    # Execute custom test
    results = await test_manager.run_e2e_test_suite(suite_id)
    return results

# Run custom test
results = await custom_underwriting_test()
```

### Parallel Test Execution
```python
import asyncio
from integration_tests import IntegrationTestManager

async def run_parallel_tests():
    test_manager = IntegrationTestManager()

    # Define test configurations
    test_configs = [
        {
            "name": "Underwriting E2E",
            "category": "e2e",
            "template": "underwriting_template"
        },
        {
            "name": "Service Integration",
            "category": "integration",
            "services": ["orchestrator", "brain_factory"]
        },
        {
            "name": "Performance Test",
            "category": "performance",
            "num_agents": 3
        }
    ]

    # Execute tests in parallel
    tasks = []
    for config in test_configs:
        suite_id = test_manager.create_test_suite(
            config["name"],
            f"{config['category']} test suite",
            config["category"]
        )

        if config["category"] == "e2e":
            task = test_manager.run_e2e_test_suite(suite_id)
        elif config["category"] == "integration":
            task = test_manager.run_service_integration_tests(suite_id)
        elif config["category"] == "performance":
            task = test_manager.run_performance_tests(suite_id)

        tasks.append(task)

    # Wait for all tests to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### CI/CD Integration
```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r services/integration-tests/requirements.txt

      - name: Start services
        run: |
          docker-compose up -d postgres redis
          sleep 30

      - name: Run integration tests
        run: |
          python -m pytest services/integration-tests/ \
            --asyncio-mode=auto \
            --tb=short \
            -v

      - name: Generate test report
        run: |
          python services/integration-tests/main.py --generate-report

      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: integration-test-results
          path: services/integration-tests/test_reports/
```

### Monitoring & Alerting
```python
# Custom monitoring setup
from integration_tests import IntegrationTestManager
import prometheus_client as prom

# Custom metrics
test_success_rate = prom.Gauge('integration_test_success_rate', 'Test success rate')
test_execution_time = prom.Histogram('integration_test_execution_time', 'Test execution time')
service_health_status = prom.Gauge('service_health_status', 'Service health status')

async def monitor_integration_tests():
    test_manager = IntegrationTestManager()

    while True:
        # Get test metrics
        summary = await test_manager.get_test_summary(days=1)

        # Update Prometheus metrics
        test_success_rate.set(summary['summary']['test_success_rate'])
        service_health_status.set(1 if summary['summary']['failed_suites'] == 0 else 0)

        # Check for alerts
        if summary['summary']['test_success_rate'] < 90:
            logger.warning("Test success rate below threshold")
            # Send alert notification

        await asyncio.sleep(300)  # Check every 5 minutes
```

## 🔐 Security Considerations

### Test Data Protection
- Use isolated test databases and data sets
- Implement data sanitization for sensitive information
- Encrypt test data in transit and at rest
- Regular cleanup of test artifacts and logs

### Access Control
```python
# Test execution permissions
test_permissions = {
    "admin": {
        "can_create_suites": true,
        "can_run_all_tests": true,
        "can_view_all_results": true,
        "can_delete_suites": true
    },
    "developer": {
        "can_create_suites": true,
        "can_run_integration_tests": true,
        "can_view_own_results": true,
        "can_delete_own_suites": true
    },
    "qa": {
        "can_create_suites": true,
        "can_run_all_tests": true,
        "can_view_all_results": true,
        "can_delete_suites": false
    }
}
```

### Secure Test Execution
- Run tests in isolated environments
- Use API authentication for service communication
- Implement rate limiting for test endpoints
- Log all test activities for audit trails
- Validate test inputs to prevent injection attacks

## 📈 Performance Optimization

### Test Execution Optimization
```python
# Optimized test configuration
optimized_config = {
    "execution": {
        "parallel_execution": true,
        "max_concurrent_tests": 5,
        "test_timeout": 300,
        "retry_failed_tests": true,
        "max_retries": 2
    },
    "resource_management": {
        "memory_limit_mb": 1024,
        "cpu_limit": 2,
        "cleanup_after_test": true,
        "reuse_connections": true
    },
    "data_management": {
        "use_test_database": true,
        "isolate_test_data": true,
        "cleanup_test_data": true,
        "cache_frequently_used_data": true
    }
}
```

### Database Optimization
```python
# Test database configuration
test_database_config = {
    "connection_pool": {
        "min_connections": 5,
        "max_connections": 20,
        "connection_timeout": 30
    },
    "query_optimization": {
        "enable_query_cache": true,
        "cache_size_mb": 256,
        "prepared_statements": true
    },
    "maintenance": {
        "auto_vacuum": true,
        "auto_analyze": true,
        "maintenance_work_mem": "128MB"
    }
}
```

### Service Mocking
```python
# Service mocking for faster testing
service_mocks = {
    "template_store": {
        "enabled": true,
        "mock_responses": {
            "/templates": {"status": 200, "data": [...]},
            "/templates/{id}/instantiate": {"status": 200, "data": {...}}
        },
        "response_delay_ms": 10
    },
    "brain_factory": {
        "enabled": true,
        "mock_responses": {
            "/generate-agent": {"status": 200, "data": {"agent_id": "mock_agent_123"}}
        }
    }
}
```

## 🐛 Troubleshooting

### Common Issues

#### Service Connectivity Problems
```bash
# Check service availability
curl -I http://localhost:8200/health
curl -I http://localhost:8301/health

# Verify service logs
docker-compose logs agent-orchestrator
docker-compose logs brain-factory

# Check network connectivity
docker network inspect agentic-network
```

#### Test Execution Failures
```bash
# Check test logs
tail -f services/integration-tests/test_logs/integration_tests.log

# Validate test data
python -c "import json; print(json.load(open('test_data/sample_workflow.json')))"

# Test individual services
curl -X POST http://localhost:8301/generate-agent \
     -H "Content-Type: application/json" \
     -d @test_data/agent_config.json
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check database performance
docker exec -it postgres pg_stat_activity;

# Analyze test execution time
python -m cProfile services/integration-tests/main.py
```

#### Database Connection Issues
```bash
# Check database connectivity
psql -h localhost -U user -d agentic_brain -c "SELECT 1;"

# Validate connection string
python -c "import sqlalchemy; engine = sqlalchemy.create_engine('postgresql://user:password@localhost:5432/agentic_brain'); print(engine.execute('SELECT 1').fetchone())"

# Check connection pool
docker exec -it postgres pg_stat_database;
```

### Debug Mode
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
export PYTHONPATH=/app

# Run tests with verbose output
python main.py --verbose --debug

# Enable service mocks for isolated testing
export USE_SERVICE_MOCKS=true
export MOCK_RESPONSE_DELAY=0
```

## 📚 API Documentation

### Complete API Reference
- [Test Suite Management](./docs/api/test-suite-management.md)
- [Test Execution](./docs/api/test-execution.md)
- [Results & Reporting](./docs/api/results-reporting.md)
- [Configuration](./docs/api/configuration.md)

### SDKs and Libraries
- **Python SDK**: `pip install agentic-brain-integration-tests`
- **REST API Client**: Comprehensive HTTP client libraries
- **Test Framework Integration**: Pytest, Jest, and Cypress plugins

## 🤝 Contributing

### Development Setup
```bash
# Clone the repository
git clone https://github.com/agentic-brain/integration-tests.git
cd integration-tests

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
2. Create a feature branch (`git checkout -b feature/new-test-scenario`)
3. Make your changes with tests
4. Ensure all tests pass (`pytest tests/ -v`)
5. Update documentation if needed
6. Submit a pull request with detailed description

### Test Coverage Requirements
- Maintain >90% test coverage for new code
- Include both positive and negative test cases
- Test error conditions and edge cases
- Validate performance requirements are met
- Ensure cross-browser compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: [docs.agenticbrain.com/integration-tests](https://docs.agenticbrain.com/integration-tests)
- **Community Forum**: [community.agenticbrain.com](https://community.agenticbrain.com)
- **Issue Tracker**: [github.com/agentic-brain/integration-tests/issues](https://github.com/agentic-brain/integration-tests/issues)
- **Email Support**: support@agenticbrain.com

### Service Level Agreements
- **Response Time**: < 4 hours for critical issues
- **Resolution Time**: < 24 hours for standard issues
- **Uptime**: 99.5% service availability
- **Test Execution Time**: < 30 minutes for standard test suites
- **Support Hours**: 24/7 enterprise support available

---

**Built with ❤️ for the Agentic Brain Platform**

*Ensuring system reliability through comprehensive integration testing*
