# Automated Testing Service

A comprehensive automated testing service for the Agentic Brain platform that orchestrates unit tests, integration tests, API tests, performance tests, and regression testing across all services with detailed analytics and reporting capabilities.

## 🎯 Features

### Core Testing Capabilities
- **Unit Testing**: Automated unit test generation and execution for individual service components
- **Integration Testing**: Cross-service integration testing with workflow validation
- **API Testing**: Comprehensive API endpoint testing with validation and error handling
- **Performance Testing**: Load testing using Locust with configurable scenarios
- **Regression Testing**: Automated regression detection and reporting

### Advanced Testing Features
- **Test Orchestration**: Intelligent test suite execution with dependency management
- **Multi-environment Testing**: Support for development, staging, and production environments
- **Parallel Test Execution**: Concurrent test execution for faster results
- **Test Result Analytics**: Comprehensive analytics and trend analysis
- **Coverage Analysis**: Code coverage reporting and optimization
- **Custom Test Scenarios**: Flexible framework for creating domain-specific tests

### Testing Orchestration
- **Test Suite Management**: Predefined and custom test suite creation
- **Scheduled Testing**: Automated test execution scheduling
- **CI/CD Integration**: Seamless integration with continuous integration pipelines
- **Result Aggregation**: Unified test result collection and reporting
- **Failure Analysis**: Root cause analysis and debugging support
- **Performance Benchmarking**: Historical performance trend analysis

### Monitoring and Analytics
- **Real-time Test Monitoring**: Live test execution status and progress
- **Test Metrics Collection**: Detailed metrics collection for all test types
- **Trend Analysis**: Historical test result and performance trend analysis
- **Coverage Reporting**: Code coverage visualization and optimization
- **Custom Dashboards**: Interactive dashboards for test analytics
- **Alert Integration**: Automated alerts for test failures and performance regressions

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Agentic Brain platform running
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
docker-compose up -d automated-testing-service

# Check service health
curl http://localhost:8390/health
```

### Basic Test Execution
```bash
# Execute unit tests for agent-orchestrator
curl -X POST http://localhost:8390/api/tests/execute \
  -H "Content-Type: application/json" \
  -d '{
    "test_type": "unit",
    "target_services": ["agent-orchestrator"]
  }'

# Execute integration tests
curl -X POST http://localhost:8390/api/tests/execute \
  -H "Content-Type: application/json" \
  -d '{
    "test_type": "integration",
    "target_services": ["agent-orchestrator", "brain-factory", "deployment-pipeline"]
  }'

# Execute performance tests
curl -X POST http://localhost:8390/api/tests/execute \
  -H "Content-Type: application/json" \
  -d '{
    "test_type": "performance",
    "target_services": ["all"],
    "config": {
      "users": 10,
      "duration": "30s",
      "host": "http://localhost:8200"
    }
  }'
```

## 📡 API Endpoints

### Test Execution
```http
POST /api/tests/execute              # Execute specific test type
POST /api/tests/execute-suite        # Execute predefined test suite
GET  /api/tests/suites               # Get available test suites
POST /api/tests/suites               # Create new test suite
GET  /api/tests/executions           # Get test execution history
GET  /api/tests/executions/{id}      # Get detailed execution results
GET  /api/tests/analytics            # Get test analytics and trends
```

### Monitoring & Dashboard
```http
GET  /dashboard                      # Interactive testing dashboard
GET  /health                        # Service health check
GET  /metrics                       # Prometheus metrics endpoint
```

## 🧪 Test Types

### 1. Unit Testing
**Purpose**: Test individual service components in isolation

**Features**:
- Automated test file generation
- Mock dependency injection
- Code coverage analysis
- Performance benchmarking
- Error condition testing

**Execution Example**:
```python
# Generated unit test structure
class TestAgentOrchestrator(unittest.TestCase):
    def test_agent_registration(self):
        # Test agent registration logic
        pass

    def test_task_routing(self):
        # Test task routing functionality
        pass

    def test_error_handling(self):
        # Test error handling scenarios
        pass
```

### 2. Integration Testing
**Purpose**: Test interactions between multiple services

**Test Scenarios**:
- Agent creation workflow (Brain Factory → Agent Orchestrator)
- Task execution pipeline (Agent Orchestrator → Brain Factory → Execution)
- Data flow validation (Input → Processing → Output)
- Service health communication
- Cross-service error handling

### 3. API Testing
**Purpose**: Validate REST API endpoints and functionality

**Coverage**:
- Endpoint availability and response codes
- Request/response validation
- Authentication and authorization
- Error handling and edge cases
- Performance and timeout handling
- CORS and security headers

### 4. Performance Testing
**Purpose**: Test system performance under load

**Configuration**:
```json
{
  "users": 10,
  "spawn_rate": 2,
  "duration": "30s",
  "host": "http://localhost:8200",
  "scenarios": [
    {
      "name": "agent_creation",
      "weight": 30
    },
    {
      "name": "task_execution",
      "weight": 40
    },
    {
      "name": "service_health",
      "weight": 30
    }
  ]
}
```

**Metrics Collected**:
- Response time (average, 95th percentile)
- Requests per second (RPS)
- Failure rate percentage
- Memory and CPU usage
- Error distribution

### 5. Comprehensive Testing Suite
**Purpose**: Execute all test types in sequence

**Execution Flow**:
1. Unit tests for all services
2. Integration tests across services
3. API endpoint validation
4. Performance load testing
5. Result aggregation and reporting

## ⚙️ Configuration

### Environment Variables
```bash
# Service Configuration
AUTOMATED_TESTING_PORT=8390
HOST=0.0.0.0

# Test Execution Configuration
MAX_CONCURRENT_TESTS=10
TEST_TIMEOUT_SECONDS=600
RESULTS_RETENTION_DAYS=30

# Service URLs
AGENT_ORCHESTRATOR_URL=http://localhost:8200
BRAIN_FACTORY_URL=http://localhost:8301
DEPLOYMENT_PIPELINE_URL=http://localhost:8303
UI_TESTING_URL=http://localhost:8310
INTEGRATION_TESTS_URL=http://localhost:8320
AUTHENTICATION_URL=http://localhost:8330

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/agentic_brain

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Test Directories
TEST_WORKSPACE=/app/test_workspace
TEST_RESULTS_DIR=/app/test_results
TEST_LOGS_DIR=/app/test_logs
```

### Test Suite Configuration
```json
{
  "name": "Agent Orchestrator Unit Tests",
  "description": "Comprehensive unit tests for agent orchestrator service",
  "test_type": "unit",
  "target_services": ["agent-orchestrator"],
  "test_config": {
    "coverage_target": 85,
    "timeout_seconds": 300,
    "parallel_execution": true
  },
  "schedule_config": {
    "enabled": true,
    "cron_expression": "0 2 * * *",
    "timezone": "UTC"
  }
}
```

## 📊 Test Results and Analytics

### Test Execution Results
```json
{
  "execution_id": "exec-12345",
  "status": "completed",
  "start_time": "2024-01-15T10:30:00Z",
  "duration_seconds": 245.8,
  "test_results": {
    "unit_tests": {
      "services_tested": ["agent-orchestrator", "brain-factory"],
      "total_tests": 45,
      "passed_tests": 42,
      "failed_tests": 3,
      "coverage_percentage": 87.5
    },
    "integration_tests": {
      "scenarios_tested": 5,
      "passed_scenarios": 4,
      "failed_scenarios": 1,
      "success_rate": 80.0
    },
    "performance_tests": {
      "total_requests": 1250,
      "requests_per_second": 41.7,
      "response_time_avg": 185.3,
      "failure_rate": 1.2
    }
  },
  "overall_status": "passed",
  "recommendations": [
    "Fix 3 failing unit tests in brain-factory",
    "Investigate integration test failure in data flow validation",
    "Performance within acceptable limits"
  ]
}
```

### Analytics Dashboard
```json
{
  "period_days": 30,
  "executions": {
    "total": 45,
    "successful": 38,
    "failed": 7,
    "success_rate": 84.4
  },
  "tests": {
    "total": 1250,
    "passed": 1187,
    "failed": 63,
    "pass_rate": 94.96
  },
  "performance": {
    "avg_execution_time_seconds": 185.3,
    "avg_response_time_ms": 245.8,
    "avg_failure_rate_percent": 1.8
  },
  "trends": {
    "success_rate_trend": "improving",
    "performance_trend": "stable",
    "coverage_trend": "increasing"
  }
}
```

## 🎨 Interactive Testing Dashboard

### Dashboard Features
- **Real-time Test Monitoring**: Live status updates of running tests
- **Test Suite Management**: Easy selection and execution of test suites
- **Result Visualization**: Charts and graphs for test results and performance metrics
- **Historical Analysis**: Trend analysis of test success rates over time
- **Coverage Reporting**: Code coverage visualization and optimization
- **Performance Analytics**: Performance metrics and bottleneck identification
- **Failure Analysis**: Detailed breakdown of test failures with diagnostics
- **Custom Test Creation**: UI for creating custom test scenarios

### Dashboard Components
```html
<!-- Test Statistics Cards -->
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value" id="total-executions">0</div>
        <div class="stat-label">Test Executions</div>
    </div>
    <div class="stat-card">
        <div class="stat-value" id="success-rate">0%</div>
        <div class="stat-label">Success Rate</div>
    </div>
    <div class="stat-card">
        <div class="stat-value" id="avg-coverage">0%</div>
        <div class="stat-label">Code Coverage</div>
    </div>
    <div class="stat-card">
        <div class="stat-value" id="performance-score">0</div>
        <div class="stat-label">Performance Score</div>
    </div>
</div>

<!-- Test Suite Cards -->
<div class="test-suites">
    <div class="suite-card" onclick="runTestSuite('unit-tests')">
        <h4>Unit Tests</h4>
        <p>Individual component testing</p>
        <button class="run-btn">Run Tests</button>
    </div>
    <div class="suite-card" onclick="runTestSuite('integration-tests')">
        <h4>Integration Tests</h4>
        <p>Cross-service interaction testing</p>
        <button class="run-btn">Run Tests</button>
    </div>
    <div class="suite-card" onclick="runTestSuite('performance-tests')">
        <h4>Performance Tests</h4>
        <p>Load and performance testing</p>
        <button class="run-btn">Run Tests</button>
    </div>
</div>

<!-- Test Results Table -->
<div class="results-table">
    <table>
        <thead>
            <tr>
                <th>Test Type</th>
                <th>Status</th>
                <th>Duration</th>
                <th>Success Rate</th>
                <th>Coverage</th>
                <th>Timestamp</th>
            </tr>
        </thead>
        <tbody id="results-body">
            <!-- Dynamic test results -->
        </tbody>
    </table>
</div>
```

## 🔧 Integration with CI/CD

### GitHub Actions Example
```yaml
name: Automated Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  automated-tests:
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

      redis:
        image: redis:6-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r services/automated-testing-service/requirements.txt

    - name: Start platform services
      run: docker-compose up -d

    - name: Wait for services
      run: |
        timeout 300 bash -c 'until curl -f http://localhost:8200/health; do sleep 5; done'

    - name: Run comprehensive test suite
      run: |
        curl -X POST http://localhost:8390/api/tests/execute-suite \
          -H "Content-Type: application/json" \
          -d '{"suite_id": "comprehensive-suite", "environment": "ci"}'

    - name: Generate test report
      run: |
        python services/automated-testing-service/generate_report.py

    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: test-results/
```

### Jenkins Pipeline Example
```groovy
pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                sh 'docker-compose up -d'
                sh 'sleep 60' // Wait for services to be ready
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    curl -X POST http://localhost:8390/api/tests/execute \
                      -H "Content-Type: application/json" \
                      -d '{"test_type": "unit", "target_services": ["all"]}' \
                      --output unit_test_results.json
                '''
            }
        }

        stage('Integration Tests') {
            steps {
                sh '''
                    curl -X POST http://localhost:8390/api/tests/execute \
                      -H "Content-Type: application/json" \
                      -d '{"test_type": "integration", "target_services": ["all"]}' \
                      --output integration_test_results.json
                '''
            }
        }

        stage('Performance Tests') {
            steps {
                sh '''
                    curl -X POST http://localhost:8390/api/tests/execute \
                      -H "Content-Type: application/json" \
                      -d '{"test_type": "performance", "target_services": ["all"], "config": {"users": 20, "duration": "60s"}}' \
                      --output performance_test_results.json
                '''
            }
        }

        stage('Generate Report') {
            steps {
                sh '''
                    python services/automated-testing-service/generate_comprehensive_report.py \
                      --unit unit_test_results.json \
                      --integration integration_test_results.json \
                      --performance performance_test_results.json \
                      --output comprehensive_test_report.html
                '''
            }
        }
    }

    post {
        always {
            sh 'docker-compose down'
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: '.',
                reportFiles: 'comprehensive_test_report.html',
                reportName: 'Comprehensive Test Report'
            ])
        }
        failure {
            sh 'curl -X POST http://localhost:8390/api/alerts/test-failure'
        }
    }
}
```

## 📚 API Integration Examples

### RESTful API Usage
```python
import requests

# Execute unit tests
response = requests.post('http://localhost:8390/api/tests/execute', json={
    "test_type": "unit",
    "target_services": ["agent-orchestrator", "brain-factory"],
    "config": {
        "coverage_target": 85,
        "parallel_execution": True
    }
})

test_result = response.json()
print(f"Test Status: {test_result['status']}")
print(f"Success Rate: {test_result['results']['success_rate']}%")

# Execute comprehensive test suite
suite_response = requests.post('http://localhost:8390/api/tests/execute-suite', json={
    "suite_id": "comprehensive-suite",
    "environment": "production",
    "triggered_by": "automated-ci"
})

suite_result = suite_response.json()
print(f"Suite Execution ID: {suite_result['execution_id']}")
print(f"Overall Status: {suite_result['overall_status']}")
```

### Python SDK
```python
from agentic_brain_automated_testing import AutomatedTestingClient

client = AutomatedTestingClient(
    base_url="http://localhost:8390",
    timeout=600
)

# Run unit tests
unit_results = await client.execute_tests(
    test_type="unit",
    target_services=["agent-orchestrator"],
    config={"coverage_target": 85}
)

# Run performance tests
perf_results = await client.execute_tests(
    test_type="performance",
    target_services=["all"],
    config={
        "users": 50,
        "duration": "120s",
        "host": "http://localhost:8200"
    }
)

# Get test analytics
analytics = await client.get_analytics(days=30)
print(f"Success Rate: {analytics['executions']['success_rate']}%")
```

### Monitoring Integration
```python
# Integration with monitoring service
monitoring_client = httpx.AsyncClient()

async def monitor_test_execution(execution_id: str):
    """Monitor test execution in real-time"""
    while True:
        response = await monitoring_client.get(f"/api/tests/executions/{execution_id}")
        execution = response.json()

        if execution["execution"]["status"] in ["completed", "failed"]:
            break

        await asyncio.sleep(10)  # Check every 10 seconds

# Usage
asyncio.create_task(monitor_test_execution("exec-12345"))
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: [docs.agenticbrain.com/automated-testing](https://docs.agenticbrain.com/automated-testing)
- **Community Forum**: [community.agenticbrain.com/automated-testing](https://community.agenticbrain.com/automated-testing)
- **Issue Tracker**: [github.com/agentic-brain/automated-testing-service/issues](https://github.com/agentic-brain/automated-testing-service/issues)
- **Email Support**: automated-testing-support@agenticbrain.com

### Service Level Agreements
- **Test Execution**: < 10 minutes for standard test suites
- **Result Generation**: < 5 minutes for comprehensive reports
- **Dashboard Load Time**: < 2 seconds for real-time updates
- **API Response Time**: < 1 second for status queries
- **Uptime**: 99.9% service availability
- **Test Coverage**: > 85% for all services
- **False Positive Rate**: < 2% for automated tests

---

**Built with ❤️ for the Agentic Brain Platform**

*Enterprise-grade automated testing for AI automation platforms*
