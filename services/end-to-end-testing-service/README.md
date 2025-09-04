# End-to-End Testing Service

A comprehensive end-to-end testing service for the Agentic Brain platform that orchestrates complete system validation from UI workflow creation through agent deployment and task execution, ensuring the entire platform works seamlessly as an integrated system with automated testing, performance validation, and comprehensive reporting.

## 🎯 Features

### Core E2E Testing Capabilities
- **UI Workflow Creation Testing**: Automated testing of drag-and-drop workflow builder interface using Selenium WebDriver
- **Agent Deployment Pipeline Testing**: Complete validation of agent creation, registration, and deployment processes
- **Task Execution Pipeline Testing**: End-to-end testing of task routing, execution, and result processing
- **Multi-Service Integration Testing**: Validation of interactions between all Agent Brain microservices
- **Performance and Load Testing**: Automated load testing with configurable user simulation and metrics collection

### Advanced Testing Features
- **Error Scenario Testing**: Comprehensive testing of error handling and recovery mechanisms
- **Data Flow Validation**: End-to-end validation of data processing pipelines across services
- **Service Health Monitoring**: Continuous health checking during test execution
- **Automated Test Reporting**: Detailed test results with screenshots, logs, and performance metrics
- **Parallel Test Execution**: Concurrent test execution for faster validation cycles
- **Custom Test Scenario Creation**: Flexible framework for creating new test scenarios

### Testing Orchestration
- **Test Suite Management**: Organized execution of multiple test scenarios
- **Dependency Management**: Automatic handling of service startup and dependency resolution
- **Test Data Management**: Automated creation and cleanup of test data
- **Result Aggregation**: Comprehensive aggregation of test results and metrics
- **CI/CD Integration**: Seamless integration with continuous integration pipelines
- **Scheduled Testing**: Automated scheduling of test execution

### Monitoring and Analytics
- **Real-time Test Monitoring**: Live monitoring of test execution progress
- **Performance Metrics Collection**: Detailed performance metrics during test execution
- **Failure Analysis**: Root cause analysis of test failures with detailed diagnostics
- **Trend Analysis**: Historical analysis of test results and performance trends
- **Custom Dashboards**: Interactive dashboards for test result visualization
- **Alert Integration**: Automated alerts for test failures and performance regressions

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Agentic Brain platform running
- Google Chrome (for UI testing)
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
docker-compose up -d end-to-end-testing-service

# Check service health
curl http://localhost:8380/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install Chrome WebDriver
webdriver-manager update

# Run the service
python main.py
```

### Basic Test Execution
```bash
# Execute UI workflow builder test
curl -X POST http://localhost:8380/tests/execute \
  -H "Content-Type: application/json" \
  -d '{"scenario": "ui_workflow_builder"}'

# Execute end-to-end pipeline test
curl -X POST http://localhost:8380/tests/execute \
  -H "Content-Type: application/json" \
  -d '{"scenario": "end_to_end_pipeline"}'

# Execute complete E2E test suite
curl -X POST http://localhost:8380/tests/e2e-suite \
  -H "Content-Type: application/json" \
  -d '{
    "scenarios": ["ui_workflow_builder", "end_to_end_pipeline", "performance_load_test"],
    "parallel_execution": true
  }'
```

## 📡 API Endpoints

### Test Execution
```http
POST /tests/execute                    # Execute specific test scenario
POST /tests/e2e-suite                  # Execute complete E2E test suite
GET /tests/scenarios                   # Get available test scenarios
GET /tests/results                     # Get test execution results
```

### Monitoring & Dashboard
```http
GET /dashboard                         # Interactive test dashboard
GET /health                           # Service health check
```

## 🧪 Test Scenarios

### 1. UI Workflow Builder Test
**Purpose**: Validates the drag-and-drop workflow creation interface

**Test Steps**:
1. Navigate to Agent Builder UI
2. Verify canvas loading and component palette
3. Test component drag-and-drop functionality
4. Validate workflow creation and saving
5. Check real-time validation feedback

**Success Criteria**:
- Page loads within 5 seconds
- All UI components are accessible
- Drag-and-drop operations work correctly
- Workflow validation passes
- No JavaScript errors occur

### 2. End-to-End Pipeline Test
**Purpose**: Tests complete agent lifecycle from creation to task execution

**Test Steps**:
1. Create agent configuration via Brain Factory
2. Register agent with Orchestrator
3. Deploy agent to execution environment
4. Submit test task for execution
5. Validate task completion and results
6. Clean up test resources

**Success Criteria**:
- Agent creation completes successfully
- Agent registration succeeds
- Task execution produces expected results
- All service interactions work correctly
- Cleanup completes without errors

### 3. Performance Load Test
**Purpose**: Tests system performance under concurrent load

**Test Configuration**:
- Concurrent users: 10 (configurable)
- Test duration: 60 seconds (configurable)
- Ramp-up period: 10 seconds (configurable)

**Performance Metrics**:
- Response time (average, 95th percentile)
- Throughput (requests per second)
- Error rate percentage
- Resource utilization (CPU, memory)

### 4. Integration Test
**Purpose**: Validates service-to-service integrations

**Test Coverage**:
- Brain Factory ↔ Orchestrator communication
- Orchestrator ↔ Plugin Registry integration
- Workflow Engine ↔ Memory Manager interaction
- UI-to-Brain Mapper ↔ Brain Factory coordination
- Audit Logging across all services

### 5. Error Recovery Test
**Purpose**: Tests error handling and recovery mechanisms

**Error Scenarios**:
- Service unavailability simulation
- Network timeout handling
- Database connection failures
- Invalid input validation
- Resource exhaustion scenarios

## ⚙️ Configuration

### Environment Variables
```bash
# Service Configuration
E2E_TESTING_PORT=8380
E2E_TESTING_HOST=0.0.0.0

# Test Configuration
MAX_CONCURRENT_TESTS=5
TEST_TIMEOUT_SECONDS=300
CLEANUP_AFTER_TEST=true

# UI Testing Configuration
SELENIUM_HUB_URL=http://localhost:4444/wd/hub
UI_TEST_TIMEOUT=60
BROWSER_HEADLESS=true

# Load Testing Configuration
LOAD_TEST_USERS=10
LOAD_TEST_DURATION=60
LOAD_TEST_RAMP_UP=10

# Agent Brain Service URLs
AGENT_BUILDER_UI_URL=http://localhost:8300
AGENT_ORCHESTRATOR_URL=http://localhost:8200
BRAIN_FACTORY_URL=http://localhost:8301
DEPLOYMENT_PIPELINE_URL=http://localhost:8303
UI_TO_BRAIN_MAPPER_URL=http://localhost:8302

# Monitoring Configuration
ENABLE_METRICS=true
TEST_RESULTS_RETENTION_DAYS=30

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/agentic_brain

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### Test Scenario Configuration
```json
{
  "ui_workflow_builder": {
    "timeout": 120,
    "retry_attempts": 2,
    "cleanup_on_failure": true,
    "screenshots_on_failure": true
  },
  "end_to_end_pipeline": {
    "timeout": 300,
    "parallel_execution": false,
    "validate_data_flow": true,
    "performance_monitoring": true
  },
  "performance_load_test": {
    "users": 10,
    "duration": 60,
    "ramp_up": 10,
    "think_time": 2,
    "collect_metrics": true
  }
}
```

## 🖥️ Test Execution Examples

### UI Workflow Builder Test
```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_workflow_creation():
    driver = webdriver.Chrome()
    driver.get("http://localhost:8300")

    # Wait for canvas to load
    wait = WebDriverWait(driver, 10)
    canvas = wait.until(EC.presence_of_element_located((By.ID, "canvas-container")))

    # Verify component palette
    palette = driver.find_element(By.CLASS_NAME, "component-palette")
    assert palette.is_displayed()

    # Test component drag and drop
    component = driver.find_element(By.CLASS_NAME, "component[data-type='llm-processor']")
    # Simulate drag and drop operation

    driver.quit()
```

### API Integration Test
```python
import httpx
import asyncio

async def test_agent_lifecycle():
    async with httpx.AsyncClient() as client:
        # Create agent
        agent_config = {
            "agent_id": "test-agent-001",
            "agent_name": "Test Agent",
            "domain": "testing",
            "persona": {"name": "Test Agent"},
            "reasoning_pattern": "ReAct",
            "memory_config": {"working_memory_size": 100},
            "plugin_config": {},
            "service_connectors": {}
        }

        response = await client.post(
            "http://localhost:8301/generate-agent",
            json={"agent_config": agent_config}
        )
        assert response.status_code == 200

        # Register agent
        reg_response = await client.post(
            "http://localhost:8200/orchestrator/register-agent",
            json={
                "agent_id": "test-agent-001",
                "agent_name": "Test Agent",
                "domain": "testing",
                "deployment_id": "test-deployment-001"
            }
        )
        assert reg_response.status_code == 200

        # Execute task
        task_response = await client.post(
            "http://localhost:8200/orchestrator/execute-task",
            json={
                "agent_id": "test-agent-001",
                "task_type": "test",
                "task_data": {"message": "Hello from E2E test"},
                "priority": 1
            }
        )
        assert task_response.status_code == 200
```

### Load Testing Example
```python
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor

async def simulate_user_session(user_id: int):
    """Simulate a single user session"""
    async with httpx.AsyncClient() as client:
        # Create agent
        agent_config = {
            "agent_id": f"user-{user_id}-agent",
            "agent_name": f"User {user_id} Agent",
            "domain": "testing"
        }

        response = await client.post(
            "http://localhost:8301/generate-agent",
            json={"agent_config": agent_config}
        )

        # Execute tasks
        for i in range(5):
            await client.post(
                "http://localhost:8200/orchestrator/execute-task",
                json={
                    "agent_id": agent_config["agent_id"],
                    "task_type": "test",
                    "task_data": {"iteration": i}
                }
            )

def run_load_test():
    """Execute load test with multiple users"""
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for user_id in range(10):
            future = executor.submit(asyncio.run, simulate_user_session(user_id))
            futures.append(future)

        # Wait for all users to complete
        for future in futures:
            future.result()
```

## 📊 Test Results and Analytics

### Test Result Structure
```json
{
  "test_id": "550e8400-e29b-41d4-a716-446655440000",
  "scenario": "end_to_end_pipeline",
  "status": "passed",
  "start_time": "2024-01-15T10:30:00Z",
  "end_time": "2024-01-15T10:32:15Z",
  "duration_seconds": 135.5,
  "steps_completed": [
    {
      "step": "agent_creation",
      "result": {"success": true, "agent_id": "test-agent-001"},
      "timestamp": "2024-01-15T10:30:15Z"
    },
    {
      "step": "agent_registration",
      "result": {"success": true, "status": "registered"},
      "timestamp": "2024-01-15T10:31:00Z"
    },
    {
      "step": "task_execution",
      "result": {"success": true, "task_id": "task-123"},
      "timestamp": "2024-01-15T10:32:00Z"
    }
  ],
  "errors": [],
  "metrics": {
    "agent_creation_time": 15.2,
    "task_execution_time": 45.8,
    "memory_usage_mb": 128.5
  },
  "performance_data": {
    "cpu_utilization": 65.2,
    "memory_utilization": 78.9,
    "network_io": 1024.5
  }
}
```

### Performance Metrics
```json
{
  "load_test_results": {
    "total_requests": 500,
    "successful_requests": 485,
    "failed_requests": 15,
    "error_rate": 3.0,
    "average_response_time": 245.8,
    "min_response_time": 89.2,
    "max_response_time": 1250.5,
    "percentile_95_response_time": 567.3,
    "throughput_rps": 8.3,
    "memory_peak_mb": 1024.7,
    "cpu_average_percent": 72.4
  }
}
```

## 🎨 Interactive Test Dashboard

### Dashboard Features
- **Real-time Test Monitoring**: Live status updates of running tests
- **Test Scenario Management**: Easy selection and execution of test scenarios
- **Result Visualization**: Charts and graphs for test results and performance metrics
- **Historical Analysis**: Trend analysis of test success rates over time
- **Failure Analysis**: Detailed breakdown of test failures with diagnostics
- **Performance Trends**: Visualization of performance metrics across test runs

### Dashboard Components
```html
<!-- Test Statistics Cards -->
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value" id="total-tests">0</div>
        <div class="stat-label">Tests Executed</div>
    </div>
    <div class="stat-card">
        <div class="stat-value" id="success-rate">0%</div>
        <div class="stat-label">Success Rate</div>
    </div>
    <div class="stat-card">
        <div class="stat-value" id="avg-duration">0s</div>
        <div class="stat-label">Avg Test Duration</div>
    </div>
</div>

<!-- Test Scenario Cards -->
<div class="test-scenarios">
    <div class="scenario-card" onclick="runTest('ui_workflow_builder')">
        <h4>UI Workflow Builder</h4>
        <p>Test drag-and-drop workflow creation</p>
        <button class="run-btn">Run Test</button>
    </div>
    <div class="scenario-card" onclick="runTest('end_to_end_pipeline')">
        <h4>End-to-End Pipeline</h4>
        <p>Test complete agent lifecycle</p>
        <button class="run-btn">Run Test</button>
    </div>
</div>

<!-- Test Results Table -->
<div class="results-table">
    <table>
        <thead>
            <tr>
                <th>Test Scenario</th>
                <th>Status</th>
                <th>Duration</th>
                <th>Errors</th>
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
name: E2E Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  e2e-tests:
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
        pip install -r services/end-to-end-testing-service/requirements.txt

    - name: Start services
      run: docker-compose up -d

    - name: Wait for services
      run: |
        timeout 300 bash -c 'until curl -f http://localhost:8200/health; do sleep 5; done'

    - name: Run E2E tests
      run: |
        python -m pytest services/end-to-end-testing-service/tests/ -v

    - name: Generate test report
      run: |
        python services/end-to-end-testing-service/generate_report.py
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

        stage('E2E Tests') {
            steps {
                sh '''
                    python -m pytest services/end-to-end-testing-service/tests/ \
                        --junitxml=test-results.xml \
                        --html=test-report.html
                '''
            }
            post {
                always {
                    junit 'test-results.xml'
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: '.',
                        reportFiles: 'test-report.html',
                        reportName: 'E2E Test Report'
                    ])
                }
            }
        }

        stage('Performance Tests') {
            steps {
                sh '''
                    python services/end-to-end-testing-service/load_test.py \
                        --users 20 \
                        --duration 120 \
                        --output performance-results.json
                '''
            }
        }
    }

    post {
        always {
            sh 'docker-compose down'
            archiveArtifacts artifacts: 'performance-results.json', allowEmptyArchive: true
        }
    }
}
```

## 📚 API Integration Examples

### RESTful API Usage
```python
import requests

# Execute individual test scenario
response = requests.post('http://localhost:8380/tests/execute', json={
    "scenario": "end_to_end_pipeline",
    "parameters": {
        "cleanup_after_test": True,
        "performance_monitoring": True
    }
})

test_result = response.json()
print(f"Test Status: {test_result['status']}")
print(f"Duration: {test_result['duration_seconds']}s")

# Execute comprehensive test suite
suite_response = requests.post('http://localhost:8380/tests/e2e-suite', json={
    "scenarios": ["ui_workflow_builder", "end_to_end_pipeline", "performance_load_test"],
    "parallel_execution": True
})

suite_result = suite_response.json()
print(f"Suite Status: {suite_result['overall_status']}")
print(f"Success Rate: {suite_result['summary']['success_rate']}%")
```

### Python SDK
```python
from agentic_brain_e2e_testing import E2ETestingClient

client = E2ETestingClient(
    base_url="http://localhost:8380",
    timeout=300
)

# Run individual test
result = await client.execute_test(
    scenario="end_to_end_pipeline",
    parameters={"validate_data_flow": True}
)

# Run test suite
suite_result = await client.execute_test_suite(
    scenarios=["ui_workflow_builder", "end_to_end_pipeline"],
    parallel=True
)

# Get test results
results = await client.get_test_results(limit=50)
```

### Monitoring Integration
```python
# Integration with monitoring service
monitoring_client = httpx.AsyncClient()

async def monitor_test_execution(test_id: str):
    """Monitor test execution in real-time"""
    while True:
        response = await monitoring_client.get(f"/tests/results/{test_id}")
        status = response.json()

        if status["status"] in ["passed", "failed"]:
            break

        await asyncio.sleep(5)  # Check every 5 seconds

# Usage
asyncio.create_task(monitor_test_execution("test-123"))
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: [docs.agenticbrain.com/e2e-testing](https://docs.agenticbrain.com/e2e-testing)
- **Community Forum**: [community.agenticbrain.com/e2e-testing](https://community.agenticbrain.com/e2e-testing)
- **Issue Tracker**: [github.com/agentic-brain/e2e-testing-service/issues](https://github.com/agentic-brain/e2e-testing-service/issues)
- **Email Support**: e2e-testing-support@agenticbrain.com

### Service Level Agreements
- **Test Execution**: < 5 minutes for standard test scenarios
- **Load Test Setup**: < 2 minutes for load test initialization
- **Result Generation**: < 30 seconds for test result processing
- **Dashboard Load Time**: < 3 seconds for dashboard rendering
- **API Response Time**: < 500ms for test status queries
- **Uptime**: 99.5% service availability
- **Data Retention**: 30 days of test results and metrics

---

**Built with ❤️ for the Agentic Brain Platform**

*Enterprise-grade end-to-end testing for AI automation platforms*
