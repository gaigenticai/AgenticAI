# UI Testing Service

A comprehensive automated UI testing service for the Agentic Brain Platform's Agent Builder interface. This service provides enterprise-grade testing capabilities including canvas interactions, workflow validation, deployment simulation, and performance monitoring.

## 🎯 Features

### Core Testing Capabilities
- **Canvas Interaction Tests**: Drag-and-drop, component placement, connection management
- **Workflow Validation**: Data flow validation, component compatibility, error detection
- **Deployment Simulation**: Agent validation, deployment pipeline testing, rollback scenarios
- **Performance Testing**: Load testing, response time measurement, resource utilization
- **Cross-Browser Testing**: Chrome, Firefox, Safari, and WebKit compatibility
- **Visual Regression Testing**: Screenshot comparison and UI consistency validation

### Advanced Features
- **Real-time Monitoring**: Live test execution tracking and progress reporting
- **Automated Screenshots**: Failure capture and visual debugging
- **Video Recording**: Test session recording for detailed analysis
- **Performance Metrics**: Response times, memory usage, error rates
- **Accessibility Testing**: WCAG compliance and usability validation
- **Mobile Testing**: Responsive design and touch interaction testing

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Agent Builder UI running (port 8300)
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
docker-compose up -d ui-testing-service

# Check service health
curl http://localhost:8310/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# Run the service
python main.py
```

## 📡 API Endpoints

### Test Execution
```http
POST /api/tests/execute
Content-Type: application/json

{
  "test_type": "canvas",
  "browser": "chrome",
  "headless": true,
  "timeout_seconds": 30,
  "take_screenshots": true,
  "record_video": false,
  "custom_parameters": {
    "workflow_config": {...},
    "agent_config": {...}
  }
}
```

### Test Session Management
```http
# Get test session results
GET /api/tests/sessions/{session_id}

# List all test sessions
GET /api/tests/sessions?limit=50&offset=0

# Get testing metrics
GET /api/tests/metrics
```

### Health and Monitoring
```http
# Service health check
GET /health

# Prometheus metrics
GET /metrics
```

## 🧪 Test Types

### 1. Canvas Interaction Tests
Tests for drag-and-drop functionality, component placement, and canvas manipulation.

```json
{
  "test_type": "canvas",
  "browser": "chrome",
  "custom_parameters": {
    "component_to_drag": "data_input_csv",
    "drop_position": {"x": 100, "y": 100},
    "expected_component_count": 1
  }
}
```

### 2. Workflow Validation Tests
Validates workflow logic, data flow, and component compatibility.

```json
{
  "test_type": "workflow",
  "browser": "firefox",
  "custom_parameters": {
    "workflow_config": {
      "components": [...],
      "connections": [...],
      "validation_rules": {...}
    }
  }
}
```

### 3. Deployment Simulation Tests
Tests agent validation, deployment pipeline, and production readiness.

```json
{
  "test_type": "deployment",
  "browser": "chrome",
  "custom_parameters": {
    "agent_config": {
      "name": "Test Agent",
      "components": [...],
      "deployment_strategy": "canary"
    }
  }
}
```

### 4. Performance Tests
Load testing, response time measurement, and resource utilization analysis.

```json
{
  "test_type": "performance",
  "browser": "chrome",
  "custom_parameters": {
    "test_duration_minutes": 5,
    "concurrent_users": 10,
    "actions_per_minute": 60,
    "measure_memory_usage": true,
    "measure_response_times": true
  }
}
```

## ⚙️ Configuration

### Environment Variables
```bash
# Service Configuration
UI_TESTING_PORT=8310
AGENT_BUILDER_UI_HOST=localhost
AGENT_BUILDER_UI_PORT=8300

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/agentic_brain

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Testing Configuration
DEFAULT_BROWSER=chrome
HEADLESS_MODE=true
TEST_TIMEOUT=30
SCREENSHOT_ON_FAILURE=true

# Performance Thresholds
PAGE_LOAD_THRESHOLD_MS=3000
INTERACTION_RESPONSE_THRESHOLD_MS=1000
MEMORY_USAGE_THRESHOLD_MB=500

# Security
REQUIRE_AUTH=false
JWT_SECRET=your-secret-key
```

### Test Configuration Files
```json
// test_scenarios.json
{
  "canvas_drag_drop": {
    "name": "Canvas Drag and Drop Test",
    "description": "Test component drag and drop functionality",
    "test_steps": [
      {"action": "drag_component", "component": "data_input", "position": {"x": 100, "y": 100}},
      {"action": "verify_component", "component_id": "data_input_1"},
      {"action": "take_screenshot", "filename": "drag_drop_result.png"}
    ]
  }
}
```

## 📊 Test Reporting

### Real-time Dashboard
Access the test dashboard at `http://localhost:8310/dashboard` for:
- Live test execution monitoring
- Test results visualization
- Performance metrics dashboard
- Historical test trends
- Failure analysis reports

### Test Reports
```bash
# Generate HTML test report
curl -X GET "http://localhost:8310/api/tests/sessions/{session_id}/report" \
     -H "Accept: text/html" > test_report.html

# Export test results as JSON
curl -X GET "http://localhost:8310/api/tests/sessions/{session_id}/export" \
     -H "Accept: application/json" > test_results.json
```

### Metrics and Analytics
```json
{
  "period": "last_7_days",
  "summary": {
    "total_sessions": 45,
    "total_tests": 892,
    "passed_tests": 876,
    "failed_tests": 16,
    "success_rate": 98.2
  },
  "test_types": {
    "canvas": 234,
    "workflow": 189,
    "deployment": 145,
    "performance": 98
  }
}
```

## 🔧 Advanced Usage

### Custom Test Scenarios
```python
from ui_testing_service import UITestManager

# Create custom test scenario
test_manager = UITestManager()

custom_scenario = {
    "name": "Complex Workflow Test",
    "steps": [
        {"action": "navigate", "url": "/agent-builder"},
        {"action": "drag_component", "component": "llm_processor", "position": {"x": 200, "y": 150}},
        {"action": "connect_components", "source": "data_input_1", "target": "llm_processor_1"},
        {"action": "validate_workflow"},
        {"action": "simulate_deployment"},
        {"action": "measure_performance"}
    ]
}

# Execute custom scenario
result = await test_manager.execute_scenario(custom_scenario)
```

### Integration with CI/CD
```yaml
# .github/workflows/ui-tests.yml
name: UI Tests
on: [push, pull_request]

jobs:
  ui-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r services/ui-testing-service/requirements.txt
          playwright install
      - name: Run UI Tests
        run: |
          python -m pytest services/ui-testing-service/tests/ \
            --html=test-results.html --self-contained-html
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: test-results.html
```

### Parallel Test Execution
```python
import asyncio
from ui_testing_service import UITestManager

async def run_parallel_tests():
    test_manager = UITestManager()

    # Define test configurations
    test_configs = [
        {"test_type": "canvas", "browser": "chrome"},
        {"test_type": "workflow", "browser": "firefox"},
        {"test_type": "deployment", "browser": "webkit"}
    ]

    # Execute tests in parallel
    tasks = [
        test_manager.execute_test(config)
        for config in test_configs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

## 🐛 Troubleshooting

### Common Issues

#### Browser Launch Failures
```bash
# Install missing browser dependencies
playwright install-deps

# Check browser installation
playwright install chromium

# Run with verbose logging
DEBUG=1 python main.py
```

#### Connection Timeouts
```bash
# Increase timeout settings
export TEST_TIMEOUT=60
export PAGE_LOAD_THRESHOLD_MS=5000

# Check network connectivity
curl -I http://localhost:8300/health
```

#### Screenshot/Video Issues
```bash
# Ensure directories exist and are writable
mkdir -p screenshots test_videos
chmod 755 screenshots test_videos

# Check disk space
df -h
```

#### Memory Issues
```bash
# Increase memory limits
export MEMORY_USAGE_THRESHOLD_MB=1024

# Monitor memory usage
docker stats ui-testing-service

# Restart with increased memory
docker-compose up -d --scale ui-testing-service=1
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with visible browser
export HEADLESS_MODE=false

# Capture all screenshots
export SCREENSHOT_ON_FAILURE=true
```

## 📈 Performance Optimization

### Browser Configuration
```json
{
  "browser_options": {
    "chromium": {
      "args": [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-web-security",
        "--disable-features=VizDisplayCompositor"
      ]
    }
  }
}
```

### Test Parallelization
```python
# Optimize test execution
test_config = {
    "parallel_execution": true,
    "max_concurrent_tests": 5,
    "browser_pool_size": 3,
    "test_timeout": 30,
    "retry_failed_tests": true,
    "retry_count": 2
}
```

### Resource Management
```python
# Memory optimization
memory_config = {
    "max_memory_per_test": "256MB",
    "cleanup_after_test": true,
    "cache_browser_context": true,
    "reuse_browser_instances": true
}
```

## 🔐 Security Considerations

### Test Data Protection
- Never use production data in automated tests
- Sanitize sensitive information in test reports
- Use encrypted connections for remote testing
- Implement proper access controls for test results

### Browser Security
```python
# Secure browser configuration
secure_browser_config = {
    "disable_web_security": false,
    "ignore_https_errors": false,
    "accept_downloads": false,
    "bypass_csp": false,
    "permissions": ["none"]
}
```

### Authentication & Authorization
```python
# Test user management
test_auth_config = {
    "test_users": {
        "admin": {"role": "administrator", "permissions": ["all"]},
        "user": {"role": "user", "permissions": ["read", "write"]},
        "viewer": {"role": "viewer", "permissions": ["read"]}
    },
    "session_management": {
        "session_timeout": 3600,
        "max_sessions_per_user": 5,
        "force_logout_on_test_end": true
    }
}
```

## 📚 API Documentation

### Complete API Reference
- [Test Execution API](./docs/api/test-execution.md)
- [Session Management API](./docs/api/session-management.md)
- [Metrics & Analytics API](./docs/api/metrics-analytics.md)
- [Configuration API](./docs/api/configuration.md)

### SDKs and Libraries
- **Python SDK**: `pip install agentic-brain-ui-testing`
- **JavaScript SDK**: `npm install @agentic-brain/ui-testing`
- **REST API Client**: Comprehensive HTTP client libraries

## 🤝 Contributing

### Development Setup
```bash
# Clone the repository
git clone https://github.com/agentic-brain/ui-testing-service.git
cd ui-testing-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Start development server
python main.py
```

### Code Standards
- Follow PEP 8 style guidelines
- Write comprehensive docstrings
- Include type hints for all functions
- Add unit tests for new features
- Update documentation for API changes

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: [docs.agenticbrain.com](https://docs.agenticbrain.com)
- **Community Forum**: [community.agenticbrain.com](https://community.agenticbrain.com)
- **Issue Tracker**: [github.com/agentic-brain/ui-testing-service/issues](https://github.com/agentic-brain/ui-testing-service/issues)
- **Email Support**: support@agenticbrain.com

### Service Level Agreement
- **Response Time**: < 24 hours for critical issues
- **Resolution Time**: < 72 hours for standard issues
- **Uptime**: 99.9% service availability
- **Support Hours**: 24/7 enterprise support available

---

**Built with ❤️ for the Agentic Brain Platform**

*Ensuring UI quality and reliability through comprehensive automated testing*
