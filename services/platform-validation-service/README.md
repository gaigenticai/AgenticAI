# Platform Validation Service

A comprehensive microservice for validating the health, configuration, and production readiness of the Agentic Brain platform.

## Overview

The Platform Validation Service provides automated validation and assessment of the entire Agentic Brain platform, ensuring all components are functioning correctly and the system is production-ready.

## Features

- **Service Health Validation**: Monitors health of all platform microservices
- **Configuration Validation**: Ensures proper environment variables and settings
- **Database Integrity Checks**: Validates database schema and data consistency
- **API Endpoint Testing**: Tests critical API endpoints across services
- **Production Readiness Assessment**: Comprehensive production deployment validation
- **Automated Reporting**: Generates detailed validation reports with recommendations
- **Web Dashboard**: Interactive dashboard for real-time platform monitoring
- **Prometheus Metrics**: Integration with monitoring and alerting systems

## API Endpoints

### Health Check
```
GET /health
```
Returns the service health status.

### Root
```
GET /
```
Returns service information and capabilities.

### Run Validation
```
POST /api/validation/run
Content-Type: application/json

{
  "validation_type": "comprehensive",
  "include_detailed_results": true,
  "store_results": true
}
```
Runs comprehensive platform validation.

### Validation History
```
GET /api/validation/history?limit=10
```
Returns recent validation results.

### Validation Status
```
GET /api/validation/status
```
Returns current platform validation status.

### Dashboard
```
GET /dashboard
```
Interactive web dashboard for platform monitoring.

### Metrics
```
GET /metrics
```
Prometheus metrics endpoint.

## Validation Types

### Comprehensive Validation
Runs all validation components:
- Service Health
- Configuration
- Database Integrity
- API Endpoints
- Production Readiness

### Individual Validations
- `service_health`: Service availability and response times
- `configuration`: Environment variables and settings
- `database`: Schema and data integrity
- `api_endpoints`: API functionality and performance
- `production_readiness`: Deployment readiness assessment

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PLATFORM_VALIDATION_PORT` | Service port | `8430` |
| `HOST` | Service host | `0.0.0.0` |
| `DATABASE_URL` | PostgreSQL connection URL | Required |
| `REDIS_HOST` | Redis host | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `REDIS_DB` | Redis database | `0` |
| `VALIDATION_TIMEOUT` | Validation timeout (seconds) | `300` |
| `HEALTH_CHECK_INTERVAL` | Health check interval (seconds) | `60` |

## Usage

### Running Validation

```python
import httpx

async def run_validation():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8430/api/validation/run",
            json={"validation_type": "comprehensive"}
        )
        result = response.json()
        print(f"Overall Score: {result['overall_score']}")
        print(f"Status: {result['status']}")
```

### Docker Usage

```bash
# Build the service
docker build -t platform-validation-service .

# Run the service
docker run -p 8430:8430 \
  -e DATABASE_URL="postgresql://user:password@db:5432/agentic_brain" \
  -e REDIS_HOST="redis" \
  platform-validation-service
```

### Docker Compose Integration

```yaml
platform-validation-service:
  build: ./services/platform-validation-service
  ports:
    - "8430:8430"
  environment:
    - DATABASE_URL=postgresql://user:password@postgres:5432/agentic_brain
    - REDIS_HOST=redis
  depends_on:
    - postgres
    - redis
  networks:
    - agentic-network
  volumes:
    - platform_validation_data:/app/data
```

## Validation Results

### Result Structure

```json
{
  "validation_id": "uuid",
  "status": "passed|failed|warning",
  "overall_score": 85.5,
  "validation_components": {
    "service_health": {
      "total_services": 21,
      "healthy_services": 20,
      "health_percentage": 95.2,
      "overall_status": "passed"
    },
    "configuration": {
      "database_status": "connected",
      "redis_status": "connected",
      "overall_status": "passed"
    }
  },
  "critical_issues": [],
  "recommendations": [
    "Review warning-level issues for potential improvements"
  ],
  "next_steps": [
    "Platform is production-ready",
    "Set up automated validation in CI/CD pipeline"
  ]
}
```

### Status Types

- **passed**: All validations successful
- **warning**: Some non-critical issues found
- **failed**: Critical issues require attention

## Database Schema

The service creates the following tables:

- `validation_results`: Stores validation execution results
- `service_health`: Tracks service health status
- `platform_metrics`: Platform-wide performance metrics
- `compliance_checks`: Compliance validation results

## Monitoring

### Health Checks
- Service responds to `/health` endpoint
- Database connectivity verified
- Redis connectivity verified
- Validation engine operational

### Metrics
- Request count and latency
- Validation execution time
- Service health status
- Error rates

### Logging
- Structured logging with JSON format
- Request/response logging
- Error tracking with context
- Validation result logging

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/
```

### Code Quality

```bash
# Run linting
flake8 .

# Run type checking
mypy .

# Run security checks
bandit .
```

## Security Considerations

- Service-to-service authentication required
- Environment variable validation
- Secure database connections
- Input validation and sanitization
- Rate limiting for API endpoints
- Audit logging for all validations

## Contributing

1. Follow the established coding standards
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure all validations pass before submitting
5. Add appropriate logging and error handling

## License

This service is part of the Agentic Brain platform and follows the same licensing terms.
