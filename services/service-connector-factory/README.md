# Service Connector Factory Service

## Overview

The Service Connector Factory Service is a comprehensive factory pattern implementation for creating and managing connections to various data ingestion and output services in the Agentic AI platform. It provides unified interfaces for connecting to CSV, API, PDF, PostgreSQL, Qdrant, Elasticsearch, MinIO, and RabbitMQ services with built-in connection pooling, retry logic, and health monitoring.

## Key Responsibilities

- **Service Discovery**: Automatic discovery and registration of data services
- **Connection Management**: Connection pooling and lifecycle management
- **Authentication Handling**: Support for various authentication methods
- **Data Operations**: Unified interface for data ingestion and output operations
- **Health Monitoring**: Continuous monitoring of service connectivity and performance
- **Error Recovery**: Automatic retry and failover mechanisms
- **Performance Metrics**: Comprehensive monitoring and analytics

## Architecture

### Factory Pattern Implementation
```python
# Create service connector
config = ServiceConnection(
    service_type="postgresql",
    service_name="prod_db",
    host="localhost",
    port=5432,
    database="analytics"
)
connector = await factory.create_connector(config)

# Execute operation
result = await factory.execute_operation("prod_db", operation_request)
```

### Service Connector Hierarchy
```
ServiceConnector (Abstract Base Class)
├── CSVIngestionConnector
├── PostgreSQLOutputConnector
├── QdrantVectorConnector
├── ElasticsearchConnector
├── MinIOStorageConnector
└── RabbitMQConnector
```

### Connection Pooling Architecture
```python
# Connection pool management
- Automatic connection creation and reuse
- Connection health monitoring
- Pool size limits and timeout handling
- Graceful connection cleanup
```

## Supported Services

### Ingestion Services
- **CSV Ingestion**: File upload, format validation, data preview
- **API Ingestion**: REST API data fetching, authentication handling
- **PDF Ingestion**: Document processing, OCR support, text extraction
- **MinIO Storage**: Object storage integration, file management

### Output Services
- **PostgreSQL**: Data insertion, querying, transaction management
- **Qdrant Vector**: Vector similarity search, data indexing
- **Elasticsearch**: Full-text search, analytics, aggregation
- **RabbitMQ**: Message queuing, pub/sub patterns

### Processing Services
- **Redis**: Caching, session management, data structures
- **RabbitMQ**: Message processing, job queues, event streaming

## API Endpoints

### Core Endpoints

#### `POST /connectors`
Create a new service connector with complete configuration.

**Request Body:**
```json
{
  "service_type": "postgresql",
  "service_name": "analytics_db",
  "host": "localhost",
  "port": 5432,
  "database": "analytics",
  "username": "analyst",
  "password": "secure_password",
  "connection_pool_size": 5,
  "connection_timeout": 30,
  "retry_attempts": 3,
  "ssl_enabled": true,
  "custom_config": {
    "schema": "public",
    "auto_commit": true
  }
}
```

**Response:**
```json
{
  "service_name": "analytics_db",
  "status": "created",
  "connector_info": {
    "service_type": "postgresql",
    "host": "localhost",
    "port": 5432,
    "connection_status": "connected"
  }
}
```

#### `POST /connectors/{service_name}/execute`
Execute data operations on connected services.

**Supported Operations:**
- `ingest`: Data ingestion from various sources
- `query`: Data retrieval and querying
- `insert`: Data insertion into databases
- `update`: Data modification operations
- `delete`: Data removal operations
- `validate`: Data format validation
- `preview`: Data preview and sampling

**Request Body:**
```json
{
  "connection_id": "analytics_db",
  "operation": "insert",
  "data_format": "json",
  "data": {
    "table_name": "user_analytics",
    "records": [
      {"user_id": 123, "action": "login", "timestamp": "2024-01-01T10:00:00Z"}
    ]
  },
  "parameters": {
    "batch_size": 100,
    "on_conflict": "update"
  }
}
```

**Response:**
```json
{
  "operation_id": "op_abc123",
  "status": "success",
  "records_processed": 1,
  "records_affected": 1,
  "execution_time": 0.245,
  "result_data": {
    "inserted_rows": 1,
    "updated_rows": 0
  }
}
```

#### `POST /connectors/{service_name}/test`
Test service connectivity and configuration.

**Response:**
```json
{
  "service_name": "analytics_db",
  "status": "success",
  "response_time": 0.123,
  "test_results": {
    "connection_test": "passed",
    "authentication_test": "passed",
    "permissions_test": "passed"
  }
}
```

### Management Endpoints

#### `GET /connectors`
List all active service connectors.

#### `DELETE /connectors/{service_name}`
Remove and cleanup service connector.

#### `GET /connectors/{service_name}/metrics`
Get performance metrics for specific connector.

#### `GET /metrics`
Get aggregated metrics for all connectors.

## Configuration

### Environment Variables

```bash
# Service Configuration
SERVICE_CONNECTOR_FACTORY_PORT=8306
SERVICE_HOST=localhost

# Connection Pool Settings
MAX_CONNECTIONS_PER_SERVICE=10
CONNECTION_TIMEOUT=30
CONNECTION_RETRY_ATTEMPTS=3
CONNECTION_RETRY_DELAY=1.0

# Health Monitoring
HEALTH_CHECK_INTERVAL=30
SERVICE_TIMEOUT_THRESHOLD=60

# Service Endpoints
CSV_INGESTION_PORT=8001
API_INGESTION_PORT=8002
PDF_INGESTION_PORT=8003
POSTGRES_OUTPUT_PORT=8004
QDRANT_VECTOR_PORT=6333
ELASTICSEARCH_PORT=9200
MINIO_PORT=9000
RABBITMQ_PORT=5672
```

### Service-Specific Configuration

#### PostgreSQL Configuration
```json
{
  "connection_string": "postgresql://user:pass@localhost:5432/db",
  "database": "analytics",
  "schema": "public",
  "ssl_mode": "require",
  "connection_pool_size": 5,
  "max_connections": 20
}
```

#### Qdrant Configuration
```json
{
  "collection_name": "vectors",
  "vector_dimension": 384,
  "index_type": "HNSW",
  "metric": "cosine",
  "replication_factor": 1
}
```

#### MinIO Configuration
```json
{
  "endpoint": "localhost:9000",
  "access_key": "minio_access_key",
  "secret_key": "minio_secret_key",
  "secure": false,
  "bucket_name": "data-lake"
}
```

## Connection Management

### Connection Pooling
```python
# Automatic connection management
- Connection creation and reuse
- Pool size monitoring and limits
- Connection health validation
- Automatic cleanup and renewal
- Load balancing across pool
```

### Retry and Failover
```python
# Intelligent retry logic
- Exponential backoff
- Circuit breaker pattern
- Service-specific retry strategies
- Automatic failover to backup services
- Graceful degradation
```

### Authentication Methods
```python
# Supported authentication
- Username/Password
- API Key/Token
- Certificate-based
- OAuth2/JWT
- AWS IAM roles
- Service account keys
```

## Health Monitoring

### Service Health Checks
```python
# Continuous monitoring
- Connection status validation
- Response time measurement
- Error rate tracking
- Resource utilization monitoring
- Automatic alert generation
```

### Performance Metrics
```python
metrics = {
    "service_name": "analytics_db",
    "total_connections": 150,
    "active_connections": 8,
    "total_operations": 1250,
    "successful_operations": 1235,
    "failed_operations": 15,
    "average_response_time": 0.245,
    "uptime_percentage": 99.7,
    "error_count": 15
}
```

## Error Handling

### Comprehensive Error Management
```python
# Error handling strategies
- Connection timeout handling
- Authentication failure recovery
- Network interruption recovery
- Service unavailability handling
- Data format validation errors
- Resource exhaustion handling
```

### Error Classification
```python
# Error types and handling
- Transient errors (automatic retry)
- Permanent errors (user notification)
- Configuration errors (validation feedback)
- Resource errors (load shedding)
- Security errors (access denial)
```

## Data Operations

### Ingestion Operations
```python
# CSV Ingestion
await connector.execute_operation({
    "operation": "ingest",
    "data": {"file_data": csv_content},
    "parameters": {
        "delimiter": ",",
        "has_headers": true,
        "encoding": "utf-8"
    }
})

# API Data Fetching
await connector.execute_operation({
    "operation": "ingest",
    "data": {"api_endpoint": "https://api.example.com/data"},
    "parameters": {
        "method": "GET",
        "headers": {"Authorization": "Bearer token"},
        "pagination": true
    }
})
```

### Output Operations
```python
# Database Insertion
await connector.execute_operation({
    "operation": "insert",
    "data": {
        "table_name": "user_events",
        "records": user_event_data
    },
    "parameters": {
        "batch_size": 100,
        "on_conflict": "update"
    }
})

# Vector Storage
await connector.execute_operation({
    "operation": "insert",
    "data": {
        "vectors": embedding_vectors,
        "metadata": vector_metadata
    },
    "parameters": {
        "collection": "embeddings",
        "batch_size": 50
    }
})
```

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run service
uvicorn main:app --host 0.0.0.0 --port 8306 --reload
```

### Docker Development
```bash
# Build image
docker build -t service-connector-factory .

# Run container
docker run -p 8306:8306 service-connector-factory
```

## Testing

### Unit Tests
```python
# Test connector creation
def test_create_postgresql_connector():
    config = ServiceConnection(
        service_type="postgresql",
        service_name="test_db",
        host="localhost",
        port=5432
    )
    connector = PostgreSQLOutputConnector(config)
    assert connector.service_name == "test_db"

# Test connection testing
@pytest.mark.asyncio
async def test_connection_test():
    connector = PostgreSQLOutputConnector(config)
    result = await connector.test_connection()
    assert result.status in ["success", "failed"]
```

### Integration Tests
```python
# Test full data pipeline
@pytest.mark.asyncio
async def test_data_pipeline():
    # Create connector
    connector = await factory.create_connector(config)

    # Test connection
    test_result = await connector.test_connection()
    assert test_result.status == "success"

    # Execute operation
    result = await connector.execute_operation(operation_request)
    assert result.status == "success"
```

### Performance Tests
```python
# Test concurrent operations
@pytest.mark.asyncio
async def test_concurrent_operations():
    operations = [connector.execute_operation(op) for op in operation_requests]
    results = await asyncio.gather(*operations)
    assert all(result.status == "success" for result in results)

# Test connection pooling
@pytest.mark.asyncio
async def test_connection_pooling():
    # Create multiple concurrent connections
    connections = await asyncio.gather(*[
        factory.create_connector(config) for _ in range(5)
    ])
    assert len(connections) == 5
```

## Usage Examples

### Basic Service Connection
```python
from main import ServiceConnectorFactory, ServiceConnection

# Initialize factory
factory = ServiceConnectorFactory()

# Create PostgreSQL connection
pg_config = ServiceConnection(
    service_type="postgresql",
    service_name="analytics_db",
    host="localhost",
    port=5432,
    database="analytics",
    username="analyst",
    password="secure_password"
)

connector = await factory.create_connector(pg_config)
print(f"Connected to {connector.service_name}")
```

### Data Ingestion Pipeline
```python
# CSV data ingestion
csv_operation = {
    "connection_id": "data_ingest",
    "operation": "ingest",
    "data": {"file_data": csv_content},
    "parameters": {"has_headers": True, "delimiter": ","}
}

result = await factory.execute_operation("data_ingest", csv_operation)
print(f"Processed {result.records_processed} records")
```

### Query and Analytics
```python
# Database querying
query_operation = {
    "connection_id": "analytics_db",
    "operation": "query",
    "data": {},
    "parameters": {
        "query": "SELECT * FROM user_events WHERE date >= $1",
        "query_params": ["2024-01-01"]
    }
}

result = await factory.execute_operation("analytics_db", query_operation)
print(f"Retrieved {len(result.result_data)} records")
```

## Security Considerations

- **Authentication Security**: Secure credential handling and storage
- **Connection Encryption**: SSL/TLS encryption for data in transit
- **Access Control**: Service-level and operation-level permissions
- **Audit Logging**: Comprehensive logging of all operations
- **Data Validation**: Input sanitization and format validation
- **Resource Protection**: Rate limiting and resource usage controls

## Performance Optimization

### Connection Optimization
- **Connection Pooling**: Reuse connections to reduce overhead
- **Lazy Loading**: Create connections only when needed
- **Connection Monitoring**: Track and optimize connection usage
- **Automatic Cleanup**: Remove idle and failed connections

### Query Optimization
- **Batch Operations**: Process multiple operations together
- **Query Caching**: Cache frequently executed queries
- **Result Streaming**: Stream large result sets
- **Parallel Processing**: Execute independent operations concurrently

## Monitoring & Observability

### Metrics Collection
```python
# Service-level metrics
- Total active connections
- Operations per second
- Average response time
- Error rate percentage
- Connection pool utilization
- Memory usage statistics
```

### Health Monitoring
```python
# Continuous health checks
- Service availability
- Connection status
- Performance degradation
- Error rate monitoring
- Resource utilization
- Automatic alerting
```

## Future Enhancements

- **Additional Service Types**: Support for more data services and APIs
- **Advanced Authentication**: Multi-factor authentication and SSO integration
- **Data Transformation**: Built-in ETL and data transformation capabilities
- **Real-time Streaming**: Support for streaming data operations
- **Machine Learning Integration**: ML model serving and inference
- **Multi-Region Support**: Cross-region and multi-cloud deployments
- **Advanced Monitoring**: AI-powered anomaly detection and predictive analytics

## API Documentation

Complete API documentation is available at `/docs` when the service is running.

## Contributing

1. Follow the established connector interface patterns
2. Add comprehensive error handling for all operations
3. Include performance monitoring for new connectors
4. Write thorough unit and integration tests
5. Update documentation for configuration changes
6. Ensure backward compatibility with existing connectors

## License and Support

This service is part of the AgenticAI platform and follows the same licensing and support policies.
