# Memory Manager Service

The Memory Manager Service provides comprehensive memory management capabilities for AI agents in the Agentic Brain Platform. It implements a multi-tier memory architecture supporting working memory, episodic memory, semantic memory, and vector memory for similarity-based retrieval.

## Features

- **Multi-Tier Memory Architecture**: Working, episodic, semantic, and vector memory types
- **TTL-Based Expiration**: Automatic cleanup of expired working memories
- **Memory Consolidation**: Intelligent memory optimization and deduplication
- **Vector Similarity Search**: Embedding-based memory retrieval for semantic understanding
- **Memory Performance Monitoring**: Comprehensive analytics and usage statistics
- **Automatic Cleanup**: Background processes for memory maintenance
- **RESTful API**: Complete API for memory operations and management
- **Authentication Integration**: Support for JWT-based authentication
- **Monitoring & Metrics**: Prometheus metrics for observability

## Memory Types

### Working Memory
- **Purpose**: Short-term memory for immediate processing
- **TTL**: Configurable expiration (default: 1 hour)
- **Use Case**: Temporary data during agent execution
- **Cleanup**: Automatic expiration-based removal

### Episodic Memory
- **Purpose**: Event-based memory for experience recall
- **Retention**: Configurable retention period (default: 30 days)
- **Use Case**: Remembering past interactions and decisions
- **Structure**: Time-stamped events with context

### Semantic Memory
- **Purpose**: Long-term knowledge and pattern storage
- **Persistence**: Indefinite retention (configurable)
- **Use Case**: Learned patterns, rules, and knowledge
- **Optimization**: Consolidation and generalization

### Vector Memory
- **Purpose**: Similarity-based memory retrieval
- **Technology**: Vector embeddings for semantic search
- **Dimension**: Configurable vector dimensions (default: 768)
- **Use Case**: Finding related memories and context

## API Endpoints

### Memory Management
- `POST /memories` - Store a memory item
- `GET /memories/{agent_id}/{memory_type}/{content_key}` - Retrieve specific memory
- `POST /memories/search` - Search memories with filtering
- `POST /memories/consolidate` - Consolidate memories using strategies
- `POST /memories/cleanup` - Trigger cleanup of expired memories
- `GET /memories/stats` - Get memory statistics
- `GET /memories/types` - Get available memory types

### Vector Memory
- `POST /vectors` - Store vector memory item
- `POST /vectors/search` - Search similar vector memories

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

### Vector Database Configuration
- `VECTOR_DB_HOST` - Vector database host (default: qdrant_vector)
- `VECTOR_DB_PORT` - Vector database port (default: 6333)

### Service Configuration
- `MEMORY_MANAGER_HOST` - Service host (default: 0.0.0.0)
- `MEMORY_MANAGER_PORT` - Service port (default: 8205)
- `REQUIRE_AUTH` - Enable authentication (default: false)
- `JWT_SECRET` - JWT secret key for authentication

### Memory Configuration
- `WORKING_MEMORY_TTL_SECONDS` - Working memory TTL (default: 3600)
- `EPISODIC_MEMORY_RETENTION_DAYS` - Episodic memory retention (default: 30)
- `SEMANTIC_MEMORY_ENABLED` - Enable semantic memory (default: true)
- `LONG_TERM_MEMORY_ENABLED` - Enable long-term memory (default: true)
- `MEMORY_VECTOR_DIMENSION` - Vector dimension (default: 768)
- `MEMORY_SIMILARITY_THRESHOLD` - Similarity threshold (default: 0.85)

### Performance Configuration
- `MAX_MEMORY_ITEMS_PER_AGENT` - Max items per agent (default: 10000)
- `MEMORY_CLEANUP_INTERVAL_SECONDS` - Cleanup interval (default: 3600)
- `VECTOR_SEARCH_BATCH_SIZE` - Vector search batch size (default: 100)

## Usage Examples

### Store Working Memory

```json
POST /memories
{
  "agent_id": "agent_123",
  "memory_type": "working",
  "content_key": "current_task",
  "content_value": {
    "task_id": "task_456",
    "description": "Process customer inquiry",
    "status": "in_progress",
    "customer_data": {
      "id": "cust_789",
      "name": "John Doe",
      "inquiry_type": "policy_quote"
    }
  },
  "importance_score": 0.8,
  "ttl_seconds": 7200,
  "tags": ["task", "customer", "inquiry"],
  "metadata": {
    "source": "customer_portal",
    "priority": "high"
  }
}
```

**Response:**
```json
{
  "memory_id": "mem_12345678-1234-1234-1234-123456789abc",
  "status": "stored",
  "message": "Memory item stored successfully"
}
```

### Retrieve Memory

```bash
GET /memories/agent_123/working/current_task
```

**Response:**
```json
{
  "memory_id": "mem_12345678-1234-1234-1234-123456789abc",
  "agent_id": "agent_123",
  "memory_type": "working",
  "content_key": "current_task",
  "content_value": {
    "task_id": "task_456",
    "description": "Process customer inquiry",
    "status": "in_progress",
    "customer_data": {
      "id": "cust_789",
      "name": "John Doe",
      "inquiry_type": "policy_quote"
    }
  },
  "importance_score": 0.8,
  "access_count": 3,
  "last_accessed": "2024-01-15T10:30:00Z",
  "created_at": "2024-01-15T10:00:00Z",
  "expires_at": "2024-01-15T12:00:00Z",
  "tags": ["task", "customer", "inquiry"],
  "metadata": {
    "source": "customer_portal",
    "priority": "high"
  }
}
```

### Search Memories

```json
POST /memories/search
{
  "agent_id": "agent_123",
  "query": "customer",
  "memory_type": "working",
  "tags": ["inquiry"],
  "date_from": "2024-01-15T00:00:00Z",
  "limit": 10,
  "include_expired": false
}
```

**Response:**
```json
{
  "memories": [
    {
      "memory_id": "mem_12345678-1234-1234-1234-123456789abc",
      "memory_type": "working",
      "content_key": "current_task",
      "content_value": {...},
      "importance_score": 0.8,
      "access_count": 3,
      "last_accessed": "2024-01-15T10:30:00Z",
      "created_at": "2024-01-15T10:00:00Z",
      "expires_at": "2024-01-15T12:00:00Z",
      "tags": ["task", "customer", "inquiry"],
      "metadata": {...}
    }
  ],
  "total_count": 1,
  "has_more": false
}
```

### Consolidate Memories

```json
POST /memories/consolidate
{
  "agent_id": "agent_123",
  "source_memory_types": ["working", "episodic"],
  "target_memory_type": "semantic",
  "consolidation_strategy": "importance_based",
  "min_importance_score": 0.7,
  "consolidation_criteria": {
    "max_items": 100,
    "similarity_threshold": 0.8
  }
}
```

**Response:**
```json
{
  "agent_id": "agent_123",
  "consolidation_strategy": "importance_based",
  "source_memories_count": 50,
  "consolidated_memories_count": 15,
  "stored_memories_count": 15,
  "consolidation_completed_at": "2024-01-15T10:30:00Z"
}
```

### Store Vector Memory

```json
POST /vectors
{
  "memory_id": "vec_12345678-1234-1234-1234-123456789abc",
  "agent_id": "agent_123",
  "content_key": "customer_inquiry_pattern",
  "content_text": "Customer frequently asks about policy coverage and claims process",
  "vector_embedding": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "pattern_type": "behavioral",
    "frequency": 15,
    "last_occurrence": "2024-01-15T10:00:00Z"
  },
  "importance_score": 0.9
}
```

### Search Vector Memories

```json
POST /vectors/search
{
  "agent_id": "agent_123",
  "query_vector": [0.1, 0.2, 0.3, ...],
  "limit": 5,
  "threshold": 0.8
}
```

**Response:**
```json
{
  "results": [
    {
      "memory_id": "vec_12345678-1234-1234-1234-123456789abc",
      "content_key": "customer_inquiry_pattern",
      "content_text": "Customer frequently asks about policy coverage and claims process",
      "similarity_score": 0.92,
      "importance_score": 0.9,
      "metadata": {...},
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "query_vector_dimension": 768,
  "search_completed_at": "2024-01-15T10:30:00Z"
}
```

## Memory Consolidation Strategies

### Importance-Based Consolidation
- **Criteria**: Memory importance score
- **Purpose**: Retain high-value memories
- **Algorithm**: Sort by importance, keep top N items

### Recency-Based Consolidation
- **Criteria**: Memory creation/access time
- **Purpose**: Keep recent memories
- **Algorithm**: Filter by time window, sort by recency

### Frequency-Based Consolidation
- **Criteria**: Memory access frequency
- **Purpose**: Retain frequently accessed memories
- **Algorithm**: Sort by access count, keep most accessed

### Similarity-Based Consolidation
- **Criteria**: Content similarity
- **Purpose**: Reduce redundancy
- **Algorithm**: Group similar memories, merge duplicates

## Memory Performance Optimization

- **Redis Caching**: Fast access to frequently used memories
- **Batch Processing**: Efficient bulk operations
- **Background Cleanup**: Automatic expired memory removal
- **Index Optimization**: Database indexes for fast queries
- **Memory Limits**: Configurable limits per agent
- **Compression**: Automatic content compression for large items

## Monitoring and Metrics

The service exposes comprehensive metrics via Prometheus:

- `memory_manager_items_total`: Total memory items by type
- `memory_manager_access_total`: Total memory accesses by type
- `memory_manager_expiration_total`: Total memory expirations
- `memory_manager_consolidation_total`: Total memory consolidations
- `memory_manager_requests_total`: Total API requests
- `memory_manager_errors_total`: Error count by type

## Integration Points

The Memory Manager integrates with:

- **Agent Orchestrator**: Provides memory for agent state management
- **Workflow Engine**: Stores workflow execution state and results
- **Brain Factory**: Accesses agent memory during instantiation
- **Plugin Registry**: Caches plugin execution results
- **Template Store**: Stores template usage patterns
- **Vector Database**: External vector storage for similarity search

## Security Considerations

- **Access Control**: Agent-scoped memory isolation
- **Data Encryption**: Sensitive memory content encryption
- **Audit Logging**: Complete audit trail of memory operations
- **TTL Enforcement**: Strict expiration time enforcement
- **Content Validation**: Memory content validation and sanitization
- **Rate Limiting**: Protection against memory abuse

## Performance Characteristics

- **Working Memory**: Fast access, short retention, high throughput
- **Episodic Memory**: Medium access speed, medium retention, event-based
- **Semantic Memory**: Medium access speed, long retention, knowledge-based
- **Vector Memory**: Slower access, similarity-based, context-aware

## Automatic Maintenance

- **TTL Cleanup**: Automatic removal of expired memories
- **Consolidation**: Periodic memory optimization
- **Archival**: Long-term memory archival to external storage
- **Defragmentation**: Memory fragmentation cleanup
- **Statistics Update**: Performance statistics maintenance

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

The service is designed to run in Docker containers and integrates with the broader Agentic Platform. It automatically discovers and communicates with other platform services through the shared Docker network.

## API Reference

For complete API documentation, visit `/docs` when the service is running, or `/redoc` for the ReDoc interface.
