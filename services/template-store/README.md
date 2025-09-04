# Template Store Service

The Template Store Service is a core component of the Agentic Brain Platform that manages prebuilt agent templates. It enables no-code agent creation by providing a library of professionally designed templates that users can instantiate with their own parameters and customizations.

## Features

- **Template Management**: Store, retrieve, and manage agent templates
- **Template Instantiation**: Create customized agents from templates
- **Built-in Templates**: Pre-loaded templates for common use cases
- **Template Validation**: Ensure template integrity and compatibility
- **Template Search**: Find templates by domain, category, tags, and popularity
- **Usage Analytics**: Track template usage and popularity metrics
- **Version Control**: Manage template versions and updates
- **Export/Import**: Support for template backup and sharing
- **RESTful API**: Complete API for template operations
- **Authentication Integration**: Support for JWT-based authentication
- **Monitoring & Metrics**: Prometheus metrics for observability

## Built-in Templates

### Underwriting Agent Template
- **ID**: `underwriting_template`
- **Domain**: Underwriting
- **Components**:
  - CSV Data Input for policy data
  - LLM Processor for risk assessment
  - Decision Node for approval/rejection logic
  - Database Output for policy storage
- **Use Case**: Automated insurance underwriting with risk analysis

### Claims Processing Agent Template
- **ID**: `claims_template`
- **Domain**: Claims
- **Components**:
  - API Data Input for claim data
  - Fraud Detector plugin
  - LLM Processor for settlement calculation
  - Email Output for notifications
- **Use Case**: Intelligent claims processing with fraud detection

### Fraud Detection Agent Template
- **ID**: `fraud_detection_template`
- **Domain**: Fraud
- **Components**:
  - Database Input for transaction data
  - Fraud Detector plugin
  - Rule Engine for fraud rules
  - Database Output for fraud alerts
- **Use Case**: Real-time fraud detection and alerting

### Customer Service Agent Template
- **ID**: `customer_service_template`
- **Domain**: Customer Service
- **Components**:
  - API Input for customer queries
  - LLM Processor for intent analysis
  - LLM Processor for response generation
  - Database Output for response storage
- **Use Case**: Automated customer service with intelligent responses

## API Endpoints

### Template Management
- `GET /templates` - List templates with filtering and search
- `GET /templates/{template_id}` - Get detailed template information
- `POST /templates/instantiate` - Instantiate template with custom parameters
- `POST /templates/validate` - Validate template structure
- `GET /templates/{template_id}/export` - Export template in JSON or ZIP format

### Discovery
- `GET /templates/categories` - Get available template categories
- `GET /templates/domains` - Get available business domains
- `GET /templates/stats` - Get template usage statistics

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

### Service Configuration
- `TEMPLATE_STORE_HOST` - Service host (default: 0.0.0.0)
- `TEMPLATE_STORE_PORT` - Service port (default: 8203)
- `REQUIRE_AUTH` - Enable authentication (default: false)
- `JWT_SECRET` - JWT secret key for authentication

### Template Configuration
- `TEMPLATE_CACHE_ENABLED` - Enable template caching (default: true)
- `TEMPLATE_CACHE_TTL_SECONDS` - Cache TTL in seconds (default: 3600)
- `TEMPLATE_VERSION_CONTROL_ENABLED` - Enable version control (default: true)
- `TEMPLATE_AUTO_BACKUP_ENABLED` - Enable automatic backups (default: true)
- `MAX_TEMPLATE_SIZE_MB` - Maximum template size (default: 10)

### Storage Configuration
- `TEMPLATE_STORAGE_PATH` - Path for template storage (default: /app/templates)
- `BACKUP_STORAGE_PATH` - Path for template backups (default: /app/backups)

## Usage Examples

### List Templates

```bash
GET /templates?domain=underwriting&limit=10
```

**Response:**
```json
{
  "templates": [
    {
      "template_id": "underwriting_template",
      "name": "Underwriting Agent",
      "domain": "underwriting",
      "description": "Comprehensive underwriting agent with risk assessment",
      "category": "business_process",
      "tags": ["underwriting", "risk", "insurance"],
      "rating": 4.5,
      "usage_count": 150,
      "created_by": "system",
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "total_count": 1,
  "limit": 10,
  "offset": 0
}
```

### Get Template Details

```bash
GET /templates/underwriting_template
```

**Response:**
```json
{
  "template_id": "underwriting_template",
  "name": "Underwriting Agent",
  "domain": "underwriting",
  "description": "Comprehensive underwriting agent with risk assessment and decision making",
  "version": "1.0.0",
  "author": "system",
  "category": "business_process",
  "tags": ["underwriting", "risk", "insurance", "decision-making"],
  "is_public": true,
  "is_featured": true,
  "rating": 4.5,
  "usage_count": 150,
  "components": [...],
  "connections": [...],
  "configuration": {...},
  "parameters": {...}
}
```

### Instantiate Template

```json
POST /templates/instantiate
{
  "template_id": "underwriting_template",
  "parameters": {
    "data_source": "/data/custom_policies.csv",
    "risk_threshold": 0.8
  },
  "customizations": {
    "configuration": {
      "persona": {
        "role": "Senior Underwriting Analyst"
      }
    }
  },
  "target_agent_id": "my_underwriting_agent"
}
```

**Response:**
```json
{
  "instantiated_template": {
    "template_id": "my_underwriting_agent_instance",
    "name": "Underwriting Agent (Instance)",
    "components": [...],
    "connections": [...],
    "configuration": {...}
  },
  "message": "Template instantiated successfully"
}
```

### Validate Template

```json
POST /templates/validate
{
  "template_id": "custom_template",
  "name": "Custom Agent",
  "domain": "general",
  "components": [
    {
      "component_id": "input1",
      "component_type": "data_input",
      "config": {
        "source_type": "csv"
      }
    }
  ],
  "connections": []
}
```

**Response:**
```json
{
  "is_valid": true,
  "errors": [],
  "warnings": [
    "Consider adding a more detailed description",
    "Consider adding tags for better discoverability"
  ],
  "validation_type": "structure"
}
```

### Search Templates

```bash
GET /templates?query=risk&domain=underwriting&tags=fraud&sort_by=usage_count&limit=5
```

## Template Structure

Templates follow a standardized JSON structure:

```json
{
  "template_id": "unique_template_id",
  "name": "Human-readable name",
  "domain": "business_domain",
  "description": "Detailed description",
  "version": "semantic_version",
  "author": "template_author",
  "category": "template_category",
  "tags": ["tag1", "tag2"],
  "is_public": true,
  "is_featured": false,
  "components": [
    {
      "component_id": "unique_component_id",
      "component_type": "component_type",
      "config": {
        "parameter1": "value1",
        "parameter2": "value2"
      }
    }
  ],
  "connections": [
    {
      "from": "source_component_id",
      "to": "target_component_id"
    }
  ],
  "configuration": {
    "persona": {
      "role": "Agent Role",
      "expertise": ["skill1", "skill2"],
      "personality": "personality_type"
    },
    "reasoningPattern": "react",
    "memoryConfig": {
      "workingMemoryTTL": 3600,
      "episodic": true,
      "longTerm": true
    }
  },
  "parameters": {
    "parameter_name": {
      "type": "string|number|boolean",
      "description": "Parameter description",
      "default": "default_value",
      "required": false
    }
  }
}
```

## Template Categories

- **Business Process**: Templates for automating business workflows
- **Customer Support**: Templates for customer service automation
- **Security**: Templates for security monitoring and compliance
- **Data Processing**: Templates for data transformation and analytics
- **General**: General-purpose templates for common tasks

## Business Domains

- **Underwriting**: Insurance underwriting and risk assessment
- **Claims**: Insurance claims processing and fraud detection
- **Fraud**: Financial fraud detection and prevention
- **Customer Service**: Customer support and service automation
- **Compliance**: Regulatory compliance and auditing
- **Finance**: Financial services and transaction processing
- **Healthcare**: Healthcare and medical services

## Parameter Substitution

Templates support parameter substitution using `{{parameter_name}}` syntax:

```json
{
  "components": [
    {
      "component_id": "data_input",
      "component_type": "data_input",
      "config": {
        "source_type": "csv",
        "file_path": "{{data_source}}"
      }
    }
  ],
  "parameters": {
    "data_source": {
      "type": "string",
      "description": "Path to data source",
      "default": "/data/input.csv"
    }
  }
}
```

## Version Control

Templates support semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes that are not backward compatible
- **MINOR**: New features that are backward compatible
- **PATCH**: Bug fixes and minor improvements

## Export and Import

Templates can be exported in multiple formats:

- **JSON**: Complete template definition
- **ZIP**: Compressed archive with template and metadata

## Monitoring and Metrics

The service exposes comprehensive metrics via Prometheus:

- `template_store_total_templates`: Total number of templates
- `template_store_usage_total`: Template usage by template and type
- `template_store_instantiations_total`: Total template instantiations
- `template_store_search_requests_total`: Total search requests
- `template_store_validation_requests_total`: Total validation requests
- `template_store_requests_total`: Total API requests
- `template_store_errors_total`: Error count by type

## Security Considerations

- **Input Validation**: Comprehensive validation of template structures
- **Access Control**: Role-based access to template operations
- **Parameter Sanitization**: Safe parameter substitution
- **Audit Logging**: Complete audit trail of template operations
- **Version Control**: Track template changes and authorship
- **Backup Security**: Secure template backup storage

## Performance Optimization

- **Caching**: Redis-based template caching for fast retrieval
- **Indexing**: Database indexes for efficient search and filtering
- **Pagination**: Efficient handling of large template lists
- **Async Processing**: Non-blocking template operations
- **Connection Pooling**: Efficient database and cache connections
- **File System Optimization**: Optimized template storage and retrieval

## Integration Points

The Template Store integrates with:

- **Agent Builder UI**: Provides templates for no-code agent creation
- **Brain Factory**: Supplies templates for agent instantiation
- **Workflow Engine**: Uses templates to create workflow definitions
- **Agent Orchestrator**: Registers instantiated agents
- **Plugin Registry**: References plugins used in templates

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
