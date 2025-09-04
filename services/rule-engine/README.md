# Rule Engine Service

The Rule Engine Service is a powerful component of the Agentic Brain Platform that provides sophisticated business rule processing and automated decision-making capabilities. It supports complex rule evaluation, conditional logic, and automated actions based on configurable rule sets.

## Features

- **Complex Rule Evaluation**: Support for multiple condition types and logical operators
- **Rule Set Management**: Organize rules into hierarchical rule sets by domain
- **Automated Actions**: Execute actions based on rule evaluation results
- **Performance Monitoring**: Track rule execution statistics and performance
- **Built-in Rule Sets**: Pre-configured rule sets for common business scenarios
- **Real-time Evaluation**: Synchronous and asynchronous rule evaluation
- **Rule Versioning**: Track rule changes and maintain version history
- **RESTful API**: Complete API for rule management and evaluation
- **Authentication Integration**: Support for JWT-based authentication
- **Monitoring & Metrics**: Prometheus metrics for observability

## Built-in Rule Sets

### Fraud Detection Rules (`fraud_detection`)
- **High Amount Transaction**: Flags transactions above $50,000 threshold
- **Rapid Transactions**: Detects high-frequency transaction patterns
- **Suspicious Patterns**: Identifies multiple suspicious indicators

### Compliance Rules (`compliance_rules`)
- **Data Retention Check**: Ensures data retention policies are followed
- **Regulatory Requirements**: Validates compliance with regulatory standards
- **Audit Trail**: Maintains compliance audit trails

### Risk Assessment Rules (`risk_assessment`)
- **Credit Score Evaluation**: Assesses risk based on credit scores
- **Financial Ratios**: Evaluates debt-to-income and loan-to-value ratios
- **Risk Scoring**: Calculates comprehensive risk scores

### Data Validation Rules (`data_validation`)
- **Required Fields**: Ensures mandatory fields are present
- **Data Type Validation**: Validates field data types
- **Format Checking**: Validates data format and structure

## Rule Structure

Rules consist of conditions and actions:

```json
{
  "rule_id": "high_amount_transaction",
  "name": "High Amount Transaction",
  "description": "Flag transactions above threshold",
  "rule_type": "decision",
  "priority": 1,
  "conditions": [
    {
      "field": "amount",
      "operator": "greater_than",
      "value": 50000
    }
  ],
  "actions": [
    {
      "type": "flag_record",
      "flag_value": "high_amount",
      "reason": "Transaction amount exceeds threshold"
    },
    {
      "type": "update_status",
      "new_status": "requires_review"
    }
  ]
}
```

## Condition Operators

The rule engine supports various condition operators:

### Comparison Operators
- `equals` - Exact equality check
- `not_equals` - Inequality check
- `greater_than` - Greater than comparison
- `less_than` - Less than comparison
- `greater_equal` - Greater than or equal
- `less_equal` - Less than or equal

### String Operators
- `contains` - Check if string contains substring
- `not_contains` - Check if string does not contain substring
- `starts_with` - Check if string starts with prefix
- `ends_with` - Check if string ends with suffix
- `regex_match` - Regular expression matching

### List Operators
- `in_list` - Check if value is in a list
- `not_in_list` - Check if value is not in a list

### Null Checks
- `is_null` - Check if field is null
- `is_not_null` - Check if field is not null

### Range Operators
- `between` - Check if value is between two values
- `not_between` - Check if value is not between two values

## Action Types

The rule engine supports various action types:

### Data Actions
- `set_field` - Set a field to a specific value
- `add_to_list` - Add value to a list field
- `remove_from_list` - Remove value from a list field
- `transform_data` - Transform data using mapping rules

### Decision Actions
- `flag_record` - Add a flag to the record
- `update_status` - Update the record status
- `calculate_score` - Calculate a score based on conditions

### Notification Actions
- `send_notification` - Send notification (email, SMS, webhook)
- `log_event` - Log an event with specified level

## API Endpoints

### Rule Management
- `GET /rules/sets` - List available rule sets
- `GET /rules/sets/{rule_set_id}` - Get detailed rule set information
- `POST /rules/evaluate` - Evaluate rules against input data
- `GET /rules/performance` - Get rule performance statistics

### Discovery
- `GET /rules/operators` - Get available condition operators
- `GET /rules/actions` - Get available action types
- `GET /rules/domains` - Get available business domains

### Monitoring
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics endpoint

## Usage Examples

### Evaluate Fraud Detection Rules

```json
POST /rules/evaluate
{
  "rule_set_id": "fraud_detection",
  "input_data": {
    "transaction_id": "TXN_123456",
    "amount": 75000,
    "transaction_frequency": 8,
    "customer_history": "good",
    "location": "unusual"
  }
}
```

**Response:**
```json
{
  "execution_id": "exec_12345678-1234-1234-1234-123456789abc",
  "rule_set_id": "fraud_detection",
  "input_data": {...},
  "output_data": {
    "transaction_id": "TXN_123456",
    "amount": 75000,
    "transaction_frequency": 8,
    "flags": [
      {
        "flag": "high_amount",
        "reason": "Transaction amount exceeds threshold",
        "timestamp": "2024-01-15T10:30:00Z",
        "rule_id": "high_amount_transaction"
      },
      {
        "flag": "rapid_transactions",
        "reason": "High frequency of transactions detected",
        "timestamp": "2024-01-15T10:30:00Z",
        "rule_id": "rapid_transactions"
      }
    ],
    "status": "requires_review"
  },
  "results": [
    {
      "rule_id": "high_amount_transaction",
      "rule_name": "High Amount Transaction",
      "conditions_matched": true,
      "matched_conditions": [...],
      "failed_conditions": [],
      "executed_actions": [...]
    }
  ],
  "matched_rules": ["high_amount_transaction", "rapid_transactions"],
  "execution_time_ms": 45
}
```

### Evaluate Compliance Rules

```json
POST /rules/evaluate
{
  "rule_set_id": "compliance_rules",
  "input_data": {
    "record_id": "REC_789",
    "data_age_days": 3000,
    "data_type": "financial",
    "retention_policy": "7_years"
  }
}
```

### Get Rule Performance

```bash
GET /rules/performance?rule_set_id=fraud_detection
```

**Response:**
```json
{
  "performance_data": [
    {
      "rule_id": "high_amount_transaction",
      "rule_set_id": "fraud_detection",
      "execution_count": 1250,
      "success_count": 1200,
      "success_rate": 96.0,
      "average_execution_time_ms": 15.5,
      "last_executed": "2024-01-15T10:30:00Z"
    }
  ],
  "total_rules": 3,
  "generated_at": "2024-01-15T10:30:00Z"
}
```

## Rule Set Structure

Rule sets are organized collections of rules:

```json
{
  "rule_set_id": "fraud_detection",
  "name": "Fraud Detection Rules",
  "description": "Rules for detecting potential fraudulent activities",
  "domain": "fraud",
  "version": "1.0.0",
  "rules": [
    {
      "rule_id": "rule_1",
      "name": "Rule Name",
      "description": "Rule description",
      "rule_type": "decision",
      "priority": 1,
      "conditions": [...],
      "actions": [...]
    }
  ]
}
```

## Logical Operators

Rules support combining conditions with logical operators:

- `AND` - All conditions must be true
- `OR` - At least one condition must be true

```json
{
  "rule_id": "complex_rule",
  "name": "Complex Rule",
  "conditions": [
    {
      "field": "amount",
      "operator": "greater_than",
      "value": 10000
    },
    {
      "field": "risk_score",
      "operator": "greater_than",
      "value": 0.8
    }
  ],
  "logical_operator": "and"
}
```

## Rule Types

### Validation Rules
Validate data integrity and compliance:
- Check required fields
- Validate data types
- Ensure format compliance
- Verify business rules

### Decision Rules
Make automated decisions:
- Approve/reject applications
- Flag suspicious activities
- Route records for review
- Determine processing paths

### Transformation Rules
Transform data:
- Normalize values
- Calculate derived fields
- Apply business logic
- Format output data

### Scoring Rules
Calculate scores and ratings:
- Risk scoring
- Credit scoring
- Compliance scoring
- Performance scoring

## Nested Field Access

Rules support nested field access using dot notation:

```json
{
  "field": "customer.address.city",
  "operator": "equals",
  "value": "New York"
}
```

And array access:

```json
{
  "field": "transactions[0].amount",
  "operator": "greater_than",
  "value": 5000
}
```

## Performance Optimization

- **Rule Caching**: Frequently used rules are cached in Redis
- **Execution Optimization**: Rules are sorted by priority for optimal evaluation
- **Batch Processing**: Support for processing multiple records simultaneously
- **Index Optimization**: Database indexes for fast rule retrieval
- **Memory Management**: Efficient memory usage for large rule sets

## Monitoring and Metrics

The service exposes comprehensive metrics via Prometheus:

- `rule_engine_rule_sets_active`: Number of active rule sets
- `rule_engine_rule_evaluations_total`: Total rule evaluations by rule set and result
- `rule_engine_evaluation_duration_seconds`: Rule evaluation duration
- `rule_engine_rule_executions_total`: Total rule executions by rule and result
- `rule_engine_requests_total`: Total API requests
- `rule_engine_errors_total`: Error count by type

## Error Handling

The Rule Engine implements robust error handling:

- **Rule-Level Errors**: Isolated failure handling per rule
- **Condition Evaluation Errors**: Graceful handling of invalid conditions
- **Action Execution Errors**: Rollback and error reporting for failed actions
- **Timeout Protection**: Automatic timeout for long-running evaluations
- **Circuit Breaker**: Protection against cascading failures
- **Audit Logging**: Complete audit trail of all rule evaluations

## Security Considerations

- **Input Validation**: Comprehensive validation of rule definitions and input data
- **Access Control**: Role-based access to rule evaluation and management
- **Rule Sandboxing**: Isolated execution environment for custom rules
- **Audit Logging**: Complete audit trail of rule changes and evaluations
- **Data Protection**: Secure handling of sensitive data in rule evaluation
- **Rate Limiting**: Protection against abuse with configurable rate limits

## Integration Points

The Rule Engine integrates with:

- **Workflow Engine**: Provides rule evaluation within workflow components
- **Agent Orchestrator**: Enables rule-based agent decision making
- **Plugin Registry**: Rules can trigger plugin execution
- **Memory Manager**: Rules can access and update agent memory
- **Audit System**: Comprehensive logging of rule evaluations
- **Monitoring System**: Real-time performance tracking and alerting

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
