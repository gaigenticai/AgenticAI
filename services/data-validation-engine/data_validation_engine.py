#!/usr/bin/env python3
"""
Data Validation Engine Service for Agentic Platform

This service provides comprehensive data validation, quality assessment,
and transformation capabilities including:

- Schema validation and type checking
- Data quality metrics calculation
- Duplicate detection and removal
- Missing value analysis and imputation
- Outlier detection and handling
- Data transformation and normalization
- Statistical profiling and reporting
- Custom validation rule engine
- Real-time validation pipelines
"""

import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pika
import pandas as pd
import psycopg2
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import uvicorn

# Configure structured logging
logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# FastAPI app
app = FastAPI(
    title="Data Validation Engine Service",
    description="Comprehensive data validation, quality assessment, and transformation service",
    version="1.0.0"
)

# Prometheus metrics
VALIDATION_REQUESTS_TOTAL = Counter('validation_requests_total', 'Total validation requests', ['validation_type'])
VALIDATION_ERRORS_TOTAL = Counter('validation_errors_total', 'Total validation errors', ['error_type'])
VALIDATION_DURATION = Histogram('validation_duration_seconds', 'Validation duration', ['operation'])
DATA_QUALITY_SCORE = Gauge('data_quality_score', 'Current data quality score', ['dataset'])
TRANSFORMATION_OPERATIONS = Counter('transformation_operations_total', 'Total transformation operations', ['operation_type'])

# Global variables
database_connection = None
message_queue_channel = None

# Pydantic models for API
class ValidationRule(BaseModel):
    """Validation rule model"""
    rule_name: str = Field(..., description="Unique name for the validation rule")
    rule_type: str = Field(..., description="Type of validation rule (schema, range, pattern, custom)")
    field_name: Optional[str] = Field(None, description="Field to apply validation to")
    parameters: Dict[str, Any] = Field(..., description="Rule parameters")
    severity: str = Field("error", description="Rule severity (error, warning, info)")
    description: Optional[str] = Field(None, description="Rule description")

class ValidationRequest(BaseModel):
    """Validation request model"""
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    data: Union[List[Dict[str, Any]], pd.DataFrame] = Field(..., description="Data to validate")
    validation_rules: List[ValidationRule] = Field(..., description="List of validation rules")
    quality_checks: List[str] = Field(["completeness", "accuracy", "consistency"], description="Quality checks to perform")
    generate_report: bool = Field(True, description="Generate detailed validation report")

class TransformationRule(BaseModel):
    """Data transformation rule model"""
    operation: str = Field(..., description="Transformation operation type")
    field_name: str = Field(..., description="Field to transform")
    parameters: Dict[str, Any] = Field(..., description="Transformation parameters")
    conditions: Optional[Dict[str, Any]] = Field(None, description="Conditions for applying transformation")

class DataQualityReport(BaseModel):
    """Data quality report model"""
    dataset_id: str
    overall_score: float
    total_records: int
    valid_records: int
    invalid_records: int
    validation_results: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime

class ValidationEngine:
    """Core validation engine with comprehensive data quality capabilities"""

    def __init__(self):
        self.custom_rules = {}
        self.quality_thresholds = {
            "completeness": 0.95,
            "accuracy": 0.90,
            "consistency": 0.85,
            "timeliness": 0.80,
            "validity": 0.90
        }

    def validate_data(self, data: Union[List[Dict[str, Any]], pd.DataFrame],
                     rules: List[ValidationRule]) -> Dict[str, Any]:
        """
        Comprehensive data validation with multiple rule types

        Args:
            data: Data to validate (list of dicts or DataFrame)
            rules: List of validation rules to apply

        Returns:
            Validation results with detailed error reporting
        """
        start_time = time.time()

        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        VALIDATION_REQUESTS_TOTAL.labels(validation_type="comprehensive").inc()

        results = {
            "is_valid": True,
            "total_records": len(df),
            "valid_records": 0,
            "invalid_records": 0,
            "errors": [],
            "warnings": [],
            "field_validation": {},
            "rule_results": {}
        }

        try:
            # Apply each validation rule
            for rule in rules:
                rule_result = self._apply_validation_rule(df, rule)
                results["rule_results"][rule.rule_name] = rule_result

                if not rule_result["passed"]:
                    if rule.severity == "error":
                        results["errors"].extend(rule_result["issues"])
                        results["is_valid"] = False
                    elif rule.severity == "warning":
                        results["warnings"].extend(rule_result["issues"])

            # Calculate record-level validity
            results["valid_records"] = results["total_records"] - len(results["errors"])
            results["invalid_records"] = len(results["errors"])

            # Field-level validation summary
            results["field_validation"] = self._analyze_field_validity(df, rules)

        except Exception as e:
            logger.error("Validation failed", error=str(e))
            VALIDATION_ERRORS_TOTAL.labels(error_type="validation_failure").inc()
            results["errors"].append(f"Validation engine error: {str(e)}")
            results["is_valid"] = False

        duration = time.time() - start_time
        VALIDATION_DURATION.labels(operation="comprehensive_validation").observe(duration)

        return results

    def _apply_validation_rule(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Apply a single validation rule"""
        try:
            if rule.rule_type == "schema":
                return self._validate_schema(df, rule)
            elif rule.rule_type == "range":
                return self._validate_range(df, rule)
            elif rule.rule_type == "pattern":
                return self._validate_pattern(df, rule)
            elif rule.rule_type == "uniqueness":
                return self._validate_uniqueness(df, rule)
            elif rule.rule_type == "completeness":
                return self._validate_completeness(df, rule)
            elif rule.rule_type == "custom":
                return self._apply_custom_rule(df, rule)
            else:
                return {
                    "passed": False,
                    "issues": [f"Unknown rule type: {rule.rule_type}"],
                    "details": {}
                }

        except Exception as e:
            return {
                "passed": False,
                "issues": [f"Rule application failed: {str(e)}"],
                "details": {"error": str(e)}
            }

    def _validate_schema(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate data schema against expected structure"""
        expected_schema = rule.parameters.get("schema", {})
        issues = []

        for field, expected_type in expected_schema.items():
            if field not in df.columns:
                issues.append(f"Missing required field: {field}")
                continue

            actual_type = str(df[field].dtype)
            if not self._types_compatible(actual_type, expected_type):
                issues.append(f"Type mismatch for {field}: expected {expected_type}, got {actual_type}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "details": {
                "expected_fields": list(expected_schema.keys()),
                "actual_fields": df.columns.tolist(),
                "missing_fields": [f for f in expected_schema.keys() if f not in df.columns]
            }
        }

    def _validate_range(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate numeric ranges"""
        field = rule.field_name
        min_val = rule.parameters.get("min")
        max_val = rule.parameters.get("max")

        if field not in df.columns:
            return {"passed": False, "issues": [f"Field {field} not found"], "details": {}}

        issues = []
        out_of_range = 0

        if min_val is not None:
            below_min = df[field] < min_val
            out_of_range += below_min.sum()
            if below_min.any():
                issues.append(f"{below_min.sum()} values below minimum {min_val}")

        if max_val is not None:
            above_max = df[field] > max_val
            out_of_range += above_max.sum()
            if above_max.any():
                issues.append(f"{above_max.sum()} values above maximum {max_val}")

        return {
            "passed": out_of_range == 0,
            "issues": issues,
            "details": {
                "out_of_range_count": int(out_of_range),
                "valid_count": len(df) - int(out_of_range),
                "range_checked": f"{min_val} - {max_val}"
            }
        }

    def _validate_pattern(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate string patterns using regex"""
        field = rule.field_name
        pattern = rule.parameters.get("pattern", "")

        if field not in df.columns:
            return {"passed": False, "issues": [f"Field {field} not found"], "details": {}}

        try:
            regex = re.compile(pattern)
            matches = df[field].astype(str).str.match(regex)
            invalid_count = (~matches).sum()

            return {
                "passed": invalid_count == 0,
                "issues": [f"{invalid_count} values don't match pattern {pattern}"] if invalid_count > 0 else [],
                "details": {
                    "pattern": pattern,
                    "valid_count": len(df) - int(invalid_count),
                    "invalid_count": int(invalid_count)
                }
            }

        except re.error as e:
            return {
                "passed": False,
                "issues": [f"Invalid regex pattern: {str(e)}"],
                "details": {"pattern_error": str(e)}
            }

    def _validate_uniqueness(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate uniqueness constraints"""
        fields = rule.parameters.get("fields", [rule.field_name])

        # Check if all fields exist
        missing_fields = [f for f in fields if f not in df.columns]
        if missing_fields:
            return {
                "passed": False,
                "issues": [f"Missing fields: {missing_fields}"],
                "details": {}
            }

        # Check for duplicates
        duplicate_count = df.duplicated(subset=fields).sum()

        return {
            "passed": duplicate_count == 0,
            "issues": [f"{duplicate_count} duplicate records found"] if duplicate_count > 0 else [],
            "details": {
                "duplicate_count": int(duplicate_count),
                "unique_count": len(df) - int(duplicate_count),
                "fields_checked": fields
            }
        }

    def _validate_completeness(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate data completeness (non-null values)"""
        threshold = rule.parameters.get("threshold", 0.95)
        fields = rule.parameters.get("fields", df.columns.tolist())

        issues = []
        completeness_scores = {}

        for field in fields:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                completeness = non_null_count / len(df)
                completeness_scores[field] = float(completeness)

                if completeness < threshold:
                    issues.append(".2%"
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "details": {
                "completeness_threshold": threshold,
                "field_scores": completeness_scores,
                "failing_fields": [f for f in fields if completeness_scores.get(f, 0) < threshold]
            }
        }

    def _apply_custom_rule(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Apply custom validation rule"""
        rule_function = self.custom_rules.get(rule.rule_name)
        if not rule_function:
            return {
                "passed": False,
                "issues": [f"Custom rule {rule.rule_name} not found"],
                "details": {}
            }

        try:
            return rule_function(df, rule.parameters)
        except Exception as e:
            return {
                "passed": False,
                "issues": [f"Custom rule execution failed: {str(e)}"],
                "details": {"execution_error": str(e)}
            }

    def _analyze_field_validity(self, df: pd.DataFrame, rules: List[ValidationRule]) -> Dict[str, Any]:
        """Analyze field-level validation results"""
        field_analysis = {}

        for column in df.columns:
            field_rules = [r for r in rules if r.field_name == column or r.field_name is None]
            if field_rules:
                null_count = df[column].isnull().sum()
                unique_count = df[column].nunique()

                field_analysis[column] = {
                    "total_values": len(df),
                    "null_count": int(null_count),
                    "null_percentage": float(null_count / len(df)) * 100,
                    "unique_count": int(unique_count),
                    "data_type": str(df[column].dtype),
                    "rules_applied": len(field_rules)
                }

        return field_analysis

    def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual data type is compatible with expected type"""
        type_mapping = {
            "int64": ["integer", "int", "number"],
            "float64": ["float", "number", "decimal"],
            "object": ["string", "text", "varchar"],
            "bool": ["boolean", "bool"],
            "datetime64[ns]": ["datetime", "timestamp", "date"]
        }

        compatible_types = type_mapping.get(actual_type.lower(), [])
        return expected_type.lower() in compatible_types or actual_type.lower() == expected_type.lower()

    def calculate_data_quality_score(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive data quality score"""
        total_records = validation_results["total_records"]
        if total_records == 0:
            return {"overall_score": 0.0, "metrics": {}}

        # Calculate individual quality metrics
        metrics = {}

        # Completeness score
        valid_records = validation_results["valid_records"]
        metrics["completeness"] = valid_records / total_records

        # Accuracy score (inverse of error rate)
        error_rate = len(validation_results["errors"]) / total_records
        metrics["accuracy"] = max(0, 1 - error_rate)

        # Consistency score (based on field validation)
        field_scores = []
        for field_data in validation_results["field_validation"].values():
            completeness = 1 - (field_data["null_percentage"] / 100)
            uniqueness_ratio = field_data["unique_count"] / field_data["total_values"]
            field_scores.append((completeness + uniqueness_ratio) / 2)

        metrics["consistency"] = sum(field_scores) / len(field_scores) if field_scores else 1.0

        # Overall quality score (weighted average)
        weights = {
            "completeness": 0.3,
            "accuracy": 0.4,
            "consistency": 0.3
        }

        overall_score = sum(metrics[metric] * weights[metric] for metric in metrics.keys())

        return {
            "overall_score": round(overall_score, 3),
            "metrics": {k: round(v, 3) for k, v in metrics.items()},
            "thresholds": self.quality_thresholds,
            "passed_thresholds": {
                metric: score >= self.quality_thresholds.get(metric, 0.8)
                for metric, score in metrics.items()
            }
        }

    def generate_quality_report(self, dataset_id: str, validation_results: Dict[str, Any]) -> DataQualityReport:
        """Generate comprehensive data quality report"""
        quality_scores = self.calculate_data_quality_score(validation_results)

        recommendations = []
        if quality_scores["overall_score"] < 0.8:
            recommendations.append("Overall data quality is below acceptable threshold")
        if quality_scores["metrics"]["completeness"] < 0.9:
            recommendations.append("High rate of missing values detected - consider data imputation")
        if quality_scores["metrics"]["accuracy"] < 0.85:
            recommendations.append("Data accuracy issues found - review validation rules")
        if len(validation_results["errors"]) > 0:
            recommendations.append(f"{len(validation_results['errors'])} validation errors need attention")

        return DataQualityReport(
            dataset_id=dataset_id,
            overall_score=quality_scores["overall_score"],
            total_records=validation_results["total_records"],
            valid_records=validation_results["valid_records"],
            invalid_records=validation_results["invalid_records"],
            validation_results=validation_results,
            quality_metrics=quality_scores["metrics"],
            recommendations=recommendations,
            generated_at=datetime.utcnow()
        )

def setup_rabbitmq():
    """Setup RabbitMQ connection and consumer"""
    global message_queue_channel

    try:
        credentials = pika.PlainCredentials(
            os.getenv("RABBITMQ_USER", "agentic_user"),
            os.getenv("RABBITMQ_PASSWORD", "agentic123")
        )
        parameters = pika.ConnectionParameters(
            host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
            port=int(os.getenv("RABBITMQ_PORT", 5672)),
            credentials=credentials
        )

        connection = pika.BlockingConnection(parameters)
        message_queue_channel = connection.channel()

        # Declare queues for validation requests
        queues = ['data_validation', 'quality_check', 'transformation_request']
        for queue in queues:
            message_queue_channel.queue_declare(queue=queue, durable=True)

        # Set up consumer
        message_queue_channel.basic_qos(prefetch_count=1)
        message_queue_channel.basic_consume(
            queue='data_validation',
            on_message_callback=process_validation_message
        )

        logger.info("RabbitMQ consumer setup completed")
        message_queue_channel.start_consuming()

    except Exception as e:
        logger.error("Failed to setup RabbitMQ consumer", error=str(e))
        raise

def process_validation_message(ch, method, properties, body):
    """Process incoming validation message"""
    try:
        message = json.loads(body)
        dataset_id = message["dataset_id"]
        data = message["data"]
        rules = message.get("rules", [])
        quality_checks = message.get("quality_checks", [])

        logger.info("Received validation request", dataset_id=dataset_id)

        # Convert rules to ValidationRule objects
        validation_rules = [ValidationRule(**rule) for rule in rules]

        # Perform validation
        engine = ValidationEngine()
        results = engine.validate_data(data, validation_rules)

        # Generate quality report if requested
        if message.get("generate_report", True):
            report = engine.generate_quality_report(dataset_id, results)
            results["quality_report"] = report.dict()

        # Store results in database if available
        if database_connection:
            store_validation_results(dataset_id, results)

        logger.info("Validation completed",
                   dataset_id=dataset_id,
                   is_valid=results["is_valid"],
                   valid_records=results["valid_records"])

        # Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.error("Failed to process validation message", error=str(e))
        # Negative acknowledge - requeue message
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

def store_validation_results(dataset_id: str, results: Dict[str, Any]):
    """Store validation results in database"""
    try:
        with database_connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO data_validation_results
                (job_id, validation_type, is_valid, error_message, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (
                dataset_id,
                "comprehensive_validation",
                results["is_valid"],
                "; ".join(results["errors"]) if results["errors"] else None,
                json.dumps(results)
            ))
            database_connection.commit()

    except Exception as e:
        logger.error("Failed to store validation results", error=str(e))

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data-validation-engine",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/validate", response_model=Dict[str, Any])
async def validate_endpoint(request: ValidationRequest):
    """Validate data endpoint"""
    try:
        engine = ValidationEngine()

        # Convert data to list of dicts if it's a DataFrame
        if hasattr(request.data, 'to_dict'):
            data = request.data.to_dict('records')
        else:
            data = request.data

        results = engine.validate_data(data, request.validation_rules)

        if request.generate_report:
            report = engine.generate_quality_report(request.dataset_id, results)
            results["quality_report"] = report.dict()

        return results

    except Exception as e:
        logger.error("Validation endpoint error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/quality-report", response_model=DataQualityReport)
async def quality_report_endpoint(dataset_id: str, data: Union[List[Dict[str, Any]], pd.DataFrame]):
    """Generate data quality report"""
    try:
        engine = ValidationEngine()

        # Perform basic validation
        basic_rules = [
            ValidationRule(
                rule_name="completeness_check",
                rule_type="completeness",
                parameters={"threshold": 0.9}
            )
        ]

        results = engine.validate_data(data, basic_rules)
        report = engine.generate_quality_report(dataset_id, results)

        return report

    except Exception as e:
        logger.error("Quality report generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/rules")
async def get_validation_rules():
    """Get available validation rules"""
    return {
        "schema_rules": ["schema", "uniqueness", "completeness"],
        "data_rules": ["range", "pattern", "custom"],
        "quality_checks": ["completeness", "accuracy", "consistency", "timeliness", "validity"],
        "severity_levels": ["error", "warning", "info"]
    }

@app.get("/stats")
async def get_stats():
    """Get validation engine statistics"""
    return {
        "service": "data-validation-engine",
        "metrics": {
            "validation_requests_total": VALIDATION_REQUESTS_TOTAL._value.get(),
            "validation_errors_total": VALIDATION_ERRORS_TOTAL._value.get(),
            "data_quality_score": DATA_QUALITY_SCORE._value.get(),
            "transformation_operations_total": TRANSFORMATION_OPERATIONS._value.get()
        }
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global database_connection

    logger.info("Data Validation Engine starting up...")

    # Setup database connection
    try:
        db_config = {
            "host": os.getenv("POSTGRES_HOST", "postgresql_ingestion"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agentic_ingestion"),
            "user": os.getenv("POSTGRES_USER", "agentic_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "agentic123")
        }

        database_connection = psycopg2.connect(**db_config)
        logger.info("Database connection established")

    except Exception as e:
        logger.warning("Database connection failed", error=str(e))

    # Setup RabbitMQ consumer in background thread
    rabbitmq_thread = threading.Thread(target=setup_rabbitmq, daemon=True)
    rabbitmq_thread.start()

    logger.info("Data Validation Engine startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global database_connection

    logger.info("Data Validation Engine shutting down...")

    # Close database connection
    if database_connection:
        database_connection.close()

    logger.info("Data Validation Engine shutdown complete")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "data_validation_engine:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8088)),
        reload=False,
        log_level="info"
    )
