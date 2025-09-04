#!/usr/bin/env python3
"""
JSON Ingestion Service for Agentic Platform

This service processes JSON files and API responses with the following capabilities:
- JSON schema validation
- Nested object flattening
- Data type inference
- Array handling
- Error handling and recovery
- Performance monitoring
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pika
import pandas as pd
import psycopg2
import structlog
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from sqlalchemy import create_engine, text
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
    title="JSON Ingestion Service",
    description="Processes JSON files and API responses with schema validation",
    version="1.0.0"
)

# Prometheus metrics
JSON_FILES_PROCESSED = Counter('json_files_processed_total', 'Total JSON files processed', ['status'])
JSON_RECORDS_PROCESSED = Counter('json_records_processed_total', 'Total records processed')
JSON_PROCESSING_TIME = Histogram('json_processing_duration_seconds', 'JSON processing duration', ['operation'])
JSON_VALIDATION_ERRORS = Counter('json_validation_errors_total', 'Total validation errors', ['error_type'])
ACTIVE_JSON_JOBS = Gauge('active_json_jobs', 'Number of active JSON processing jobs')

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://agentic_user:agentic123@postgresql_ingestion:5432/agentic_ingestion")
engine = create_engine(DATABASE_URL)

# Message queue connection
rabbitmq_connection = None
rabbitmq_channel = None

class JSONProcessor:
    """JSON file processor with schema validation and flattening"""

    def __init__(self):
        self.db_connection = None

    def connect_db(self):
        """Connect to database"""
        try:
            self.db_connection = psycopg2.connect(DATABASE_URL)
            logger.info("Database connection established")
        except Exception as e:
            logger.error("Failed to connect to database", error=str(e))
            raise

    def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status in database"""
        try:
            with self.db_connection.cursor() as cursor:
                update_query = """
                UPDATE ingestion_jobs
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                """

                params = [status]

                if 'total_records' in kwargs:
                    update_query += ", total_records = %s"
                    params.append(kwargs['total_records'])

                if 'processed_records' in kwargs:
                    update_query += ", processed_records = %s"
                    params.append(kwargs['processed_records'])

                if 'failed_records' in kwargs:
                    update_query += ", failed_records = %s"
                    params.append(kwargs['failed_records'])

                if 'error_message' in kwargs:
                    update_query += ", metadata = metadata || %s"
                    params.append(json.dumps({"error_message": kwargs['error_message']}))

                if status == 'processing':
                    update_query += ", start_time = CURRENT_TIMESTAMP"
                elif status in ['completed', 'failed']:
                    update_query += ", end_time = CURRENT_TIMESTAMP"

                update_query += " WHERE job_id = %s"
                params.append(job_id)

                cursor.execute(update_query, params)
                self.db_connection.commit()

                logger.info("Job status updated", job_id=job_id, status=status)

        except Exception as e:
            logger.error("Failed to update job status", error=str(e), job_id=job_id)
            if self.db_connection:
                self.db_connection.rollback()

    def validate_json_structure(self, data: Any) -> Dict[str, Any]:
        """Validate JSON structure and extract schema information"""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "schema_info": {}
        }

        try:
            if data is None:
                validation_results["is_valid"] = False
                validation_results["errors"].append("JSON data is null")
                return validation_results

            # Determine data type
            if isinstance(data, dict):
                validation_results["schema_info"]["data_type"] = "object"
                validation_results["schema_info"]["field_count"] = len(data)
                validation_results["schema_info"]["fields"] = list(data.keys())
                self._analyze_object_structure(data, validation_results)

            elif isinstance(data, list):
                validation_results["schema_info"]["data_type"] = "array"
                validation_results["schema_info"]["array_length"] = len(data)
                if data:
                    validation_results["schema_info"]["element_type"] = type(data[0]).__name__
                    self._analyze_array_structure(data, validation_results)
                else:
                    validation_results["warnings"].append("JSON array is empty")

            else:
                validation_results["schema_info"]["data_type"] = "primitive"
                validation_results["schema_info"]["primitive_type"] = type(data).__name__

            # Check for deeply nested structures
            max_depth = self._calculate_nesting_depth(data)
            if max_depth > 10:
                validation_results["warnings"].append(f"Deeply nested structure detected (depth: {max_depth})")

        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"JSON validation error: {str(e)}")

        return validation_results

    def _analyze_object_structure(self, obj: Dict[str, Any], results: Dict[str, Any]):
        """Analyze object structure for field types and patterns"""
        field_types = {}
        nested_objects = 0
        arrays = 0

        for key, value in obj.items():
            field_types[key] = type(value).__name__

            if isinstance(value, dict):
                nested_objects += 1
            elif isinstance(value, list):
                arrays += 1

        results["schema_info"]["field_types"] = field_types
        results["schema_info"]["nested_objects"] = nested_objects
        results["schema_info"]["arrays"] = arrays

    def _analyze_array_structure(self, arr: List[Any], results: Dict[str, Any]):
        """Analyze array structure for consistency"""
        if not arr:
            return

        first_element_type = type(arr[0]).__name__
        type_consistency = all(type(item).__name__ == first_element_type for item in arr)

        if not type_consistency:
            results["warnings"].append("Array contains mixed data types")

        # Sample a few elements for structure analysis
        sample_size = min(5, len(arr))
        element_samples = []

        for i in range(sample_size):
            if isinstance(arr[i], dict):
                element_samples.append({
                    "type": "object",
                    "fields": list(arr[i].keys())
                })
            else:
                element_samples.append({
                    "type": type(arr[i]).__name__,
                    "value": str(arr[i])[:50]  # Truncate for display
                })

        results["schema_info"]["element_samples"] = element_samples

    def _calculate_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of JSON structure"""
        if current_depth > 20:  # Prevent infinite recursion
            return current_depth

        if isinstance(data, dict):
            return max((self._calculate_nesting_depth(value, current_depth + 1) for value in data.values()), default=current_depth)
        elif isinstance(data, list):
            return max((self._calculate_nesting_depth(item, current_depth + 1) for item in data), default=current_depth)
        else:
            return current_depth

    def flatten_json(self, data: Any, config: Dict[str, Any]) -> pd.DataFrame:
        """Flatten JSON data into tabular format"""
        flatten_config = config.get("flatten_config", {})

        try:
            if isinstance(data, dict):
                # Single object - convert to single row
                flattened = self._flatten_object(data, flatten_config)
                return pd.DataFrame([flattened])

            elif isinstance(data, list):
                # Array of objects - convert to multiple rows
                flattened_rows = []
                for item in data:
                    if isinstance(item, dict):
                        flattened = self._flatten_object(item, flatten_config)
                        flattened_rows.append(flattened)
                    else:
                        # Handle primitive arrays
                        flattened_rows.append({"value": item})

                return pd.DataFrame(flattened_rows)

            else:
                # Primitive value
                return pd.DataFrame([{"value": data}])

        except Exception as e:
            logger.error("JSON flattening failed", error=str(e))
            raise

    def _flatten_object(self, obj: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a JSON object using dot notation for nested fields"""
        flattened = {}
        max_depth = config.get("max_depth", 5)

        def _flatten_recursive(data: Any, prefix: str = "", depth: int = 0):
            if depth >= max_depth:
                flattened[prefix.rstrip(".")] = str(data)
                return

            if isinstance(data, dict):
                if not data:  # Empty dict
                    flattened[prefix.rstrip(".")] = None
                else:
                    for key, value in data.items():
                        new_prefix = f"{prefix}{key}."
                        _flatten_recursive(value, new_prefix, depth + 1)
            elif isinstance(data, list):
                if not data:  # Empty list
                    flattened[prefix.rstrip(".")] = None
                else:
                    # Store array as JSON string or expand if small
                    if len(data) <= 10 and config.get("expand_arrays", False):
                        for i, item in enumerate(data):
                            new_prefix = f"{prefix}{i}."
                            _flatten_recursive(item, new_prefix, depth + 1)
                    else:
                        flattened[prefix.rstrip(".")] = json.dumps(data)
            else:
                flattened[prefix.rstrip(".")] = data

        _flatten_recursive(obj)
        return flattened

    def process_json_file(self, job_id: str, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a JSON file"""
        start_time = time.time()
        ACTIVE_JSON_JOBS.inc()

        try:
            logger.info("Starting JSON processing", job_id=job_id, file_path=file_path)

            # Update job status to processing
            self.update_job_status(job_id, "processing")

            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate JSON structure
            validation_results = self.validate_json_structure(data)

            if not validation_results["is_valid"]:
                error_msg = "; ".join(validation_results["errors"])
                self.update_job_status(job_id, "failed", error_message=error_msg)
                JSON_FILES_PROCESSED.labels(status="failed").inc()
                return {"status": "failed", "error": error_msg}

            # Flatten JSON data
            df = self.flatten_json(data, config)

            JSON_RECORDS_PROCESSED.inc(len(df))

            # Store validation results
            self.store_validation_results(job_id, validation_results)

            # Process data in batches
            batch_size = config.get("batch_size", 1000)
            total_processed = 0
            total_failed = 0

            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                try:
                    processed_count = self.process_batch(job_id, batch_df, config)
                    total_processed += processed_count
                except Exception as e:
                    logger.error("Batch processing failed", error=str(e), batch=i)
                    total_failed += len(batch_df)
                    JSON_VALIDATION_ERRORS.labels(error_type="batch_processing").inc()

            # Update final status
            if total_failed == 0:
                self.update_job_status(
                    job_id,
                    "completed",
                    total_records=len(df),
                    processed_records=total_processed
                )
                JSON_FILES_PROCESSED.labels(status="success").inc()
                status = "completed"
            else:
                self.update_job_status(
                    job_id,
                    "completed_with_errors",
                    total_records=len(df),
                    processed_records=total_processed,
                    failed_records=total_failed
                )
                JSON_FILES_PROCESSED.labels(status="partial").inc()
                status = "completed_with_errors"

            processing_time = time.time() - start_time
            JSON_PROCESSING_TIME.labels(operation="full_process").observe(processing_time)

            logger.info("JSON processing completed",
                       job_id=job_id,
                       status=status,
                       total_records=len(df),
                       processed=total_processed,
                       failed=total_failed,
                       duration=processing_time)

            return {
                "status": status,
                "total_records": len(df),
                "processed_records": total_processed,
                "failed_records": total_failed,
                "processing_time": processing_time,
                "validation_results": validation_results
            }

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"JSON processing failed: {str(e)}"

            self.update_job_status(job_id, "failed", error_message=error_msg)
            JSON_FILES_PROCESSED.labels(status="failed").inc()
            JSON_PROCESSING_TIME.labels(operation="full_process").observe(processing_time)

            logger.error("JSON processing failed",
                        job_id=job_id,
                        error=str(e),
                        duration=processing_time)

            return {"status": "failed", "error": error_msg}

        finally:
            ACTIVE_JSON_JOBS.dec()

    def store_validation_results(self, job_id: str, validation_results: Dict[str, Any]):
        """Store validation results in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO data_validation_results
                    (job_id, validation_type, is_valid, error_message, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    job_id,
                    "json_structure_validation",
                    validation_results["is_valid"],
                    "; ".join(validation_results["errors"]) if validation_results["errors"] else None,
                    json.dumps(validation_results)
                ))
                self.db_connection.commit()

        except Exception as e:
            logger.error("Failed to store validation results", error=str(e), job_id=job_id)

    def process_batch(self, job_id: str, batch_df: pd.DataFrame, config: Dict[str, Any]) -> int:
        """Process a batch of JSON data"""
        # This is where you would implement the actual data processing logic
        # For now, just simulate processing
        time.sleep(0.01)  # Simulate processing time
        return len(batch_df)

# Global processor instance
processor = JSONProcessor()

def setup_rabbitmq():
    """Setup RabbitMQ connection and consumer"""
    global rabbitmq_connection, rabbitmq_channel

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

        rabbitmq_connection = pika.BlockingConnection(parameters)
        rabbitmq_channel = rabbitmq_connection.channel()

        # Declare queue
        rabbitmq_channel.queue_declare(queue='json_ingestion', durable=True)

        # Set up consumer
        rabbitmq_channel.basic_qos(prefetch_count=1)
        rabbitmq_channel.basic_consume(
            queue='json_ingestion',
            on_message_callback=process_message
        )

        logger.info("RabbitMQ consumer setup completed")
        rabbitmq_channel.start_consuming()

    except Exception as e:
        logger.error("Failed to setup RabbitMQ consumer", error=str(e))
        raise

def process_message(ch, method, properties, body):
    """Process incoming RabbitMQ message"""
    try:
        message = json.loads(body)
        job_id = message["job_id"]
        source_config = message["source_config"]

        logger.info("Received JSON ingestion job", job_id=job_id)

        # Process the JSON file
        file_path = source_config["connection_config"]["file_path"]
        result = processor.process_json_file(job_id, file_path, source_config)

        logger.info("JSON ingestion job completed", job_id=job_id, result=result)

        # Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.error("Failed to process message", error=str(e))
        # Negative acknowledge - requeue message
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "json-ingestion-service",
        "timestamp": datetime.utcnow().isoformat(),
        "active_jobs": ACTIVE_JSON_JOBS._value.get()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

# Startup and shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("JSON Ingestion Service starting up...")

    # Connect to database
    processor.connect_db()

    # Setup RabbitMQ consumer in background thread
    import threading
    rabbitmq_thread = threading.Thread(target=setup_rabbitmq, daemon=True)
    rabbitmq_thread.start()

    logger.info("JSON Ingestion Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("JSON Ingestion Service shutting down...")

    # Close database connection
    if processor.db_connection:
        processor.db_connection.close()

    # Close RabbitMQ connection
    if rabbitmq_connection:
        rabbitmq_connection.close()

    logger.info("JSON Ingestion Service shutdown complete")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "json_ingestion_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8085)),
        reload=False,
        log_level="info"
    )
