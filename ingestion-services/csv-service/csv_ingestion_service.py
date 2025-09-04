#!/usr/bin/env python3
"""
CSV Ingestion Service for Agentic Platform

This service processes CSV files with the following capabilities:
- Automatic data type detection
- Schema validation
- Data quality checks
- Duplicate detection
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
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
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

# Create custom registry for CSV service to avoid metric name conflicts
CSV_REGISTRY = CollectorRegistry()

# FastAPI app
app = FastAPI(
    title="CSV Ingestion Service",
    description="Processes CSV files with data validation and quality checks",
    version="1.0.0"
)

# Prometheus metrics - using custom registry to avoid conflicts
CSV_FILES_PROCESSED = Counter('csv_files_processed_total', 'Total CSV files processed', ['status'], registry=CSV_REGISTRY)
CSV_RECORDS_PROCESSED = Counter('csv_records_processed_total', 'Total records processed', registry=CSV_REGISTRY)
CSV_PROCESSING_TIME = Histogram('csv_processing_duration_seconds', 'CSV processing duration', ['operation'], registry=CSV_REGISTRY)
CSV_VALIDATION_ERRORS = Counter('csv_validation_errors_total', 'Total validation errors', ['error_type'], registry=CSV_REGISTRY)
ACTIVE_CSV_JOBS = Gauge('active_csv_jobs', 'Number of active CSV processing jobs', registry=CSV_REGISTRY)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://agentic_user:agentic123@postgresql_ingestion:5432/agentic_ingestion")
engine = create_engine(DATABASE_URL)

# Message queue connection
rabbitmq_connection = None
rabbitmq_channel = None

class CSVProcessor:
    """CSV file processor with validation and quality checks"""

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
                    params.append(job_id)
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

    def validate_csv_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate CSV structure and data quality"""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }

        try:
            # Check for empty dataframe
            if df.empty:
                validation_results["is_valid"] = False
                validation_results["errors"].append("CSV file is empty")
                return validation_results

            # Check for duplicate columns
            if df.columns.duplicated().any():
                validation_results["warnings"].append("Duplicate column names found")

            # Basic statistics
            validation_results["statistics"] = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "duplicate_rows": df.duplicated().sum()
            }

            # Check for high null percentage
            null_percentages = (df.isnull().sum() / len(df)) * 100
            high_null_columns = null_percentages[null_percentages > 50].index.tolist()
            if high_null_columns:
                validation_results["warnings"].append(
                    f"High null percentage in columns: {high_null_columns}"
                )

            # Data type inference validation
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if numeric columns are stored as strings
                    try:
                        pd.to_numeric(df[col], errors='coerce')
                        validation_results["warnings"].append(
                            f"Column '{col}' contains numeric data but is stored as text"
                        )
                    except:
                        pass

        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")

        return validation_results

    def process_csv_file(self, job_id: str, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a CSV file"""
        start_time = time.time()
        ACTIVE_CSV_JOBS.inc()

        try:
            logger.info("Starting CSV processing", job_id=job_id, file_path=file_path)

            # Update job status to processing
            self.update_job_status(job_id, "processing")

            # Read CSV file with error handling
            read_options = config.get("read_options", {})
            df = pd.read_csv(file_path, **read_options)

            CSV_RECORDS_PROCESSED.inc(len(df))

            # Validate CSV structure
            validation_results = self.validate_csv_structure(df)

            if not validation_results["is_valid"]:
                error_msg = "; ".join(validation_results["errors"])
                self.update_job_status(
                    job_id,
                    "failed",
                    error_message=error_msg
                )
                CSV_FILES_PROCESSED.labels(status="failed").inc()
                return {"status": "failed", "error": error_msg}

            # Store validation results in database
            self.store_validation_results(job_id, validation_results)

            # Process data in batches if specified
            batch_size = config.get("batch_size", 1000)
            total_processed = 0
            total_failed = 0

            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                try:
                    # Process batch (this would include transformation and storage logic)
                    processed_count = self.process_batch(job_id, batch_df, config)
                    total_processed += processed_count

                    # Update progress
                    self.update_job_status(
                        job_id,
                        "processing",
                        processed_records=total_processed
                    )

                except Exception as e:
                    logger.error("Batch processing failed", error=str(e), batch=i)
                    total_failed += len(batch_df)
                    CSV_VALIDATION_ERRORS.labels(error_type="batch_processing").inc()

            # Update final status
            if total_failed == 0:
                self.update_job_status(
                    job_id,
                    "completed",
                    total_records=len(df),
                    processed_records=total_processed
                )
                CSV_FILES_PROCESSED.labels(status="success").inc()
                status = "completed"
            else:
                self.update_job_status(
                    job_id,
                    "completed_with_errors",
                    total_records=len(df),
                    processed_records=total_processed,
                    failed_records=total_failed
                )
                CSV_FILES_PROCESSED.labels(status="partial").inc()
                status = "completed_with_errors"

            processing_time = time.time() - start_time
            CSV_PROCESSING_TIME.labels(operation="full_process").observe(processing_time)

            logger.info("CSV processing completed",
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
            error_msg = f"CSV processing failed: {str(e)}"

            self.update_job_status(
                job_id,
                "failed",
                error_message=error_msg
            )

            CSV_FILES_PROCESSED.labels(status="failed").inc()
            CSV_PROCESSING_TIME.labels(operation="full_process").observe(processing_time)

            logger.error("CSV processing failed",
                        job_id=job_id,
                        error=str(e),
                        duration=processing_time)

            return {"status": "failed", "error": error_msg}

        finally:
            ACTIVE_CSV_JOBS.dec()

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
                    "csv_structure_validation",
                    validation_results["is_valid"],
                    "; ".join(validation_results["errors"]) if validation_results["errors"] else None,
                    json.dumps(validation_results)
                ))
                self.db_connection.commit()

        except Exception as e:
            logger.error("Failed to store validation results", error=str(e), job_id=job_id)

    def process_batch(self, job_id: str, batch_df: pd.DataFrame, config: Dict[str, Any]) -> int:
        """Process a batch of CSV data"""
        # This is where you would implement the actual data processing logic
        # For now, just simulate processing
        time.sleep(0.01)  # Simulate processing time
        return len(batch_df)

# Global processor instance
processor = CSVProcessor()

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
        rabbitmq_channel.queue_declare(queue='csv_ingestion', durable=True)

        # Set up consumer
        rabbitmq_channel.basic_qos(prefetch_count=1)
        rabbitmq_channel.basic_consume(
            queue='csv_ingestion',
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

        logger.info("Received CSV ingestion job", job_id=job_id)

        # Process the CSV file
        file_path = source_config["connection_config"]["file_path"]
        result = processor.process_csv_file(job_id, file_path, source_config)

        logger.info("CSV ingestion job completed", job_id=job_id, result=result)

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
        "service": "csv-ingestion-service",
        "timestamp": datetime.utcnow().isoformat(),
        "active_jobs": ACTIVE_CSV_JOBS._value.get()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(CSV_REGISTRY)

# Startup and shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("CSV Ingestion Service starting up...")

    # Connect to database
    processor.connect_db()

    # Setup RabbitMQ consumer in background thread
    import threading
    rabbitmq_thread = threading.Thread(target=setup_rabbitmq, daemon=True)
    rabbitmq_thread.start()

    logger.info("CSV Ingestion Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("CSV Ingestion Service shutting down...")

    # Close database connection
    if processor.db_connection:
        processor.db_connection.close()

    # Close RabbitMQ connection
    if rabbitmq_connection:
        rabbitmq_connection.close()

    logger.info("CSV Ingestion Service shutdown complete")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "csv_ingestion_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=False,
        log_level="info"
    )
