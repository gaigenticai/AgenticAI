#!/usr/bin/env python3
"""
PostgreSQL Output Service for Agentic Platform

This service handles data output to PostgreSQL databases with advanced features:
- Bulk insert operations for performance
- Automatic schema management and table creation
- Data type mapping and conversion
- Transaction management with rollback support
- Connection pooling and health monitoring
- Comprehensive error handling and recovery
- Performance optimization and query tuning
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pika
import psycopg2
import psycopg2.extras
import psycopg2.pool
import structlog
from fastapi import FastAPI
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
    title="PostgreSQL Output Service",
    description="Handles data output to PostgreSQL databases with advanced performance features",
    version="1.0.0"
)

# Prometheus metrics
POSTGRESQL_CONNECTIONS_ACTIVE = Gauge('postgresql_connections_active', 'Active PostgreSQL connections')
POSTGRESQL_RECORDS_INSERTED = Counter('postgresql_records_inserted_total', 'Total records inserted', ['table_name'])
POSTGRESQL_BATCH_DURATION = Histogram('postgresql_batch_duration_seconds', 'Batch insert duration', ['table_name'])
POSTGRESQL_ERRORS = Counter('postgresql_errors_total', 'Total PostgreSQL errors', ['error_type'])
POSTGRESQL_CONNECTION_POOL_USAGE = Gauge('postgresql_connection_pool_usage', 'Connection pool usage percentage')

# Global variables
connection_pool = None
message_queue_channel = None

class PostgreSQLWriter:
    """PostgreSQL writer with advanced features"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self.table_name = config.get("table_name", "output_data")
        self.batch_size = config.get("batch_size", 1000)
        self.create_table_if_not_exists = config.get("create_table", True)
        self.schema_name = config.get("schema_name", "public")

    def connect(self):
        """Establish connection to PostgreSQL"""
        try:
            if connection_pool:
                self.connection = connection_pool.getconn()
            else:
                self.connection = psycopg2.connect(**self.config)
            self.connection.autocommit = False  # Use transactions
            POSTGRESQL_CONNECTIONS_ACTIVE.inc()
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL", error=str(e))
            raise

    def disconnect(self):
        """Close database connection"""
        if self.connection:
            try:
                if connection_pool:
                    connection_pool.putconn(self.connection)
                else:
                    self.connection.close()
                POSTGRESQL_CONNECTIONS_ACTIVE.dec()
            except Exception as e:
                logger.warning("Error closing connection", error=str(e))

    def ensure_table_exists(self, sample_data: Dict[str, Any]):
        """Create table if it doesn't exist based on sample data"""
        if not self.create_table_if_not_exists:
            return

        try:
            with self.connection.cursor() as cursor:
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = %s AND table_name = %s
                    )
                """, (self.schema_name, self.table_name))

                if not cursor.fetchone()[0]:
                    # Create table based on sample data
                    create_table_sql = self._generate_create_table_sql(sample_data)
                    cursor.execute(create_table_sql)
                    logger.info("Created table", table=self.table_name, schema=self.schema_name)

        except Exception as e:
            logger.error("Failed to create table", error=str(e), table=self.table_name)
            raise

    def _generate_create_table_sql(self, sample_data: Dict[str, Any]) -> str:
        """Generate CREATE TABLE SQL based on sample data"""
        columns = []

        # Add ID column as primary key
        columns.append("id SERIAL PRIMARY KEY")

        # Add timestamp columns
        columns.append("created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP")
        columns.append("updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP")

        # Add data columns based on sample
        for key, value in sample_data.items():
            if key in ['id', 'created_at', 'updated_at']:
                continue  # Skip reserved columns

            column_type = self._infer_sql_type(value)
            columns.append(f'"{key}" {column_type}')

        # Add JSON column for unstructured data
        columns.append("metadata JSONB")

        table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.table_name} (
            {', '.join(columns)}
        )
        """

        return table_sql

    def _infer_sql_type(self, value: Any) -> str:
        """Infer SQL column type from Python value"""
        if isinstance(value, bool):
            return "BOOLEAN"
        elif isinstance(value, int):
            return "BIGINT"
        elif isinstance(value, float):
            return "DOUBLE PRECISION"
        elif isinstance(value, str):
            # Estimate string length
            if len(value) > 1000:
                return "TEXT"
            elif len(value) > 255:
                return "VARCHAR(1000)"
            else:
                return "VARCHAR(255)"
        elif isinstance(value, (list, dict)):
            return "JSONB"
        elif isinstance(value, datetime):
            return "TIMESTAMP WITH TIME ZONE"
        else:
            return "TEXT"  # Default fallback

    def insert_batch(self, data_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert a batch of records with optimized performance"""
        if not data_batch:
            return {"records_inserted": 0}

        start_time = time.time()

        try:
            # Ensure table exists
            if data_batch:
                self.ensure_table_exists(data_batch[0])

            # Prepare data for bulk insert
            columns = self._get_columns_from_data(data_batch[0])
            column_names = [f'"{col}"' for col in columns]

            # Build INSERT statement
            placeholders = ", ".join(["%s"] * len(columns))
            insert_sql = f"""
            INSERT INTO {self.schema_name}.{self.table_name}
            ({', '.join(column_names)}, created_at, updated_at, metadata)
            VALUES ({placeholders}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %s)
            """

            # Prepare data values
            values = []
            for record in data_batch:
                row_values = []
                metadata = {}

                for col in columns:
                    if col in record:
                        row_values.append(record[col])
                    else:
                        row_values.append(None)
                        metadata[col] = None

                # Add metadata for any extra fields
                for key, value in record.items():
                    if key not in columns and key not in ['id', 'created_at', 'updated_at']:
                        metadata[key] = value

                row_values.append(json.dumps(metadata))
                values.append(row_values)

            # Execute batch insert
            with self.connection.cursor() as cursor:
                psycopg2.extras.execute_batch(cursor, insert_sql, values)
                self.connection.commit()

            records_count = len(data_batch)
            duration = time.time() - start_time

            POSTGRESQL_RECORDS_INSERTED.labels(table_name=self.table_name).inc(records_count)
            POSTGRESQL_BATCH_DURATION.labels(table_name=self.table_name).observe(duration)

            logger.info("Batch insert completed",
                       table=self.table_name,
                       records=records_count,
                       duration=duration)

            return {
                "records_inserted": records_count,
                "duration_seconds": duration
            }

        except Exception as e:
            logger.error("Batch insert failed", error=str(e), table=self.table_name)
            if self.connection:
                self.connection.rollback()
            POSTGRESQL_ERRORS.labels(error_type="batch_insert").inc()
            raise

    def _get_columns_from_data(self, sample_record: Dict[str, Any]) -> List[str]:
        """Extract column names from sample record"""
        columns = []
        for key in sample_record.keys():
            if key not in ['id', 'created_at', 'updated_at', 'metadata']:
                columns.append(key)
        return columns

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a custom query"""
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params or ())
                results = cursor.fetchall()
                return [dict(row) for row in results]

        except Exception as e:
            logger.error("Query execution failed", error=str(e), query=query)
            POSTGRESQL_ERRORS.labels(error_type="query_execution").inc()
            raise

    def get_table_stats(self) -> Dict[str, Any]:
        """Get statistics about the target table"""
        try:
            with self.connection.cursor() as cursor:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {self.schema_name}.{self.table_name}")
                row_count = cursor.fetchone()[0]

                # Get column information
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (self.schema_name, self.table_name))

                columns = cursor.fetchall()

                return {
                    "table_name": self.table_name,
                    "schema_name": self.schema_name,
                    "row_count": row_count,
                    "columns": [
                        {
                            "name": col[0],
                            "type": col[1],
                            "nullable": col[2] == 'YES'
                        } for col in columns
                    ]
                }

        except Exception as e:
            logger.error("Failed to get table stats", error=str(e), table=self.table_name)
            return {"error": str(e)}

def setup_rabbitmq():
    """Setup RabbitMQ connection and consumer"""
    global message_queue_channel

    try:
        credentials = pika.PlainCredentials(
            os.getenv("RABBITMQ_USER", "agentic_user"),
            os.getenv("RABBITMQ_PASSWORD", "")
        )
        parameters = pika.ConnectionParameters(
            host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
            port=int(os.getenv("RABBITMQ_PORT", 5672)),
            credentials=credentials
        )

        connection = pika.BlockingConnection(parameters)
        message_queue_channel = connection.channel()

        # Declare queue
        message_queue_channel.queue_declare(queue='postgresql_output', durable=True)

        # Set up consumer
        message_queue_channel.basic_qos(prefetch_count=1)
        message_queue_channel.basic_consume(
            queue='postgresql_output',
            on_message_callback=process_message
        )

        logger.info("RabbitMQ consumer setup completed")
        message_queue_channel.start_consuming()

    except Exception as e:
        logger.error("Failed to setup RabbitMQ consumer", error=str(e))
        raise

def process_message(ch, method, properties, body):
    """Process incoming RabbitMQ message"""
    try:
        message = json.loads(body)
        job_id = message["job_id"]
        target_config = message["target_config"]
        data = message["data"]

        logger.info("Received PostgreSQL output job", job_id=job_id)

        # Create writer and process data
        writer = PostgreSQLWriter(target_config)
        writer.connect()

        try:
            if isinstance(data, list):
                # Process in batches
                batch_size = target_config.get("batch_size", 1000)
                total_processed = 0

                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    result = writer.insert_batch(batch)
                    total_processed += result["records_inserted"]

                result = {"records_inserted": total_processed}
            else:
                # Single record
                result = writer.insert_batch([data])

            logger.info("PostgreSQL output job completed",
                       job_id=job_id,
                       result=result)

        finally:
            writer.disconnect()

        # Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.error("Failed to process PostgreSQL output message", error=str(e))
        # Negative acknowledge - requeue message
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    pool_status = "healthy" if connection_pool else "not_initialized"

    return {
        "status": "healthy",
        "service": "postgresql-output-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "connection_pool": pool_status,
        "active_connections": POSTGRESQL_CONNECTIONS_ACTIVE._value.get()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    try:
        # Get connection pool stats if available
        pool_stats = {}
        if connection_pool:
            pool_stats = {
                "minconn": connection_pool.minconn,
                "maxconn": connection_pool.maxconn,
                "idle_connections": len(connection_pool._used) if hasattr(connection_pool, '_used') else 0,
                "used_connections": len(connection_pool._rused) if hasattr(connection_pool, '_rused') else 0
            }

        return {
            "service": "postgresql-output-service",
            "metrics": {
                "active_connections": POSTGRESQL_CONNECTIONS_ACTIVE._value.get(),
                "records_inserted_total": POSTGRESQL_RECORDS_INSERTED._value.get(),
                "errors_total": POSTGRESQL_ERRORS._value.get()
            },
            "connection_pool": pool_stats
        }

    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        return {"error": str(e)}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global connection_pool

    logger.info("PostgreSQL Output Service starting up...")

    # Setup database connection pool
    try:
        db_config = {
            "host": os.getenv("POSTGRES_HOST", "postgresql_output"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agentic_output"),
            "user": os.getenv("POSTGRES_USER", "agentic_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "")
        }

        connection_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=int(os.getenv("POSTGRES_MAX_CONNECTIONS", 10)),
            **db_config
        )

        logger.info("PostgreSQL connection pool initialized")

    except Exception as e:
        logger.error("Failed to initialize connection pool", error=str(e))
        raise

    # Setup RabbitMQ consumer in background thread
    rabbitmq_thread = threading.Thread(target=setup_rabbitmq, daemon=True)
    rabbitmq_thread.start()

    logger.info("PostgreSQL Output Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global connection_pool

    logger.info("PostgreSQL Output Service shutting down...")

    # Close connection pool
    if connection_pool:
        connection_pool.closeall()

    logger.info("PostgreSQL Output Service shutdown complete")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "postgresql_output_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8083)),
        reload=False,
        log_level="info"
    )
