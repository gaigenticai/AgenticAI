#!/usr/bin/env python3
"""
Metadata Management Service for Agentic Platform

This service provides intelligent data profiling and metadata management including:
- Automatic data profiling and statistical analysis
- Metadata cataloging and discovery
- Data lineage tracking
- Schema evolution monitoring
- Data classification and tagging
- Metadata search and retrieval
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pika
import pandas as pd
import psycopg2
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, generate_latest
import uvicorn

# Configure structured logging
logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
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
    title="Metadata Management Service",
    description="Intelligent data profiling and metadata management service",
    version="1.0.0"
)

# Prometheus metrics
METADATA_PROFILES_TOTAL = Counter('metadata_profiles_total', 'Total metadata profiles generated')
METADATA_CATALOG_SIZE = Gauge('metadata_catalog_size', 'Size of metadata catalog')
METADATA_SEARCH_QUERIES = Counter('metadata_search_queries_total', 'Total metadata search queries')

# Global variables
database_connection = None
message_queue_channel = None

# Pydantic models
class MetadataProfile(BaseModel):
    """Metadata profile model"""
    dataset_id: str
    dataset_name: str
    profile_data: Dict[str, Any]
    statistical_summary: Dict[str, Any]
    data_quality_metrics: Dict[str, Any]
    generated_at: datetime

class MetadataEntry(BaseModel):
    """Metadata entry model"""
    source_id: str
    table_name: str
    column_name: str
    data_type: str
    description: Optional[str]
    tags: List[str] = []
    sensitivity_level: str = "public"
    last_updated: datetime

class MetadataManager:
    """Metadata management and profiling engine"""

    def __init__(self):
        self.profiling_cache = {}

    def generate_profile(self, data: Union[List[Dict[str, Any]], pd.DataFrame],
                        dataset_name: str) -> MetadataProfile:
        """Generate comprehensive metadata profile for dataset"""

        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        profile_data = {
            "dataset_name": dataset_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": {}
        }

        # Analyze each column
        for column in df.columns:
            column_profile = self._analyze_column(df[column])
            profile_data["columns"][column] = column_profile

        # Statistical summary
        statistical_summary = {
            "numeric_columns": len([c for c in df.columns if df[c].dtype in ['int64', 'float64']]),
            "categorical_columns": len([c for c in df.columns if df[c].dtype == 'object']),
            "datetime_columns": len([c for c in df.columns if 'datetime' in str(df[c].dtype)]),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
        }

        # Data quality metrics
        data_quality_metrics = {
            "completeness": float((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
            "uniqueness": float(df.nunique().sum() / len(df.columns)),
            "consistency": self._calculate_consistency_score(df)
        }

        METADATA_PROFILES_TOTAL.inc()

        return MetadataProfile(
            dataset_id=f"{dataset_name}_{int(time.time())}",
            dataset_name=dataset_name,
            profile_data=profile_data,
            statistical_summary=statistical_summary,
            data_quality_metrics=data_quality_metrics,
            generated_at=datetime.utcnow()
        )

    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze individual column characteristics"""
        analysis = {
            "data_type": str(series.dtype),
            "null_count": int(series.isnull().sum()),
            "null_percentage": float(series.isnull().sum() / len(series) * 100),
            "unique_count": int(series.nunique()),
            "unique_percentage": float(series.nunique() / len(series) * 100)
        }

        # Type-specific analysis
        if series.dtype in ['int64', 'float64']:
            analysis.update({
                "min": float(series.min()) if not series.empty else None,
                "max": float(series.max()) if not series.empty else None,
                "mean": float(series.mean()) if not series.empty else None,
                "median": float(series.median()) if not series.empty else None,
                "std": float(series.std()) if not series.empty else None
            })
        elif series.dtype == 'object':
            non_null_values = series.dropna()
            if len(non_null_values) > 0:
                analysis.update({
                    "most_common": non_null_values.value_counts().head(5).to_dict(),
                    "avg_length": float(non_null_values.str.len().mean()),
                    "contains_digits": bool(non_null_values.str.contains(r'\d').any()),
                    "contains_alpha": bool(non_null_values.str.contains(r'[a-zA-Z]').any())
                })

        return analysis

    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        consistency_factors = []

        # Check for consistent data types within columns
        for column in df.columns:
            unique_types = df[column].dropna().apply(type).nunique()
            type_consistency = 1.0 if unique_types <= 1 else 1.0 / unique_types
            consistency_factors.append(type_consistency)

        return float(sum(consistency_factors) / len(consistency_factors)) if consistency_factors else 1.0

    def store_metadata(self, profile: MetadataProfile):
        """Store metadata profile in database"""
        try:
            with database_connection.cursor() as cursor:
                # Store profile summary
                cursor.execute("""
                    INSERT INTO metadata_catalog
                    (source_id, table_name, column_name, data_type, description, tags, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    profile.dataset_id,
                    profile.dataset_name,
                    'PROFILE_SUMMARY',
                    'jsonb',
                    f"Metadata profile for {profile.dataset_name}",
                    ['profile', 'summary']
                ))

                # Store column metadata
                for column_name, column_data in profile.profile_data.get("columns", {}).items():
                    cursor.execute("""
                        INSERT INTO metadata_catalog
                        (source_id, table_name, column_name, data_type, description, tags, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (
                        profile.dataset_id,
                        profile.dataset_name,
                        column_name,
                        column_data.get("data_type", "unknown"),
                        f"Column profile: {column_name}",
                        ['column', 'profile']
                    ))

                database_connection.commit()
                METADATA_CATALOG_SIZE.inc()

        except Exception as e:
            logger.error("Failed to store metadata", error=str(e))
            if database_connection:
                database_connection.rollback()

    def search_metadata(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search metadata catalog"""
        try:
            METADATA_SEARCH_QUERIES.inc()

            with database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                sql = """
                    SELECT * FROM metadata_catalog
                    WHERE table_name ILIKE %s OR column_name ILIKE %s OR description ILIKE %s
                """
                search_pattern = f"%{query}%"
                params = [search_pattern, search_pattern, search_pattern]

                # Add filters if provided
                if filters:
                    if "data_type" in filters:
                        sql += " AND data_type = %s"
                        params.append(filters["data_type"])
                    if "sensitivity_level" in filters:
                        sql += " AND sensitivity_level = %s"
                        params.append(filters["sensitivity_level"])

                sql += " ORDER BY created_at DESC LIMIT 50"

                cursor.execute(sql, params)
                results = cursor.fetchall()

                return [dict(row) for row in results]

        except Exception as e:
            logger.error("Metadata search failed", error=str(e))
            return []

    def get_dataset_lineage(self, dataset_id: str) -> Dict[str, Any]:
        """Get data lineage information for a dataset"""
        try:
            with database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM data_lineage
                    WHERE source_record_id = %s OR target_record_id = %s
                    ORDER BY created_at DESC
                """, (dataset_id, dataset_id))

                lineage_records = cursor.fetchall()

                return {
                    "dataset_id": dataset_id,
                    "lineage_records": [dict(record) for record in lineage_records],
                    "upstream_sources": [
                        r for r in lineage_records
                        if r['target_record_id'] == dataset_id
                    ],
                    "downstream_targets": [
                        r for r in lineage_records
                        if r['source_record_id'] == dataset_id
                    ]
                }

        except Exception as e:
            logger.error("Failed to get dataset lineage", error=str(e))
            return {"dataset_id": dataset_id, "lineage_records": [], "error": str(e)}

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

        # Declare queues
        queues = ['metadata_profiling', 'metadata_search']
        for queue in queues:
            message_queue_channel.queue_declare(queue=queue, durable=True)

        # Set up consumer
        message_queue_channel.basic_qos(prefetch_count=1)
        message_queue_channel.basic_consume(
            queue='metadata_profiling',
            on_message_callback=process_metadata_message
        )

        logger.info("RabbitMQ consumer setup completed")
        message_queue_channel.start_consuming()

    except Exception as e:
        logger.error("Failed to setup RabbitMQ consumer", error=str(e))
        raise

def process_metadata_message(ch, method, properties, body):
    """Process incoming metadata message"""
    try:
        message = json.loads(body)
        dataset_name = message["dataset_name"]
        data = message["data"]

        logger.info("Received metadata profiling request", dataset_name=dataset_name)

        # Generate metadata profile
        manager = MetadataManager()
        profile = manager.generate_profile(data, dataset_name)

        # Store profile
        manager.store_metadata(profile)

        logger.info("Metadata profiling completed", dataset_name=dataset_name)

        # Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.error("Failed to process metadata message", error=str(e))
        # Negative acknowledge - requeue message
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "metadata-management",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/profile", response_model=MetadataProfile)
async def generate_profile(
    dataset_name: str,
    data: Union[List[Dict[str, Any]], pd.DataFrame]
):
    """Generate metadata profile for dataset"""
    try:
        manager = MetadataManager()
        profile = manager.generate_profile(data, dataset_name)
        return profile

    except Exception as e:
        logger.error("Profile generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Profile generation failed: {str(e)}")

@app.get("/search")
async def search_metadata(query: str, data_type: Optional[str] = None):
    """Search metadata catalog"""
    try:
        manager = MetadataManager()
        filters = {"data_type": data_type} if data_type else None
        results = manager.search_metadata(query, filters)

        return {"query": query, "results": results, "count": len(results)}

    except Exception as e:
        logger.error("Metadata search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/lineage/{dataset_id}")
async def get_lineage(dataset_id: str):
    """Get data lineage for dataset"""
    try:
        manager = MetadataManager()
        lineage = manager.get_dataset_lineage(dataset_id)
        return lineage

    except Exception as e:
        logger.error("Lineage retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lineage retrieval failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get metadata service statistics"""
    return {
        "service": "metadata-management",
        "metrics": {
            "profiles_generated": METADATA_PROFILES_TOTAL._value.get(),
            "catalog_size": METADATA_CATALOG_SIZE._value.get(),
            "search_queries": METADATA_SEARCH_QUERIES._value.get()
        }
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global database_connection

    logger.info("Metadata Management Service starting up...")

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

    logger.info("Metadata Management Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global database_connection

    logger.info("Metadata Management Service shutting down...")

    # Close database connection
    if database_connection:
        database_connection.close()

    logger.info("Metadata Management Service shutdown complete")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "metadata_management:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8089)),
        reload=False,
        log_level="info"
    )
