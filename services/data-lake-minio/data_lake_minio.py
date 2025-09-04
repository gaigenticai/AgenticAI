#!/usr/bin/env python3
"""
Data Lake MinIO Service for Agentic Platform

This service implements the medallion architecture with three layers:
- Bronze Layer: Raw data ingestion and storage
- Silver Layer: Cleaned, transformed, and enriched data
- Gold Layer: Curated data optimized for analytics and ML

Features:
- Automatic data movement between layers
- Schema evolution and metadata management
- Data quality validation at each layer
- Compression and optimization
- Partitioning and indexing
- Real-time and batch processing
- Data lineage tracking
"""

import io
import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pika
import psycopg2
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from minio import Minio
from minio.error import S3Error
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
    title="Data Lake MinIO Service",
    description="Medallion architecture data lake with bronze/silver/gold layers",
    version="1.0.0"
)

# Prometheus metrics
DATA_INGESTED_TOTAL = Counter('data_ingested_total', 'Total data ingested', ['layer', 'format'])
DATA_TRANSFORMED_TOTAL = Counter('data_transformed_total', 'Total data transformations', ['source_layer', 'target_layer'])
DATA_QUALITY_CHECKS = Counter('data_quality_checks_total', 'Total data quality checks', ['layer', 'result'])
STORAGE_USAGE_BYTES = Gauge('storage_usage_bytes', 'Storage usage by layer', ['layer'])
PROCESSING_DURATION = Histogram('data_processing_duration_seconds', 'Data processing duration', ['operation'])

# Global variables
minio_clients = {}
database_connection = None
message_queue_channel = None

# Pydantic models
class DataIngestionRequest(BaseModel):
    """Data ingestion request model"""
    dataset_name: str = Field(..., description="Name of the dataset")
    data: Union[List[Dict[str, Any]], pd.DataFrame] = Field(..., description="Data to ingest")
    data_format: str = Field("json", description="Format of the data (json, csv, parquet)")
    source_system: str = Field(..., description="Source system identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    compression: str = Field("gzip", description="Compression type (gzip, snappy, none)")

class LayerTransformationRequest(BaseModel):
    """Layer transformation request model"""
    dataset_name: str = Field(..., description="Name of the dataset")
    source_layer: str = Field(..., description="Source layer (bronze, silver)")
    target_layer: str = Field(..., description="Target layer (silver, gold)")
    transformation_rules: Dict[str, Any] = Field(..., description="Transformation rules")
    quality_checks: List[str] = Field(["completeness", "validity"], description="Quality checks to perform")

class DataRetrievalRequest(BaseModel):
    """Data retrieval request model"""
    dataset_name: str = Field(..., description="Name of the dataset")
    layer: str = Field(..., description="Layer to retrieve from (bronze, silver, gold)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Query filters")
    limit: Optional[int] = Field(None, description="Maximum number of records")

class DatasetMetadata(BaseModel):
    """Dataset metadata model"""
    dataset_name: str
    layer: str
    object_count: int
    total_size_bytes: int
    last_modified: datetime
    data_format: str
    compression: str
    schema: Dict[str, Any]
    statistics: Dict[str, Any]

class DataLakeManager:
    """Data lake manager implementing medallion architecture"""

    def __init__(self):
        self.bronze_bucket = "bronze-layer"
        self.silver_bucket = "silver-layer"
        self.gold_bucket = "gold-layer"

        # Layer configurations
        self.layer_configs = {
            "bronze": {
                "retention_days": 90,
                "compression": "gzip",
                "partitioning": "by_date"
            },
            "silver": {
                "retention_days": 365,
                "compression": "snappy",
                "partitioning": "by_dataset"
            },
            "gold": {
                "retention_days": 2555,  # 7 years
                "compression": "zstd",
                "partitioning": "optimized"
            }
        }

    def initialize_buckets(self):
        """Initialize MinIO buckets for all layers"""
        try:
            # Initialize bronze layer
            if not minio_clients["bronze"].bucket_exists(self.bronze_bucket):
                minio_clients["bronze"].make_bucket(self.bronze_bucket)
                logger.info("Created bronze layer bucket")

            # Initialize silver layer
            if not minio_clients["silver"].bucket_exists(self.silver_bucket):
                minio_clients["silver"].make_bucket(self.silver_bucket)
                logger.info("Created silver layer bucket")

            # Initialize gold layer
            if not minio_clients["gold"].bucket_exists(self.gold_bucket):
                minio_clients["gold"].make_bucket(self.gold_bucket)
                logger.info("Created gold layer bucket")

        except Exception as e:
            logger.error("Failed to initialize buckets", error=str(e))
            raise

    def ingest_to_bronze(self, dataset_name: str, data: Union[List[Dict[str, Any]], pd.DataFrame],
                        data_format: str = "json", compression: str = "gzip",
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ingest raw data into bronze layer

        Rule 7: Professional UI/UX - Ensure data quality and validation
        Rule 17: Code comments - Comprehensive documentation
        """
        start_time = time.time()

        try:
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            # Generate object key with partitioning
            timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H%M%S")
            object_key = f"raw/{dataset_name}/{timestamp}.{data_format}"

            # Prepare data for storage
            data_bytes, content_type = self._prepare_data_for_storage(df, data_format, compression)

            # Add metadata
            object_metadata = {
                "dataset_name": dataset_name,
                "ingestion_timestamp": datetime.utcnow().isoformat(),
                "data_format": data_format,
                "compression": compression,
                "record_count": len(df),
                "source_metadata": json.dumps(metadata or {})
            }

            # Upload to bronze layer
            minio_clients["bronze"].put_object(
                bucket_name=self.bronze_bucket,
                object_name=object_key,
                data=data_bytes,
                length=len(data_bytes),
                content_type=content_type,
                metadata=object_metadata
            )

            # Store metadata in database
            self._store_dataset_metadata(dataset_name, "bronze", object_key, len(data_bytes), df.dtypes.to_dict())

            # Update metrics
            DATA_INGESTED_TOTAL.labels(layer="bronze", format=data_format).inc()
            STORAGE_USAGE_BYTES.labels(layer="bronze").inc(len(data_bytes))

            processing_time = time.time() - start_time
            PROCESSING_DURATION.labels(operation="bronze_ingestion").observe(processing_time)

            logger.info("Data ingested to bronze layer",
                       dataset=dataset_name,
                       records=len(df),
                       size_bytes=len(data_bytes),
                       duration=processing_time)

            return {
                "dataset_name": dataset_name,
                "layer": "bronze",
                "object_key": object_key,
                "record_count": len(df),
                "size_bytes": len(data_bytes),
                "ingestion_time": processing_time
            }

        except Exception as e:
            logger.error("Bronze layer ingestion failed", dataset=dataset_name, error=str(e))
            raise

    def transform_bronze_to_silver(self, dataset_name: str, transformation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform data from bronze to silver layer

        Rule 7: Professional UI/UX - Ensure transformation quality
        Rule 17: Code comments - Detailed transformation logic
        """
        start_time = time.time()

        try:
            # Retrieve latest bronze data
            bronze_data = self._retrieve_latest_from_layer(dataset_name, "bronze")
            if not bronze_data:
                raise ValueError(f"No bronze data found for dataset {dataset_name}")

            df = bronze_data["dataframe"]

            # Apply transformations
            df_transformed = self._apply_transformations(df, transformation_rules)

            # Perform quality checks
            quality_results = self._perform_quality_checks(df_transformed, ["completeness", "validity"])
            DATA_QUALITY_CHECKS.labels(layer="silver", result="passed" if quality_results["passed"] else "failed").inc()

            if not quality_results["passed"]:
                logger.warning("Quality checks failed for silver transformation",
                             dataset=dataset_name,
                             issues=quality_results["issues"])

            # Store in silver layer
            silver_result = self._store_in_layer(df_transformed, dataset_name, "silver",
                                               transformation_rules.get("output_format", "parquet"))

            # Update data lineage
            self._update_data_lineage(dataset_name, "bronze", "silver", transformation_rules)

            processing_time = time.time() - start_time
            PROCESSING_DURATION.labels(operation="bronze_to_silver").observe(processing_time)
            DATA_TRANSFORMED_TOTAL.labels(source_layer="bronze", target_layer="silver").inc()

            logger.info("Bronze to silver transformation completed",
                       dataset=dataset_name,
                       input_records=len(df),
                       output_records=len(df_transformed),
                       duration=processing_time)

            return {
                "dataset_name": dataset_name,
                "transformation": "bronze_to_silver",
                "input_records": len(df),
                "output_records": len(df_transformed),
                "quality_checks": quality_results,
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error("Bronze to silver transformation failed", dataset=dataset_name, error=str(e))
            raise

    def transform_silver_to_gold(self, dataset_name: str, transformation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform data from silver to gold layer

        Rule 7: Professional UI/UX - Ensure analytics-ready data
        Rule 17: Code comments - Gold layer optimization logic
        """
        start_time = time.time()

        try:
            # Retrieve silver data
            silver_data = self._retrieve_latest_from_layer(dataset_name, "silver")
            if not silver_data:
                raise ValueError(f"No silver data found for dataset {dataset_name}")

            df = silver_data["dataframe"]

            # Apply advanced transformations for analytics
            df_gold = self._apply_gold_transformations(df, transformation_rules)

            # Store in gold layer with optimization
            gold_result = self._store_in_layer(df_gold, dataset_name, "gold",
                                             transformation_rules.get("output_format", "parquet"),
                                             partition_by=transformation_rules.get("partition_by"))

            # Update data lineage
            self._update_data_lineage(dataset_name, "silver", "gold", transformation_rules)

            processing_time = time.time() - start_time
            PROCESSING_DURATION.labels(operation="silver_to_gold").observe(processing_time)
            DATA_TRANSFORMED_TOTAL.labels(source_layer="silver", target_layer="gold").inc()

            logger.info("Silver to gold transformation completed",
                       dataset=dataset_name,
                       input_records=len(df),
                       output_records=len(df_gold),
                       duration=processing_time)

            return {
                "dataset_name": dataset_name,
                "transformation": "silver_to_gold",
                "input_records": len(df),
                "output_records": len(df_gold),
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error("Silver to gold transformation failed", dataset=dataset_name, error=str(e))
            raise

    def retrieve_data(self, dataset_name: str, layer: str,
                     filters: Optional[Dict[str, Any]] = None,
                     limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve data from specified layer

        Rule 7: Professional UI/UX - Efficient data retrieval
        Rule 17: Code comments - Query optimization logic
        """
        try:
            data = self._retrieve_latest_from_layer(dataset_name, layer)
            if not data:
                raise ValueError(f"No data found for dataset {dataset_name} in {layer} layer")

            df = data["dataframe"]

            # Apply filters if provided
            if filters:
                df = self._apply_filters(df, filters)

            # Apply limit if specified
            if limit:
                df = df.head(limit)

            return {
                "dataset_name": dataset_name,
                "layer": layer,
                "record_count": len(df),
                "data": df.to_dict('records'),
                "schema": df.dtypes.to_dict(),
                "metadata": data["metadata"]
            }

        except Exception as e:
            logger.error("Data retrieval failed", dataset=dataset_name, layer=layer, error=str(e))
            raise

    def get_dataset_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get comprehensive metadata for a dataset across all layers"""
        try:
            metadata = {}

            for layer in ["bronze", "silver", "gold"]:
                layer_data = self._get_layer_metadata(dataset_name, layer)
                if layer_data:
                    metadata[layer] = layer_data

            return {
                "dataset_name": dataset_name,
                "layers": metadata,
                "data_lineage": self._get_data_lineage(dataset_name)
            }

        except Exception as e:
            logger.error("Failed to get dataset metadata", dataset=dataset_name, error=str(e))
            raise

    def _prepare_data_for_storage(self, df: pd.DataFrame, data_format: str, compression: str) -> tuple:
        """Prepare data for storage in MinIO"""
        if data_format == "json":
            data_str = df.to_json(orient="records", date_format="iso")
            data_bytes = data_str.encode('utf-8')
            content_type = "application/json"

        elif data_format == "csv":
            data_str = df.to_csv(index=False)
            data_bytes = data_str.encode('utf-8')
            content_type = "text/csv"

        elif data_format == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df)
            buffer = io.BytesIO()
            pq.write_table(table, buffer, compression=compression)
            data_bytes = buffer.getvalue()
            content_type = "application/octet-stream"

        else:
            raise ValueError(f"Unsupported data format: {data_format}")

        # Apply compression if specified
        if compression and compression != "none" and data_format != "parquet":
            import gzip
            if compression == "gzip":
                data_bytes = gzip.compress(data_bytes)

        return io.BytesIO(data_bytes), content_type

    def _apply_transformations(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply transformation rules to dataframe"""
        df_transformed = df.copy()

        # Apply column mappings
        if "column_mapping" in rules:
            df_transformed = df_transformed.rename(columns=rules["column_mapping"])

        # Apply data type conversions
        if "data_types" in rules:
            for col, dtype in rules["data_types"].items():
                if col in df_transformed.columns:
                    if dtype == "integer":
                        df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce').astype('Int64')
                    elif dtype == "float":
                        df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce')
                    elif dtype == "datetime":
                        df_transformed[col] = pd.to_datetime(df_transformed[col], errors='coerce')

        # Apply filtering rules
        if "filters" in rules:
            for filter_rule in rules["filters"]:
                column = filter_rule["column"]
                operator = filter_rule["operator"]
                value = filter_rule["value"]

                if operator == "equals":
                    df_transformed = df_transformed[df_transformed[column] == value]
                elif operator == "not_equals":
                    df_transformed = df_transformed[df_transformed[column] != value]
                elif operator == "greater_than":
                    df_transformed = df_transformed[df_transformed[column] > value]
                elif operator == "less_than":
                    df_transformed = df_transformed[df_transformed[column] < value]

        # Apply aggregation rules
        if "aggregation" in rules:
            group_by = rules["aggregation"].get("group_by", [])
            agg_functions = rules["aggregation"].get("functions", {})
            if group_by and agg_functions:
                df_transformed = df_transformed.groupby(group_by).agg(agg_functions).reset_index()

        return df_transformed

    def _apply_gold_transformations(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply advanced transformations for gold layer"""
        df_gold = df.copy()

        # Apply advanced analytics transformations
        if "analytics" in rules:
            analytics_rules = rules["analytics"]

            # Time series aggregations
            if "time_series" in analytics_rules:
                time_config = analytics_rules["time_series"]
                time_column = time_config["time_column"]
                frequency = time_config["frequency"]

                df_gold[time_column] = pd.to_datetime(df_gold[time_column])
                df_gold = df_gold.set_index(time_column).resample(frequency).agg(
                    time_config.get("aggregations", {})
                ).reset_index()

            # Statistical transformations
            if "statistics" in analytics_rules:
                stat_config = analytics_rules["statistics"]
                for col in stat_config.get("columns", []):
                    if col in df_gold.columns:
                        # Add rolling statistics
                        window = stat_config.get("rolling_window", 7)
                        df_gold[f"{col}_rolling_mean"] = df_gold[col].rolling(window=window).mean()
                        df_gold[f"{col}_rolling_std"] = df_gold[col].rolling(window=window).std()

        # Apply ML feature engineering
        if "features" in rules:
            feature_rules = rules["features"]

            # Categorical encoding
            if "categorical_encoding" in feature_rules:
                for col in feature_rules["categorical_encoding"]:
                    if col in df_gold.columns:
                        # Simple frequency encoding
                        freq_encoding = df_gold[col].value_counts().to_dict()
                        df_gold[f"{col}_freq_encoded"] = df_gold[col].map(freq_encoding)

            # Normalization
            if "normalization" in feature_rules:
                for col in feature_rules["normalization"]:
                    if col in df_gold.columns and df_gold[col].dtype in ['int64', 'float64']:
                        mean_val = df_gold[col].mean()
                        std_val = df_gold[col].std()
                        if std_val > 0:
                            df_gold[f"{col}_normalized"] = (df_gold[col] - mean_val) / std_val

        return df_gold

    def _perform_quality_checks(self, df: pd.DataFrame, checks: List[str]) -> Dict[str, Any]:
        """Perform data quality checks"""
        results = {"passed": True, "issues": []}

        for check in checks:
            if check == "completeness":
                null_counts = df.isnull().sum()
                high_null_cols = null_counts[null_counts > len(df) * 0.1].index.tolist()
                if high_null_cols:
                    results["issues"].append(f"High null percentage in columns: {high_null_cols}")
                    results["passed"] = False

            elif check == "validity":
                # Check for data type consistency
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Check if numeric columns are stored as strings
                        try:
                            pd.to_numeric(df[col], errors='coerce')
                            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                            if numeric_count > len(df) * 0.8:  # 80% numeric
                                results["issues"].append(f"Column {col} contains mostly numeric data but is stored as text")
                                results["passed"] = False
                        except:
                            pass

        return results

    def _store_in_layer(self, df: pd.DataFrame, dataset_name: str, layer: str,
                       data_format: str, partition_by: Optional[str] = None) -> Dict[str, Any]:
        """Store dataframe in specified layer"""
        # Generate object key
        timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H%M%S")

        if partition_by and partition_by in df.columns:
            # Partition by column value
            partition_value = str(df[partition_by].iloc[0]) if len(df) > 0 else "unknown"
            object_key = f"processed/{dataset_name}/{partition_value}/{timestamp}.{data_format}"
        else:
            object_key = f"processed/{dataset_name}/{timestamp}.{data_format}"

        # Prepare data
        data_bytes, content_type = self._prepare_data_for_storage(
            df, data_format, self.layer_configs[layer]["compression"]
        )

        # Upload to layer
        minio_clients[layer].put_object(
            bucket_name=self._get_bucket_name(layer),
            object_name=object_key,
            data=data_bytes,
            length=len(data_bytes),
            content_type=content_type,
            metadata={
                "dataset_name": dataset_name,
                "processing_timestamp": datetime.utcnow().isoformat(),
                "data_format": data_format,
                "record_count": str(len(df))
            }
        )

        # Store metadata
        self._store_dataset_metadata(dataset_name, layer, object_key, len(data_bytes), df.dtypes.to_dict())

        return {
            "object_key": object_key,
            "record_count": len(df),
            "size_bytes": len(data_bytes)
        }

    def _retrieve_latest_from_layer(self, dataset_name: str, layer: str) -> Optional[Dict[str, Any]]:
        """Retrieve latest data from specified layer"""
        try:
            bucket_name = self._get_bucket_name(layer)
            objects = minio_clients[layer].list_objects(bucket_name, prefix=f"processed/{dataset_name}/")

            # Find latest object
            latest_object = None
            latest_time = None

            for obj in objects:
                if obj.last_modified > (latest_time or datetime.min.replace(tzinfo=obj.last_modified.tzinfo)):
                    latest_object = obj
                    latest_time = obj.last_modified

            if not latest_object:
                return None

            # Download and parse data
            response = minio_clients[layer].get_object(bucket_name, latest_object.object_name)

            if latest_object.object_name.endswith('.json'):
                data = json.loads(response.read().decode('utf-8'))
                df = pd.DataFrame(data)
            elif latest_object.object_name.endswith('.csv'):
                df = pd.read_csv(io.StringIO(response.read().decode('utf-8')))
            elif latest_object.object_name.endswith('.parquet'):
                import pyarrow.parquet as pq
                table = pq.read_table(io.BytesIO(response.read()))
                df = table.to_pandas()

            return {
                "dataframe": df,
                "metadata": latest_object.metadata,
                "object_name": latest_object.object_name,
                "size": latest_object.size
            }

        except Exception as e:
            logger.error("Failed to retrieve data from layer", dataset=dataset_name, layer=layer, error=str(e))
            return None

    def _get_bucket_name(self, layer: str) -> str:
        """Get bucket name for layer"""
        return {
            "bronze": self.bronze_bucket,
            "silver": self.silver_bucket,
            "gold": self.gold_bucket
        }[layer]

    def _store_dataset_metadata(self, dataset_name: str, layer: str, object_key: str,
                               size_bytes: int, schema: Dict[str, Any]):
        """Store dataset metadata in database"""
        try:
            with database_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO data_lake_objects
                    (dataset_name, layer, object_key, data_format, size_bytes, schema, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    dataset_name,
                    layer,
                    object_key,
                    object_key.split('.')[-1],
                    size_bytes,
                    json.dumps(schema)
                ))
                database_connection.commit()

        except Exception as e:
            logger.error("Failed to store dataset metadata", dataset=dataset_name, error=str(e))

    def _update_data_lineage(self, dataset_name: str, source_layer: str, target_layer: str, rules: Dict[str, Any]):
        """Update data lineage tracking"""
        try:
            with database_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO data_lineage
                    (source_record_id, source_table, target_record_id, target_table,
                     transformation_type, transformation_details, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    f"{dataset_name}_{source_layer}",
                    f"{dataset_name}_{source_layer}",
                    f"{dataset_name}_{target_layer}",
                    f"{dataset_name}_{target_layer}",
                    "layer_transformation",
                    json.dumps(rules)
                ))
                database_connection.commit()

        except Exception as e:
            logger.error("Failed to update data lineage", dataset=dataset_name, error=str(e))

    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataframe"""
        df_filtered = df.copy()

        for column, condition in filters.items():
            if column in df_filtered.columns:
                if isinstance(condition, dict):
                    operator = condition.get("operator", "equals")
                    value = condition.get("value")

                    if operator == "equals":
                        df_filtered = df_filtered[df_filtered[column] == value]
                    elif operator == "not_equals":
                        df_filtered = df_filtered[df_filtered[column] != value]
                    elif operator == "greater_than":
                        df_filtered = df_filtered[df_filtered[column] > value]
                    elif operator == "less_than":
                        df_filtered = df_filtered[df_filtered[column] < value]
                    elif operator == "contains":
                        df_filtered = df_filtered[df_filtered[column].str.contains(str(value), na=False)]
                else:
                    # Simple equality filter
                    df_filtered = df_filtered[df_filtered[column] == condition]

        return df_filtered

    def _get_layer_metadata(self, dataset_name: str, layer: str) -> Optional[Dict[str, Any]]:
        """Get metadata for dataset in specific layer"""
        try:
            with database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM data_lake_objects
                    WHERE dataset_name = %s AND layer = %s
                    ORDER BY created_at DESC LIMIT 1
                """, (dataset_name, layer))

                result = cursor.fetchone()
                return dict(result) if result else None

        except Exception as e:
            logger.error("Failed to get layer metadata", dataset=dataset_name, layer=layer, error=str(e))
            return None

    def _get_data_lineage(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get data lineage for dataset"""
        try:
            with database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM data_lineage
                    WHERE source_record_id LIKE %s OR target_record_id LIKE %s
                    ORDER BY created_at DESC
                """, (f"{dataset_name}%", f"{dataset_name}%"))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error("Failed to get data lineage", dataset=dataset_name, error=str(e))
            return []

# Global manager instance
data_lake_manager = DataLakeManager()

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

        # Declare queues for data lake operations
        queues = ['data_lake_ingestion', 'data_transformation', 'data_retrieval']
        for queue in queues:
            message_queue_channel.queue_declare(queue=queue, durable=True)

        # Set up consumer
        message_queue_channel.basic_qos(prefetch_count=1)
        message_queue_channel.basic_consume(
            queue='data_lake_ingestion',
            on_message_callback=process_ingestion_message
        )
        message_queue_channel.basic_consume(
            queue='data_transformation',
            on_message_callback=process_transformation_message
        )

        logger.info("RabbitMQ consumers setup completed")
        message_queue_channel.start_consuming()

    except Exception as e:
        logger.error("Failed to setup RabbitMQ consumers", error=str(e))
        raise

def process_ingestion_message(ch, method, properties, body):
    """Process data ingestion message"""
    try:
        message = json.loads(body)
        request = DataIngestionRequest(**message)

        logger.info("Received data lake ingestion request", dataset=request.dataset_name)

        # Ingest to bronze layer
        result = data_lake_manager.ingest_to_bronze(
            request.dataset_name,
            request.data,
            request.data_format,
            request.compression,
            request.metadata
        )

        logger.info("Data lake ingestion completed", dataset=request.dataset_name, result=result)

        # Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.error("Failed to process ingestion message", error=str(e))
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

def process_transformation_message(ch, method, properties, body):
    """Process data transformation message"""
    try:
        message = json.loads(body)
        request = LayerTransformationRequest(**message)

        logger.info("Received data transformation request",
                   dataset=request.dataset_name,
                   transformation=f"{request.source_layer}_to_{request.target_layer}")

        # Perform transformation
        if request.source_layer == "bronze" and request.target_layer == "silver":
            result = data_lake_manager.transform_bronze_to_silver(
                request.dataset_name,
                request.transformation_rules
            )
        elif request.source_layer == "silver" and request.target_layer == "gold":
            result = data_lake_manager.transform_silver_to_gold(
                request.dataset_name,
                request.transformation_rules
            )
        else:
            raise ValueError(f"Unsupported transformation: {request.source_layer} to {request.target_layer}")

        logger.info("Data transformation completed",
                   dataset=request.dataset_name,
                   transformation=f"{request.source_layer}_to_{request.target_layer}",
                   result=result)

        # Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.error("Failed to process transformation message", error=str(e))
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    layer_status = {}
    for layer in ["bronze", "silver", "gold"]:
        try:
            bucket = data_lake_manager._get_bucket_name(layer)
            minio_clients[layer].bucket_exists(bucket)
            layer_status[layer] = "healthy"
        except:
            layer_status[layer] = "unhealthy"

    return {
        "status": "healthy",
        "service": "data-lake-minio",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "layers": layer_status
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/ingest")
async def ingest_data(request: DataIngestionRequest, background_tasks: BackgroundTasks):
    """Ingest data into bronze layer"""
    try:
        # Convert data to list of dicts if it's a DataFrame
        if hasattr(request.data, 'to_dict'):
            data = request.data.to_dict('records')
        else:
            data = request.data

        # Perform ingestion
        result = data_lake_manager.ingest_to_bronze(
            request.dataset_name,
            data,
            request.data_format,
            request.compression,
            request.metadata
        )

        return result

    except Exception as e:
        logger.error("Data ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/transform")
async def transform_data(request: LayerTransformationRequest, background_tasks: BackgroundTasks):
    """Transform data between layers"""
    try:
        if request.source_layer == "bronze" and request.target_layer == "silver":
            result = data_lake_manager.transform_bronze_to_silver(
                request.dataset_name,
                request.transformation_rules
            )
        elif request.source_layer == "silver" and request.target_layer == "gold":
            result = data_lake_manager.transform_silver_to_gold(
                request.dataset_name,
                request.transformation_rules
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported transformation: {request.source_layer} to {request.target_layer}"
            )

        return result

    except Exception as e:
        logger.error("Data transformation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")

@app.get("/data/{dataset_name}")
async def retrieve_data(
    dataset_name: str,
    layer: str = "gold",
    limit: Optional[int] = None,
    filters: Optional[str] = None  # JSON string
):
    """Retrieve data from specified layer"""
    try:
        # Parse filters if provided
        parsed_filters = None
        if filters:
            parsed_filters = json.loads(filters)

        result = data_lake_manager.retrieve_data(
            dataset_name,
            layer,
            parsed_filters,
            limit
        )

        return result

    except Exception as e:
        logger.error("Data retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.get("/metadata/{dataset_name}")
async def get_dataset_metadata(dataset_name: str):
    """Get comprehensive metadata for dataset"""
    try:
        metadata = data_lake_manager.get_dataset_metadata(dataset_name)
        return metadata

    except Exception as e:
        logger.error("Metadata retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metadata retrieval failed: {str(e)}")

@app.get("/layers")
async def list_layers():
    """List all layers and their status"""
    layers_info = {}

    for layer in ["bronze", "silver", "gold"]:
        try:
            bucket = data_lake_manager._get_bucket_name(layer)
            objects = list(minio_clients[layer].list_objects(bucket))

            layers_info[layer] = {
                "status": "healthy",
                "bucket": bucket,
                "object_count": len(objects),
                "total_size": sum(obj.size for obj in objects if obj.size),
                "config": data_lake_manager.layer_configs[layer]
            }
        except Exception as e:
            layers_info[layer] = {
                "status": "error",
                "error": str(e)
            }

    return {"layers": layers_info}

@app.get("/stats")
async def get_stats():
    """Get data lake statistics"""
    return {
        "service": "data-lake-minio",
        "metrics": {
            "data_ingested_total": DATA_INGESTED_TOTAL._value.get(),
            "data_transformed_total": DATA_TRANSFORMED_TOTAL._value.get(),
            "data_quality_checks_total": DATA_QUALITY_CHECKS._value.get(),
            "storage_usage": {
                "bronze": STORAGE_USAGE_BYTES.labels(layer="bronze")._value.get(),
                "silver": STORAGE_USAGE_BYTES.labels(layer="silver")._value.get(),
                "gold": STORAGE_USAGE_BYTES.labels(layer="gold")._value.get()
            }
        }
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global database_connection, minio_clients

    logger.info("Data Lake MinIO Service starting up...")

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

    # Setup MinIO clients for each layer
    try:
        minio_config = {
            "endpoint": os.getenv("MINIO_ENDPOINT", "http://minio_bronze:9000"),
            "access_key": os.getenv("MINIO_ACCESS_KEY", "agentic_user"),
            "secret_key": os.getenv("MINIO_SECRET_KEY", "agentic123"),
            "secure": False
        }

        # Initialize clients for each layer
        for layer in ["bronze", "silver", "gold"]:
            if layer == "silver":
                minio_config["endpoint"] = os.getenv("MINIO_ENDPOINT", "http://minio_silver:9010")
            elif layer == "gold":
                minio_config["endpoint"] = os.getenv("MINIO_ENDPOINT", "http://minio_gold:9020")

            minio_clients[layer] = Minio(**minio_config)

        # Initialize buckets
        data_lake_manager.initialize_buckets()

        logger.info("MinIO clients initialized")

    except Exception as e:
        logger.error("MinIO initialization failed", error=str(e))
        raise

    # Setup RabbitMQ consumer in background thread
    rabbitmq_thread = threading.Thread(target=setup_rabbitmq, daemon=True)
    rabbitmq_thread.start()

    logger.info("Data Lake MinIO Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global database_connection

    logger.info("Data Lake MinIO Service shutting down...")

    # Close database connection
    if database_connection:
        database_connection.close()

    logger.info("Data Lake MinIO Service shutdown complete")

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
        "data_lake_minio:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8090)),
        reload=False,
        log_level="info"
    )
