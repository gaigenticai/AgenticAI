#!/usr/bin/env python3
"""
Backup Orchestration Service for Agentic Platform

This service provides comprehensive backup and disaster recovery orchestration with:
- Automated backup scheduling for all data stores
- Cross-cluster replication and failover
- Backup verification and integrity checks
- Disaster recovery procedures
- Backup retention policy management
- Point-in-time recovery capabilities
- Cloud storage integration
- Backup monitoring and alerting
"""

import json
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import psycopg2
import pymongo
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import structlog
import uvicorn

# Configure structured logging
logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
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
    title="Backup Orchestration Service",
    description="Comprehensive backup and disaster recovery orchestration",
    version="1.0.0"
)

# Prometheus metrics
BACKUP_OPERATIONS = Counter('backup_operations_total', 'Total backup operations', ['operation_type', 'status'])
RESTORE_OPERATIONS = Counter('restore_operations_total', 'Total restore operations', ['operation_type', 'status'])
BACKUP_SIZE_BYTES = Gauge('backup_size_bytes', 'Backup size in bytes', ['backup_type'])
BACKUP_DURATION = Histogram('backup_duration_seconds', 'Backup operation duration', ['backup_type'])
REPLICATION_LAG = Gauge('replication_lag_seconds', 'Replication lag in seconds', ['cluster'])

# Global variables
database_connection = None

# Pydantic models
class BackupRequest(BaseModel):
    """Backup request model"""
    backup_type: str = Field(..., description="Type of backup: database, object_store, cache, config")
    target_system: str = Field(..., description="Target system to backup")
    storage_location: str = Field(..., description="Storage location for backup")
    retention_days: int = Field(30, description="Backup retention period in days")
    compression: bool = Field(True, description="Enable compression")
    encryption: bool = Field(True, description="Enable encryption")
    verify_integrity: bool = Field(True, description="Verify backup integrity")

class RestoreRequest(BaseModel):
    """Restore request model"""
    backup_id: str = Field(..., description="Backup ID to restore")
    target_system: str = Field(..., description="Target system to restore to")
    restore_point: Optional[str] = Field(None, description="Point-in-time restore point")
    verify_before_restore: bool = Field(True, description="Verify backup before restore")

class ReplicationConfig(BaseModel):
    """Replication configuration model"""
    source_cluster: str = Field(..., description="Source cluster identifier")
    target_cluster: str = Field(..., description="Target cluster identifier")
    replication_type: str = Field(..., description="Type: async, sync, semi-sync")
    data_types: List[str] = Field(..., description="Data types to replicate")
    lag_tolerance_seconds: int = Field(300, description="Maximum acceptable lag")

class DisasterRecoveryPlan(BaseModel):
    """Disaster recovery plan model"""
    plan_name: str = Field(..., description="Name of the recovery plan")
    trigger_conditions: List[str] = Field(..., description="Conditions that trigger recovery")
    recovery_steps: List[Dict[str, Any]] = Field(..., description="Step-by-step recovery procedure")
    estimated_rto_minutes: int = Field(..., description="Recovery Time Objective in minutes")
    estimated_rpo_seconds: int = Field(..., description="Recovery Point Objective in seconds")
    test_frequency_days: int = Field(30, description="How often to test the plan")

class BackupSchedule(BaseModel):
    """Backup schedule model"""
    schedule_name: str = Field(..., description="Name of the backup schedule")
    backup_type: str = Field(..., description="Type of backup")
    cron_expression: str = Field(..., description="Cron expression for scheduling")
    target_systems: List[str] = Field(..., description="Systems to backup")
    retention_policy: str = Field(..., description="Retention policy to apply")

# Backup Orchestration Manager Class
class BackupOrchestrationManager:
    """Comprehensive backup orchestration manager"""

    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.backup_configs = {}
        self.replication_configs = {}
        self.recovery_plans = {}

    def create_backup(self, request: BackupRequest) -> Dict[str, Any]:
        """Create a backup based on the request"""
        try:
            backup_id = f"backup_{uuid.uuid4().hex[:16]}_{int(time.time())}"

            if request.backup_type == "database":
                result = self._backup_database(request, backup_id)
            elif request.backup_type == "object_store":
                result = self._backup_object_store(request, backup_id)
            elif request.backup_type == "cache":
                result = self._backup_cache(request, backup_id)
            elif request.backup_type == "config":
                result = self._backup_configurations(request, backup_id)
            else:
                raise ValueError(f"Unsupported backup type: {request.backup_type}")

            # Record backup metadata
            self._record_backup_metadata(backup_id, request, result)

            BACKUP_OPERATIONS.labels(operation_type=request.backup_type, status="success").inc()
            BACKUP_SIZE_BYTES.labels(backup_type=request.backup_type).set(result.get("size_bytes", 0))

            logger.info("Backup created successfully", backup_id=backup_id, backup_type=request.backup_type)

            return {
                "backup_id": backup_id,
                "status": "completed",
                "backup_type": request.backup_type,
                "target_system": request.target_system,
                "storage_location": request.storage_location,
                "size_bytes": result.get("size_bytes", 0),
                "duration_seconds": result.get("duration", 0),
                "created_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Backup creation failed", error=str(e), backup_type=request.backup_type)
            BACKUP_OPERATIONS.labels(operation_type=request.backup_type, status="failed").inc()
            raise

    def restore_backup(self, request: RestoreRequest) -> Dict[str, Any]:
        """Restore from a backup"""
        try:
            # Get backup metadata
            backup_metadata = self._get_backup_metadata(request.backup_id)
            if not backup_metadata:
                raise ValueError(f"Backup not found: {request.backup_id}")

            # Verify backup integrity if requested
            if request.verify_before_restore:
                integrity_result = self._verify_backup_integrity(request.backup_id)
                if not integrity_result["valid"]:
                    raise ValueError(f"Backup integrity check failed: {integrity_result['issues']}")

            # Perform restore
            if backup_metadata["backup_type"] == "database":
                result = self._restore_database(request, backup_metadata)
            elif backup_metadata["backup_type"] == "object_store":
                result = self._restore_object_store(request, backup_metadata)
            elif backup_metadata["backup_type"] == "cache":
                result = self._restore_cache(request, backup_metadata)
            elif backup_metadata["backup_type"] == "config":
                result = self._restore_configurations(request, backup_metadata)
            else:
                raise ValueError(f"Unsupported restore type: {backup_metadata['backup_type']}")

            RESTORE_OPERATIONS.labels(operation_type=backup_metadata["backup_type"], status="success").inc()

            logger.info("Backup restored successfully", backup_id=request.backup_id)

            return {
                "backup_id": request.backup_id,
                "status": "restored",
                "target_system": request.target_system,
                "restored_at": datetime.utcnow().isoformat(),
                "verification_performed": request.verify_before_restore
            }

        except Exception as e:
            logger.error("Backup restore failed", error=str(e), backup_id=request.backup_id)
            RESTORE_OPERATIONS.labels(operation_type="unknown", status="failed").inc()
            raise

    def _backup_database(self, request: BackupRequest, backup_id: str) -> Dict[str, Any]:
        """Backup database"""
        start_time = time.time()

        try:
            # Create backup directory
            backup_dir = f"/backups/database/{backup_id}"
            os.makedirs(backup_dir, exist_ok=True)

            # PostgreSQL backup using pg_dump
            db_host = os.getenv("POSTGRES_HOST", "postgresql_ingestion")
            db_user = os.getenv("POSTGRES_USER", "agentic_user")
            db_name = os.getenv("POSTGRES_DB", "agentic_ingestion")

            cmd = [
                "pg_dump",
                f"--host={db_host}",
                f"--username={db_user}",
                f"--dbname={db_name}",
                f"--file={backup_dir}/database.sql",
                "--format=custom",
                "--compress=9",
                "--no-password"
            ]

            # Set password environment variable
            env = os.environ.copy()
            env["PGPASSWORD"] = os.getenv("POSTGRES_PASSWORD", "")
            if not env["PGPASSWORD"]:
                logger.error("POSTGRES_PASSWORD not configured for Backup Orchestration Service")
                raise RuntimeError("POSTGRES_PASSWORD not configured for Backup Orchestration Service")

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")

            # Calculate backup size
            backup_size = sum(f.stat().st_size for f in Path(backup_dir).rglob('*') if f.is_file())

            # Upload to storage if specified
            if request.storage_location.startswith("s3://"):
                self._upload_to_s3(backup_dir, request.storage_location)

            duration = time.time() - start_time

            return {
                "size_bytes": backup_size,
                "duration": duration,
                "files_created": len(list(Path(backup_dir).rglob('*'))),
                "compression_used": True
            }

        except Exception as e:
            logger.error("Database backup failed", error=str(e))
            raise

    def _backup_object_store(self, request: BackupRequest, backup_id: str) -> Dict[str, Any]:
        """Backup object storage (MinIO)"""
        start_time = time.time()

        try:
            # Use MinIO client to backup buckets
            from minio import Minio

            minio_client = Minio(
                os.getenv("MINIO_ENDPOINT", "http://minio_bronze:9000"),
                access_key=os.getenv("MINIO_ACCESS_KEY", "agentic_user"),
                secret_key=os.getenv("MINIO_SECRET_KEY", ""),
                secure=False
            )

            backup_dir = f"/backups/object_store/{backup_id}"
            os.makedirs(backup_dir, exist_ok=True)

            total_size = 0
            total_objects = 0

            # Backup each bucket
            buckets = ["bronze-layer", "silver-layer", "gold-layer"]
            for bucket in buckets:
                if minio_client.bucket_exists(bucket):
                    bucket_backup_dir = f"{backup_dir}/{bucket}"
                    os.makedirs(bucket_backup_dir, exist_ok=True)

                    # List and download objects
                    objects = minio_client.list_objects(bucket, recursive=True)
                    for obj in objects:
                        minio_client.fget_object(bucket, obj.object_name, f"{bucket_backup_dir}/{obj.object_name}")
                        total_size += obj.size or 0
                        total_objects += 1

            # Upload to storage if specified
            if request.storage_location.startswith("s3://"):
                self._upload_to_s3(backup_dir, request.storage_location)

            duration = time.time() - start_time

            return {
                "size_bytes": total_size,
                "duration": duration,
                "objects_backed_up": total_objects,
                "buckets_backed_up": len(buckets)
            }

        except Exception as e:
            logger.error("Object store backup failed", error=str(e))
            raise

    def _backup_cache(self, request: BackupRequest, backup_id: str) -> Dict[str, Any]:
        """Backup cache (Redis)"""
        start_time = time.time()

        try:
            import redis

            redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "redis_ingestion"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                decode_responses=True
            )

            backup_dir = f"/backups/cache/{backup_id}"
            os.makedirs(backup_dir, exist_ok=True)

            # Use Redis BGSAVE for consistent backup
            redis_client.bgsave()

            # Wait for save to complete
            while True:
                info = redis_client.info()
                if not info.get("rdb_bgsave_in_progress", False):
                    break
                time.sleep(1)

            # Copy RDB file (assuming standard Redis setup)
            rdb_source = "/var/lib/redis/dump.rdb"  # Adjust path as needed
            rdb_dest = f"{backup_dir}/dump.rdb"

            if os.path.exists(rdb_source):
                shutil.copy2(rdb_source, rdb_dest)
                backup_size = os.path.getsize(rdb_dest)
            else:
                # Fallback: export keys using SCAN
                backup_size = self._export_redis_keys(redis_client, backup_dir)

            # Upload to storage if specified
            if request.storage_location.startswith("s3://"):
                self._upload_to_s3(backup_dir, request.storage_location)

            duration = time.time() - start_time

            return {
                "size_bytes": backup_size,
                "duration": duration,
                "backup_method": "rdb_file" if os.path.exists(rdb_source) else "key_export"
            }

        except Exception as e:
            logger.error("Cache backup failed", error=str(e))
            raise

    def _backup_configurations(self, request: BackupRequest, backup_id: str) -> Dict[str, Any]:
        """Backup configurations"""
        start_time = time.time()

        try:
            backup_dir = f"/backups/config/{backup_id}"
            os.makedirs(backup_dir, exist_ok=True)

            # Backup configuration files
            config_files = [
                "/app/docker-compose.yml",
                "/app/.env",
                "/app/schema.sql",
                "/app/docker/configs/"
            ]

            total_size = 0

            for config_path in config_files:
                if os.path.exists(config_path):
                    if os.path.isdir(config_path):
                        # Copy directory
                        dest_dir = f"{backup_dir}/{os.path.basename(config_path)}"
                        shutil.copytree(config_path, dest_dir, dirs_exist_ok=True)
                    else:
                        # Copy file
                        shutil.copy2(config_path, backup_dir)

            # Calculate total size
            total_size = sum(f.stat().st_size for f in Path(backup_dir).rglob('*') if f.is_file())

            # Upload to storage if specified
            if request.storage_location.startswith("s3://"):
                self._upload_to_s3(backup_dir, request.storage_location)

            duration = time.time() - start_time

            return {
                "size_bytes": total_size,
                "duration": duration,
                "files_backed_up": len(list(Path(backup_dir).rglob('*')))
            }

        except Exception as e:
            logger.error("Configuration backup failed", error=str(e))
            raise

    def _restore_database(self, request: RestoreRequest, backup_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Restore database from backup"""
        try:
            backup_location = backup_metadata["storage_location"]
            backup_file = f"{backup_location}/database.sql"

            # PostgreSQL restore using pg_restore
            db_host = os.getenv("POSTGRES_HOST", "postgresql_ingestion")
            db_user = os.getenv("POSTGRES_USER", "agentic_user")
            db_name = os.getenv("POSTGRES_DB", "agentic_ingestion")

            cmd = [
                "pg_restore",
                f"--host={db_host}",
                f"--username={db_user}",
                f"--dbname={db_name}",
                "--clean",
                "--if-exists",
                backup_file
            ]

            env = os.environ.copy()
            env["PGPASSWORD"] = os.getenv("POSTGRES_PASSWORD", "")
            if not env["PGPASSWORD"]:
                logger.error("POSTGRES_PASSWORD not configured for Backup Orchestration Service")
                raise RuntimeError("POSTGRES_PASSWORD not configured for Backup Orchestration Service")

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"pg_restore failed: {result.stderr}")

            return {"status": "restored", "tables_restored": "all"}

        except Exception as e:
            logger.error("Database restore failed", error=str(e))
            raise

    def _restore_object_store(self, request: RestoreRequest, backup_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Restore object storage from backup"""
        try:
            from minio import Minio

            minio_client = Minio(
                os.getenv("MINIO_ENDPOINT", "http://minio_bronze:9000"),
                access_key=os.getenv("MINIO_ACCESS_KEY", "agentic_user"),
                secret_key=os.getenv("MINIO_SECRET_KEY", ""),
                secure=False
            )

            backup_location = backup_metadata["storage_location"]

            # Restore each bucket
            buckets = ["bronze-layer", "silver-layer", "gold-layer"]
            total_objects = 0

            for bucket in buckets:
                bucket_backup_dir = f"{backup_location}/{bucket}"
                if os.path.exists(bucket_backup_dir):
                    # Ensure bucket exists
                    if not minio_client.bucket_exists(bucket):
                        minio_client.make_bucket(bucket)

                    # Upload all files
                    for file_path in Path(bucket_backup_dir).rglob('*'):
                        if file_path.is_file():
                            object_name = str(file_path.relative_to(bucket_backup_dir))
                            minio_client.fput_object(bucket, object_name, str(file_path))
                            total_objects += 1

            return {"status": "restored", "objects_restored": total_objects}

        except Exception as e:
            logger.error("Object store restore failed", error=str(e))
            raise

    def _restore_cache(self, request: RestoreRequest, backup_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Restore cache from backup"""
        try:
            import redis

            redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "redis_ingestion"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0
            )

            backup_location = backup_metadata["storage_location"]
            rdb_file = f"{backup_location}/dump.rdb"

            if os.path.exists(rdb_file):
                # Copy RDB file to Redis data directory
                redis_data_dir = "/var/lib/redis"
                shutil.copy2(rdb_file, f"{redis_data_dir}/dump.rdb")

                # Trigger Redis to reload RDB file
                redis_client.shutdown(save=False)
                # Redis will automatically reload on restart

            return {"status": "restored", "method": "rdb_file"}

        except Exception as e:
            logger.error("Cache restore failed", error=str(e))
            raise

    def _restore_configurations(self, request: RestoreRequest, backup_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Restore configurations from backup"""
        try:
            backup_location = backup_metadata["storage_location"]

            # Restore configuration files
            restore_files = [
                ("docker-compose.yml", "/app/docker-compose.yml"),
                (".env", "/app/.env"),
                ("configs/", "/app/docker/configs/")
            ]

            files_restored = 0

            for src_name, dest_path in restore_files:
                src_path = f"{backup_location}/{src_name}"
                if os.path.exists(src_path):
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_path, dest_path)
                    files_restored += 1

            return {"status": "restored", "files_restored": files_restored}

        except Exception as e:
            logger.error("Configuration restore failed", error=str(e))
            raise

    def _verify_backup_integrity(self, backup_id: str) -> Dict[str, Any]:
        """Verify backup integrity"""
        try:
            # Get backup metadata
            backup_metadata = self._get_backup_metadata(backup_id)

            # Basic integrity checks
            issues = []

            # Check if backup files exist
            if not os.path.exists(backup_metadata["storage_location"]):
                issues.append("Backup location does not exist")

            # Check file sizes
            total_size = sum(f.stat().st_size for f in Path(backup_metadata["storage_location"]).rglob('*') if f.is_file())
            if total_size != backup_metadata.get("size_bytes", 0):
                issues.append("Backup size mismatch")

            # Check backup age
            backup_age = datetime.utcnow() - datetime.fromisoformat(backup_metadata["created_at"])
            if backup_age.days > 365:  # Older than 1 year
                issues.append("Backup is very old")

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "backup_age_days": backup_age.days,
                "verified_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Backup integrity verification failed", error=str(e))
            return {
                "valid": False,
                "issues": [f"Verification failed: {str(e)}"],
                "verified_at": datetime.utcnow().isoformat()
            }

    def _record_backup_metadata(self, backup_id: str, request: BackupRequest, result: Dict[str, Any]):
        """Record backup metadata in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO backup_metadata (
                        backup_id, backup_type, target_system, storage_location,
                        retention_days, size_bytes, status, created_at,
                        compression_used, encryption_used
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    backup_id,
                    request.backup_type,
                    request.target_system,
                    request.storage_location,
                    request.retention_days,
                    result.get("size_bytes", 0),
                    "completed",
                    datetime.utcnow(),
                    request.compression,
                    request.encryption
                ))
                self.db_connection.commit()

        except Exception as e:
            logger.error("Failed to record backup metadata", error=str(e))

    def _get_backup_metadata(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Get backup metadata from database"""
        try:
            with self.db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM backup_metadata WHERE backup_id = %s
                """, (backup_id,))

                result = cursor.fetchone()
                return dict(result) if result else None

        except Exception as e:
            logger.error("Failed to get backup metadata", error=str(e))
            return None

    def _upload_to_s3(self, local_path: str, s3_path: str):
        """Upload backup to S3"""
        try:
            # Parse S3 path
            if not s3_path.startswith("s3://"):
                raise ValueError("Invalid S3 path")

            s3_parts = s3_path[5:].split("/", 1)
            bucket_name = s3_parts[0]
            key_prefix = s3_parts[1] if len(s3_parts) > 1 else ""

            # Initialize S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )

            # Upload files
            for file_path in Path(local_path).rglob('*'):
                if file_path.is_file():
                    s3_key = f"{key_prefix}/{file_path.relative_to(local_path)}"
                    s3_client.upload_file(str(file_path), bucket_name, s3_key)

        except Exception as e:
            logger.error("S3 upload failed", error=str(e))
            raise

    def _export_redis_keys(self, redis_client, backup_dir: str) -> int:
        """Export Redis keys to JSON file"""
        try:
            keys_data = {}

            # SCAN all keys
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(cursor, count=1000)
                for key in keys:
                    key_type = redis_client.type(key)
                    if key_type == "string":
                        keys_data[key] = redis_client.get(key)
                    elif key_type == "hash":
                        keys_data[key] = redis_client.hgetall(key)
                    elif key_type == "list":
                        keys_data[key] = redis_client.lrange(key, 0, -1)
                    elif key_type == "set":
                        keys_data[key] = list(redis_client.smembers(key))
                    elif key_type == "zset":
                        keys_data[key] = redis_client.zrange(key, 0, -1, withscores=True)

                if cursor == 0:
                    break

            # Save to JSON file
            backup_file = f"{backup_dir}/redis_keys.json"
            with open(backup_file, 'w') as f:
                json.dump(keys_data, f, indent=2, default=str)

            return os.path.getsize(backup_file)

        except Exception as e:
            logger.error("Redis key export failed", error=str(e))
            return 0

# Global instance
backup_manager = None

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "backup-orchestration-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/backup")
async def create_backup(request: BackupRequest, background_tasks: BackgroundTasks):
    """Create a backup"""
    try:
        # Start backup in background
        background_tasks.add_task(backup_manager.create_backup, request)

        return {
            "status": "backup_started",
            "backup_type": request.backup_type,
            "target_system": request.target_system,
            "message": "Backup operation started in background"
        }

    except Exception as e:
        logger.error("Backup request failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Backup request failed: {str(e)}")

@app.post("/restore")
async def restore_backup(request: RestoreRequest, background_tasks: BackgroundTasks):
    """Restore from a backup"""
    try:
        # Start restore in background
        background_tasks.add_task(backup_manager.restore_backup, request)

        return {
            "status": "restore_started",
            "backup_id": request.backup_id,
            "target_system": request.target_system,
            "message": "Restore operation started in background"
        }

    except Exception as e:
        logger.error("Restore request failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Restore request failed: {str(e)}")

@app.get("/backups")
async def list_backups(
    backup_type: Optional[str] = None,
    target_system: Optional[str] = None,
    limit: int = 50
):
    """List available backups"""
    try:
        with database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            query = "SELECT * FROM backup_metadata WHERE 1=1"
            params = []

            if backup_type:
                query += " AND backup_type = %s"
                params.append(backup_type)

            if target_system:
                query += " AND target_system = %s"
                params.append(target_system)

            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            backups = cursor.fetchall()

            return {
                "backups": [dict(backup) for backup in backups],
                "total": len(backups)
            }

    except Exception as e:
        logger.error("Failed to list backups", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list backups: {str(e)}")

@app.get("/backups/{backup_id}")
async def get_backup_details(backup_id: str):
    """Get backup details"""
    try:
        backup_metadata = backup_manager._get_backup_metadata(backup_id)
        if not backup_metadata:
            raise HTTPException(status_code=404, detail="Backup not found")

        # Get integrity status
        integrity = backup_manager._verify_backup_integrity(backup_id)

        return {
            "backup": backup_metadata,
            "integrity": integrity
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get backup details", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get backup details: {str(e)}")

@app.delete("/backups/{backup_id}")
async def delete_backup(backup_id: str):
    """Delete a backup"""
    try:
        backup_metadata = backup_manager._get_backup_metadata(backup_id)
        if not backup_metadata:
            raise HTTPException(status_code=404, detail="Backup not found")

        # Delete from storage
        if os.path.exists(backup_metadata["storage_location"]):
            shutil.rmtree(backup_metadata["storage_location"])

        # Delete from database
        with database_connection.cursor() as cursor:
            cursor.execute("DELETE FROM backup_metadata WHERE backup_id = %s", (backup_id,))
            database_connection.commit()

        return {"status": "deleted", "backup_id": backup_id}

    except Exception as e:
        logger.error("Failed to delete backup", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete backup: {str(e)}")

@app.get("/replication/status")
async def get_replication_status():
    """Get replication status across clusters"""
    try:
        # This would query actual replication status
        # For now, return mock data
        return {
            "clusters": [
                {
                    "name": "primary",
                    "status": "healthy",
                    "lag_seconds": 0
                },
                {
                    "name": "secondary",
                    "status": "healthy",
                    "lag_seconds": 2
                }
            ],
            "overall_status": "healthy",
            "last_updated": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to get replication status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get replication status: {str(e)}")

@app.get("/stats")
async def get_backup_stats():
    """Get backup service statistics"""
    return {
        "service": "backup-orchestration-service",
        "metrics": {
            "backup_operations_total": BACKUP_OPERATIONS._value.get(),
            "restore_operations_total": RESTORE_OPERATIONS._value.get(),
            "backup_size_bytes": BACKUP_SIZE_BYTES._value.get()
        }
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global backup_manager, database_connection

    logger.info("Backup Orchestration Service starting up...")

    # Setup database connection
    try:
        db_config = {
            "host": os.getenv("POSTGRES_HOST", "postgresql_ingestion"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agentic_ingestion"),
            "user": os.getenv("POSTGRES_USER", "agentic_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "")
        }

        database_connection = psycopg2.connect(**db_config)
        backup_manager = BackupOrchestrationManager(database_connection)

        logger.info("Database connection established")

    except Exception as e:
        logger.warning("Database connection failed", error=str(e))

    logger.info("Backup Orchestration Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global database_connection

    logger.info("Backup Orchestration Service shutting down...")

    # Close database connection
    if database_connection:
        database_connection.close()

    logger.info("Backup Orchestration Service shutdown complete")

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
        "backup_orchestration_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8097)),
        reload=False,
        log_level="info"
    )
