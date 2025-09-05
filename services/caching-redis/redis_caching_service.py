#!/usr/bin/env python3
"""
Redis Caching Service for Agentic Platform

This service provides advanced caching capabilities with Redis including:
- Intelligent cache key generation and management
- Cache warming strategies from database
- TTL (Time-To-Live) management and cache invalidation
- Distributed caching with Redis Cluster support
- Cache compression for large objects
- Cache monitoring and performance metrics
- Cache hit/miss statistics and analytics
- Multi-level caching strategies
- Cache warming from data sources
"""

import hashlib
import json
import os
import threading
import time
import zlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import redis
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
    title="Redis Caching Service",
    description="Advanced caching layer with Redis for ingestion and output operations",
    version="1.0.0"
)

# Prometheus metrics
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits', ['cache_type'])
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses', ['cache_type'])
CACHE_SETS = Counter('cache_sets_total', 'Total cache sets', ['cache_type'])
CACHE_DELETES = Counter('cache_deletes_total', 'Total cache deletes', ['cache_type'])
CACHE_SIZE_BYTES = Gauge('cache_size_bytes', 'Cache size in bytes', ['cache_type'])
CACHE_COMPRESSION_RATIO = Gauge('cache_compression_ratio', 'Cache compression ratio', ['cache_type'])
CACHE_WARMUP_TIME = Histogram('cache_warmup_duration_seconds', 'Cache warmup duration', ['data_source'])

# Global variables
redis_client = None
database_connection = None

# Pydantic models
class CacheEntry(BaseModel):
    """Cache entry model"""
    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Cached value")
    ttl_seconds: Optional[int] = Field(None, description="Time-to-live in seconds")
    tags: List[str] = Field(default=[], description="Cache tags for grouping")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class CacheConfig(BaseModel):
    """Cache configuration model"""
    cache_type: str = Field(..., description="Type of cache (ingestion, output, metadata)")
    max_memory_mb: int = Field(256, description="Maximum memory in MB")
    ttl_default: int = Field(3600, description="Default TTL in seconds")
    compression_enabled: bool = Field(True, description="Enable compression for large objects")
    cluster_enabled: bool = Field(False, description="Enable Redis Cluster mode")

class CacheWarmupRequest(BaseModel):
    """Cache warmup request model"""
    data_source: str = Field(..., description="Data source to warmup from")
    cache_type: str = Field(..., description="Type of cache to warmup")
    query: Optional[str] = Field(None, description="Query to fetch data for warmup")
    batch_size: int = Field(1000, description="Batch size for warmup")
    ttl_seconds: Optional[int] = Field(None, description="TTL for warmed up entries")

class CacheStats(BaseModel):
    """Cache statistics model"""
    cache_type: str
    total_keys: int
    memory_usage_bytes: int
    hit_ratio: float
    avg_ttl_seconds: float
    compression_ratio: float
    warmup_status: str

class RedisCacheManager:
    """Advanced Redis cache manager with comprehensive features"""

    def __init__(self):
        self.cache_configs = {}
        self.cache_stats = {}
        self.compression_threshold = 1024  # Compress objects larger than 1KB
        self.default_ttl = 3600  # 1 hour default TTL

    def initialize_cache(self, config: CacheConfig):
        """Initialize cache with specific configuration"""
        try:
            cache_key = f"{config.cache_type}_cache"

            # Configure Redis connection
            redis_config = {
                "host": os.getenv("REDIS_HOST", "redis_ingestion"),
                "port": int(os.getenv("REDIS_PORT", 6379)),
                "db": self._get_db_for_cache_type(config.cache_type),
                "decode_responses": True,
                "socket_timeout": 5,
                "socket_connect_timeout": 5,
                "retry_on_timeout": True,
                "max_connections": 20
            }

            if config.cluster_enabled:
                # Redis Cluster configuration
                startup_nodes = [
                    {"host": os.getenv("REDIS_HOST", "redis_ingestion"), "port": int(os.getenv("REDIS_PORT", 6379))}
                ]
                redis_client = redis.RedisCluster(startup_nodes=startup_nodes, **redis_config)
            else:
                redis_client = redis.Redis(**redis_config)

            # Test connection
            redis_client.ping()

            self.cache_configs[cache_key] = {
                "client": redis_client,
                "config": config,
                "stats": {
                    "hits": 0,
                    "misses": 0,
                    "sets": 0,
                    "deletes": 0,
                    "last_updated": time.time()
                }
            }

            logger.info("Cache initialized successfully", cache_type=config.cache_type)

        except Exception as e:
            logger.error("Failed to initialize cache", error=str(e), cache_type=config.cache_type)
            raise

    def _get_db_for_cache_type(self, cache_type: str) -> int:
        """Get Redis database number for cache type"""
        db_mapping = {
            "ingestion": 0,
            "output": 1,
            "metadata": 2,
            "session": 3,
            "temporary": 4
        }
        return db_mapping.get(cache_type, 0)

    def set_cache_entry(self, cache_type: str, key: str, value: Any,
                       ttl_seconds: Optional[int] = None, tags: List[str] = None) -> bool:
        """Set cache entry with intelligent key generation and compression"""
        try:
            cache_key = f"{cache_type}_cache"
            if cache_key not in self.cache_configs:
                raise ValueError(f"Cache {cache_type} not initialized")

            cache_config = self.cache_configs[cache_key]
            client = cache_config["client"]

            # Generate intelligent cache key
            intelligent_key = self._generate_cache_key(cache_type, key)

            # Serialize and optionally compress value
            serialized_value = self._serialize_value(value)

            if cache_config["config"].compression_enabled and len(serialized_value) > self.compression_threshold:
                serialized_value = self._compress_value(serialized_value)

            # Set TTL
            ttl = ttl_seconds or cache_config["config"].ttl_default

            # Store in cache
            client.setex(intelligent_key, ttl, serialized_value)

            # Store tags for cache invalidation
            if tags:
                tag_key = f"tag:{cache_type}:{','.join(sorted(tags))}"
                client.sadd(tag_key, intelligent_key)

            # Update statistics
            cache_config["stats"]["sets"] += 1
            CACHE_SETS.labels(cache_type=cache_type).inc()

            logger.info("Cache entry set", cache_type=cache_type, key=intelligent_key, ttl=ttl)
            return True

        except Exception as e:
            logger.error("Failed to set cache entry", error=str(e), cache_type=cache_type, key=key)
            return False

    def get_cache_entry(self, cache_type: str, key: str) -> Optional[Any]:
        """Get cache entry with automatic decompression"""
        try:
            cache_key = f"{cache_type}_cache"
            if cache_key not in self.cache_configs:
                raise ValueError(f"Cache {cache_type} not initialized")

            cache_config = self.cache_configs[cache_key]
            client = cache_config["client"]

            # Generate intelligent cache key
            intelligent_key = self._generate_cache_key(cache_type, key)

            # Get from cache
            cached_value = client.get(intelligent_key)

            if cached_value is None:
                # Cache miss
                cache_config["stats"]["misses"] += 1
                CACHE_MISSES.labels(cache_type=cache_type).inc()
                return None

            # Cache hit
            cache_config["stats"]["hits"] += 1
            CACHE_HITS.labels(cache_type=cache_type).inc()

            # Decompress if needed and deserialize
            if isinstance(cached_value, bytes) and cached_value.startswith(b'compressed:'):
                cached_value = self._decompress_value(cached_value)

            return self._deserialize_value(cached_value)

        except Exception as e:
            logger.error("Failed to get cache entry", error=str(e), cache_type=cache_type, key=key)
            return None

    def delete_cache_entry(self, cache_type: str, key: str) -> bool:
        """Delete cache entry"""
        try:
            cache_key = f"{cache_type}_cache"
            if cache_key not in self.cache_configs:
                raise ValueError(f"Cache {cache_type} not initialized")

            cache_config = self.cache_configs[cache_key]
            client = cache_config["client"]

            intelligent_key = self._generate_cache_key(cache_type, key)
            result = client.delete(intelligent_key)

            if result > 0:
                cache_config["stats"]["deletes"] += 1
                CACHE_DELETES.labels(cache_type=cache_type).inc()

            return result > 0

        except Exception as e:
            logger.error("Failed to delete cache entry", error=str(e), cache_type=cache_type, key=key)
            return False

    def invalidate_by_tags(self, cache_type: str, tags: List[str]) -> int:
        """Invalidate cache entries by tags"""
        try:
            cache_key = f"{cache_type}_cache"
            if cache_key not in self.cache_configs:
                raise ValueError(f"Cache {cache_type} not initialized")

            cache_config = self.cache_configs[cache_key]
            client = cache_config["client"]

            tag_key = f"tag:{cache_type}:{','.join(sorted(tags))}"
            keys_to_delete = client.smembers(tag_key)

            if keys_to_delete:
                client.delete(*keys_to_delete)
                client.delete(tag_key)  # Clean up tag set

                deleted_count = len(keys_to_delete)
                cache_config["stats"]["deletes"] += deleted_count
                CACHE_DELETES.labels(cache_type=cache_type).inc(deleted_count)

                logger.info("Cache entries invalidated by tags",
                          cache_type=cache_type, tags=tags, deleted=deleted_count)
                return deleted_count

            return 0

        except Exception as e:
            logger.error("Failed to invalidate by tags", error=str(e), cache_type=cache_type, tags=tags)
            return 0

    def warmup_cache(self, request: CacheWarmupRequest) -> Dict[str, Any]:
        """Warmup cache from data source"""
        start_time = time.time()

        try:
            if request.data_source == "database":
                result = self._warmup_from_database(request)
            elif request.data_source == "file":
                result = self._warmup_from_file(request)
            else:
                raise ValueError(f"Unsupported data source: {request.data_source}")

            warmup_time = time.time() - start_time
            CACHE_WARMUP_TIME.labels(data_source=request.data_source).observe(warmup_time)

            logger.info("Cache warmup completed",
                       data_source=request.data_source,
                       cache_type=request.cache_type,
                       entries_cached=result["entries_cached"],
                       duration=warmup_time)

            return result

        except Exception as e:
            logger.error("Cache warmup failed", error=str(e), data_source=request.data_source)
            raise

    def _warmup_from_database(self, request: CacheWarmupRequest) -> Dict[str, Any]:
        """Warmup cache from database query"""
        try:
            with database_connection.cursor() as cursor:
                cursor.execute(request.query or "SELECT * FROM information_schema.tables LIMIT 100")

                rows = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]

                entries_cached = 0

                for row in rows:
                    # Convert row to dict
                    row_dict = dict(zip(column_names, row))

                    # Generate cache key
                    cache_key = f"db:{request.query}:{row[column_names[0]]}"

                    # Cache the row
                    self.set_cache_entry(
                        request.cache_type,
                        cache_key,
                        row_dict,
                        request.ttl_seconds,
                        ["database", "warmup"]
                    )

                    entries_cached += 1

                    # Process in batches to avoid memory issues
                    if entries_cached % request.batch_size == 0:
                        logger.info("Cache warmup progress", entries_cached=entries_cached)

                return {
                    "status": "completed",
                    "entries_cached": entries_cached,
                    "data_source": "database",
                    "query": request.query
                }

        except Exception as e:
            logger.error("Database warmup failed", error=str(e))
            raise

    def _warmup_from_file(self, request: CacheWarmupRequest) -> Dict[str, Any]:
        """Warmup cache from file (placeholder for file-based warmup)"""
        # This would implement file-based cache warming
        # For now, return a placeholder response
        return {
            "status": "completed",
            "entries_cached": 0,
            "data_source": "file",
            "note": "File-based warmup not yet implemented"
        }

    def get_cache_stats(self, cache_type: str) -> Optional[CacheStats]:
        """Get comprehensive cache statistics"""
        try:
            cache_key = f"{cache_type}_cache"
            if cache_key not in self.cache_configs:
                return None

            cache_config = self.cache_configs[cache_key]
            client = cache_config["client"]
            stats = cache_config["stats"]

            # Get Redis stats
            info = client.info()

            total_keys = client.dbsize()
            memory_usage = info.get("used_memory", 0)

            # Calculate hit ratio
            total_requests = stats["hits"] + stats["misses"]
            hit_ratio = stats["hits"] / total_requests if total_requests > 0 else 0

            # Estimate compression ratio (simplified)
            compression_ratio = 1.0  # Would need more sophisticated calculation

            return CacheStats(
                cache_type=cache_type,
                total_keys=total_keys,
                memory_usage_bytes=memory_usage,
                hit_ratio=hit_ratio,
                avg_ttl_seconds=self.default_ttl,
                compression_ratio=compression_ratio,
                warmup_status="ready"
            )

        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e), cache_type=cache_type)
            return None

    def _generate_cache_key(self, cache_type: str, key: str) -> str:
        """Generate intelligent cache key with hash for consistency"""
        # Create a consistent hash of the key for cache key generation
        key_hash = hashlib.md5(key.encode()).hexdigest()[:8]
        return f"{cache_type}:{key_hash}:{key}"

    def _serialize_value(self, value: Any) -> str:
        """Serialize value for caching"""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, (int, float, bool)):
            return str(value)
        else:
            return str(value)

    def _deserialize_value(self, value: str) -> Any:
        """Deserialize cached value"""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def _compress_value(self, value: str) -> bytes:
        """Compress value for storage"""
        compressed = zlib.compress(value.encode('utf-8'))
        return b'compressed:' + compressed

    def _decompress_value(self, compressed_value: bytes) -> str:
        """Decompress cached value"""
        if compressed_value.startswith(b'compressed:'):
            decompressed = zlib.decompress(compressed_value[11:])  # Remove 'compressed:' prefix
            return decompressed.decode('utf-8')
        return compressed_value.decode('utf-8')

# Global manager instance
cache_manager = RedisCacheManager()

def initialize_caches():
    """Initialize default caches"""
    cache_configs = [
        CacheConfig(cache_type="ingestion", max_memory_mb=256, ttl_default=1800),  # 30 min
        CacheConfig(cache_type="output", max_memory_mb=256, ttl_default=3600),    # 1 hour
        CacheConfig(cache_type="metadata", max_memory_mb=128, ttl_default=7200),  # 2 hours
        CacheConfig(cache_type="session", max_memory_mb=64, ttl_default=86400),   # 24 hours
    ]

    for config in cache_configs:
        try:
            cache_manager.initialize_cache(config)
            logger.info("Cache initialized", cache_type=config.cache_type)
        except Exception as e:
            logger.warning("Failed to initialize cache", cache_type=config.cache_type, error=str(e))

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    cache_status = {}
    for cache_type in ["ingestion", "output", "metadata"]:
        try:
            stats = cache_manager.get_cache_stats(cache_type)
            cache_status[cache_type] = {
                "status": "healthy" if stats else "not_initialized",
                "keys": stats.total_keys if stats else 0
            }
        except:
            cache_status[cache_type] = {"status": "error"}

    return {
        "status": "healthy",
        "service": "redis-caching-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "caches": cache_status
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/cache/{cache_type}")
async def set_cache(cache_type: str, entry: CacheEntry):
    """Set cache entry"""
    try:
        success = cache_manager.set_cache_entry(
            cache_type,
            entry.key,
            entry.value,
            entry.ttl_seconds,
            entry.tags
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to set cache entry")

        return {"status": "cached", "key": entry.key}

    except Exception as e:
        logger.error("Cache set failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache operation failed: {str(e)}")

@app.get("/cache/{cache_type}/{key}")
async def get_cache(cache_type: str, key: str):
    """Get cache entry"""
    try:
        value = cache_manager.get_cache_entry(cache_type, key)

        if value is None:
            raise HTTPException(status_code=404, detail="Cache entry not found")

        return {"key": key, "value": value, "cache_type": cache_type}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Cache get failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache operation failed: {str(e)}")

@app.delete("/cache/{cache_type}/{key}")
async def delete_cache(cache_type: str, key: str):
    """Delete cache entry"""
    try:
        success = cache_manager.delete_cache_entry(cache_type, key)

        return {"status": "deleted" if success else "not_found", "key": key}

    except Exception as e:
        logger.error("Cache delete failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache operation failed: {str(e)}")

@app.post("/cache/invalidate")
async def invalidate_cache(cache_type: str, tags: List[str]):
    """Invalidate cache entries by tags"""
    try:
        deleted_count = cache_manager.invalidate_by_tags(cache_type, tags)

        return {
            "status": "invalidated",
            "cache_type": cache_type,
            "tags": tags,
            "entries_deleted": deleted_count
        }

    except Exception as e:
        logger.error("Cache invalidation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache invalidation failed: {str(e)}")

@app.post("/cache/warmup")
async def warmup_cache(request: CacheWarmupRequest, background_tasks: BackgroundTasks):
    """Warmup cache from data source"""
    try:
        # Start warmup in background
        background_tasks.add_task(cache_manager.warmup_cache, request)

        return {
            "status": "warming_up",
            "data_source": request.data_source,
            "cache_type": request.cache_type,
            "message": "Cache warmup started in background"
        }

    except Exception as e:
        logger.error("Cache warmup failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache warmup failed: {str(e)}")

@app.get("/cache/stats/{cache_type}")
async def get_cache_stats(cache_type: str):
    """Get cache statistics"""
    try:
        stats = cache_manager.get_cache_stats(cache_type)

        if stats is None:
            raise HTTPException(status_code=404, detail=f"Cache {cache_type} not found")

        return stats.dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get cache stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

@app.get("/cache/stats")
async def get_all_cache_stats():
    """Get statistics for all caches"""
    stats = {}
    for cache_type in ["ingestion", "output", "metadata", "session"]:
        try:
            cache_stats = cache_manager.get_cache_stats(cache_type)
            if cache_stats:
                stats[cache_type] = cache_stats.dict()
        except:
            stats[cache_type] = {"error": "Failed to get stats"}

    return {"caches": stats}

@app.post("/cache/flush/{cache_type}")
async def flush_cache(cache_type: str):
    """Flush entire cache"""
    try:
        cache_key = f"{cache_type}_cache"
        if cache_key not in cache_manager.cache_configs:
            raise HTTPException(status_code=404, detail=f"Cache {cache_type} not found")

        client = cache_manager.cache_configs[cache_key]["client"]
        client.flushdb()

        return {"status": "flushed", "cache_type": cache_type}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Cache flush failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache flush failed: {str(e)}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global database_connection

    logger.info("Redis Caching Service starting up...")

    # Setup database connection
    try:
        db_config = {
            "host": os.getenv("POSTGRES_HOST", "postgresql_ingestion"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agentic_ingestion"),
            "user": os.getenv("POSTGRES_USER", "agentic_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "")
        if not os.getenv("POSTGRES_PASSWORD"):
            logger.error("POSTGRES_PASSWORD not configured for Redis Caching Service")
            raise RuntimeError("POSTGRES_PASSWORD not configured for Redis Caching Service")
        }

        database_connection = psycopg2.connect(**db_config)
        logger.info("Database connection established")

    except Exception as e:
        logger.warning("Database connection failed", error=str(e))

    # Initialize caches
    initialize_caches()

    logger.info("Redis Caching Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global database_connection

    logger.info("Redis Caching Service shutting down...")

    # Close database connection
    if database_connection:
        database_connection.close()

    logger.info("Redis Caching Service shutdown complete")

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
        "redis_caching_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8092)),
        reload=False,
        log_level="info"
    )
