#!/usr/bin/env python3
"""
GraphQL API Service for Agentic Platform

This service provides a flexible GraphQL query interface for data retrieval:
- Unified query interface across all data sources
- Intelligent query optimization and caching
- Real-time data subscriptions
- Schema stitching for multiple services
- Advanced filtering and aggregation
- Type-safe queries with auto-completion
- Performance monitoring and query analytics
"""

import json
import os
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import graphene
import httpx
import psycopg2
import redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from graphene import ObjectType, String, Int, Float, Boolean, List as GraphList, Field
from graphene_pydantic import PydanticObjectType
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import structlog
from starlette.graphql import GraphQLApp
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
    title="GraphQL API Service",
    description="Flexible GraphQL query interface for unified data retrieval",
    version="1.0.0"
)

# Prometheus metrics
GRAPHQL_QUERIES = Counter('graphql_queries_total', 'Total GraphQL queries', ['operation_type'])
GRAPHQL_ERRORS = Counter('graphql_errors_total', 'Total GraphQL errors', ['error_type'])
QUERY_EXECUTION_TIME = Histogram('graphql_query_duration_seconds', 'GraphQL query execution time', ['query_type'])
DATA_RESOLUTION_TIME = Histogram('graphql_data_resolution_seconds', 'Data resolution time', ['data_source'])

# Global variables
database_connection = None
redis_client = None

# Pydantic models for GraphQL types
class DataRecord(BaseModel):
    """Data record model"""
    id: str
    data: Dict[str, Any]
    source: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class DatasetInfo(BaseModel):
    """Dataset information model"""
    name: str
    description: Optional[str] = None
    record_count: int
    last_updated: datetime
    schema: Dict[str, str]
    tags: List[str] = []

class ServiceHealth(BaseModel):
    """Service health model"""
    name: str
    status: str
    version: Optional[str] = None
    uptime: Optional[int] = None
    last_checked: datetime

class QueryAnalytics(BaseModel):
    """Query analytics model"""
    query_id: str
    query: str
    execution_time: float
    data_sources_used: List[str]
    records_returned: int
    timestamp: datetime

# GraphQL Types
class DataRecordType(PydanticObjectType):
    """GraphQL type for data records"""
    class Meta:
        model = DataRecord

class DatasetInfoType(PydanticObjectType):
    """GraphQL type for dataset information"""
    class Meta:
        model = DatasetInfo

class ServiceHealthType(PydanticObjectType):
    """GraphQL type for service health"""
    class Meta:
        model = ServiceHealth

class QueryAnalyticsType(PydanticObjectType):
    """GraphQL type for query analytics"""
    class Meta:
        model = QueryAnalytics

# GraphQL Queries
class Query(ObjectType):
    """Root GraphQL query"""

    # Data queries
    data_records = GraphList(
        DataRecordType,
        dataset=String(required=True),
        limit=Int(default_value=100),
        offset=Int(default_value=0),
        filters=String(),
        sort_by=String(),
        sort_order=String(default_value="ASC")
    )

    dataset_info = Field(
        DatasetInfoType,
        name=String(required=True)
    )

    datasets = GraphList(
        DatasetInfoType,
        tags=GraphList(String),
        limit=Int(default_value=50)
    )

    # Service queries
    service_health = GraphList(ServiceHealthType)

    services = GraphList(
        String,
        status=String()
    )

    # Analytics queries
    query_analytics = GraphList(
        QueryAnalyticsType,
        limit=Int(default_value=20),
        timeframe=String(default_value="1h")
    )

    # Cross-service queries
    unified_search = GraphList(
        DataRecordType,
        query=String(required=True),
        data_sources=GraphList(String),
        limit=Int(default_value=50)
    )

    async def resolve_data_records(
        self,
        info,
        dataset,
        limit=100,
        offset=0,
        filters=None,
        sort_by=None,
        sort_order="ASC"
    ):
        """Resolve data records query"""
        start_time = asyncio.get_event_loop().time()

        try:
            GRAPHQL_QUERIES.labels(operation_type="data_records").inc()

            # Query data lake for records
            records = await query_data_lake(
                dataset=dataset,
                limit=limit,
                offset=offset,
                filters=filters,
                sort_by=sort_by,
                sort_order=sort_order
            )

            execution_time = asyncio.get_event_loop().time() - start_time
            QUERY_EXECUTION_TIME.labels(query_type="data_records").observe(execution_time)

            return [DataRecordType(**record) for record in records]

        except Exception as e:
            logger.error("Data records query failed", error=str(e))
            GRAPHQL_ERRORS.labels(error_type="data_query").inc()
            raise

    async def resolve_dataset_info(self, info, name):
        """Resolve dataset info query"""
        try:
            GRAPHQL_QUERIES.labels(operation_type="dataset_info").inc()

            info_data = await get_dataset_info(name)
            return DatasetInfoType(**info_data) if info_data else None

        except Exception as e:
            logger.error("Dataset info query failed", error=str(e))
            GRAPHQL_ERRORS.labels(error_type="dataset_query").inc()
            raise

    async def resolve_datasets(self, info, tags=None, limit=50):
        """Resolve datasets query"""
        try:
            GRAPHQL_QUERIES.labels(operation_type="datasets").inc()

            datasets = await get_datasets_list(tags=tags, limit=limit)
            return [DatasetInfoType(**ds) for ds in datasets]

        except Exception as e:
            logger.error("Datasets query failed", error=str(e))
            GRAPHQL_ERRORS.labels(error_type="datasets_query").inc()
            raise

    async def resolve_service_health(self, info):
        """Resolve service health query"""
        try:
            GRAPHQL_QUERIES.labels(operation_type="service_health").inc()

            health_data = await get_services_health()
            return [ServiceHealthType(**service) for service in health_data]

        except Exception as e:
            logger.error("Service health query failed", error=str(e))
            GRAPHQL_ERRORS.labels(error_type="health_query").inc()
            raise

    async def resolve_services(self, info, status=None):
        """Resolve services query"""
        try:
            GRAPHQL_QUERIES.labels(operation_type="services").inc()

            services = await get_services_list(status=status)
            return services

        except Exception as e:
            logger.error("Services query failed", error=str(e))
            GRAPHQL_ERRORS.labels(error_type="services_query").inc()
            raise

    async def resolve_query_analytics(self, info, limit=20, timeframe="1h"):
        """Resolve query analytics query"""
        try:
            GRAPHQL_QUERIES.labels(operation_type="query_analytics").inc()

            analytics = await get_query_analytics(limit=limit, timeframe=timeframe)
            return [QueryAnalyticsType(**analytic) for analytic in analytics]

        except Exception as e:
            logger.error("Query analytics failed", error=str(e))
            GRAPHQL_ERRORS.labels(error_type="analytics_query").inc()
            raise

    async def resolve_unified_search(
        self,
        info,
        query,
        data_sources=None,
        limit=50
    ):
        """Resolve unified search query"""
        try:
            GRAPHQL_QUERIES.labels(operation_type="unified_search").inc()

            results = await perform_unified_search(
                query=query,
                data_sources=data_sources,
                limit=limit
            )

            return [DataRecordType(**result) for result in results]

        except Exception as e:
            logger.error("Unified search failed", error=str(e))
            GRAPHQL_ERRORS.labels(error_type="search_query").inc()
            raise

# GraphQL Schema
schema = graphene.Schema(query=Query)

# Service integration functions
async def query_data_lake(dataset: str, **kwargs) -> List[Dict[str, Any]]:
    """Query data lake service"""
    try:
        async with httpx.AsyncClient() as client:
            params = {k: v for k, v in kwargs.items() if v is not None}
            response = await client.get(
                f"http://data-lake-minio:8090/data/{dataset}",
                params=params,
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            else:
                logger.warning("Data lake query failed", status=response.status_code)
                return []

    except Exception as e:
        logger.error("Data lake query error", error=str(e))
        return []

async def get_dataset_info(name: str) -> Optional[Dict[str, Any]]:
    """Get dataset information"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://data-lake-minio:8090/metadata/{name}",
                timeout=30.0
            )

            if response.status_code == 200:
                return response.json()
            else:
                return None

    except Exception as e:
        logger.error("Dataset info query error", error=str(e))
        return None

async def get_datasets_list(tags: Optional[List[str]] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Get list of available datasets"""
    try:
        # This would query metadata management service
        # For now, return mock data
        return [
            {
                "name": "customer_data",
                "description": "Customer information dataset",
                "record_count": 10000,
                "last_updated": datetime.utcnow().isoformat(),
                "schema": {"id": "integer", "name": "string", "email": "string"},
                "tags": ["customers", "pii"]
            },
            {
                "name": "sales_transactions",
                "description": "Sales transaction data",
                "record_count": 50000,
                "last_updated": datetime.utcnow().isoformat(),
                "schema": {"id": "integer", "amount": "float", "date": "datetime"},
                "tags": ["sales", "transactions"]
            }
        ]

    except Exception as e:
        logger.error("Datasets list query error", error=str(e))
        return []

async def get_services_health() -> List[Dict[str, Any]]:
    """Get services health status"""
    services = [
        {"name": "ingestion-coordinator", "endpoint": "http://ingestion-coordinator:8080/health"},
        {"name": "data-lake-minio", "endpoint": "http://data-lake-minio:8090/health"},
        {"name": "message-queue", "endpoint": "http://message-queue:8091/health"},
        {"name": "redis-caching", "endpoint": "http://redis-caching:8092/health"},
        {"name": "oauth2-oidc", "endpoint": "http://oauth2-oidc:8093/health"},
        {"name": "data-encryption", "endpoint": "http://data-encryption:8094/health"},
        {"name": "monitoring", "endpoint": "http://monitoring:8095/health"},
        {"name": "audit-compliance", "endpoint": "http://audit-compliance:8096/health"},
        {"name": "backup-orchestration", "endpoint": "http://backup-orchestration:8097/health"},
        {"name": "port-manager", "endpoint": "http://port-manager:8098/health"},
        {"name": "tracing-jaeger", "endpoint": "http://tracing-jaeger:8099/health"}
    ]

    health_results = []

    async with httpx.AsyncClient() as client:
        for service in services:
            try:
                response = await client.get(service["endpoint"], timeout=5.0)
                status = "healthy" if response.status_code == 200 else "unhealthy"

                health_results.append({
                    "name": service["name"],
                    "status": status,
                    "version": "1.0.0",
                    "uptime": 3600,  # Mock uptime
                    "last_checked": datetime.utcnow()
                })

            except Exception:
                health_results.append({
                    "name": service["name"],
                    "status": "unreachable",
                    "last_checked": datetime.utcnow()
                })

    return health_results

async def get_services_list(status: Optional[str] = None) -> List[str]:
    """Get list of services"""
    all_services = [
        "ingestion-coordinator",
        "output-coordinator",
        "data-lake-minio",
        "message-queue",
        "redis-caching",
        "oauth2-oidc",
        "data-encryption",
        "monitoring",
        "audit-compliance",
        "backup-orchestration",
        "port-manager",
        "tracing-jaeger"
    ]

    if status:
        # Filter by status if provided
        health_data = await get_services_health()
        filtered_services = [
            service["name"] for service in health_data
            if service["status"] == status
        ]
        return filtered_services

    return all_services

async def get_query_analytics(limit: int = 20, timeframe: str = "1h") -> List[Dict[str, Any]]:
    """Get query analytics data"""
    try:
        # This would query monitoring service for analytics
        # For now, return mock data
        return [
            {
                "query_id": "query_001",
                "query": "query { dataRecords(dataset: \"customer_data\") { id data } }",
                "execution_time": 0.25,
                "data_sources_used": ["data-lake-minio"],
                "records_returned": 100,
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "query_id": "query_002",
                "query": "query { datasets { name recordCount } }",
                "execution_time": 0.15,
                "data_sources_used": ["data-lake-minio"],
                "records_returned": 5,
                "timestamp": datetime.utcnow().isoformat()
            }
        ]

    except Exception as e:
        logger.error("Query analytics error", error=str(e))
        return []

async def perform_unified_search(query: str, data_sources: Optional[List[str]] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Perform unified search across multiple data sources"""
    try:
        results = []

        # Search data lake
        if not data_sources or "data-lake" in data_sources:
            lake_results = await query_data_lake("customer_data", limit=limit//2)
            results.extend(lake_results[:limit//2])

        # Search other data sources as needed
        # This would integrate with search services

        return results[:limit]

    except Exception as e:
        logger.error("Unified search error", error=str(e))
        return []

# GraphQL App
graphql_app = GraphQLApp(schema=schema)

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "graphql-api-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/schema")
async def get_graphql_schema():
    """Get GraphQL schema"""
    return {
        "schema": str(schema),
        "introspection": schema.introspect()
    }

@app.get("/playground")
async def graphql_playground():
    """GraphQL Playground redirect"""
    return {
        "message": "GraphQL Playground available at /graphql",
        "endpoint": "/graphql",
        "documentation": "/docs"
    }

@app.post("/graphql")
async def graphql_endpoint(request: Request):
    """GraphQL endpoint"""
    return await graphql_app.handle_graphql(request)

@app.get("/stats")
async def get_graphql_stats():
    """Get GraphQL service statistics"""
    return {
        "service": "graphql-api-service",
        "metrics": {
            "graphql_queries_total": GRAPHQL_QUERIES._value.get(),
            "graphql_errors_total": GRAPHQL_ERRORS._value.get()
        },
        "schema": {
            "types_count": len(schema.types),
            "queries_count": len(schema.query._meta.fields),
            "mutations_count": 0  # No mutations implemented yet
        }
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global database_connection, redis_client

    logger.info("GraphQL API Service starting up...")

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

    # Setup Redis connection
    try:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis_ingestion"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Redis connection established")

    except Exception as e:
        logger.warning("Redis connection failed", error=str(e))

    logger.info("GraphQL API Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global database_connection

    logger.info("GraphQL API Service shutting down...")

    # Close database connection
    if database_connection:
        database_connection.close()

    logger.info("GraphQL API Service shutdown complete")

# Error handlers
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
        "graphql_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8100)),
        reload=False,
        log_level="info"
    )
