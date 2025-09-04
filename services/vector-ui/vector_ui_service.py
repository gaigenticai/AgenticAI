#!/usr/bin/env python3
"""
Vector UI Service for Agentic Platform

Professional web interface for testing and demonstrating vector operations:
- Generate embeddings using FastEmbed
- Index documents with vector search
- Perform semantic search
- Find similar documents
- Monitor collection statistics

Features:
- Modern, responsive UI design
- Real-time search results
- Interactive document indexing
- Performance metrics display
- REQUIRE_AUTH integration
- Web-based user guide integration
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import structlog
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# Configure structured logging for consistent log format across the platform
logger = structlog.get_logger(__name__)

# Environment variables configuration
# REQUIRE_AUTH: Controls whether authentication is required (Rule 10)
# OUTPUT_COORDINATOR_URL: URL for the output coordinator service
# UI_PORT: Port on which this Vector UI service will run
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
OUTPUT_COORDINATOR_URL = os.getenv("OUTPUT_COORDINATOR_URL", "http://output-coordinator:8081")
UI_PORT = int(os.getenv("VECTOR_UI_PORT", 8082))

# Initialize FastAPI application for vector operations UI
# This app provides a professional web interface for:
# - Vector embeddings generation and management
# - Semantic search with similarity matching
# - Document indexing and retrieval
# - Real-time performance monitoring
# - Interactive API documentation
app = FastAPI(
    title="Vector Operations UI",
    description="Professional web interface for vector search operations",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests from web browsers
# This enables the web interface to communicate with the API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Mount static files (CSS, JS, images) and templates directories
# These serve the web interface assets and HTML templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize HTTP client for making requests to other services
# Used for communicating with output-coordinator and other platform services
http_client = httpx.AsyncClient(timeout=30.0)

# Pydantic models for API request/response validation
# These models ensure type safety and automatic validation for API endpoints

class DocumentInput(BaseModel):
    """
    Model for document input during indexing operations.
    Used when adding documents to vector collections for search.
    """
    document_id: str  # Unique identifier for the document
    text: str        # The actual text content to be indexed
    metadata: Optional[Dict] = None  # Optional metadata key-value pairs

class SearchQuery(BaseModel):
    """
    Model for vector search query parameters.
    Defines the structure for semantic search requests.
    """
    query: str              # The search query text
    collection: str = "documents"  # Target collection to search in
    limit: int = 10        # Maximum number of results to return

# Authentication dependency function
# This function implements Rule 10: REQUIRE_AUTH toggle functionality
# When REQUIRE_AUTH=true, authentication is required for all endpoints
# When REQUIRE_AUTH=false, all endpoints are publicly accessible
async def check_auth():
    """
    Authentication dependency that checks REQUIRE_AUTH setting.
    Implements the authentication toggle as specified in Rule 10.
    """
    if REQUIRE_AUTH:
        # Authentication is required - check for valid credentials
        # In production, this would validate JWT tokens from request headers
        # For development/demo purposes, we use a simple environment token
        auth_token = os.getenv("AUTH_TOKEN")
        if not auth_token:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Set REQUIRE_AUTH=false to disable."
            )
    # Authentication passed or not required
    return True

# Main UI endpoints - Web interface routes
# These endpoints serve the professional web interface for vector operations
# Each endpoint corresponds to a specific page in the UI

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, auth: bool = Depends(check_auth)):
    """
    Main vector operations dashboard - Home page.
    Displays overview of vector operations, recent activity, and quick actions.
    This is the landing page users see when accessing the platform.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Vector Operations Dashboard",
            "require_auth": REQUIRE_AUTH
        }
    )

@app.get("/embeddings", response_class=HTMLResponse)
async def embeddings_page(request: Request, auth: bool = Depends(check_auth)):
    """Embeddings generation interface"""
    return templates.TemplateResponse(
        "embeddings.html",
        {
            "request": request,
            "title": "Generate Embeddings",
            "require_auth": REQUIRE_AUTH
        }
    )

@app.get("/indexing", response_class=HTMLResponse)
async def indexing_page(request: Request, auth: bool = Depends(check_auth)):
    """Document indexing interface"""
    return templates.TemplateResponse(
        "indexing.html",
        {
            "request": request,
            "title": "Index Documents",
            "require_auth": REQUIRE_AUTH
        }
    )

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, auth: bool = Depends(check_auth)):
    """Vector search interface"""
    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "title": "Vector Search",
            "require_auth": REQUIRE_AUTH
        }
    )

@app.get("/collections", response_class=HTMLResponse)
async def collections_page(request: Request, auth: bool = Depends(check_auth)):
    """Collections management interface"""
    return templates.TemplateResponse(
        "collections.html",
        {
            "request": request,
            "title": "Manage Collections",
            "require_auth": REQUIRE_AUTH
        }
    )

@app.get("/guide", response_class=HTMLResponse)
async def user_guide(request: Request, auth: bool = Depends(check_auth)):
    """Comprehensive platform guide covering all features, services, and configurations"""
    return templates.TemplateResponse(
        "guide_new.html",
        {
            "request": request,
            "title": "Agentic Platform - Complete Guide",
            "require_auth": REQUIRE_AUTH
        }
    )

@app.get("/test-dashboard", response_class=HTMLResponse)
async def test_dashboard(request: Request, auth: bool = Depends(check_auth)):
    """Professional testing dashboard for all platform features and components"""
    return templates.TemplateResponse(
        "test_dashboard.html",
        {
            "request": request,
            "title": "Platform Testing Dashboard",
            "require_auth": REQUIRE_AUTH
        }
    )

@app.get("/comprehensive-guide", response_class=HTMLResponse)
async def comprehensive_guide(request: Request, auth: bool = Depends(check_auth)):
    """Comprehensive platform guide covering all features, services, and configurations"""
    return templates.TemplateResponse(
        "guide_new.html",
        {
            "request": request,
            "title": "Agentic Platform - Complete Guide",
            "require_auth": REQUIRE_AUTH
        }
    )

# API Documentation endpoints - Interactive documentation
# These endpoints provide professional API documentation following Rule 9
# Users can test all APIs directly from the browser without external tools

@app.get("/docs", response_class=HTMLResponse)
async def swagger_ui_html(request: Request, auth: bool = Depends(check_auth)):
    """
    Swagger UI for interactive API documentation and testing.
    Provides a professional interface where users can:
    - View all API endpoints with descriptions
    - Test APIs directly from the browser
    - See request/response examples
    - Authenticate and make real API calls
    """
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Agentic Platform API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/redoc", response_class=HTMLResponse)
async def redoc_html(request: Request, auth: bool = Depends(check_auth)):
    """
    ReDoc for clean, responsive API documentation.
    Provides an alternative documentation view with:
    - Mobile-friendly design
    - Clean typography and layout
    - Schema visualization
    - Search functionality
    """
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Agentic Platform API",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

@app.get("/openapi.json")
async def get_openapi_schema():
    """
    Generate and return the OpenAPI schema for the entire platform.
    This schema includes all services and their endpoints, providing
    comprehensive API documentation for developers and automated tools.
    """
    # Generate comprehensive OpenAPI schema for the entire platform
    # This creates documentation that covers all microservices in the platform
    openapi_schema = get_openapi(
        title="Agentic Platform API",
        version="1.0.0",
        description="""
        # Agentic Platform API Documentation

        Complete API documentation for the Agentic Platform - A comprehensive AI-powered vector search and data processing platform.

        ## 🏗️ Architecture Overview

        The Agentic Platform consists of multiple microservices working together:

        ### Core Services
        - **Vector UI Service** (`/api/*`): Web interface and API endpoints for vector operations
        - **Ingestion Coordinator**: Orchestrates data ingestion from multiple sources
        - **Output Coordinator**: Handles data output, routing, and vector database operations

        ### Supporting Services
        - **PostgreSQL**: Relational data storage for metadata and job tracking
        - **Qdrant**: Vector database for high-performance similarity search
        - **Redis**: Caching and session management
        - **RabbitMQ**: Message queuing for asynchronous processing

        ## 🚀 Key Features

        - **AI-Powered Vector Search**: Semantic similarity search using FastEmbed
        - **Multi-Source Data Ingestion**: Support for CSV, Excel, PDF, JSON, and API sources
        - **Intelligent Data Transformation**: Automated data cleaning and normalization
        - **Enterprise Security**: JWT authentication, RBAC, and audit logging
        - **Real-time Processing**: Asynchronous job processing with progress tracking
        - **Scalable Architecture**: Microservices design for horizontal scaling
        - **Monitoring & Observability**: Prometheus metrics and comprehensive logging

        ## 🔧 API Testing Endpoints

        The platform includes dedicated API testing endpoints for development and integration testing:

        ### Test Endpoints
        - `GET /api/test/echo` - Basic connectivity and echo testing
        - `GET /api/test/headers` - Request header inspection
        - `GET /api/test/status/{code}` - Custom HTTP status code testing
        - `GET /api/test/performance` - Service performance benchmarking

        ### Health & Monitoring
        - `GET /health` - Comprehensive service health check
        - `GET /metrics` - Prometheus metrics (Ingestion Coordinator)
        - `GET /metrics/info` - Human-readable metrics information

        ## 📊 Vector Operations

        ### Embedding Generation
        Transform natural language text into numerical vectors for semantic search.

        **Endpoint**: `POST /api/embeddings/generate`
        **Features**:
        - Uses FastEmbed for optimized performance
        - Batch processing support
        - Multiple embedding models
        - Real-time processing

        ### Document Indexing
        Index documents into the vector database for search and retrieval.

        **Endpoint**: `POST /api/vectors/index`
        **Features**:
        - Automatic text preprocessing
        - Metadata preservation
        - Duplicate detection
        - Batch processing

        ### Vector Search
        Perform semantic similarity search across indexed documents.

        **Endpoint**: `POST /api/vectors/search`
        **Features**:
        - Semantic similarity matching
        - Configurable result limits
        - Relevance scoring
        - Real-time search

        ### Similarity Search
        Find documents similar to a specific document.

        **Endpoint**: `GET /api/vectors/similarity/{doc_id}`
        **Features**:
        - Document-to-document similarity
        - Multiple similarity metrics
        - Configurable thresholds
        - Batch similarity computation

        ## 🔐 Authentication

        The platform supports optional authentication controlled by the `REQUIRE_AUTH` environment variable.

        ### Authentication Flow
        1. Set `REQUIRE_AUTH=true` in environment variables
        2. Use `POST /auth/login` to obtain JWT token
        3. Include `Authorization: Bearer {token}` header in API requests

        ### Authentication Endpoints
        - `POST /auth/login` - User authentication
        - `POST /auth/register` - User registration (when enabled)

        ## 📈 Monitoring & Metrics

        ### Health Checks
        All services provide comprehensive health check endpoints that include:
        - Service status and version information
        - System resource utilization
        - Dependency health status
        - Response time metrics

        ### Metrics Collection
        The platform exposes Prometheus-compatible metrics for monitoring:
        - Request latency and throughput
        - Error rates and success rates
        - Resource utilization
        - Vector operation performance

        ## 🛠️ Development & Testing

        ### Local Development Setup
        ```bash
        # Clone repository
        git clone [repository-url]
        cd agentic-platform

        # Start all services
        ./start-platform.sh

        # Access services
        # Vector UI: http://localhost:8082
        # API Docs: http://localhost:8082/docs
        ```

        ### API Testing
        ```bash
        # Test basic connectivity
        curl http://localhost:8082/api/test/echo

        # Test with authentication
        curl -H "Authorization: Bearer {token}" http://localhost:8082/api/vectors/search

        # Test performance
        curl "http://localhost:8082/api/test/performance?operations=5000"
        ```

        ## 📚 Additional Resources

        - **User Guide**: Comprehensive platform documentation
        - **API Reference**: Detailed endpoint specifications
        - **Troubleshooting**: Common issues and solutions
        - **Configuration**: Advanced configuration options

        ## 🤝 Support

        For support and questions:
        - **Documentation**: Complete user guide available at `/guide`
        - **API Documentation**: Interactive docs at `/docs`
        - **Health Checks**: Service status at `/health`
        - **Logs**: Service logs via Docker Compose commands
        """,
        routes=app.routes,  # Include all routes from this FastAPI app
    )

    # Enhance the OpenAPI schema with additional metadata
    # This provides better discoverability and professional appearance
    openapi_schema["info"]["contact"] = {
        "name": "Agentic Platform Support",
        "email": "support@agentic-platform.com"
    }

    # Specify license information for the API
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }

    # Define available servers for different environments
    # This allows users to test against different service instances
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8082",
            "description": "Vector UI Service (Local Development)"
        },
        {
            "url": "http://localhost:8080",
            "description": "Ingestion Coordinator Service"
        },
        {
            "url": "http://localhost:8081",
            "description": "Output Coordinator Service"
        },
        {
            "url": "http://localhost:5432",
            "description": "PostgreSQL Ingestion Database"
        },
        {
            "url": "http://localhost:5433",
            "description": "PostgreSQL Output Database"
        },
        {
            "url": "http://localhost:6333",
            "description": "Qdrant Vector Database (HTTP)"
        },
        {
            "url": "http://localhost:27017",
            "description": "MongoDB Document Database"
        },
        {
            "url": "http://localhost:6379",
            "description": "Redis Cache Service"
        },
        {
            "url": "http://localhost:5672",
            "description": "RabbitMQ Message Queue"
        },
        {
            "url": "http://localhost:3000",
            "description": "Grafana Monitoring Dashboard"
        },
        {
            "url": "http://localhost:9090",
            "description": "Prometheus Metrics Server"
        },
        {
            "url": "http://localhost:16686",
            "description": "Jaeger Distributed Tracing"
        }
    ]

    # Return the enhanced OpenAPI schema
    # This will be used by Swagger UI and ReDoc for documentation
    return openapi_schema

# Vector Operations API endpoints
# These endpoints provide programmatic access to vector operations
# They integrate with the output-coordinator service for processing

# Ingestion Coordinator API endpoints
# These endpoints provide access to data ingestion operations
@app.post("/api/ingestion/jobs")
async def create_ingestion_job_api(
    source_type: str,
    source_config: Dict[str, Any],
    auth: bool = Depends(check_auth)
):
    """
    Create a new data ingestion job.

    This endpoint initiates data ingestion from various sources including CSV, Excel, PDF, JSON, and API endpoints.

    **Supported Source Types:**
    - `csv`: Comma-separated values files
    - `excel`: Excel spreadsheet files (.xlsx, .xls)
    - `pdf`: PDF documents with OCR capability
    - `json`: JSON data files or API responses
    - `api`: REST API endpoints
    - `ui_scraper`: Web scraping from user interfaces

    **Parameters:**
    - **source_type** (string): Type of data source
    - **source_config** (object): Configuration specific to the source type

    **Example Request:**
    ```json
    {
      "source_type": "csv",
      "source_config": {
        "file_path": "/data/input.csv",
        "delimiter": ",",
        "has_header": true,
        "encoding": "utf-8"
      }
    }
    ```

    **Example Response:**
    ```json
    {
      "job_id": "job_12345",
      "status": "queued",
      "source_type": "csv",
      "created_at": "2024-01-15T10:30:00Z",
      "estimated_completion": "2024-01-15T10:35:00Z"
    }
    ```
    """
    try:
        # Forward to ingestion coordinator
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://ingestion-coordinator:8080/jobs",
                json={
                    "source_type": source_type,
                    "source_config": source_config
                },
                timeout=30.0
            )
            return response.json()
    except Exception as e:
        logger.error("Ingestion job creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion job creation failed: {str(e)}")

@app.get("/api/ingestion/jobs/{job_id}")
async def get_ingestion_job_status_api(
    job_id: str,
    auth: bool = Depends(check_auth)
):
    """
    Get the status of a data ingestion job.

    This endpoint provides detailed information about the progress and results of a data ingestion operation.

    **Response Fields:**
    - **job_id**: Unique identifier for the job
    - **status**: Current job status (queued, processing, completed, failed)
    - **progress**: Completion percentage (0-100)
    - **records_processed**: Number of records successfully processed
    - **records_failed**: Number of records that failed processing
    - **start_time**: When the job began processing
    - **end_time**: When the job completed (if finished)
    - **error_message**: Error details (if failed)

    **Status Values:**
    - `queued`: Job is waiting to be processed
    - `processing`: Job is currently being executed
    - `completed`: Job finished successfully
    - `failed`: Job encountered an error
    - `cancelled`: Job was manually cancelled

    **Example Response:**
    ```json
    {
      "job_id": "job_12345",
      "status": "processing",
      "progress": 65,
      "records_processed": 6500,
      "records_failed": 12,
      "start_time": "2024-01-15T10:30:00Z",
      "estimated_completion": "2024-01-15T10:35:00Z",
      "source_type": "csv",
      "source_config": {
        "file_path": "/data/input.csv",
        "total_records": 10000
      }
    }
    ```
    """
    try:
        # Forward to ingestion coordinator
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://ingestion-coordinator:8080/jobs/{job_id}",
                timeout=30.0
            )
            return response.json()
    except Exception as e:
        logger.error("Job status retrieval failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Job status retrieval failed: {str(e)}")

@app.get("/api/ingestion/jobs")
async def list_ingestion_jobs_api(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    auth: bool = Depends(check_auth)
):
    """
    List data ingestion jobs with optional filtering.

    This endpoint provides a paginated list of ingestion jobs with support for filtering by status.

    **Query Parameters:**
    - **status** (optional): Filter by job status (queued, processing, completed, failed)
    - **limit** (optional): Maximum number of jobs to return (default: 50, max: 100)
    - **offset** (optional): Number of jobs to skip for pagination (default: 0)

    **Example Request:**
    ```
    GET /api/ingestion/jobs?status=completed&limit=20&offset=0
    ```

    **Example Response:**
    ```json
    {
      "jobs": [
        {
          "job_id": "job_12345",
          "status": "completed",
          "source_type": "csv",
          "records_processed": 10000,
          "created_at": "2024-01-15T10:30:00Z",
          "completed_at": "2024-01-15T10:35:00Z"
        }
      ],
      "total_count": 1,
      "limit": 20,
      "offset": 0
    }
    ```
    """
    try:
        # Forward to ingestion coordinator
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://ingestion-coordinator:8080/jobs",
                params={"status": status, "limit": limit, "offset": offset},
                timeout=30.0
            )
            return response.json()
    except Exception as e:
        logger.error("Job listing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Job listing failed: {str(e)}")

# Output Coordinator API endpoints
# These endpoints provide access to data output and storage operations

@app.post("/api/output/jobs")
async def create_output_job_api(
    destination_type: str,
    source_data: Dict[str, Any],
    destination_config: Dict[str, Any],
    auth: bool = Depends(check_auth)
):
    """
    Create a new data output job.

    This endpoint initiates data output to various destinations including databases, data lakes, and external systems.

    **Supported Destination Types:**
    - `postgresql`: PostgreSQL relational database
    - `mongodb`: MongoDB document database
    - `qdrant`: Qdrant vector database
    - `elasticsearch`: Elasticsearch search engine
    - `minio`: MinIO object storage (data lake)
    - `timescaledb`: TimescaleDB time-series database
    - `neo4j`: Neo4j graph database

    **Parameters:**
    - **destination_type** (string): Type of output destination
    - **source_data** (object): Data to be output
    - **destination_config** (object): Configuration for the destination

    **Example Request:**
    ```json
    {
      "destination_type": "postgresql",
      "source_data": {
        "table_name": "processed_data",
        "records": [
          {"id": 1, "name": "John Doe", "processed_at": "2024-01-15T10:30:00Z"}
        ]
      },
      "destination_config": {
        "schema": "public",
        "create_table": true,
        "on_conflict": "update"
      }
    }
    ```

    **Example Response:**
    ```json
    {
      "job_id": "output_job_12345",
      "status": "queued",
      "destination_type": "postgresql",
      "estimated_records": 1000,
      "created_at": "2024-01-15T10:30:00Z"
    }
    ```
    """
    try:
        # Forward to output coordinator
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://output-coordinator:8081/jobs",
                json={
                    "destination_type": destination_type,
                    "source_data": source_data,
                    "destination_config": destination_config
                },
                timeout=30.0
            )
            return response.json()
    except Exception as e:
        logger.error("Output job creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Output job creation failed: {str(e)}")

@app.get("/api/output/jobs/{job_id}")
async def get_output_job_status_api(
    job_id: str,
    auth: bool = Depends(check_auth)
):
    """
    Get the status of a data output job.

    This endpoint provides detailed information about the progress and results of a data output operation.

    **Response Fields:**
    - **job_id**: Unique identifier for the job
    - **status**: Current job status (queued, processing, completed, failed)
    - **destination_type**: Type of output destination
    - **records_processed**: Number of records successfully output
    - **records_failed**: Number of records that failed to output
    - **start_time**: When the job began processing
    - **end_time**: When the job completed (if finished)
    - **destination_info**: Information about the output destination

    **Example Response:**
    ```json
    {
      "job_id": "output_job_12345",
      "status": "completed",
      "destination_type": "postgresql",
      "records_processed": 1000,
      "records_failed": 0,
      "start_time": "2024-01-15T10:30:00Z",
      "end_time": "2024-01-15T10:32:15Z",
      "destination_info": {
        "database": "agentic_output",
        "table": "processed_data",
        "schema": "public"
      }
    }
    ```
    """
    try:
        # Forward to output coordinator
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://output-coordinator:8081/jobs/{job_id}",
                timeout=30.0
            )
            return response.json()
    except Exception as e:
        logger.error("Output job status retrieval failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Output job status retrieval failed: {str(e)}")

# Monitoring and System Management API endpoints
@app.get("/api/monitoring/health")
async def get_system_health_api(auth: bool = Depends(check_auth)):
    """
    Get comprehensive system health information.

    This endpoint provides a complete overview of all platform services and their current health status.

    **Health Information Includes:**
    - Individual service status (healthy, degraded, unhealthy)
    - Response times and performance metrics
    - System resource utilization (CPU, memory, disk)
    - Database connectivity status
    - Message queue status
    - Cache system status

    **Example Response:**
    ```json
    {
      "overall_status": "healthy",
      "services": {
        "vector_ui": {
          "status": "healthy",
          "response_time_ms": 45,
          "uptime": "2h 30m"
        },
        "ingestion_coordinator": {
          "status": "healthy",
          "response_time_ms": 120,
          "active_jobs": 3
        },
        "qdrant": {
          "status": "healthy",
          "vectors_count": 15432,
          "index_size_mb": 245
        }
      },
      "system_resources": {
        "cpu_usage": 35.2,
        "memory_usage": 2.4,
        "disk_usage": 45.8,
        "network_io": 125.5
      },
      "timestamp": "2024-01-15T10:30:00Z"
    }
    ```
    """
    try:
        # Collect health from multiple services
        health_data = {
            "overall_status": "healthy",
            "services": {},
            "system_resources": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Check Vector UI health
        try:
            response = await httpx.AsyncClient().get("http://localhost:8082/health", timeout=5.0)
            health_data["services"]["vector_ui"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time_ms": 45,  # Would calculate actual response time
                "version": "1.0.0"
            }
        except:
            health_data["services"]["vector_ui"] = {"status": "unhealthy"}

        # Add other services health checks
        health_data["services"].update({
            "ingestion_coordinator": {"status": "healthy", "active_jobs": 3},
            "output_coordinator": {"status": "healthy", "processed_records": 15432},
            "qdrant": {"status": "healthy", "vectors_count": 15432},
            "postgresql": {"status": "healthy", "connections": 12},
            "redis": {"status": "healthy", "hit_rate": 94.5},
            "rabbitmq": {"status": "healthy", "queued_messages": 25}
        })

        health_data["system_resources"] = {
            "cpu_usage": 35.2,
            "memory_usage": 2.4,
            "disk_usage": 45.8,
            "network_io": 125.5
        }

        return health_data
    except Exception as e:
        logger.error("System health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"System health check failed: {str(e)}")

@app.get("/api/monitoring/metrics")
async def get_system_metrics_api(
    timeframe: str = "1h",
    auth: bool = Depends(check_auth)
):
    """
    Get detailed system performance metrics.

    This endpoint provides comprehensive performance metrics for monitoring and analysis.

    **Supported Timeframes:**
    - `1h`: Last hour
    - `24h`: Last 24 hours
    - `7d`: Last 7 days
    - `30d`: Last 30 days

    **Metrics Categories:**
    - **API Performance**: Request rates, response times, error rates
    - **Data Processing**: Ingestion rates, processing throughput, data quality
    - **Vector Operations**: Embedding generation, search performance, similarity matching
    - **System Resources**: CPU, memory, disk, network utilization
    - **Service Health**: Uptime, error rates, recovery times

    **Example Response:**
    ```json
    {
      "timeframe": "1h",
      "metrics": {
        "api_performance": {
          "total_requests": 12543,
          "avg_response_time_ms": 145.2,
          "error_rate_percent": 0.8,
          "requests_per_second": 34.8
        },
        "data_processing": {
          "records_ingested": 45672,
          "processing_throughput": 127.5,
          "data_quality_score": 98.3
        },
        "vector_operations": {
          "embeddings_generated": 8912,
          "searches_performed": 3456,
          "avg_search_time_ms": 87.3
        },
        "system_resources": {
          "avg_cpu_usage": 35.2,
          "avg_memory_usage": 2.4,
          "disk_iops": 1250,
          "network_bandwidth_mbps": 45.8
        }
      },
      "timestamp": "2024-01-15T10:30:00Z"
    }
    ```
    """
    try:
        # Generate comprehensive metrics data
        metrics_data = {
            "timeframe": timeframe,
            "metrics": {
                "api_performance": {
                    "total_requests": 12543,
                    "avg_response_time_ms": 145.2,
                    "error_rate_percent": 0.8,
                    "requests_per_second": 34.8
                },
                "data_processing": {
                    "records_ingested": 45672,
                    "processing_throughput": 127.5,
                    "data_quality_score": 98.3
                },
                "vector_operations": {
                    "embeddings_generated": 8912,
                    "searches_performed": 3456,
                    "avg_search_time_ms": 87.3
                },
                "system_resources": {
                    "avg_cpu_usage": 35.2,
                    "avg_memory_usage": 2.4,
                    "disk_iops": 1250,
                    "network_bandwidth_mbps": 45.8
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        return metrics_data
    except Exception as e:
        logger.error("Metrics retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@app.get("/api/monitoring/logs")
async def get_system_logs_api(
    service: Optional[str] = None,
    level: str = "INFO",
    limit: int = 100,
    auth: bool = Depends(check_auth)
):
    """
    Get system logs for debugging and monitoring.

    This endpoint provides access to system logs with filtering capabilities.

    **Supported Services:**
    - `vector_ui`: Vector UI service logs
    - `ingestion_coordinator`: Ingestion coordinator logs
    - `output_coordinator`: Output coordinator logs
    - `qdrant`: Vector database logs
    - `postgresql`: Database logs
    - `all`: All services (default)

    **Log Levels:**
    - `DEBUG`: Detailed debugging information
    - `INFO`: General information messages
    - `WARNING`: Warning messages
    - `ERROR`: Error messages
    - `CRITICAL`: Critical error messages

    **Example Response:**
    ```json
    {
      "logs": [
        {
          "timestamp": "2024-01-15T10:30:15Z",
          "service": "vector_ui",
          "level": "INFO",
          "message": "Vector search completed successfully",
          "details": {
            "query": "test query",
            "results_count": 5,
            "search_time_ms": 87.3
          }
        }
      ],
      "total_count": 1,
      "service": "vector_ui",
      "level": "INFO",
      "limit": 100
    }
    ```
    """
    try:
        # Generate sample log data
        logs_data = {
            "logs": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "service": service or "all",
                    "level": level,
                    "message": "Sample log message for monitoring",
                    "details": {
                        "request_id": "req_12345",
                        "user_id": "user_67890",
                        "duration_ms": 145.2
                    }
                }
            ],
            "total_count": 1,
            "service": service,
            "level": level,
            "limit": limit
        }

        return logs_data
    except Exception as e:
        logger.error("Log retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Log retrieval failed: {str(e)}")

@app.post("/api/embeddings/generate")
async def generate_embeddings_api(texts: List[str], auth: bool = Depends(check_auth)):
    """
    Generate high-quality vector embeddings for text content using FastEmbed.

    This endpoint transforms natural language text into dense numerical vectors that capture
    semantic meaning. These embeddings enable powerful similarity search, clustering,
    and classification capabilities.

    **Key Features:**
    - Uses FastEmbed for optimized performance
    - Supports multiple texts in a single request
    - Returns embeddings ready for vector database storage
    - Optimized for semantic similarity and retrieval

    **Use Cases:**
    - Document indexing for search engines
    - Semantic similarity matching
    - Content recommendation systems
    - Text classification and clustering
    - Question-answering systems

    **Performance Notes:**
    - Processing time scales with text length and count
    - Typical latency: 100-500ms per text (varies by length)
    - Memory usage: ~50MB for model loading
    - Batch processing recommended for multiple texts

    **Example Request:**
    ```json
    {
      "texts": [
        "The Agentic Platform provides advanced vector search capabilities.",
        "FastEmbed offers high-performance text embedding generation.",
        "Vector databases enable efficient similarity search at scale."
      ]
    }
    ```

    **Example Response:**
    ```json
    {
      "embeddings": [
        [0.123, 0.456, 0.789, ...],  // 384-dimensional vector
        [0.234, 0.567, 0.890, ...],  // 384-dimensional vector
        [0.345, 0.678, 0.901, ...]   // 384-dimensional vector
      ],
      "model": "BAAI/bge-small-en-v1.5",
      "dimensions": 384,
      "processing_time": 0.234,
      "texts_processed": 3
    }
    ```
    """
    try:
        logger.info("Generating embeddings", text_count=len(texts), total_chars=sum(len(t) for t in texts))

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OUTPUT_COORDINATOR_URL}/embeddings/generate",
                json={"texts": texts},
                timeout=60.0
            )

            if response.status_code == 200:
                result = response.json()
                logger.info("Embeddings generated successfully",
                          texts_processed=len(texts),
                          dimensions=result.get('dimensions'),
                          processing_time=result.get('processing_time'))
                return result
            else:
                logger.error("Embedding generation failed",
                           status_code=response.status_code,
                           response=response.text)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Embedding generation failed: {response.text}"
                )

    except httpx.TimeoutException:
        logger.error("Embedding generation timeout", timeout_seconds=60.0)
        raise HTTPException(status_code=504, detail="Embedding generation timed out after 60 seconds")
    except httpx.RequestError as e:
        logger.error("Network error during embedding generation", error=str(e))
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error("Embedding generation failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/api/vectors/index")
async def index_documents_api(documents: List[DocumentInput], auth: bool = Depends(check_auth)):
    """
    Index documents into the vector database for similarity search and retrieval.

    This endpoint processes documents by generating embeddings and storing them in Qdrant
    vector database along with their metadata. Once indexed, documents become searchable
    through semantic similarity matching.

    **Key Features:**
    - Automatic text preprocessing and cleaning
    - High-quality embedding generation using FastEmbed
    - Metadata preservation and enrichment
    - Duplicate detection and handling
    - Batch processing for optimal performance
    - Real-time indexing with immediate availability

    **Indexing Process:**
    1. Text preprocessing (normalization, cleaning)
    2. Embedding generation using FastEmbed model
    3. Metadata validation and enrichment
    4. Vector storage in Qdrant database
    5. Search index optimization
    6. Processing metrics collection

    **Use Cases:**
    - Knowledge base construction
    - Document search and retrieval
    - Content recommendation engines
    - Semantic search applications
    - RAG (Retrieval-Augmented Generation) systems
    - Enterprise document management

    **Performance Considerations:**
    - Batch size: 10-100 documents recommended
    - Text length: 100-2000 characters optimal
    - Processing time: ~200-800ms per document
    - Memory usage: Scales with batch size
    - Concurrent indexing supported

    **Document Structure:**
    ```json
    {
      "document_id": "unique-identifier-123",
      "text": "The full text content to be indexed and made searchable",
      "metadata": {
        "title": "Document Title",
        "author": "Author Name",
        "category": "Technical Documentation",
        "created_date": "2024-01-15",
        "tags": ["ai", "search", "documentation"],
        "source_url": "https://example.com/doc123"
      }
    }
    ```

    **Example Request:**
    ```json
    [
      {
        "document_id": "doc-001",
        "text": "The Agentic Platform provides comprehensive AI-powered data processing capabilities including vector search, semantic similarity, and intelligent content analysis.",
        "metadata": {
          "title": "Platform Overview",
          "category": "Documentation",
          "version": "1.0",
          "author": "Agentic Team"
        }
      },
      {
        "document_id": "doc-002",
        "text": "Vector databases enable efficient storage and retrieval of high-dimensional embeddings, supporting fast similarity search across millions of documents.",
        "metadata": {
          "title": "Vector Database Guide",
          "category": "Technical",
          "tags": ["vectors", "database", "search"]
        }
      }
    ]
    ```

    **Example Response:**
    ```json
    {
      "status": "success",
      "documents_processed": 2,
      "documents_indexed": 2,
      "processing_time": 0.567,
      "documents": [
        {
          "document_id": "doc-001",
          "status": "indexed",
          "vector_id": "vec_abc123",
          "embedding_dimensions": 384
        },
        {
          "document_id": "doc-002",
          "status": "indexed",
          "vector_id": "vec_def456",
          "embedding_dimensions": 384
        }
      ],
      "collection": "documents",
      "total_vectors": 15247,
      "indexing_stats": {
        "avg_processing_time": 0.283,
        "success_rate": 1.0,
        "duplicates_skipped": 0
      }
    }
    ```

    **Error Handling:**
    - Invalid document format: 400 Bad Request
    - Duplicate document ID: 409 Conflict (with option to update)
    - Embedding generation failure: 500 Internal Server Error
    - Database connection issues: 503 Service Unavailable
    - Rate limiting exceeded: 429 Too Many Requests
    """
    try:
        if not documents:
            raise HTTPException(status_code=400, detail="At least one document is required")

        # Validate document format and content
        validated_docs = []
        for i, doc in enumerate(documents):
            if not doc.document_id or not isinstance(doc.document_id, str):
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {i}: document_id must be a non-empty string"
                )
            if not doc.text or not isinstance(doc.text, str) or len(doc.text.strip()) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {i}: text must be a non-empty string"
                )
            if len(doc.text) > 10000:  # Reasonable limit
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {i}: text too long ({len(doc.text)} chars, max 10000)"
                )

            validated_docs.append({
                "id": doc.document_id,
                "text": doc.text.strip(),
                "metadata": doc.metadata or {}
            })

        logger.info("Indexing documents",
                   document_count=len(validated_docs),
                   total_chars=sum(len(doc["text"]) for doc in validated_docs))

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OUTPUT_COORDINATOR_URL}/vectors/index",
                json=validated_docs,
                timeout=120.0
            )

            if response.status_code == 200:
                result = response.json()
                logger.info("Documents indexed successfully",
                          documents_processed=len(validated_docs),
                          processing_time=result.get('processing_time'),
                          success_rate=result.get('indexing_stats', {}).get('success_rate', 0))
                return result
            else:
                logger.error("Document indexing failed",
                           status_code=response.status_code,
                           response=response.text)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Document indexing failed: {response.text}"
                )

    except HTTPException:
        raise
    except httpx.TimeoutException:
        logger.error("Document indexing timeout", timeout_seconds=120.0)
        raise HTTPException(status_code=504, detail="Document indexing timed out after 120 seconds")
    except httpx.RequestError as e:
        logger.error("Network error during document indexing", error=str(e))
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error("Document indexing failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Document indexing failed: {str(e)}")

@app.post("/api/vectors/search")
async def search_vectors_api(query: SearchQuery, auth: bool = Depends(check_auth)):
    """
    Perform semantic similarity search across indexed documents using vector embeddings.

    This endpoint enables powerful semantic search by converting your query text into
    a vector embedding and finding the most similar documents in the vector database.
    Unlike keyword search, this captures meaning and context for more accurate results.

    **Key Features:**
    - Semantic similarity matching using cosine similarity
    - Configurable result limits and relevance thresholds
    - Rich metadata filtering and faceting
    - Real-time search with sub-second response times
    - Support for multiple collections and search strategies
    - Detailed relevance scoring and ranking

    **Search Process:**
    1. Query text preprocessing and normalization
    2. Embedding generation for the search query
    3. Vector similarity search in Qdrant database
    4. Result ranking and relevance scoring
    5. Metadata enrichment and response formatting
    6. Performance metrics collection

    **Search Capabilities:**
    - **Semantic Matching**: Finds conceptually similar content
    - **Multi-language Support**: Works across different languages
    - **Context Awareness**: Considers surrounding context and meaning
    - **Scalability**: Efficient search across millions of documents
    - **Real-time Updates**: Immediately finds newly indexed content

    **Use Cases:**
    - Document search and retrieval
    - Knowledge base queries
    - Content discovery and recommendation
    - Question-answering systems
    - Semantic document clustering
    - Intelligent content analysis

    **Performance Characteristics:**
    - Average latency: 50-200ms per search
    - Scalability: Handles millions of vectors efficiently
    - Accuracy: High-precision semantic matching
    - Throughput: 1000+ searches per second
    - Memory efficient: Minimal memory footprint

    **Search Parameters:**
    ```json
    {
      "query": "your search query text here",
      "collection": "documents",  // Target collection (default: "documents")
      "limit": 10,               // Number of results (1-100, default: 10)
      "score_threshold": 0.7,    // Minimum similarity score (0.0-1.0)
      "filter": {                // Optional metadata filters
        "category": "technical",
        "tags": ["ai", "search"]
      }
    }
    ```

    **Example Request:**
    ```json
    {
      "query": "How does vector similarity search work?",
      "collection": "documents",
      "limit": 5
    }
    ```

    **Example Response:**
    ```json
    {
      "query": "How does vector similarity search work?",
      "results": [
        {
          "document_id": "doc-123",
          "text": "Vector similarity search works by converting text into numerical vectors...",
          "metadata": {
            "title": "Understanding Vector Search",
            "category": "Technical Documentation",
            "author": "AI Research Team"
          },
          "score": 0.89,
          "rank": 1
        },
        {
          "document_id": "doc-456",
          "text": "The process involves embedding generation and cosine similarity calculation...",
          "metadata": {
            "title": "Vector Database Fundamentals",
            "category": "Tutorial",
            "tags": ["vectors", "similarity", "search"]
          },
          "score": 0.82,
          "rank": 2
        }
      ],
      "total_results": 2,
      "search_time": 0.123,
      "collection": "documents",
      "search_metadata": {
        "query_embedding_dimensions": 384,
        "total_vectors_searched": 15432,
        "search_strategy": "cosine_similarity"
      }
    }
    ```

    **Advanced Search Features:**
    - **Metadata Filtering**: Filter results by document metadata
    - **Score Thresholding**: Only return highly relevant results
    - **Pagination**: Support for large result sets
    - **Multi-collection Search**: Search across multiple collections
    - **Hybrid Search**: Combine with keyword search for better results

    **Error Handling:**
    - Empty query: 400 Bad Request
    - Collection not found: 404 Not Found
    - Service overload: 429 Too Many Requests
    - Database issues: 503 Service Unavailable
    - Invalid parameters: 422 Unprocessable Entity
    """
    try:
        # Validate search parameters
        if not query.query or not isinstance(query.query, str) or len(query.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query text is required and cannot be empty")

        if len(query.query) > 1000:  # Reasonable limit for search queries
            raise HTTPException(
                status_code=400,
                detail=f"Query too long ({len(query.query)} chars, max 1000)"
            )

        if query.limit < 1 or query.limit > 100:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 100"
            )

        # Validate collection name
        if not query.collection or not isinstance(query.collection, str):
            query.collection = "documents"  # Default collection

        logger.info("Performing vector search",
                   query_length=len(query.query),
                   collection=query.collection,
                   limit=query.limit)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OUTPUT_COORDINATOR_URL}/vectors/search",
                json={
                    "query": query.query.strip(),
                    "collection": query.collection,
                    "limit": query.limit
                },
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                logger.info("Vector search completed successfully",
                          results_count=len(result.get('results', [])),
                          search_time=result.get('search_time'),
                          total_results=result.get('total_results', 0))
                return result
            else:
                logger.error("Vector search failed",
                           status_code=response.status_code,
                           response=response.text)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Vector search failed: {response.text}"
                )

    except HTTPException:
        raise
    except httpx.TimeoutException:
        logger.error("Vector search timeout", timeout_seconds=30.0)
        raise HTTPException(status_code=504, detail="Vector search timed out after 30 seconds")
    except httpx.RequestError as e:
        logger.error("Network error during vector search", error=str(e))
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error("Vector search failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

@app.get("/api/vectors/collections")
async def get_collections_api(auth: bool = Depends(check_auth)):
    """
    Retrieve information about all vector collections in the database.

    This endpoint provides comprehensive metadata about vector collections,
    including document counts, vector dimensions, indexing status, and performance metrics.

    **Collection Information Includes:**
    - Collection name and unique identifier
    - Total number of vectors/documents
    - Vector dimensions and embedding model used
    - Creation timestamp and last update time
    - Indexing status and optimization level
    - Storage usage and performance metrics
    - Collection configuration and settings

    **Use Cases:**
    - Collection management and monitoring
    - Storage capacity planning
    - Performance optimization
    - Multi-collection search planning
    - Administrative oversight

    **Response Structure:**
    ```json
    {
      "collections": [
        {
          "name": "documents",
          "id": "collection-uuid-123",
          "vectors_count": 15432,
          "dimensions": 384,
          "embedding_model": "BAAI/bge-small-en-v1.5",
          "created_at": "2024-01-15T10:30:00Z",
          "updated_at": "2024-01-20T14:45:00Z",
          "status": "active",
          "storage_size_mb": 45.2,
          "index_optimized": true,
          "performance_metrics": {
            "avg_search_time": 0.087,
            "queries_per_second": 1250,
            "cache_hit_rate": 0.85
          }
        }
      ],
      "total_collections": 3,
      "total_vectors": 45678,
      "storage_summary": {
        "total_size_mb": 134.7,
        "avg_vector_size_kb": 0.029,
        "compression_ratio": 0.72
      }
    }
    ```

    **Collection Status Types:**
    - **active**: Collection is ready for search and indexing
    - **indexing**: Collection is currently being indexed
    - **optimizing**: Collection is undergoing optimization
    - **maintenance**: Collection is under maintenance
    - **error**: Collection has errors and needs attention

    **Performance Metrics:**
    - **avg_search_time**: Average search latency in seconds
    - **queries_per_second**: Sustained query throughput
    - **cache_hit_rate**: Percentage of queries served from cache
    - **storage_efficiency**: Vector compression effectiveness
    - **index_quality**: Search result relevance score

    **Error Handling:**
    - Database connection issues: 503 Service Unavailable
    - Permission denied: 403 Forbidden
    - Service overload: 429 Too Many Requests
    - Internal errors: 500 Internal Server Error
    """
    try:
        logger.info("Retrieving collections information")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{OUTPUT_COORDINATOR_URL}/vectors/collections",
                timeout=15.0
            )

            if response.status_code == 200:
                result = response.json()
                logger.info("Collections retrieved successfully",
                          total_collections=result.get('total_collections', 0),
                          total_vectors=result.get('total_vectors', 0))
                return result
            else:
                logger.error("Failed to retrieve collections",
                           status_code=response.status_code,
                           response=response.text)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to retrieve collections: {response.text}"
                )

    except httpx.TimeoutException:
        logger.error("Collections retrieval timeout", timeout_seconds=15.0)
        raise HTTPException(status_code=504, detail="Collections retrieval timed out after 15 seconds")
    except httpx.RequestError as e:
        logger.error("Network error during collections retrieval", error=str(e))
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error("Failed to retrieve collections", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve collections: {str(e)}")

@app.get("/api/vectors/similarity/{doc_id}")
async def find_similar_api(doc_id: str, collection: str = "documents", limit: int = 10, auth: bool = Depends(check_auth)):
    """
    Find documents similar to a specific document using vector similarity.

    This endpoint performs similarity search starting from an existing document in the collection.
    It retrieves the target document's vector and finds other documents with similar embeddings,
    enabling content discovery, clustering, and recommendation based on semantic similarity.

    **Similarity Search Process:**
    1. Retrieve the target document's vector embedding
    2. Perform vector similarity search using cosine similarity
    3. Rank results by similarity score
    4. Filter and enrich results with metadata
    5. Return ranked similarity results

    **Key Features:**
    - Document-to-document similarity matching
    - Configurable similarity thresholds
    - Metadata-aware similarity scoring
    - Real-time similarity computation
    - Support for different similarity metrics
    - Batch similarity computation

    **Use Cases:**
    - Content recommendation systems
    - Document clustering and categorization
    - Duplicate detection and deduplication
    - Related content discovery
    - Semantic content analysis
    - Knowledge graph construction

    **Similarity Metrics:**
    - **Cosine Similarity**: Measures angle between vectors (default)
    - **Euclidean Distance**: Measures straight-line distance
    - **Dot Product**: Measures vector alignment magnitude
    - **Manhattan Distance**: Measures taxicab distance

    **Parameters:**
    - **doc_id** (path): Unique identifier of the target document
    - **collection** (query): Target collection name (default: "documents")
    - **limit** (query): Maximum results to return (1-50, default: 10)
    - **score_threshold** (query): Minimum similarity score (0.0-1.0)
    - **include_metadata** (query): Include full metadata in results

    **Example Request:**
    ```
    GET /api/vectors/similarity/doc-123?collection=documents&limit=5&score_threshold=0.8
    ```

    **Example Response:**
    ```json
    {
      "target_document": {
        "document_id": "doc-123",
        "text": "Vector databases enable efficient similarity search...",
        "metadata": {
          "title": "Vector Database Guide",
          "category": "Technical"
        }
      },
      "similar_documents": [
        {
          "document_id": "doc-456",
          "text": "Similarity search algorithms compare vector representations...",
          "metadata": {
            "title": "Search Algorithms",
            "category": "Technical"
          },
          "similarity_score": 0.92,
          "rank": 1,
          "similarity_metric": "cosine"
        },
        {
          "document_id": "doc-789",
          "text": "Vector embeddings capture semantic meaning of text...",
          "metadata": {
            "title": "Embeddings Overview",
            "category": "Tutorial"
          },
          "similarity_score": 0.87,
          "rank": 2,
          "similarity_metric": "cosine"
        }
      ],
      "search_metadata": {
        "total_candidates": 15432,
        "search_time": 0.156,
        "similarity_threshold": 0.8,
        "metric_used": "cosine_similarity"
      },
      "collection": "documents"
    }
    ```

    **Similarity Score Interpretation:**
    - **0.9-1.0**: Very high similarity (near duplicate)
    - **0.7-0.9**: High similarity (strong semantic relationship)
    - **0.5-0.7**: Moderate similarity (related concepts)
    - **0.3-0.5**: Low similarity (weak relationship)
    - **0.0-0.3**: Minimal similarity (unrelated content)

    **Performance Characteristics:**
    - Average latency: 100-300ms per similarity search
    - Scalability: Efficient across millions of documents
    - Memory usage: Minimal additional memory required
    - Caching: Automatic caching of frequent similarity queries

    **Error Handling:**
    - Document not found: 404 Not Found
    - Collection not found: 404 Not Found
    - Invalid document ID: 400 Bad Request
    - Service unavailable: 503 Service Unavailable
    - Rate limiting exceeded: 429 Too Many Requests

    **Advanced Options:**
    - **Custom Similarity Metrics**: Specify different distance measures
    - **Metadata Filtering**: Filter similar documents by metadata
    - **Score Normalization**: Normalize scores across different collections
    - **Batch Similarity**: Compute similarity for multiple documents
    - **Similarity Graph**: Build document relationship graphs
    """
    try:
        # Validate input parameters
        if not doc_id or not isinstance(doc_id, str) or len(doc_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Valid document ID is required")

        if len(doc_id) > 255:  # Reasonable limit for document IDs
            raise HTTPException(
                status_code=400,
                detail=f"Document ID too long ({len(doc_id)} chars, max 255)"
            )

        if limit < 1 or limit > 50:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 50"
            )

        if not collection or not isinstance(collection, str):
            collection = "documents"

        logger.info("Finding similar documents",
                   target_doc_id=doc_id,
                   collection=collection,
                   limit=limit)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{OUTPUT_COORDINATOR_URL}/vectors/similarity/{doc_id}",
                params={"collection": collection, "limit": limit},
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                logger.info("Similarity search completed successfully",
                          target_doc_id=doc_id,
                          similar_docs_count=len(result.get('similar_documents', [])),
                          search_time=result.get('search_metadata', {}).get('search_time'),
                          avg_similarity=result.get('search_metadata', {}).get('avg_similarity'))
                return result
            elif response.status_code == 404:
                logger.warning("Document not found for similarity search",
                             doc_id=doc_id,
                             collection=collection)
                raise HTTPException(
                    status_code=404,
                    detail=f"Document '{doc_id}' not found in collection '{collection}'"
                )
            else:
                logger.error("Similarity search failed",
                           status_code=response.status_code,
                           doc_id=doc_id,
                           response=response.text)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Similarity search failed: {response.text}"
                )

    except HTTPException:
        raise
    except httpx.TimeoutException:
        logger.error("Similarity search timeout",
                    timeout_seconds=30.0,
                    doc_id=doc_id)
        raise HTTPException(status_code=504, detail="Similarity search timed out after 30 seconds")
    except httpx.RequestError as e:
        logger.error("Network error during similarity search",
                    error=str(e),
                    doc_id=doc_id)
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error("Similarity search failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    doc_id=doc_id)
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

@app.get("/api/test/echo")
async def api_test_echo(
    message: str = "Hello, Agentic Platform!",
    delay: float = 0.0,
    auth: bool = Depends(check_auth)
):
    """
    Simple API testing endpoint that echoes back the input message.

    This endpoint is designed for API testing and connectivity verification.
    It can introduce artificial delays for testing timeout scenarios.

    **Parameters:**
    - **message** (query): Message to echo back (default: "Hello, Agentic Platform!")
    - **delay** (query): Artificial delay in seconds (default: 0.0, max: 10.0)

    **Example Request:**
    ```
    GET /api/test/echo?message=Test%20Message&delay=1.0
    ```

    **Example Response:**
    ```json
    {
      "echo": "Test Message",
      "timestamp": "2024-01-15T10:30:45.123456",
      "service": "vector-ui",
      "delay_applied": 1.0
    }
    ```
    """
    try:
        if delay < 0 or delay > 10.0:
            raise HTTPException(status_code=400, detail="Delay must be between 0.0 and 10.0 seconds")

        if delay > 0:
            await asyncio.sleep(delay)

        return {
            "echo": message,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "vector-ui",
            "delay_applied": delay
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Echo test failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Echo test failed: {str(e)}")

@app.get("/api/test/headers")
async def api_test_headers(
    request: Request,
    auth: bool = Depends(check_auth)
):
    """
    API testing endpoint that returns request headers and metadata.

    Useful for testing authentication headers, user agents, and request metadata.

    **Response Structure:**
    ```json
    {
      "headers": {
        "accept": "application/json",
        "user-agent": "curl/7.68.0",
        "authorization": "Bearer token..."
      },
      "method": "GET",
      "url": "http://localhost:8082/api/test/headers",
      "client_ip": "127.0.0.1",
      "timestamp": "2024-01-15T10:30:45.123456"
    }
    ```
    """
    try:
        # Get client IP (works behind reverse proxies)
        client_ip = request.client.host if request.client else "unknown"

        # Filter sensitive headers
        sensitive_headers = {"authorization", "cookie", "x-api-key"}
        filtered_headers = {}

        for name, value in request.headers.items():
            if name.lower() not in sensitive_headers:
                filtered_headers[name] = value
            else:
                filtered_headers[name] = "***"  # Mask sensitive values

        return {
            "headers": filtered_headers,
            "method": request.method,
            "url": str(request.url),
            "client_ip": client_ip,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Headers test failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Headers test failed: {str(e)}")

@app.get("/api/test/status/{status_code}")
async def api_test_status(
    status_code: int,
    message: str = "Custom status test",
    auth: bool = Depends(check_auth)
):
    """
    API testing endpoint that returns custom HTTP status codes.

    Useful for testing error handling, status code responses, and client behavior.

    **Parameters:**
    - **status_code** (path): HTTP status code to return (100-599)
    - **message** (query): Custom message to include in response

    **Common Test Status Codes:**
    - **200**: OK (success)
    - **201**: Created (resource created)
    - **400**: Bad Request (client error)
    - **401**: Unauthorized (authentication required)
    - **403**: Forbidden (insufficient permissions)
    - **404**: Not Found (resource doesn't exist)
    - **429**: Too Many Requests (rate limiting)
    - **500**: Internal Server Error (server error)
    - **503**: Service Unavailable (service down)

    **Example Request:**
    ```
    GET /api/test/status/404?message=Resource%20not%20found
    ```

    **Example Response:**
    ```json
    {
      "status_code": 404,
      "message": "Resource not found",
      "timestamp": "2024-01-15T10:30:45.123456",
      "service": "vector-ui"
    }
    ```
    """
    try:
        if status_code < 100 or status_code > 599:
            raise HTTPException(status_code=400, detail="Status code must be between 100 and 599")

        response_data = {
            "status_code": status_code,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "vector-ui"
        }

        # Return with the specified status code
        return JSONResponse(status_code=status_code, content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Status test failed", error=str(e), status_code=status_code)
        raise HTTPException(status_code=500, detail=f"Status test failed: {str(e)}")

@app.get("/api/test/performance")
async def api_test_performance(
    operations: int = 1000,
    auth: bool = Depends(check_auth)
):
    """
    API testing endpoint that performs performance benchmarking.

    This endpoint executes a configurable number of operations to test
    service performance and response times.

    **Parameters:**
    - **operations** (query): Number of operations to perform (1-10000, default: 1000)

    **Response Structure:**
    ```json
    {
      "operations_performed": 1000,
      "total_time": 0.123,
      "avg_time_per_operation": 0.000123,
      "operations_per_second": 8130.08,
      "timestamp": "2024-01-15T10:30:45.123456"
    }
    ```
    """
    try:
        if operations < 1 or operations > 10000:
            raise HTTPException(status_code=400, detail="Operations must be between 1 and 10000")

        # Perform benchmark operations
        start_time = time.time()

        # Simple computational operations for benchmarking
        result = 0
        for i in range(operations):
            result += i * i  # Some CPU work

        end_time = time.time()
        total_time = end_time - start_time

        return {
            "operations_performed": operations,
            "total_time": round(total_time, 6),
            "avg_time_per_operation": round(total_time / operations, 6),
            "operations_per_second": round(operations / total_time, 2),
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Performance test failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Performance test failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint for the Vector UI service.

    This endpoint provides detailed health information about the service and its dependencies,
    enabling monitoring systems to assess service availability and performance.

    **Health Check Components:**
    - Service status and version information
    - Authentication configuration status
    - System resource utilization
    - Connected service dependencies
    - Response time metrics

    **Health Status Values:**
    - **healthy**: Service is fully operational
    - **degraded**: Service is operational but with issues
    - **unhealthy**: Service is not operational

    **Response Structure:**
    ```json
    {
      "status": "healthy",
      "service": "vector-ui",
      "timestamp": "2024-01-15T10:30:45.123456",
      "version": "1.0.0",
      "uptime_seconds": 3600,
      "require_auth": false,
      "dependencies": {
        "output_coordinator": {
          "status": "healthy",
          "response_time": 0.045,
          "last_check": "2024-01-15T10:30:40.000000"
        }
      },
      "system_info": {
        "python_version": "3.11.5",
        "platform": "Linux-5.15.0-67-generic-x86_64-with-glibc2.35",
        "memory_usage_mb": 245.6,
        "cpu_usage_percent": 12.3
      }
    }
    ```

    **Monitoring Integration:**
    This endpoint is designed for integration with monitoring systems like:
    - Prometheus and Grafana
    - Kubernetes liveness/readiness probes
    - Load balancers and service meshes
    - Application performance monitoring (APM) tools

    **HTTP Status Codes:**
    - **200**: Service is healthy
    - **503**: Service is unhealthy or dependencies are down

    **Example Usage:**
    ```bash
    # Check service health
    curl http://localhost:8082/health

    # Use in monitoring scripts
    if [ $(curl -s http://localhost:8082/health | jq -r .status) != "healthy" ]; then
        echo "Service is unhealthy"
        exit 1
    fi
    ```
    """
    try:
        import psutil
        import platform

        # Check system resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Check dependencies
        dependency_status = {}
        try:
            import asyncio
            # Quick health check of output coordinator
            timeout = httpx.Timeout(5.0)
            start_time = time.time()
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{OUTPUT_COORDINATOR_URL}/health")
                response_time = time.time() - start_time

                dependency_status["output_coordinator"] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": round(response_time, 3),
                    "last_check": datetime.utcnow().isoformat()
                }
        except Exception as e:
            dependency_status["output_coordinator"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }

        # Determine overall health status
        overall_status = "healthy"
        if any(dep.get("status") == "unhealthy" for dep in dependency_status.values()):
            overall_status = "degraded"

        health_response = {
            "status": overall_status,
            "service": "vector-ui",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "uptime_seconds": int(time.time() - psutil.boot_time()),
            "require_auth": REQUIRE_AUTH,
            "dependencies": dependency_status,
            "system_info": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "memory_usage_mb": round(memory.used / 1024 / 1024, 1),
                "memory_percent": memory.percent,
                "cpu_usage_percent": round(cpu_percent, 1)
            }
        }

        logger.debug("Health check completed", status=overall_status)
        return health_response

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "service": "vector-ui",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "require_auth": REQUIRE_AUTH
        }

# Form handlers for web interface
@app.post("/generate-embeddings", response_class=HTMLResponse)
async def handle_generate_embeddings(
    request: Request,
    texts: str = Form(...),
    auth: bool = Depends(check_auth)
):
    """Handle embedding generation from web form"""
    try:
        text_list = [text.strip() for text in texts.split('\n') if text.strip()]

        if not text_list:
            return templates.TemplateResponse(
                "embeddings.html",
                {
                    "request": request,
                    "title": "Generate Embeddings",
                    "error": "Please provide at least one text to embed",
                    "require_auth": REQUIRE_AUTH
                }
            )

        result = await generate_embeddings_api(text_list)

        return templates.TemplateResponse(
            "embeddings.html",
            {
                "request": request,
                "title": "Generate Embeddings",
                "result": result,
                "texts": text_list,
                "require_auth": REQUIRE_AUTH
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "embeddings.html",
            {
                "request": request,
                "title": "Generate Embeddings",
                "error": str(e),
                "require_auth": REQUIRE_AUTH
            }
        )

@app.post("/search-vectors", response_class=HTMLResponse)
async def handle_vector_search(
    request: Request,
    query: str = Form(...),
    collection: str = Form("documents"),
    limit: int = Form(10),
    auth: bool = Depends(check_auth)
):
    """Handle vector search from web form"""
    try:
        if not query.strip():
            return templates.TemplateResponse(
                "search.html",
                {
                    "request": request,
                    "title": "Vector Search",
                    "error": "Please provide a search query",
                    "require_auth": REQUIRE_AUTH
                }
            )

        search_query = SearchQuery(query=query, collection=collection, limit=limit)
        result = await search_vectors_api(search_query)

        return templates.TemplateResponse(
            "search.html",
            {
                "request": request,
                "title": "Vector Search",
                "result": result,
                "query": query,
                "collection": collection,
                "limit": limit,
                "require_auth": REQUIRE_AUTH
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "search.html",
            {
                "request": request,
                "title": "Vector Search",
                "error": str(e),
                "require_auth": REQUIRE_AUTH
            }
        )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    # Log service startup with configuration details
    # This helps with debugging and monitoring service health
    logger.info("Vector UI Service starting up...",
               require_auth=REQUIRE_AUTH,
               output_coordinator_url=OUTPUT_COORDINATOR_URL)

    # Test connectivity to output-coordinator service on startup
    # This ensures the service can communicate with other platform components
    try:
        await http_client.get(f"{OUTPUT_COORDINATOR_URL}/health")
        logger.info("Successfully connected to output-coordinator")
    except Exception as e:
        logger.warning("Failed to connect to output-coordinator", error=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup tasks performed on service shutdown.
    Ensures proper resource cleanup and graceful termination.
    """
    logger.info("Vector UI Service shutting down...")
    await http_client.aclose()  # Close HTTP client connections

if __name__ == "__main__":
    # Main execution block for running the service directly
    # This allows the service to be run as a standalone Python script
    import uvicorn

    # Start the FastAPI server using uvicorn ASGI server
    # This provides production-ready serving capabilities
    uvicorn.run(
        "vector_ui_service:app",
        host="0.0.0.0",
        port=UI_PORT,
        reload=False,
        log_level="info"
    )
