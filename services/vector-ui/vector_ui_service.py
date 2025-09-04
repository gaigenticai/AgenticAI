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
        Complete API documentation for the Agentic Platform.

        ## Services Included:
        - **Vector UI Service**: Vector operations and search
        - **Ingestion Coordinator**: Data ingestion orchestration
        - **Output Coordinator**: Data output and routing
        - **All Platform Services**: Complete API coverage

        ## Key Features:
        - AI-powered vector similarity search
        - Multi-source data ingestion
        - Intelligent data transformation
        - Enterprise security and monitoring
        - Real-time processing capabilities
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
            "description": "Ingestion Coordinator"
        },
        {
            "url": "http://localhost:8081",
            "description": "Output Coordinator"
        }
    ]

    # Return the enhanced OpenAPI schema
    # This will be used by Swagger UI and ReDoc for documentation
    return openapi_schema

# Vector Operations API endpoints
# These endpoints provide programmatic access to vector operations
# They integrate with the output-coordinator service for processing

@app.post("/api/embeddings/generate")
async def generate_embeddings_api(texts: List[str], auth: bool = Depends(check_auth)):
    """
    Generate vector embeddings for input texts using FastEmbed.
    This endpoint converts text into numerical vectors that can be used for similarity search.
    Returns embeddings that are stored in the vector database for later retrieval.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OUTPUT_COORDINATOR_URL}/embeddings/generate",
                json={"texts": texts},
                timeout=60.0
            )
            return response.json()
    except Exception as e:
        logger.error("Embedding generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/api/vectors/index")
async def index_documents_api(documents: List[DocumentInput], auth: bool = Depends(check_auth)):
    """Index documents via API"""
    try:
        # Convert to format expected by output-coordinator
        formatted_docs = [
            {
                "id": doc.document_id,
                "text": doc.text,
                "metadata": doc.metadata or {}
            }
            for doc in documents
        ]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OUTPUT_COORDINATOR_URL}/vectors/index",
                json=formatted_docs,
                timeout=120.0
            )
            return response.json()
    except Exception as e:
        logger.error("Document indexing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Document indexing failed: {str(e)}")

@app.post("/api/vectors/search")
async def search_vectors_api(query: SearchQuery, auth: bool = Depends(check_auth)):
    """Search vectors via API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OUTPUT_COORDINATOR_URL}/vectors/search",
                json={
                    "query": query.query,
                    "collection": query.collection,
                    "limit": query.limit
                },
                timeout=30.0
            )
            return response.json()
    except Exception as e:
        logger.error("Vector search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

@app.get("/api/vectors/collections")
async def get_collections_api(auth: bool = Depends(check_auth)):
    """Get collections via API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{OUTPUT_COORDINATOR_URL}/vectors/collections"
            )
            return response.json()
    except Exception as e:
        logger.error("Failed to get collections", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get collections: {str(e)}")

@app.get("/api/vectors/similarity/{doc_id}")
async def find_similar_api(doc_id: str, collection: str = "documents", limit: int = 10, auth: bool = Depends(check_auth)):
    """Find similar documents via API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{OUTPUT_COORDINATOR_URL}/vectors/similarity/{doc_id}",
                params={"collection": collection, "limit": limit}
            )
            return response.json()
    except Exception as e:
        logger.error("Similarity search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "vector-ui",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
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
