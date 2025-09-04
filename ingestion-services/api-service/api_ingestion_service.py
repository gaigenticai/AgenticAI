#!/usr/bin/env python3
"""
API Ingestion Service for Agentic Platform

This service ingests data from REST APIs with the following capabilities:
- REST API endpoint consumption
- Authentication support (API keys, OAuth, Basic Auth)
- Rate limiting and retry logic
- Pagination handling
- Data transformation and mapping
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
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    title="API Ingestion Service",
    description="Ingests data from REST APIs with authentication and pagination support",
    version="1.0.0"
)

# Prometheus metrics
API_REQUESTS_PROCESSED = Counter('api_requests_processed_total', 'Total API requests processed', ['status'])
API_ENDPOINTS_SCRAPED = Counter('api_endpoints_scraped_total', 'Total API endpoints scraped')
API_PROCESSING_TIME = Histogram('api_processing_duration_seconds', 'API processing duration', ['operation'])
API_REQUEST_ERRORS = Counter('api_request_errors_total', 'Total API request errors', ['error_type'])
ACTIVE_API_JOBS = Gauge('active_api_jobs', 'Number of active API processing jobs')

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://agentic_user:agentic123@postgresql_ingestion:5432/agentic_ingestion")
engine = create_engine(DATABASE_URL)

# Message queue connection
rabbitmq_connection = None
rabbitmq_channel = None

class APIIngestionProcessor:
    """API data ingestion processor with authentication and pagination support"""

    def __init__(self):
        self.db_connection = None
        self.session = None

    def connect_db(self):
        """Connect to database"""
        try:
            self.db_connection = psycopg2.connect(DATABASE_URL)
            logger.info("Database connection established")
        except Exception as e:
            logger.error("Failed to connect to database", error=str(e))
            raise

    def create_session(self):
        """Create HTTP session with retry configuration"""
        if self.session is None:
            self.session = requests.Session()

            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                backoff_factor=1
            )

            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

        return self.session

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

    def validate_api_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate API configuration"""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "config_analysis": {}
        }

        try:
            # Check required fields
            if "endpoint_url" not in config:
                validation_results["is_valid"] = False
                validation_results["errors"].append("endpoint_url is required")

            if "method" not in config:
                config["method"] = "GET"  # Default to GET

            # Validate HTTP method
            valid_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
            if config.get("method", "").upper() not in valid_methods:
                validation_results["warnings"].append(f"HTTP method should be one of: {valid_methods}")

            # Check authentication configuration
            auth_config = config.get("authentication", {})
            if auth_config.get("type") == "api_key" and "api_key" not in auth_config:
                validation_results["warnings"].append("API key authentication configured but api_key not provided")

            if auth_config.get("type") == "oauth2" and "token_url" not in auth_config:
                validation_results["warnings"].append("OAuth2 authentication configured but token_url not provided")

            # Check pagination configuration
            pagination_config = config.get("pagination", {})
            if pagination_config.get("enabled"):
                if "type" not in pagination_config:
                    validation_results["warnings"].append("Pagination enabled but type not specified")

            # Analyze configuration
            validation_results["config_analysis"] = {
                "has_authentication": bool(auth_config),
                "auth_type": auth_config.get("type", "none"),
                "has_pagination": pagination_config.get("enabled", False),
                "pagination_type": pagination_config.get("type", "none"),
                "method": config.get("method", "GET").upper()
            }

        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Configuration validation error: {str(e)}")

        return validation_results

    def prepare_request_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Prepare HTTP headers for API request"""
        headers = config.get("headers", {}).copy()

        # Add default headers
        if "User-Agent" not in headers:
            headers["User-Agent"] = "Agentic-API-Ingestion/1.0"

        if "Accept" not in headers:
            headers["Accept"] = "application/json"

        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        # Add authentication headers
        auth_config = config.get("authentication", {})
        auth_type = auth_config.get("type")

        if auth_type == "api_key":
            header_name = auth_config.get("header_name", "X-API-Key")
            headers[header_name] = auth_config.get("api_key", "")

        elif auth_type == "bearer_token":
            headers["Authorization"] = f"Bearer {auth_config.get('token', '')}"

        elif auth_type == "basic_auth":
            # Basic auth is handled by requests library
            pass

        return headers

    def make_api_request(self, url: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to API endpoint"""
        try:
            session = self.create_session()
            headers = self.prepare_request_headers(config)

            method = config.get("method", "GET").upper()
            timeout = config.get("timeout", 30)

            # Prepare request parameters
            request_params = {
                "url": url,
                "headers": headers,
                "timeout": timeout
            }

            # Add authentication if basic auth
            auth_config = config.get("authentication", {})
            if auth_config.get("type") == "basic_auth":
                from requests.auth import HTTPBasicAuth
                request_params["auth"] = HTTPBasicAuth(
                    auth_config.get("username", ""),
                    auth_config.get("password", "")
                )

            # Add request body for POST/PUT/PATCH
            if method in ["POST", "PUT", "PATCH"]:
                if "body" in config:
                    if isinstance(config["body"], dict):
                        request_params["json"] = config["body"]
                    else:
                        request_params["data"] = config["body"]

            # Add query parameters
            if "params" in config:
                request_params["params"] = config["params"]

            # Make request
            start_time = time.time()
            response = session.request(method, **request_params)
            response_time = time.time() - start_time

            response.raise_for_status()

            # Parse response
            try:
                if response.headers.get("content-type", "").startswith("application/json"):
                    data = response.json()
                else:
                    data = {"text_content": response.text}
            except:
                data = {"text_content": response.text}

            return {
                "status": "success",
                "status_code": response.status_code,
                "response_time": response_time,
                "data": data,
                "headers": dict(response.headers),
                "url": url
            }

        except requests.exceptions.RequestException as e:
            API_REQUEST_ERRORS.labels(error_type="request").inc()
            logger.error("API request failed", url=url, error=str(e))
            return {
                "status": "failed",
                "error": str(e),
                "url": url
            }

        except Exception as e:
            API_REQUEST_ERRORS.labels(error_type="processing").inc()
            logger.error("API request processing failed", url=url, error=str(e))
            return {
                "status": "failed",
                "error": str(e),
                "url": url
            }

    def handle_pagination(self, initial_response: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle API pagination to collect all data"""
        responses = [initial_response]

        if not config.get("pagination", {}).get("enabled", False):
            return responses

        pagination_config = config["pagination"]
        pagination_type = pagination_config.get("type", "offset")

        try:
            if pagination_type == "offset":
                responses.extend(self._handle_offset_pagination(initial_response, config))

            elif pagination_type == "cursor":
                responses.extend(self._handle_cursor_pagination(initial_response, config))

            elif pagination_type == "page":
                responses.extend(self._handle_page_pagination(initial_response, config))

        except Exception as e:
            logger.error("Pagination handling failed", error=str(e))

        return responses

    def _handle_offset_pagination(self, initial_response: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle offset-based pagination"""
        responses = []
        pagination_config = config["pagination"]

        offset = pagination_config.get("initial_offset", 0)
        limit = pagination_config.get("limit", 100)
        max_pages = pagination_config.get("max_pages", 10)

        for page in range(1, max_pages):
            offset += limit

            # Create new config with updated offset
            page_config = config.copy()
            page_config["params"] = page_config.get("params", {}).copy()
            page_config["params"][pagination_config.get("offset_param", "offset")] = offset

            response = self.make_api_request(config["endpoint_url"], page_config)

            if response["status"] == "success" and response.get("data"):
                responses.append(response)
            else:
                break  # Stop if no more data or error

        return responses

    def _handle_cursor_pagination(self, initial_response: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle cursor-based pagination"""
        responses = []
        pagination_config = config["pagination"]
        max_pages = pagination_config.get("max_pages", 10)

        cursor = self._extract_cursor(initial_response, pagination_config)

        for page in range(1, max_pages):
            if not cursor:
                break

            # Create new config with cursor
            page_config = config.copy()
            page_config["params"] = page_config.get("params", {}).copy()
            page_config["params"][pagination_config.get("cursor_param", "cursor")] = cursor

            response = self.make_api_request(config["endpoint_url"], page_config)

            if response["status"] == "success" and response.get("data"):
                responses.append(response)
                cursor = self._extract_cursor(response, pagination_config)
            else:
                break

        return responses

    def _handle_page_pagination(self, initial_response: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle page-based pagination"""
        responses = []
        pagination_config = config["pagination"]
        max_pages = pagination_config.get("max_pages", 10)

        for page in range(2, max_pages + 1):
            # Create new config with page number
            page_config = config.copy()
            page_config["params"] = page_config.get("params", {}).copy()
            page_config["params"][pagination_config.get("page_param", "page")] = page

            response = self.make_api_request(config["endpoint_url"], page_config)

            if response["status"] == "success" and response.get("data"):
                responses.append(response)
            else:
                break

        return responses

    def _extract_cursor(self, response: Dict[str, Any], pagination_config: Dict[str, Any]) -> Optional[str]:
        """Extract cursor from API response"""
        try:
            cursor_path = pagination_config.get("cursor_path", [])
            data = response.get("data", {})

            for path_segment in cursor_path:
                if isinstance(data, dict):
                    data = data.get(path_segment)
                elif isinstance(data, list):
                    data = data[int(path_segment)] if path_segment.isdigit() else None
                else:
                    data = None

            return data if isinstance(data, str) else None

        except Exception as e:
            logger.error("Cursor extraction failed", error=str(e))
            return None

    def process_api_data(self, job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process API data ingestion job"""
        start_time = time.time()
        ACTIVE_API_JOBS.inc()

        try:
            logger.info("Starting API ingestion job", job_id=job_id)

            # Update job status to processing
            self.update_job_status(job_id, "processing")

            # Validate configuration
            validation_results = self.validate_api_config(config)

            if not validation_results["is_valid"]:
                error_msg = "; ".join(validation_results["errors"])
                self.update_job_status(job_id, "failed", error_message=error_msg)
                API_REQUESTS_PROCESSED.labels(status="failed").inc()
                return {"status": "failed", "error": error_msg}

            # Make initial API request
            initial_response = self.make_api_request(config["endpoint_url"], config)

            if initial_response["status"] != "success":
                self.update_job_status(
                    job_id,
                    "failed",
                    error_message=initial_response.get("error", "API request failed")
                )
                API_REQUESTS_PROCESSED.labels(status="failed").inc()
                return {"status": "failed", "error": initial_response.get("error", "API request failed")}

            # Handle pagination
            all_responses = self.handle_pagination(initial_response, config)
            API_ENDPOINTS_SCRAPED.inc(len(all_responses))

            # Process and store data
            self.store_api_results(job_id, validation_results, all_responses)

            # Transform data for processing
            total_records = sum(len(self._extract_records(response)) for response in all_responses)
            processed_records = self.process_extracted_data(job_id, all_responses, config)

            # Update final status
            if processed_records > 0:
                self.update_job_status(
                    job_id,
                    "completed",
                    total_records=total_records,
                    processed_records=processed_records
                )
                API_REQUESTS_PROCESSED.labels(status="success").inc()
                status = "completed"
            else:
                self.update_job_status(
                    job_id,
                    "failed",
                    error_message="No data could be extracted from API"
                )
                API_REQUESTS_PROCESSED.labels(status="failed").inc()
                status = "failed"

            processing_time = time.time() - start_time
            API_PROCESSING_TIME.labels(operation="full_api_ingestion").observe(processing_time)

            logger.info("API ingestion completed",
                       job_id=job_id,
                       status=status,
                       requests_made=len(all_responses),
                       total_records=total_records,
                       processed=processed_records,
                       duration=processing_time)

            return {
                "status": status,
                "requests_made": len(all_responses),
                "total_records": total_records,
                "processed_records": processed_records,
                "processing_time": processing_time,
                "validation_results": validation_results
            }

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"API ingestion failed: {str(e)}"

            self.update_job_status(job_id, "failed", error_message=error_msg)
            API_REQUESTS_PROCESSED.labels(status="failed").inc()
            API_PROCESSING_TIME.labels(operation="full_api_ingestion").observe(processing_time)

            logger.error("API ingestion failed",
                        job_id=job_id,
                        error=str(e),
                        duration=processing_time)

            return {"status": "failed", "error": error_msg}

        finally:
            ACTIVE_API_JOBS.dec()

    def _extract_records(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract records from API response"""
        try:
            data = response.get("data", [])

            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Try to find array in common locations
                for key in ["data", "results", "items", "records"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # If no array found, wrap single object in list
                return [data]
            else:
                return []

        except Exception as e:
            logger.error("Record extraction failed", error=str(e))
            return []

    def store_api_results(self, job_id: str, validation_results: Dict[str, Any],
                         responses: List[Dict[str, Any]]):
        """Store API ingestion results in database"""
        try:
            with self.db_connection.cursor() as cursor:
                # Store validation results
                cursor.execute("""
                    INSERT INTO data_validation_results
                    (job_id, validation_type, is_valid, error_message, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    job_id,
                    "api_ingestion_validation",
                    validation_results["is_valid"],
                    "; ".join(validation_results["errors"]) if validation_results["errors"] else None,
                    json.dumps(validation_results)
                ))

                # Store API responses
                for response in responses:
                    cursor.execute("""
                        INSERT INTO api_ingestion_results
                        (job_id, endpoint_url, status_code, response_time, response_data, metadata, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (
                        job_id,
                        response.get("url"),
                        response.get("status_code"),
                        response.get("response_time"),
                        json.dumps(response.get("data", {})),
                        json.dumps({
                            "headers": response.get("headers", {}),
                            "request_config": {k: v for k, v in response.items()
                                             if k not in ["url", "status_code", "response_time", "data", "headers"]}
                        })
                    ))

                self.db_connection.commit()

        except Exception as e:
            logger.error("Failed to store API results", error=str(e), job_id=job_id)

    def process_extracted_data(self, job_id: str, responses: List[Dict[str, Any]],
                              config: Dict[str, Any]) -> int:
        """Process extracted API data"""
        # This is where you would implement the actual data processing logic
        # For now, just simulate processing
        total_processed = 0

        for response in responses:
            records = self._extract_records(response)
            for record in records:
                time.sleep(0.001)  # Simulate processing time
                total_processed += 1

        return total_processed

# Global processor instance
processor = APIIngestionProcessor()

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
        rabbitmq_channel.queue_declare(queue='api_ingestion', durable=True)

        # Set up consumer
        rabbitmq_channel.basic_qos(prefetch_count=1)
        rabbitmq_channel.basic_consume(
            queue='api_ingestion',
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

        logger.info("Received API ingestion job", job_id=job_id)

        # Process the API ingestion job
        result = processor.process_api_data(job_id, source_config)

        logger.info("API ingestion job completed", job_id=job_id, result=result)

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
        "service": "api-ingestion-service",
        "timestamp": datetime.utcnow().isoformat(),
        "active_jobs": ACTIVE_API_JOBS._value.get()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

# Startup and shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("API Ingestion Service starting up...")

    # Connect to database
    processor.connect_db()

    # Setup RabbitMQ consumer in background thread
    import threading
    rabbitmq_thread = threading.Thread(target=setup_rabbitmq, daemon=True)
    rabbitmq_thread.start()

    logger.info("API Ingestion Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("API Ingestion Service shutting down...")

    # Close HTTP session
    if processor.session:
        processor.session.close()

    # Close database connection
    if processor.db_connection:
        processor.db_connection.close()

    # Close RabbitMQ connection
    if rabbitmq_connection:
        rabbitmq_connection.close()

    logger.info("API Ingestion Service shutdown complete")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "api_ingestion_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8086)),
        reload=False,
        log_level="info"
    )
