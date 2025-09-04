#!/usr/bin/env python3
"""
Elasticsearch Service for Agentic Platform

This service provides comprehensive search and analytics capabilities:
- Full-text search with advanced query support
- Real-time indexing and data ingestion
- Analytics and aggregations
- Schema management and mappings
- Index optimization and management
- Search result ranking and relevance tuning
- Integration with data lake and output coordinator
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pika
import psycopg2
import structlog
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
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

# Prometheus metrics
SEARCH_REQUESTS = Counter('elasticsearch_search_requests_total', 'Total search requests')
SEARCH_DURATION = Histogram('elasticsearch_search_duration_seconds', 'Search request duration')
INDEXING_REQUESTS = Counter('elasticsearch_indexing_requests_total', 'Total indexing requests')
INDEX_SIZE = Gauge('elasticsearch_index_size_bytes', 'Index size in bytes')

# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    index: str = Field("default", description="Index to search in")
    size: int = Field(10, ge=1, le=1000, description="Number of results to return")
    from_: int = Field(0, ge=0, alias="from", description="Offset for pagination")
    fields: Optional[List[str]] = Field(None, description="Fields to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    sort: Optional[List[Dict[str, str]]] = Field(None, description="Sort criteria")

class IndexDocument(BaseModel):
    index: str = Field(..., description="Index name")
    document_id: str = Field(..., description="Document ID")
    document: Dict[str, Any] = Field(..., description="Document data")
    pipeline: Optional[str] = Field(None, description="Ingest pipeline")

class CreateIndexRequest(BaseModel):
    index: str = Field(..., description="Index name")
    mappings: Optional[Dict[str, Any]] = Field(None, description="Index mappings")
    settings: Optional[Dict[str, Any]] = Field(None, description="Index settings")

class ElasticsearchService:
    """Main Elasticsearch service class"""

    def __init__(self):
        self.app = FastAPI(
            title="Elasticsearch Service",
            description="Advanced search and analytics service for Agentic Platform",
            version="1.0.0"
        )
        self.es_client = None
        self.rabbitmq_connection = None
        self.setup_clients()
        self.setup_routes()
        self.setup_message_consumer()

    def setup_clients(self):
        """Initialize Elasticsearch and RabbitMQ clients"""
        try:
            # Elasticsearch client
            es_host = os.getenv('ELASTICSEARCH_HOST', 'elasticsearch_output')
            es_port = int(os.getenv('ELASTICSEARCH_PORT', '9200'))
            self.es_client = Elasticsearch([f'http://{es_host}:{es_port}'])

            # Wait for Elasticsearch to be ready
            self.wait_for_elasticsearch()

            logger.info("Elasticsearch client initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize Elasticsearch client", error=str(e))
            self.es_client = None

        try:
            # RabbitMQ client
            rabbitmq_host = os.getenv('RABBITMQ_HOST', 'rabbitmq')
            rabbitmq_user = os.getenv('RABBITMQ_USER', 'agentic_user')
            rabbitmq_password = os.getenv('RABBITMQ_PASSWORD', 'agentic123')

            credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
            parameters = pika.ConnectionParameters(
                host=rabbitmq_host,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            self.rabbitmq_connection = pika.BlockingConnection(parameters)

            logger.info("RabbitMQ client initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize RabbitMQ client", error=str(e))
            self.rabbitmq_connection = None

    def wait_for_elasticsearch(self):
        """Wait for Elasticsearch to be ready"""
        max_retries = 30
        for attempt in range(max_retries):
            try:
                if self.es_client.ping():
                    logger.info("Elasticsearch is ready")
                    return
            except Exception as e:
                logger.warning(f"Waiting for Elasticsearch (attempt {attempt + 1}/{max_retries})", error=str(e))

            time.sleep(2)

        raise Exception("Elasticsearch is not ready after maximum retries")

    def setup_message_consumer(self):
        """Setup RabbitMQ message consumer"""
        if not self.rabbitmq_connection:
            return

        try:
            channel = self.rabbitmq_connection.channel()
            channel.exchange_declare(exchange='elasticsearch', exchange_type='direct', durable=True)
            channel.queue_declare(queue='elasticsearch_indexing', durable=True)
            channel.queue_bind(exchange='elasticsearch', queue='elasticsearch_indexing')

            channel.basic_consume(
                queue='elasticsearch_indexing',
                on_message_callback=self.handle_indexing_message,
                auto_ack=False
            )

            # Start consuming in a separate thread
            import threading
            consumer_thread = threading.Thread(target=channel.start_consuming)
            consumer_thread.daemon = True
            consumer_thread.start()

            logger.info("Message consumer started")

        except Exception as e:
            logger.error("Failed to setup message consumer", error=str(e))

    def handle_indexing_message(self, ch, method, properties, body):
        """Handle incoming indexing messages"""
        try:
            message = json.loads(body)
            logger.info("Received indexing message", message_type=message.get('type'))

            if message.get('type') == 'index_document':
                self.index_document(message['data'])
            elif message.get('type') == 'delete_document':
                self.delete_document(message['data'])
            elif message.get('type') == 'bulk_index':
                self.bulk_index(message['data'])

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error("Failed to process indexing message", error=str(e))
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def index_document(self, data: Dict[str, Any]):
        """Index a single document"""
        try:
            INDEXING_REQUESTS.inc()

            result = self.es_client.index(
                index=data['index'],
                id=data['document_id'],
                document=data['document'],
                pipeline=data.get('pipeline')
            )

            logger.info("Document indexed successfully",
                       index=data['index'],
                       document_id=data['document_id'],
                       result=result['result'])

        except Exception as e:
            logger.error("Failed to index document", error=str(e), data=data)

    def delete_document(self, data: Dict[str, Any]):
        """Delete a document"""
        try:
            result = self.es_client.delete(
                index=data['index'],
                id=data['document_id']
            )

            logger.info("Document deleted successfully",
                       index=data['index'],
                       document_id=data['document_id'])

        except Exception as e:
            logger.error("Failed to delete document", error=str(e), data=data)

    def bulk_index(self, data: Dict[str, Any]):
        """Bulk index documents"""
        try:
            operations = []
            for doc in data['documents']:
                operations.extend([
                    {"index": {"_index": data['index'], "_id": doc['id']}},
                    doc['data']
                ])

            result = self.es_client.bulk(operations=operations)

            logger.info("Bulk indexing completed",
                       index=data['index'],
                       documents=len(data['documents']),
                       errors=result['errors'])

        except Exception as e:
            logger.error("Failed to bulk index documents", error=str(e))

    def search_documents(self, request: SearchRequest) -> Dict[str, Any]:
        """Search documents in Elasticsearch"""
        try:
            SEARCH_REQUESTS.inc()
            with SEARCH_DURATION.time():

                # Build query
                query = {
                    "query": {
                        "multi_match": {
                            "query": request.query,
                            "fields": ["*"]
                        }
                    },
                    "size": request.size,
                    "from": request.from_
                }

                # Add filters if provided
                if request.filters:
                    query["query"] = {
                        "bool": {
                            "must": {
                                "multi_match": {
                                    "query": request.query,
                                    "fields": ["*"]
                                }
                            },
                            "filter": []
                        }
                    }

                    # Convert filters to Elasticsearch format
                    for key, value in request.filters.items():
                        query["query"]["bool"]["filter"].append({
                            "term": {key: value}
                        })

                # Add sort if provided
                if request.sort:
                    query["sort"] = request.sort

                # Add source filtering if fields specified
                if request.fields:
                    query["_source"] = request.fields

                result = self.es_client.search(
                    index=request.index,
                    body=query
                )

                return {
                    "total": result["hits"]["total"]["value"],
                    "hits": result["hits"]["hits"],
                    "took": result["took"],
                    "timed_out": result["timed_out"]
                }

        except Exception as e:
            logger.error("Search failed", error=str(e), request=request.dict())
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    def create_index(self, request: CreateIndexRequest):
        """Create a new index"""
        try:
            body = {}
            if request.mappings:
                body["mappings"] = request.mappings
            if request.settings:
                body["settings"] = request.settings

            result = self.es_client.indices.create(
                index=request.index,
                body=body if body else None
            )

            logger.info("Index created successfully", index=request.index)
            return result

        except es_exceptions.RequestError as e:
            if e.error == 'resource_already_exists_exception':
                raise HTTPException(status_code=409, detail="Index already exists")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Failed to create index", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to create index: {str(e)}")

    def get_index_stats(self, index: str = None) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            if index:
                stats = self.es_client.indices.stats(index=index)
            else:
                stats = self.es_client.indices.stats()

            # Update Prometheus metrics
            if "_all" in stats:
                INDEX_SIZE.set(stats["_all"]["primaries"]["store"]["size_in_bytes"])

            return stats

        except Exception as e:
            logger.error("Failed to get index stats", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            health_status = {
                "status": "healthy",
                "service": "elasticsearch",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "elasticsearch": {
                    "connected": self.es_client is not None and self.es_client.ping(),
                    "cluster_health": None
                },
                "rabbitmq": {
                    "connected": self.rabbitmq_connection is not None and not self.rabbitmq_connection.is_closed
                }
            }

            # Check Elasticsearch cluster health
            if health_status["elasticsearch"]["connected"]:
                try:
                    health = self.es_client.cluster.health()
                    health_status["elasticsearch"]["cluster_health"] = {
                        "status": health["status"],
                        "number_of_nodes": health["number_of_nodes"],
                        "active_shards": health["active_shards"]
                    }
                except Exception as e:
                    health_status["elasticsearch"]["cluster_health"] = f"Error: {str(e)}"
                    health_status["status"] = "degraded"

            # Check RabbitMQ connection
            if not health_status["rabbitmq"]["connected"]:
                health_status["status"] = "degraded"

            if health_status["status"] == "degraded":
                health_status["status"] = "degraded"
            elif health_status["status"] == "healthy":
                health_status["status"] = "healthy"

            return JSONResponse(
                content=health_status,
                status_code=200 if health_status["status"] == "healthy" else 503
            )

        @self.app.post("/search")
        async def search(request: SearchRequest):
            """Search documents"""
            return self.search_documents(request)

        @self.app.post("/index")
        async def index_document(request: IndexDocument):
            """Index a document"""
            self.index_document({
                "index": request.index,
                "document_id": request.document_id,
                "document": request.document,
                "pipeline": request.pipeline
            })
            return {"status": "indexed", "index": request.index, "id": request.document_id}

        @self.app.post("/indices")
        async def create_index(request: CreateIndexRequest):
            """Create a new index"""
            return self.create_index(request)

        @self.app.get("/indices/{index}/stats")
        async def get_stats(index: str = None):
            """Get index statistics"""
            return self.get_index_stats(index)

        @self.app.delete("/indices/{index}")
        async def delete_index(index: str):
            """Delete an index"""
            try:
                result = self.es_client.indices.delete(index=index)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to delete index: {str(e)}")

        @self.app.get("/cluster/health")
        async def cluster_health():
            """Get cluster health"""
            try:
                return self.es_client.cluster.health()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get cluster health: {str(e)}")

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()

# Main application
service = ElasticsearchService()

if __name__ == "__main__":
    port = int(os.getenv('SERVICE_PORT', '8100'))
    logger.info("Starting Elasticsearch Service", port=port)

    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
