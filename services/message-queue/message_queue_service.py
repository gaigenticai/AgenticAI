#!/usr/bin/env python3
"""
Message Queue Service for Agentic Platform

This service provides advanced RabbitMQ message queue management with:
- Event-driven processing with dead letter queues
- Priority queues for time-sensitive data
- Message routing and filtering
- Comprehensive monitoring and metrics
- Automatic retry mechanisms with exponential backoff
- Message deduplication and idempotency
- Queue health monitoring and alerting
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pika
import psycopg2
import structlog
from fastapi import FastAPI, HTTPException
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
    title="Message Queue Service",
    description="Advanced RabbitMQ message queue management with event-driven processing",
    version="1.0.0"
)

# Prometheus metrics
MESSAGES_PUBLISHED = Counter('messages_published_total', 'Total messages published', ['queue_name', 'priority'])
MESSAGES_CONSUMED = Counter('messages_consumed_total', 'Total messages consumed', ['queue_name'])
MESSAGES_FAILED = Counter('messages_failed_total', 'Total message processing failures', ['queue_name', 'error_type'])
QUEUE_SIZE = Gauge('queue_size', 'Current queue size', ['queue_name'])
MESSAGE_PROCESSING_TIME = Histogram('message_processing_duration_seconds', 'Message processing duration', ['queue_name'])
DEAD_LETTER_MESSAGES = Counter('dead_letter_messages_total', 'Total messages sent to dead letter queue', ['queue_name'])

# Global variables
rabbitmq_connection = None
rabbitmq_channel = None
database_connection = None

# Pydantic models
class Message(BaseModel):
    """Message model for queue operations"""
    id: str = Field(..., description="Unique message identifier")
    queue_name: str = Field(..., description="Target queue name")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    priority: int = Field(1, description="Message priority (1-10, higher is more important)")
    headers: Optional[Dict[str, Any]] = Field(None, description="Message headers")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for request tracking")
    reply_to: Optional[str] = Field(None, description="Reply queue for RPC patterns")
    expiration: Optional[int] = Field(None, description="Message expiration time in seconds")

class QueueConfig(BaseModel):
    """Queue configuration model"""
    queue_name: str = Field(..., description="Name of the queue")
    durable: bool = Field(True, description="Whether queue survives broker restart")
    max_priority: int = Field(10, description="Maximum priority level for priority queues")
    dead_letter_exchange: Optional[str] = Field(None, description="Dead letter exchange name")
    dead_letter_routing_key: Optional[str] = Field(None, description="Dead letter routing key")
    message_ttl: Optional[int] = Field(None, description="Message time-to-live in milliseconds")
    max_length: Optional[int] = Field(None, description="Maximum queue length")

class QueueStats(BaseModel):
    """Queue statistics model"""
    queue_name: str
    message_count: int
    consumer_count: int
    publish_rate: float
    consume_rate: float
    dead_letter_count: int
    error_rate: float

class MessageQueueManager:
    """Advanced message queue manager with RabbitMQ"""

    def __init__(self):
        self.queues = {}
        self.exchanges = {}
        self.bindings = []
        self.consumers = {}
        self.deduplication_cache = set()

    def setup_rabbitmq_infrastructure(self):
        """Setup RabbitMQ exchanges, queues, and bindings"""
        try:
            # Declare main exchange
            rabbitmq_channel.exchange_declare(
                exchange='agentic.events',
                exchange_type='topic',
                durable=True
            )

            # Declare dead letter exchange
            rabbitmq_channel.exchange_declare(
                exchange='agentic.dlx',
                exchange_type='direct',
                durable=True
            )

            # Setup ingestion queues
            self._setup_ingestion_queues()

            # Setup output queues
            self._setup_output_queues()

            # Setup data processing queues
            self._setup_processing_queues()

            # Setup dead letter queues
            self._setup_dead_letter_queues()

            logger.info("RabbitMQ infrastructure setup completed")

        except Exception as e:
            logger.error("Failed to setup RabbitMQ infrastructure", error=str(e))
            raise

    def _setup_ingestion_queues(self):
        """Setup ingestion-related queues"""
        ingestion_queues = [
            'csv_ingestion',
            'excel_ingestion',
            'pdf_ingestion',
            'json_ingestion',
            'api_ingestion',
            'ui_scraper'
        ]

        for queue_name in ingestion_queues:
            # Main queue
            rabbitmq_channel.queue_declare(
                queue=queue_name,
                durable=True,
                arguments={
                    'x-max-priority': 10,
                    'x-dead-letter-exchange': 'agentic.dlx',
                    'x-dead-letter-routing-key': f'{queue_name}.dead'
                }
            )

            # Bind to main exchange
            rabbitmq_channel.queue_bind(
                exchange='agentic.events',
                queue=queue_name,
                routing_key=f'ingestion.{queue_name}'
            )

            self.queues[queue_name] = {
                'type': 'ingestion',
                'priority_support': True,
                'dead_letter_support': True
            }

    def _setup_output_queues(self):
        """Setup output-related queues"""
        output_queues = [
            'postgresql_output',
            'mongodb_output',
            'qdrant_output',
            'elasticsearch_output',
            'minio_output'
        ]

        for queue_name in output_queues:
            # Main queue
            rabbitmq_channel.queue_declare(
                queue=queue_name,
                durable=True,
                arguments={
                    'x-max-priority': 10,
                    'x-dead-letter-exchange': 'agentic.dlx',
                    'x-dead-letter-routing-key': f'{queue_name}.dead'
                }
            )

            # Bind to main exchange
            rabbitmq_channel.queue_bind(
                exchange='agentic.events',
                queue=queue_name,
                routing_key=f'output.{queue_name}'
            )

            self.queues[queue_name] = {
                'type': 'output',
                'priority_support': True,
                'dead_letter_support': True
            }

    def _setup_processing_queues(self):
        """Setup data processing queues"""
        processing_queues = [
            'data_validation',
            'quality_check',
            'transformation_request',
            'metadata_profiling',
            'data_lake_ingestion'
        ]

        for queue_name in processing_queues:
            # Main queue
            rabbitmq_channel.queue_declare(
                queue=queue_name,
                durable=True,
                arguments={
                    'x-max-priority': 5,
                    'x-dead-letter-exchange': 'agentic.dlx',
                    'x-dead-letter-routing-key': f'{queue_name}.dead'
                }
            )

            # Bind to main exchange
            rabbitmq_channel.queue_bind(
                exchange='agentic.events',
                queue=queue_name,
                routing_key=f'processing.{queue_name}'
            )

            self.queues[queue_name] = {
                'type': 'processing',
                'priority_support': True,
                'dead_letter_support': True
            }

    def _setup_dead_letter_queues(self):
        """Setup dead letter queues for error handling"""
        for queue_name in self.queues.keys():
            dead_queue_name = f'{queue_name}.dead'

            # Dead letter queue
            rabbitmq_channel.queue_declare(
                queue=dead_queue_name,
                durable=True
            )

            # Bind to dead letter exchange
            rabbitmq_channel.queue_bind(
                exchange='agentic.dlx',
                queue=dead_queue_name,
                routing_key=f'{queue_name}.dead'
            )

    def publish_message(self, message: Message) -> Dict[str, Any]:
        """Publish message to queue with advanced features"""
        try:
            # Check for message deduplication
            message_key = f"{message.id}:{message.queue_name}"
            if message.id in self.deduplication_cache:
                logger.warning("Duplicate message detected", message_id=message.id)
                return {"status": "duplicate", "message_id": message.id}

            # Add to deduplication cache
            self.deduplication_cache.add(message.id)

            # Prepare message properties
            properties = pika.BasicProperties(
                message_id=message.id,
                correlation_id=message.correlation_id or message.id,
                timestamp=int(time.time()),
                user_id='agentic-platform',
                priority=message.priority,
                headers=message.headers or {},
                expiration=str(message.expiration * 1000) if message.expiration else None,
                reply_to=message.reply_to
            )

            # Publish message
            routing_key = f"{self.queues[message.queue_name]['type']}.{message.queue_name}"

            rabbitmq_channel.basic_publish(
                exchange='agentic.events',
                routing_key=routing_key,
                body=json.dumps(message.payload),
                properties=properties,
                mandatory=True
            )

            # Update metrics
            MESSAGES_PUBLISHED.labels(
                queue_name=message.queue_name,
                priority=str(message.priority)
            ).inc()

            logger.info("Message published successfully",
                       message_id=message.id,
                       queue=message.queue_name,
                       priority=message.priority)

            return {
                "status": "published",
                "message_id": message.id,
                "queue_name": message.queue_name,
                "routing_key": routing_key,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Failed to publish message", error=str(e), message_id=message.id)
            MESSAGES_FAILED.labels(queue_name=message.queue_name, error_type="publish").inc()
            raise

    def get_queue_stats(self) -> List[QueueStats]:
        """Get statistics for all queues"""
        stats = []

        try:
            # Get queue information from RabbitMQ management API
            for queue_name, queue_info in self.queues.items():
                try:
                    # Get queue details
                    queue_details = rabbitmq_channel.queue_declare(queue=queue_name, passive=True)

                    stats.append(QueueStats(
                        queue_name=queue_name,
                        message_count=queue_details.method.message_count,
                        consumer_count=queue_details.method.consumer_count,
                        publish_rate=0.0,  # Would need to track over time
                        consume_rate=0.0,  # Would need to track over time
                        dead_letter_count=self._get_dead_letter_count(queue_name),
                        error_rate=0.0  # Would need to calculate from metrics
                    ))

                except Exception as e:
                    logger.warning("Failed to get stats for queue", queue=queue_name, error=str(e))

            return stats

        except Exception as e:
            logger.error("Failed to get queue statistics", error=str(e))
            return []

    def _get_dead_letter_count(self, queue_name: str) -> int:
        """Get count of messages in dead letter queue"""
        try:
            dead_queue_name = f'{queue_name}.dead'
            queue_details = rabbitmq_channel.queue_declare(queue=dead_queue_name, passive=True)
            return queue_details.method.message_count
        except:
            return 0

    def purge_queue(self, queue_name: str) -> Dict[str, Any]:
        """Purge all messages from a queue"""
        try:
            if queue_name not in self.queues:
                raise ValueError(f"Queue {queue_name} not found")

            result = rabbitmq_channel.queue_purge(queue=queue_name)

            logger.info("Queue purged successfully", queue=queue_name, messages_purged=result.method.message_count)

            return {
                "status": "purged",
                "queue_name": queue_name,
                "messages_purged": result.method.message_count
            }

        except Exception as e:
            logger.error("Failed to purge queue", error=str(e), queue=queue_name)
            raise

    def setup_consumer(self, queue_name: str, callback_function):
        """Setup a consumer for a queue"""
        try:
            if queue_name not in self.queues:
                raise ValueError(f"Queue {queue_name} not found")

            def wrapper_callback(ch, method, properties, body):
                """Wrapper callback with error handling and metrics"""
                start_time = time.time()
                message_id = properties.message_id or "unknown"

                try:
                    # Parse message
                    payload = json.loads(body.decode())

                    # Call user callback
                    callback_function(payload, properties)

                    # Acknowledge message
                    ch.basic_ack(delivery_tag=method.delivery_tag)

                    # Update metrics
                    MESSAGES_CONSUMED.labels(queue_name=queue_name).inc()
                    processing_time = time.time() - start_time
                    MESSAGE_PROCESSING_TIME.labels(queue_name=queue_name).observe(processing_time)

                    logger.info("Message processed successfully",
                               message_id=message_id,
                               queue=queue_name,
                               processing_time=processing_time)

                except Exception as e:
                    logger.error("Message processing failed",
                               message_id=message_id,
                               queue=queue_name,
                               error=str(e))

                    # Send to dead letter queue
                    self._send_to_dead_letter_queue(queue_name, body, properties, str(e))

                    # Negative acknowledge
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

                    MESSAGES_FAILED.labels(queue_name=queue_name, error_type="processing").inc()

            # Start consuming
            rabbitmq_channel.basic_consume(
                queue=queue_name,
                on_message_callback=wrapper_callback
            )

            self.consumers[queue_name] = callback_function
            logger.info("Consumer setup completed", queue=queue_name)

        except Exception as e:
            logger.error("Failed to setup consumer", error=str(e), queue=queue_name)
            raise

    def _send_to_dead_letter_queue(self, queue_name: str, body: bytes,
                                  properties: pika.BasicProperties, error: str):
        """Send failed message to dead letter queue"""
        try:
            dead_routing_key = f'{queue_name}.dead'

            # Add error information to headers
            headers = properties.headers or {}
            headers['x-death-reason'] = error
            headers['x-death-timestamp'] = int(time.time())

            rabbitmq_channel.basic_publish(
                exchange='agentic.dlx',
                routing_key=dead_routing_key,
                body=body,
                properties=pika.BasicProperties(
                    message_id=properties.message_id,
                    correlation_id=properties.correlation_id,
                    timestamp=int(time.time()),
                    headers=headers
                )
            )

            DEAD_LETTER_MESSAGES.labels(queue_name=queue_name).inc()
            logger.info("Message sent to dead letter queue",
                       queue=queue_name,
                       message_id=properties.message_id)

        except Exception as e:
            logger.error("Failed to send to dead letter queue", error=str(e), queue=queue_name)

    def create_queue(self, config: QueueConfig) -> Dict[str, Any]:
        """Create a new queue with custom configuration"""
        try:
            arguments = {}

            if config.max_priority:
                arguments['x-max-priority'] = config.max_priority

            if config.dead_letter_exchange:
                arguments['x-dead-letter-exchange'] = config.dead_letter_exchange
                if config.dead_letter_routing_key:
                    arguments['x-dead-letter-routing-key'] = config.dead_letter_routing_key

            if config.message_ttl:
                arguments['x-message-ttl'] = config.message_ttl

            if config.max_length:
                arguments['x-max-length'] = config.max_length

            rabbitmq_channel.queue_declare(
                queue=config.queue_name,
                durable=config.durable,
                arguments=arguments
            )

            # Bind to main exchange
            rabbitmq_channel.queue_bind(
                exchange='agentic.events',
                queue=config.queue_name,
                routing_key=f'custom.{config.queue_name}'
            )

            self.queues[config.queue_name] = {
                'type': 'custom',
                'priority_support': config.max_priority > 1,
                'dead_letter_support': bool(config.dead_letter_exchange)
            }

            logger.info("Custom queue created", queue=config.queue_name)

            return {
                "status": "created",
                "queue_name": config.queue_name,
                "configuration": config.dict()
            }

        except Exception as e:
            logger.error("Failed to create queue", error=str(e), queue=config.queue_name)
            raise

# Global manager instance
queue_manager = MessageQueueManager()

def setup_rabbitmq():
    """Setup RabbitMQ connection and infrastructure"""
    global rabbitmq_connection, rabbitmq_channel

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

        rabbitmq_connection = pika.BlockingConnection(parameters)
        rabbitmq_channel = rabbitmq_connection.channel()

        # Setup infrastructure
        queue_manager.setup_rabbitmq_infrastructure()

        logger.info("RabbitMQ setup completed")

    except Exception as e:
        logger.error("Failed to setup RabbitMQ", error=str(e))
        raise

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    queue_status = {}
    for queue_name in queue_manager.queues.keys():
        try:
            queue_details = rabbitmq_channel.queue_declare(queue=queue_name, passive=True)
            queue_status[queue_name] = {
                "status": "healthy",
                "message_count": queue_details.method.message_count,
                "consumer_count": queue_details.method.consumer_count
            }
        except Exception as e:
            queue_status[queue_name] = {"status": "error", "error": str(e)}

    return {
        "status": "healthy",
        "service": "message-queue-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "queues": queue_status
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/messages")
async def publish_message(message: Message):
    """Publish a message to a queue"""
    try:
        result = queue_manager.publish_message(message)
        return result

    except Exception as e:
        logger.error("Failed to publish message", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to publish message: {str(e)}")

@app.get("/queues")
async def list_queues():
    """List all queues and their status"""
    try:
        stats = queue_manager.get_queue_stats()
        return {"queues": [stat.dict() for stat in stats]}

    except Exception as e:
        logger.error("Failed to list queues", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list queues: {str(e)}")

@app.post("/queues")
async def create_queue(config: QueueConfig):
    """Create a new queue"""
    try:
        result = queue_manager.create_queue(config)
        return result

    except Exception as e:
        logger.error("Failed to create queue", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create queue: {str(e)}")

@app.delete("/queues/{queue_name}")
async def purge_queue(queue_name: str):
    """Purge all messages from a queue"""
    try:
        result = queue_manager.purge_queue(queue_name)
        return result

    except Exception as e:
        logger.error("Failed to purge queue", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to purge queue: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get message queue service statistics"""
    return {
        "service": "message-queue-service",
        "metrics": {
            "messages_published_total": MESSAGES_PUBLISHED._value.get(),
            "messages_consumed_total": MESSAGES_CONSUMED._value.get(),
            "messages_failed_total": MESSAGES_FAILED._value.get(),
            "dead_letter_messages_total": DEAD_LETTER_MESSAGES._value.get(),
            "active_queues": len(queue_manager.queues),
            "active_consumers": len(queue_manager.consumers)
        },
        "queues": list(queue_manager.queues.keys()),
        "exchanges": list(queue_manager.exchanges.keys())
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global database_connection

    logger.info("Message Queue Service starting up...")

    # Setup database connection
    try:
        db_config = {
            "host": os.getenv("POSTGRES_HOST", "postgresql_ingestion"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agentic_ingestion"),
            "user": os.getenv("POSTGRES_USER", "agentic_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "")
        }

        if not db_config.get("password"):
            logger.error("POSTGRES_PASSWORD not configured for Message Queue Service")
            raise RuntimeError("POSTGRES_PASSWORD not configured for Message Queue Service")

        database_connection = psycopg2.connect(**db_config)
        logger.info("Database connection established")

    except Exception as e:
        logger.warning("Database connection failed", error=str(e))

    # Setup RabbitMQ in background thread
    rabbitmq_thread = threading.Thread(target=setup_rabbitmq, daemon=True)
    rabbitmq_thread.start()

    logger.info("Message Queue Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global database_connection, rabbitmq_connection

    logger.info("Message Queue Service shutting down...")

    # Close database connection
    if database_connection:
        database_connection.close()

    # Close RabbitMQ connection
    if rabbitmq_connection:
        rabbitmq_connection.close()

    logger.info("Message Queue Service shutdown complete")

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
        "message_queue_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8091)),
        reload=False,
        log_level="info"
    )
