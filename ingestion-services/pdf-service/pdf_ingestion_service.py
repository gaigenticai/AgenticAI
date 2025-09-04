#!/usr/bin/env python3
"""
PDF Ingestion Service for Agentic Platform

This service processes PDF files with OCR capabilities with the following features:
- Text extraction from PDF files
- OCR processing for scanned documents
- Image extraction and processing
- Table detection and extraction
- Multi-page document handling
- Error handling and recovery
- Performance monitoring
"""

import io
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
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import uvicorn

try:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PDF libraries not available. Install PyMuPDF, pytesseract, and Pillow for PDF support.")

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

# Create custom registry for PDF service to avoid metric name conflicts
PDF_REGISTRY = CollectorRegistry()

# FastAPI app
app = FastAPI(
    title="PDF Ingestion Service",
    description="Processes PDF files with OCR and text extraction capabilities",
    version="1.0.0"
)

# Prometheus metrics - using custom registry to avoid conflicts
PDF_FILES_PROCESSED = Counter('pdf_files_processed_total', 'Total PDF files processed', ['status'], registry=PDF_REGISTRY)
PDF_PAGES_PROCESSED = Counter('pdf_pages_processed_total', 'Total PDF pages processed', registry=PDF_REGISTRY)
PDF_PROCESSING_TIME = Histogram('pdf_processing_duration_seconds', 'PDF processing duration', ['operation'], registry=PDF_REGISTRY)
PDF_OCR_ERRORS = Counter('pdf_ocr_errors_total', 'Total OCR processing errors', registry=PDF_REGISTRY)
ACTIVE_PDF_JOBS = Gauge('active_pdf_jobs', 'Number of active PDF processing jobs', registry=PDF_REGISTRY)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://agentic_user:agentic123@postgresql_ingestion:5432/agentic_ingestion")
engine = create_engine(DATABASE_URL)

# Message queue connection
rabbitmq_connection = None
rabbitmq_channel = None

class PDFProcessor:
    """PDF file processor with OCR and text extraction capabilities"""

    def __init__(self):
        self.db_connection = None

    def connect_db(self):
        """Connect to database"""
        try:
            self.db_connection = psycopg2.connect(DATABASE_URL)
            logger.info("Database connection established")
        except Exception as e:
            logger.error("Failed to connect to database", error=str(e))
            raise

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

    def validate_pdf_structure(self, pdf_document) -> Dict[str, Any]:
        """Validate PDF file structure and extract metadata"""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }

        try:
            # Get basic PDF information
            page_count = len(pdf_document)
            validation_results["metadata"]["page_count"] = page_count
            validation_results["metadata"]["file_size"] = pdf_document.metadata.get("file_size", 0)

            # Extract metadata
            metadata = pdf_document.metadata
            validation_results["metadata"]["title"] = metadata.get("title", "")
            validation_results["metadata"]["author"] = metadata.get("author", "")
            validation_results["metadata"]["creator"] = metadata.get("creator", "")
            validation_results["metadata"]["producer"] = metadata.get("producer", "")
            validation_results["metadata"]["subject"] = metadata.get("subject", "")

            # Check for encryption
            if pdf_document.is_encrypted:
                validation_results["warnings"].append("PDF file is encrypted")

            # Sample first few pages for content analysis
            text_pages = 0
            image_pages = 0

            sample_pages = min(5, page_count)
            for page_num in range(sample_pages):
                page = pdf_document[page_num]

                # Check for text content
                text = page.get_text()
                if text.strip():
                    text_pages += 1

                # Check for images
                images = page.get_images(full=True)
                if images:
                    image_pages += 1

            validation_results["metadata"]["text_pages_sample"] = text_pages
            validation_results["metadata"]["image_pages_sample"] = image_pages

            # Determine if OCR will be needed
            if text_pages < image_pages:
                validation_results["warnings"].append("Document appears to be image-based, OCR processing recommended")

        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"PDF validation error: {str(e)}")

        return validation_results

    def extract_text_from_page(self, page, config: Dict[str, Any]) -> str:
        """Extract text from a PDF page using native text extraction or OCR"""
        try:
            # First try native text extraction
            text = page.get_text()

            # If little or no text found, try OCR
            if not text.strip() or len(text.strip()) < 50:
                if config.get("enable_ocr", True):
                    text = self.perform_ocr_on_page(page, config)

            return text

        except Exception as e:
            logger.error("Text extraction failed", error=str(e))
            return ""

    def perform_ocr_on_page(self, page, config: Dict[str, Any]) -> str:
        """Perform OCR on a PDF page"""
        try:
            if not PDF_SUPPORT:
                return ""

            # Get page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
            img = Image.open(io.BytesIO(pix.tobytes()))

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Perform OCR
            ocr_config = config.get("ocr_config", {})
            lang = ocr_config.get("lang", "eng")
            custom_config = ocr_config.get("config", '--oem 3 --psm 6')

            text = pytesseract.image_to_string(img, lang=lang, config=custom_config)

            PDF_PAGES_PROCESSED.inc()
            return text

        except Exception as e:
            PDF_OCR_ERRORS.inc()
            logger.error("OCR processing failed", error=str(e))
            return ""

    def extract_tables_from_page(self, page, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tables from a PDF page"""
        tables = []

        try:
            # Find tables on the page
            tabs = page.find_tables()

            for tab in tabs:
                try:
                    # Extract table data
                    table_data = tab.extract()

                    if table_data:
                        tables.append({
                            "page_number": page.number + 1,
                            "row_count": len(table_data),
                            "column_count": len(table_data[0]) if table_data else 0,
                            "data": table_data
                        })

                except Exception as e:
                    logger.error("Table extraction failed", error=str(e), page=page.number)

        except Exception as e:
            logger.error("Table detection failed", error=str(e), page=page.number)

        return tables

    def process_pdf_file(self, job_id: str, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a PDF file with text extraction and OCR"""
        start_time = time.time()
        ACTIVE_PDF_JOBS.inc()

        try:
            logger.info("Starting PDF processing", job_id=job_id, file_path=file_path)

            if not PDF_SUPPORT:
                error_msg = "PDF processing not available. Install PyMuPDF, pytesseract, and Pillow packages."
                self.update_job_status(job_id, "failed", error_message=error_msg)
                return {"status": "failed", "error": error_msg}

            # Update job status to processing
            self.update_job_status(job_id, "processing")

            # Open PDF document
            pdf_document = fitz.open(file_path)

            # Validate PDF structure
            validation_results = self.validate_pdf_structure(pdf_document)

            if not validation_results["is_valid"]:
                error_msg = "; ".join(validation_results["errors"])
                self.update_job_status(job_id, "failed", error_message=error_msg)
                PDF_FILES_PROCESSED.labels(status="failed").inc()
                pdf_document.close()
                return {"status": "failed", "error": error_msg}

            # Process pages
            page_config = config.get("page_config", {})
            total_pages = len(pdf_document)
            processed_pages = 0
            extracted_text = []
            extracted_tables = []

            # Determine page range to process
            start_page = page_config.get("start_page", 0)
            end_page = page_config.get("end_page", total_pages)
            page_range = range(start_page, min(end_page, total_pages))

            for page_num in page_range:
                try:
                    page = pdf_document[page_num]

                    # Extract text from page
                    page_text = self.extract_text_from_page(page, config)
                    if page_text.strip():
                        extracted_text.append({
                            "page_number": page_num + 1,
                            "text": page_text,
                            "char_count": len(page_text)
                        })

                    # Extract tables if enabled
                    if config.get("extract_tables", False):
                        page_tables = self.extract_tables_from_page(page, config)
                        extracted_tables.extend(page_tables)

                    processed_pages += 1

                except Exception as e:
                    logger.error("Page processing failed", error=str(e), page=page_num)

            # Store validation results and extracted content
            self.store_pdf_results(job_id, validation_results, extracted_text, extracted_tables)

            # Process extracted data
            total_records = len(extracted_text) + len(extracted_tables)
            processed_records = self.process_extracted_data(job_id, extracted_text, extracted_tables, config)

            pdf_document.close()

            # Update final status
            if processed_records > 0:
                self.update_job_status(
                    job_id,
                    "completed",
                    total_records=total_records,
                    processed_records=processed_records
                )
                PDF_FILES_PROCESSED.labels(status="success").inc()
                status = "completed"
            else:
                self.update_job_status(
                    job_id,
                    "failed",
                    error_message="No content could be extracted from PDF"
                )
                PDF_FILES_PROCESSED.labels(status="failed").inc()
                status = "failed"

            processing_time = time.time() - start_time
            PDF_PROCESSING_TIME.labels(operation="full_process").observe(processing_time)

            logger.info("PDF processing completed",
                       job_id=job_id,
                       status=status,
                       pages_processed=processed_pages,
                       text_pages=len(extracted_text),
                       tables_extracted=len(extracted_tables),
                       total_records=total_records,
                       processed=processed_records,
                       duration=processing_time)

            return {
                "status": status,
                "total_pages": total_pages,
                "processed_pages": processed_pages,
                "text_pages": len(extracted_text),
                "tables_extracted": len(extracted_tables),
                "total_records": total_records,
                "processed_records": processed_records,
                "processing_time": processing_time,
                "validation_results": validation_results
            }

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"PDF processing failed: {str(e)}"

            self.update_job_status(job_id, "failed", error_message=error_msg)
            PDF_FILES_PROCESSED.labels(status="failed").inc()
            PDF_PROCESSING_TIME.labels(operation="full_process").observe(processing_time)

            logger.error("PDF processing failed",
                        job_id=job_id,
                        error=str(e),
                        duration=processing_time)

            return {"status": "failed", "error": error_msg}

        finally:
            ACTIVE_PDF_JOBS.dec()

    def store_pdf_results(self, job_id: str, validation_results: Dict[str, Any],
                         extracted_text: List[Dict[str, Any]], extracted_tables: List[Dict[str, Any]]):
        """Store PDF processing results in database"""
        try:
            with self.db_connection.cursor() as cursor:
                # Store validation results
                cursor.execute("""
                    INSERT INTO data_validation_results
                    (job_id, validation_type, is_valid, error_message, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    job_id,
                    "pdf_structure_validation",
                    validation_results["is_valid"],
                    "; ".join(validation_results["errors"]) if validation_results["errors"] else None,
                    json.dumps(validation_results)
                ))

                # Store extracted text
                for text_item in extracted_text:
                    cursor.execute("""
                        INSERT INTO pdf_extracted_content
                        (job_id, page_number, content_type, content, metadata, created_at)
                        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (
                        job_id,
                        text_item["page_number"],
                        "text",
                        text_item["text"],
                        json.dumps({"char_count": text_item["char_count"]})
                    ))

                # Store extracted tables
                for table_item in extracted_tables:
                    cursor.execute("""
                        INSERT INTO pdf_extracted_content
                        (job_id, page_number, content_type, content, metadata, created_at)
                        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (
                        job_id,
                        table_item["page_number"],
                        "table",
                        json.dumps(table_item["data"]),
                        json.dumps({
                            "row_count": table_item["row_count"],
                            "column_count": table_item["column_count"]
                        })
                    ))

                self.db_connection.commit()

        except Exception as e:
            logger.error("Failed to store PDF results", error=str(e), job_id=job_id)

    def process_extracted_data(self, job_id: str, extracted_text: List[Dict[str, Any]],
                              extracted_tables: List[Dict[str, Any]], config: Dict[str, Any]) -> int:
        """Process extracted PDF data"""
        # This is where you would implement the actual data processing logic
        # For now, just simulate processing
        total_processed = 0

        for text_item in extracted_text:
            time.sleep(0.001)  # Simulate processing time
            total_processed += 1

        for table_item in extracted_tables:
            time.sleep(0.002)  # Simulate processing time
            total_processed += 1

        return total_processed

# Global processor instance
processor = PDFProcessor()

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
        rabbitmq_channel.queue_declare(queue='pdf_ingestion', durable=True)

        # Set up consumer
        rabbitmq_channel.basic_qos(prefetch_count=1)
        rabbitmq_channel.basic_consume(
            queue='pdf_ingestion',
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

        logger.info("Received PDF ingestion job", job_id=job_id)

        # Process the PDF file
        file_path = source_config["connection_config"]["file_path"]
        result = processor.process_pdf_file(job_id, file_path, source_config)

        logger.info("PDF ingestion job completed", job_id=job_id, result=result)

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
        "service": "pdf-ingestion-service",
        "timestamp": datetime.utcnow().isoformat(),
        "active_jobs": ACTIVE_PDF_JOBS._value.get(),
        "pdf_support": PDF_SUPPORT
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(PDF_REGISTRY)

# Startup and shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("PDF Ingestion Service starting up...")

    # Connect to database
    processor.connect_db()

    # Setup RabbitMQ consumer in background thread
    import threading
    rabbitmq_thread = threading.Thread(target=setup_rabbitmq, daemon=True)
    rabbitmq_thread.start()

    logger.info("PDF Ingestion Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("PDF Ingestion Service shutting down...")

    # Close database connection
    if processor.db_connection:
        processor.db_connection.close()

    # Close RabbitMQ connection
    if rabbitmq_connection:
        rabbitmq_connection.close()

    logger.info("PDF Ingestion Service shutdown complete")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "pdf_ingestion_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8083)),
        reload=False,
        log_level="info"
    )
