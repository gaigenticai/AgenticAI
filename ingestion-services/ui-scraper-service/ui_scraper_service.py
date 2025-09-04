#!/usr/bin/env python3
"""
UI Scraper Service for Agentic Platform

This service scrapes data from web interfaces and user interfaces with the following capabilities:
- Web page scraping and data extraction
- JavaScript-rendered content handling
- Form interaction and submission
- API endpoint discovery
- Rate limiting and anti-bot detection
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
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import uvicorn

try:
    import requests
    from bs4 import BeautifulSoup
    import selenium
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SCRAPER_SUPPORT = True
except ImportError:
    SCRAPER_SUPPORT = False
    logger.warning("Scraping libraries not available. Install requests, beautifulsoup4, and selenium for web scraping support.")

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

# Create custom registry for UI scraper service to avoid metric name conflicts
SCRAPER_REGISTRY = CollectorRegistry()

# FastAPI app
app = FastAPI(
    title="UI Scraper Service",
    description="Scrapes data from web interfaces and user interfaces",
    version="1.0.0"
)

# Prometheus metrics - using custom registry to avoid conflicts
SCRAPER_REQUESTS_PROCESSED = Counter('scraper_requests_processed_total', 'Total scraping requests processed', ['status'], registry=SCRAPER_REGISTRY)
SCRAPER_PAGES_SCRAPED = Counter('scraper_pages_scraped_total', 'Total pages scraped', registry=SCRAPER_REGISTRY)
SCRAPER_PROCESSING_TIME = Histogram('scraper_processing_duration_seconds', 'Scraping processing duration', ['operation'], registry=SCRAPER_REGISTRY)
SCRAPER_ERRORS = Counter('scraper_errors_total', 'Total scraping errors', ['error_type'], registry=SCRAPER_REGISTRY)
ACTIVE_SCRAPER_JOBS = Gauge('active_scraper_jobs', 'Number of active scraping jobs', registry=SCRAPER_REGISTRY)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://agentic_user:agentic123@postgresql_ingestion:5432/agentic_ingestion")
engine = create_engine(DATABASE_URL)

# Message queue connection
rabbitmq_connection = None
rabbitmq_channel = None

class WebScraper:
    """Web scraper with support for static and dynamic content"""

    def __init__(self):
        self.db_connection = None
        self.driver = None

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

    def validate_scraping_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scraping configuration"""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "config_analysis": {}
        }

        try:
            # Check required fields
            if "url" not in config:
                validation_results["is_valid"] = False
                validation_results["errors"].append("URL is required")

            if "selectors" not in config and "api_endpoints" not in config:
                validation_results["warnings"].append("No selectors or API endpoints specified")

            # Analyze configuration
            if "selectors" in config:
                selectors = config["selectors"]
                validation_results["config_analysis"]["selector_count"] = len(selectors)
                validation_results["config_analysis"]["selector_types"] = list(set(
                    sel.get("type", "unknown") for sel in selectors
                ))

            if "api_endpoints" in config:
                endpoints = config["api_endpoints"]
                validation_results["config_analysis"]["api_endpoint_count"] = len(endpoints)

            # Check for rate limiting
            if "rate_limit" not in config:
                validation_results["warnings"].append("No rate limiting configured")

            # Check for anti-bot measures
            if "user_agent" not in config:
                validation_results["warnings"].append("No custom user agent specified")

        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Configuration validation error: {str(e)}")

        return validation_results

    def scrape_static_content(self, url: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape static web content using requests and BeautifulSoup"""
        try:
            headers = config.get("headers", {})
            if "user_agent" in config:
                headers["User-Agent"] = config["user_agent"]

            timeout = config.get("timeout", 30)

            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            extracted_data = {}

            # Extract data using selectors
            if "selectors" in config:
                for selector in config["selectors"]:
                    selector_name = selector["name"]
                    selector_type = selector.get("type", "css")
                    selector_value = selector["value"]

                    try:
                        if selector_type == "css":
                            elements = soup.select(selector_value)
                        elif selector_type == "xpath":
                            # Note: BeautifulSoup doesn't support XPath natively
                            # This would require lxml
                            elements = []
                        else:
                            elements = soup.select(selector_value)

                        if selector.get("multiple", False):
                            extracted_data[selector_name] = [
                                elem.get_text(strip=True) if hasattr(elem, 'get_text') else str(elem)
                                for elem in elements
                            ]
                        else:
                            extracted_data[selector_name] = (
                                elements[0].get_text(strip=True) if elements
                                else None
                            )

                    except Exception as e:
                        logger.error("Selector extraction failed",
                                   selector=selector_name,
                                   error=str(e))

            return {
                "status": "success",
                "url": url,
                "response_time": response.elapsed.total_seconds(),
                "status_code": response.status_code,
                "content_length": len(response.content),
                "extracted_data": extracted_data
            }

        except Exception as e:
            logger.error("Static scraping failed", url=url, error=str(e))
            return {
                "status": "failed",
                "url": url,
                "error": str(e)
            }

    def scrape_dynamic_content(self, url: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape dynamic web content using Selenium"""
        try:
            if not self.driver:
                self.setup_selenium_driver()

            self.driver.get(url)

            # Wait for page to load
            wait_time = config.get("wait_time", 10)
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            extracted_data = {}

            # Extract data using selectors
            if "selectors" in config:
                for selector in config["selectors"]:
                    selector_name = selector["name"]
                    selector_type = selector.get("type", "css")
                    selector_value = selector["value"]

                    try:
                        if selector_type == "css":
                            elements = self.driver.find_elements(By.CSS_SELECTOR, selector_value)
                        elif selector_type == "xpath":
                            elements = self.driver.find_elements(By.XPATH, selector_value)
                        else:
                            elements = self.driver.find_elements(By.CSS_SELECTOR, selector_value)

                        if selector.get("multiple", False):
                            extracted_data[selector_name] = [
                                elem.text for elem in elements
                            ]
                        else:
                            extracted_data[selector_name] = (
                                elements[0].text if elements else None
                            )

                    except Exception as e:
                        logger.error("Dynamic selector extraction failed",
                                   selector=selector_name,
                                   error=str(e))

            # Handle form interactions if specified
            if "form_actions" in config:
                self.handle_form_actions(config["form_actions"])

            return {
                "status": "success",
                "url": url,
                "page_title": self.driver.title,
                "current_url": self.driver.current_url,
                "extracted_data": extracted_data
            }

        except Exception as e:
            logger.error("Dynamic scraping failed", url=url, error=str(e))
            return {
                "status": "failed",
                "url": url,
                "error": str(e)
            }

    def setup_selenium_driver(self):
        """Setup Selenium WebDriver with headless Chrome"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")

            # Add user agent if specified
            user_agent = os.getenv("SELENIUM_USER_AGENT")
            if user_agent:
                chrome_options.add_argument(f"--user-agent={user_agent}")

            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Selenium WebDriver initialized")

        except Exception as e:
            logger.error("Failed to setup Selenium driver", error=str(e))
            raise

    def handle_form_actions(self, form_actions: List[Dict[str, Any]]):
        """Handle form interactions"""
        try:
            for action in form_actions:
                action_type = action.get("type", "input")

                if action_type == "input":
                    selector = action["selector"]
                    value = action["value"]
                    element = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    element.clear()
                    element.send_keys(value)

                elif action_type == "click":
                    selector = action["selector"]
                    element = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    element.click()

                elif action_type == "wait":
                    time.sleep(action.get("seconds", 2))

        except Exception as e:
            logger.error("Form action failed", error=str(e))

    def process_scraping_job(self, job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a web scraping job"""
        start_time = time.time()
        ACTIVE_SCRAPER_JOBS.inc()

        try:
            logger.info("Starting web scraping job", job_id=job_id)

            if not SCRAPER_SUPPORT:
                error_msg = "Web scraping not available. Install requests, beautifulsoup4, and selenium packages."
                self.update_job_status(job_id, "failed", error_message=error_msg)
                return {"status": "failed", "error": error_msg}

            # Update job status to processing
            self.update_job_status(job_id, "processing")

            # Validate configuration
            validation_results = self.validate_scraping_config(config)

            if not validation_results["is_valid"]:
                error_msg = "; ".join(validation_results["errors"])
                self.update_job_status(job_id, "failed", error_message=error_msg)
                SCRAPER_REQUESTS_PROCESSED.labels(status="failed").inc()
                return {"status": "failed", "error": error_msg}

            url = config["url"]
            scraping_results = []

            # Determine scraping method
            use_selenium = config.get("use_selenium", False) or config.get("dynamic_content", False)

            if use_selenium:
                result = self.scrape_dynamic_content(url, config)
            else:
                result = self.scrape_static_content(url, config)

            scraping_results.append(result)

            # Process additional URLs if specified
            if "additional_urls" in config:
                for additional_url in config["additional_urls"]:
                    try:
                        if use_selenium:
                            result = self.scrape_dynamic_content(additional_url, config)
                        else:
                            result = self.scrape_static_content(additional_url, config)

                        scraping_results.append(result)

                    except Exception as e:
                        logger.error("Additional URL scraping failed",
                                   url=additional_url,
                                   error=str(e))

            # Store results
            self.store_scraping_results(job_id, validation_results, scraping_results)

            # Process extracted data
            total_records = sum(len(result.get("extracted_data", {})) for result in scraping_results)
            processed_records = self.process_extracted_data(job_id, scraping_results, config)

            # Apply rate limiting if configured
            if "rate_limit" in config:
                time.sleep(config["rate_limit"].get("delay", 1))

            # Update final status
            if processed_records > 0:
                self.update_job_status(
                    job_id,
                    "completed",
                    total_records=total_records,
                    processed_records=processed_records
                )
                SCRAPER_REQUESTS_PROCESSED.labels(status="success").inc()
                status = "completed"
            else:
                self.update_job_status(
                    job_id,
                    "failed",
                    error_message="No data could be extracted"
                )
                SCRAPER_REQUESTS_PROCESSED.labels(status="failed").inc()
                status = "failed"

            processing_time = time.time() - start_time
            SCRAPER_PROCESSING_TIME.labels(operation="full_scraping").observe(processing_time)

            logger.info("Web scraping completed",
                       job_id=job_id,
                       status=status,
                       urls_processed=len(scraping_results),
                       total_records=total_records,
                       processed=processed_records,
                       duration=processing_time)

            return {
                "status": status,
                "urls_processed": len(scraping_results),
                "total_records": total_records,
                "processed_records": processed_records,
                "processing_time": processing_time,
                "results": scraping_results,
                "validation_results": validation_results
            }

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Web scraping failed: {str(e)}"

            self.update_job_status(job_id, "failed", error_message=error_msg)
            SCRAPER_REQUESTS_PROCESSED.labels(status="failed").inc()
            SCRAPER_PROCESSING_TIME.labels(operation="full_scraping").observe(processing_time)

            logger.error("Web scraping failed",
                        job_id=job_id,
                        error=str(e),
                        duration=processing_time)

            return {"status": "failed", "error": error_msg}

        finally:
            ACTIVE_SCRAPER_JOBS.dec()

    def store_scraping_results(self, job_id: str, validation_results: Dict[str, Any],
                              scraping_results: List[Dict[str, Any]]):
        """Store scraping results in database"""
        try:
            with self.db_connection.cursor() as cursor:
                # Store validation results
                cursor.execute("""
                    INSERT INTO data_validation_results
                    (job_id, validation_type, is_valid, error_message, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    job_id,
                    "web_scraping_validation",
                    validation_results["is_valid"],
                    "; ".join(validation_results["errors"]) if validation_results["errors"] else None,
                    json.dumps(validation_results)
                ))

                # Store scraping results
                for result in scraping_results:
                    cursor.execute("""
                        INSERT INTO scraping_results
                        (job_id, url, status, response_time, status_code, extracted_data, metadata, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (
                        job_id,
                        result.get("url"),
                        result.get("status"),
                        result.get("response_time"),
                        result.get("status_code"),
                        json.dumps(result.get("extracted_data", {})),
                        json.dumps({k: v for k, v in result.items()
                                  if k not in ["url", "status", "response_time", "status_code", "extracted_data"]})
                    ))

                self.db_connection.commit()

        except Exception as e:
            logger.error("Failed to store scraping results", error=str(e), job_id=job_id)

    def process_extracted_data(self, job_id: str, scraping_results: List[Dict[str, Any]],
                              config: Dict[str, Any]) -> int:
        """Process extracted scraping data"""
        # This is where you would implement the actual data processing logic
        # For now, just simulate processing
        total_processed = 0

        for result in scraping_results:
            extracted_data = result.get("extracted_data", {})
            for key, value in extracted_data.items():
                time.sleep(0.001)  # Simulate processing time
                total_processed += 1

        return total_processed

    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
            except Exception as e:
                logger.error("Failed to cleanup Selenium driver", error=str(e))

# Global scraper instance
scraper = WebScraper()

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
        rabbitmq_channel.queue_declare(queue='ui_scraper', durable=True)

        # Set up consumer
        rabbitmq_channel.basic_qos(prefetch_count=1)
        rabbitmq_channel.basic_consume(
            queue='ui_scraper',
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

        logger.info("Received UI scraping job", job_id=job_id)

        # Process the scraping job
        result = scraper.process_scraping_job(job_id, source_config)

        logger.info("UI scraping job completed", job_id=job_id, result=result)

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
        "service": "ui-scraper-service",
        "timestamp": datetime.utcnow().isoformat(),
        "active_jobs": ACTIVE_SCRAPER_JOBS._value.get(),
        "scraper_support": SCRAPER_SUPPORT
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(SCRAPER_REGISTRY)

# Startup and shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("UI Scraper Service starting up...")

    # Connect to database
    scraper.connect_db()

    # Setup RabbitMQ consumer in background thread
    import threading
    rabbitmq_thread = threading.Thread(target=setup_rabbitmq, daemon=True)
    rabbitmq_thread.start()

    logger.info("UI Scraper Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("UI Scraper Service shutting down...")

    # Cleanup scraper resources
    scraper.cleanup()

    # Close database connection
    if scraper.db_connection:
        scraper.db_connection.close()

    # Close RabbitMQ connection
    if rabbitmq_connection:
        rabbitmq_connection.close()

    logger.info("UI Scraper Service shutdown complete")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "ui_scraper_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8087)),
        reload=False,
        log_level="info"
    )
