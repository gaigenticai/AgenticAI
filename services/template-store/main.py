#!/usr/bin/env python3
"""
Template Store Service for Agentic Brain Platform

This service manages prebuilt agent templates that enable no-code agent creation
through drag-and-drop workflows. It provides template storage, versioning, instantiation,
and usage tracking capabilities.

Features:
- Template storage and retrieval with versioning
- Template instantiation with parameter substitution
- Template validation and integrity checking
- Template search and filtering by domain/category
- Template usage analytics and popularity tracking
- Template import/export functionality
- RESTful API for template operations
- Comprehensive monitoring and metrics
- Authentication and authorization support
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import hashlib
import zipfile
import tempfile

import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON, Float, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func
import httpx
import uvicorn

# JWT support for authentication
try:
    import jwt
except ImportError:
    jwt = None

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for Template Store Service"""

    # Database Configuration
    DB_HOST = os.getenv('POSTGRES_HOST', 'postgresql_ingestion')
    DB_PORT = os.getenv('POSTGRES_PORT', '5432')
    DB_NAME = os.getenv('POSTGRES_DB', 'agentic_ingestion')
    DB_USER = os.getenv('POSTGRES_USER', 'agentic_user')
    DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'agentic123')
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Redis Configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis_ingestion')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB = 3  # Use DB 3 for template store

    # Service Configuration
    SERVICE_HOST = os.getenv('TEMPLATE_STORE_HOST', '0.0.0.0')
    SERVICE_PORT = int(os.getenv('TEMPLATE_STORE_PORT', '8203'))

    # Authentication Configuration
    REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-jwt-key-change-in-production')
    JWT_ALGORITHM = 'HS256'

    # Template Configuration
    TEMPLATE_CACHE_ENABLED = os.getenv('TEMPLATE_CACHE_ENABLED', 'true').lower() == 'true'
    TEMPLATE_CACHE_TTL_SECONDS = int(os.getenv('TEMPLATE_CACHE_TTL_SECONDS', '3600'))
    TEMPLATE_VERSION_CONTROL_ENABLED = os.getenv('TEMPLATE_VERSION_CONTROL_ENABLED', 'true').lower() == 'true'
    TEMPLATE_AUTO_BACKUP_ENABLED = os.getenv('TEMPLATE_AUTO_BACKUP_ENABLED', 'true').lower() == 'true'
    MAX_TEMPLATE_SIZE_MB = int(os.getenv('MAX_TEMPLATE_SIZE_MB', '10'))

    # File Storage Configuration
    TEMPLATE_STORAGE_PATH = os.getenv('TEMPLATE_STORAGE_PATH', '/app/templates')
    BACKUP_STORAGE_PATH = os.getenv('BACKUP_STORAGE_PATH', '/app/backups')

    # Monitoring Configuration
    METRICS_ENABLED = os.getenv('AGENT_METRICS_ENABLED', 'true').lower() == 'true'

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class TemplateVersion(Base):
    """Database model for template versions"""
    __tablename__ = 'template_versions'

    id = Column(Integer, primary_key=True)
    template_id = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    version_type = Column(String(20), default='minor')  # major, minor, patch
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(255))
    changelog = Column(Text)
    is_active = Column(Boolean, default=True)
    download_count = Column(BigInteger, default=0)
    compatibility_info = Column(JSON, default=dict)

    __table_args__ = (
        {'schema': None}
    )

class TemplateUsage(Base):
    """Database model for template usage tracking"""
    __tablename__ = 'template_usage'

    id = Column(Integer, primary_key=True)
    template_id = Column(String(100), nullable=False)
    user_id = Column(String(255))
    usage_type = Column(String(50), nullable=False)  # view, instantiate, deploy, export
    usage_context = Column(JSON, default=dict)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    session_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class TemplateValidation(Base):
    """Database model for template validation results"""
    __tablename__ = 'template_validation'

    id = Column(Integer, primary_key=True)
    template_id = Column(String(100), nullable=False)
    version = Column(String(50))
    validation_type = Column(String(50), nullable=False)  # syntax, semantic, compatibility
    is_valid = Column(Boolean, default=True)
    validation_errors = Column(JSON, default=list)
    validation_warnings = Column(JSON, default=list)
    validated_by = Column(String(255))
    validated_at = Column(DateTime, default=datetime.utcnow)

class TemplateBackup(Base):
    """Database model for template backups"""
    __tablename__ = 'template_backups'

    id = Column(Integer, primary_key=True)
    template_id = Column(String(100), nullable=False)
    backup_version = Column(String(50), nullable=False)
    backup_path = Column(String(500), nullable=False)
    backup_size_bytes = Column(BigInteger)
    backup_checksum = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

# =============================================================================
# TEMPLATE MODEL
# =============================================================================

class TemplateMetadata(BaseModel):
    """Model for template metadata"""
    template_id: str
    name: str
    domain: str
    description: str
    version: str = "1.0.0"
    author: str
    category: str = "general"
    tags: List[str] = Field(default_factory=list)
    thumbnail_url: Optional[str] = None
    documentation_url: Optional[str] = None
    license: str = "MIT"
    repository_url: Optional[str] = None
    is_public: bool = True
    is_featured: bool = False
    rating: float = 0.0
    total_ratings: int = 0
    usage_count: int = 0
    created_at: datetime
    updated_at: datetime
    compatibility: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[Dict[str, Any]] = Field(default_factory=list)

class TemplateContent(BaseModel):
    """Model for template content"""
    components: List[Dict[str, Any]]
    connections: List[Dict[str, str]]
    configuration: Dict[str, Any]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)
    metadata: TemplateMetadata

class TemplateInstantiationRequest(BaseModel):
    """Model for template instantiation request"""
    template_id: str
    version: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    customizations: Dict[str, Any] = Field(default_factory=dict)
    target_agent_id: Optional[str] = None

class TemplateSearchRequest(BaseModel):
    """Model for template search request"""
    query: Optional[str] = None
    domain: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    is_public: Optional[bool] = None
    is_featured: Optional[bool] = None
    min_rating: Optional[float] = None
    sort_by: str = "usage_count"
    sort_order: str = "desc"
    limit: int = 20
    offset: int = 0

class TemplateRatingRequest(BaseModel):
    """Model for template rating request"""
    template_id: str
    rating: float = Field(..., ge=1.0, le=5.0)
    review: Optional[str] = None

# =============================================================================
# BUSINESS LOGIC CLASSES
# =============================================================================

class TemplateManager:
    """Manages template storage, retrieval, and operations"""

    def __init__(self, db_session: Session, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.storage_path = Path(Config.TEMPLATE_STORAGE_PATH)
        self.backup_path = Path(Config.BACKUP_STORAGE_PATH)

        # Ensure storage directories exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)

        # Load built-in templates
        self._load_builtin_templates()

    def _load_builtin_templates(self):
        """Load built-in templates into the system"""
        builtin_templates = [
            self._create_underwriting_template(),
            self._create_claims_template(),
            self._create_fraud_detection_template(),
            self._create_customer_service_template(),
            self._create_data_analytics_template(),
            self._create_document_processing_template(),
            self._create_workflow_automation_template()
        ]

        for template in builtin_templates:
            try:
                self.store_template(template, "system")
            except Exception as e:
                logger.error(f"Failed to load built-in template {template['template_id']}: {str(e)}")

    def _create_underwriting_template(self) -> Dict[str, Any]:
        """Create the underwriting agent template"""
        return {
            "template_id": "underwriting_template",
            "name": "Underwriting Agent",
            "domain": "underwriting",
            "description": "Comprehensive underwriting agent with risk assessment and decision making",
            "version": "1.0.0",
            "author": "system",
            "category": "business_process",
            "tags": ["underwriting", "risk", "insurance", "decision-making"],
            "is_public": True,
            "is_featured": True,
            "components": [
                {
                    "component_id": "data_input",
                    "component_type": "data_input",
                    "config": {
                        "source_type": "csv",
                        "file_path": "/data/policies.csv",
                        "has_header": True
                    }
                },
                {
                    "component_id": "risk_assess",
                    "component_type": "llm_processor",
                    "config": {
                        "model": "gpt-4",
                        "temperature": 0.6,
                        "prompt_template": "Analyze the following applicant data for risk factors: {applicant_data}"
                    }
                },
                {
                    "component_id": "decision",
                    "component_type": "decision_node",
                    "config": {
                        "condition_field": "risk_score",
                        "operator": "less_than",
                        "threshold": 0.7,
                        "true_branch": "approve_policy",
                        "false_branch": "review_manually"
                    }
                },
                {
                    "component_id": "policy_output",
                    "component_type": "database_output",
                    "config": {
                        "table_name": "policies",
                        "operation": "INSERT"
                    }
                }
            ],
            "connections": [
                {"from": "data_input", "to": "risk_assess"},
                {"from": "risk_assess", "to": "decision"},
                {"from": "decision", "to": "policy_output"}
            ],
            "configuration": {
                "persona": {
                    "role": "Underwriting Analyst",
                    "expertise": ["risk assessment", "compliance"],
                    "personality": "balanced"
                },
                "reasoningPattern": "react",
                "memoryConfig": {
                    "workingMemoryTTL": 3600,
                    "episodic": True,
                    "longTerm": True
                }
            },
            "parameters": {
                "data_source": {
                    "type": "string",
                    "description": "Path to the CSV file containing policy data",
                    "default": "/data/policies.csv"
                },
                "risk_threshold": {
                    "type": "number",
                    "description": "Risk score threshold for automatic approval",
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0
                }
            }
        }

    def _create_claims_template(self) -> Dict[str, Any]:
        """Create the claims processing agent template"""
        return {
            "template_id": "claims_template",
            "name": "Claims Processing Agent",
            "domain": "claims",
            "description": "Intelligent claims processing with fraud detection and settlement calculation",
            "version": "1.0.0",
            "author": "system",
            "category": "business_process",
            "tags": ["claims", "fraud", "insurance", "settlement"],
            "is_public": True,
            "is_featured": True,
            "components": [
                {
                    "component_id": "claim_input",
                    "component_type": "data_input",
                    "config": {
                        "source_type": "api",
                        "endpoint": "/api/claims",
                        "method": "GET"
                    }
                },
                {
                    "component_id": "fraud_detect",
                    "component_type": "plugin_executor",
                    "config": {
                        "plugin_id": "fraudDetector",
                        "parameters": {}
                    }
                },
                {
                    "component_id": "adjust_calc",
                    "component_type": "llm_processor",
                    "config": {
                        "model": "gpt-4",
                        "temperature": 0.7,
                        "prompt_template": "Calculate fair settlement for claim: {claim_data}"
                    }
                },
                {
                    "component_id": "email_notify",
                    "component_type": "email_output",
                    "config": {
                        "template": "claim_notification",
                        "recipient_field": "customer_email"
                    }
                }
            ],
            "connections": [
                {"from": "claim_input", "to": "fraud_detect"},
                {"from": "fraud_detect", "to": "adjust_calc"},
                {"from": "adjust_calc", "to": "email_notify"}
            ],
            "configuration": {
                "persona": {
                    "role": "Claims Specialist",
                    "expertise": ["fraud detection", "settlement"],
                    "personality": "efficient"
                },
                "reasoningPattern": "reflection"
            },
            "parameters": {
                "api_endpoint": {
                    "type": "string",
                    "description": "API endpoint for claims data",
                    "default": "/api/claims"
                },
                "fraud_threshold": {
                    "type": "number",
                    "description": "Fraud score threshold for investigation",
                    "default": 0.8
                }
            }
        }

    def _create_fraud_detection_template(self) -> Dict[str, Any]:
        """Create the fraud detection agent template"""
        return {
            "template_id": "fraud_detection_template",
            "name": "Fraud Detection Agent",
            "domain": "fraud",
            "description": "Advanced fraud detection agent with machine learning and rule-based analysis",
            "version": "1.0.0",
            "author": "system",
            "category": "security",
            "tags": ["fraud", "detection", "ml", "security"],
            "is_public": True,
            "is_featured": False,
            "components": [
                {
                    "component_id": "transaction_input",
                    "component_type": "data_input",
                    "config": {
                        "source_type": "database",
                        "table": "transactions",
                        "query": "SELECT * FROM transactions WHERE created_at >= NOW() - INTERVAL '1 hour'"
                    }
                },
                {
                    "component_id": "pattern_analysis",
                    "component_type": "plugin_executor",
                    "config": {
                        "plugin_id": "fraudDetector"
                    }
                },
                {
                    "component_id": "rule_check",
                    "component_type": "rule_engine",
                    "config": {
                        "rule_set": "fraud_rules"
                    }
                },
                {
                    "component_id": "alert_output",
                    "component_type": "database_output",
                    "config": {
                        "table_name": "fraud_alerts",
                        "operation": "INSERT"
                    }
                }
            ],
            "connections": [
                {"from": "transaction_input", "to": "pattern_analysis"},
                {"from": "pattern_analysis", "to": "rule_check"},
                {"from": "rule_check", "to": "alert_output"}
            ],
            "configuration": {
                "persona": {
                    "role": "Fraud Analyst",
                    "expertise": ["fraud detection", "pattern analysis"],
                    "personality": "vigilant"
                },
                "reasoningPattern": "react"
            }
        }

    def _create_customer_service_template(self) -> Dict[str, Any]:
        """Create the customer service agent template"""
        return {
            "template_id": "customer_service_template",
            "name": "Customer Service Agent",
            "domain": "customer_service",
            "description": "Intelligent customer service agent with query understanding and response generation",
            "version": "1.0.0",
            "author": "system",
            "category": "customer_support",
            "tags": ["customer", "service", "support", "chat"],
            "is_public": True,
            "is_featured": False,
            "components": [
                {
                    "component_id": "query_input",
                    "component_type": "data_input",
                    "config": {
                        "source_type": "api",
                        "endpoint": "/api/customer_queries"
                    }
                },
                {
                    "component_id": "intent_analysis",
                    "component_type": "llm_processor",
                    "config": {
                        "model": "gpt-4",
                        "temperature": 0.3,
                        "prompt_template": "Analyze customer query intent: {query}"
                    }
                },
                {
                    "component_id": "response_gen",
                    "component_type": "llm_processor",
                    "config": {
                        "model": "gpt-4",
                        "temperature": 0.7,
                        "prompt_template": "Generate helpful response for: {analyzed_query}"
                    }
                },
                {
                    "component_id": "response_output",
                    "component_type": "database_output",
                    "config": {
                        "table_name": "customer_responses",
                        "operation": "INSERT"
                    }
                }
            ],
            "connections": [
                {"from": "query_input", "to": "intent_analysis"},
                {"from": "intent_analysis", "to": "response_gen"},
                {"from": "response_gen", "to": "response_output"}
            ],
            "configuration": {
                "persona": {
                    "role": "Customer Service Representative",
                    "expertise": ["customer support", "communication"],
                    "personality": "helpful"
                },
                "reasoningPattern": "react"
            }
        }

    def _create_data_analytics_template(self) -> Dict[str, Any]:
        """Create the data analytics agent template"""
        return {
            "template_id": "data_analytics_template",
            "name": "Data Analytics Agent",
            "domain": "data_analytics",
            "description": "Advanced data analytics agent with visualization and insights generation",
            "version": "1.0.0",
            "author": "system",
            "category": "analytics",
            "tags": ["data", "analytics", "visualization", "insights"],
            "is_public": True,
            "is_featured": True,
            "components": [
                {
                    "component_id": "data_ingest",
                    "component_type": "data_input",
                    "config": {
                        "source_type": "database",
                        "table_name": "analytics_data",
                        "query": "SELECT * FROM analytics_data WHERE date >= CURRENT_DATE - INTERVAL '30 days'"
                    }
                },
                {
                    "component_id": "data_analysis",
                    "component_type": "llm_processor",
                    "config": {
                        "model": "gpt-4",
                        "temperature": 0.4,
                        "prompt_template": "Analyze the following dataset and identify key patterns and insights: {dataset_summary}"
                    }
                },
                {
                    "component_id": "visualization",
                    "component_type": "plugin_executor",
                    "config": {
                        "plugin_id": "dataVisualizer",
                        "parameters": {
                            "chart_types": ["bar", "line", "pie"],
                            "output_format": "png"
                        }
                    }
                },
                {
                    "component_id": "report_output",
                    "component_type": "database_output",
                    "config": {
                        "table_name": "analytics_reports",
                        "operation": "INSERT"
                    }
                }
            ],
            "connections": [
                {"from": "data_ingest", "to": "data_analysis"},
                {"from": "data_analysis", "to": "visualization"},
                {"from": "visualization", "to": "report_output"}
            ],
            "configuration": {
                "persona": {
                    "role": "Data Analyst",
                    "expertise": ["data analysis", "visualization", "insights"],
                    "personality": "analytical"
                },
                "reasoningPattern": "planning"
            },
            "parameters": {
                "data_source": {
                    "type": "string",
                    "description": "Database table or query for data source",
                    "default": "analytics_data"
                },
                "analysis_focus": {
                    "type": "string",
                    "description": "Specific area of analysis focus",
                    "default": "general_trends"
                }
            }
        }

    def _create_document_processing_template(self) -> Dict[str, Any]:
        """Create the document processing agent template"""
        return {
            "template_id": "document_processing_template",
            "name": "Document Processing Agent",
            "domain": "document_processing",
            "description": "Intelligent document processing with OCR, classification, and content extraction",
            "version": "1.0.0",
            "author": "system",
            "category": "content_processing",
            "tags": ["documents", "ocr", "classification", "extraction"],
            "is_public": True,
            "is_featured": True,
            "components": [
                {
                    "component_id": "doc_input",
                    "component_type": "data_input",
                    "config": {
                        "source_type": "file",
                        "file_path": "/documents/incoming/*",
                        "file_types": ["pdf", "docx", "txt", "jpg", "png"]
                    }
                },
                {
                    "component_id": "ocr_processor",
                    "component_type": "plugin_executor",
                    "config": {
                        "plugin_id": "ocrProcessor",
                        "parameters": {
                            "languages": ["en", "es"],
                            "confidence_threshold": 0.8
                        }
                    }
                },
                {
                    "component_id": "doc_classifier",
                    "component_type": "llm_processor",
                    "config": {
                        "model": "gpt-4",
                        "temperature": 0.2,
                        "prompt_template": "Classify this document content and extract key information: {document_text}"
                    }
                },
                {
                    "component_id": "data_output",
                    "component_type": "database_output",
                    "config": {
                        "table_name": "processed_documents",
                        "operation": "INSERT"
                    }
                }
            ],
            "connections": [
                {"from": "doc_input", "to": "ocr_processor"},
                {"from": "ocr_processor", "to": "doc_classifier"},
                {"from": "doc_classifier", "to": "data_output"}
            ],
            "configuration": {
                "persona": {
                    "role": "Document Specialist",
                    "expertise": ["document processing", "content analysis", "data extraction"],
                    "personality": "precise"
                },
                "reasoningPattern": "react"
            },
            "parameters": {
                "input_directory": {
                    "type": "string",
                    "description": "Directory path for input documents",
                    "default": "/documents/incoming/"
                },
                "supported_formats": {
                    "type": "array",
                    "description": "Supported document formats",
                    "default": ["pdf", "docx", "txt"]
                }
            }
        }

    def _create_workflow_automation_template(self) -> Dict[str, Any]:
        """Create the workflow automation agent template"""
        return {
            "template_id": "workflow_automation_template",
            "name": "Workflow Automation Agent",
            "domain": "workflow_automation",
            "description": "Intelligent workflow automation with decision routing and task orchestration",
            "version": "1.0.0",
            "author": "system",
            "category": "automation",
            "tags": ["workflow", "automation", "orchestration", "decision"],
            "is_public": True,
            "is_featured": True,
            "components": [
                {
                    "component_id": "workflow_trigger",
                    "component_type": "data_input",
                    "config": {
                        "source_type": "api",
                        "endpoint": "/api/workflow-triggers",
                        "method": "GET"
                    }
                },
                {
                    "component_id": "condition_check",
                    "component_type": "rule_engine",
                    "config": {
                        "rules": [
                            {
                                "condition": "priority == 'high'",
                                "action": "escalate"
                            },
                            {
                                "condition": "deadline < CURRENT_DATE + INTERVAL '1 day'",
                                "action": "urgent"
                            }
                        ]
                    }
                },
                {
                    "component_id": "task_router",
                    "component_type": "decision_node",
                    "config": {
                        "condition_field": "workflow_type",
                        "branches": {
                            "approval": "approval_workflow",
                            "review": "review_workflow",
                            "processing": "processing_workflow"
                        }
                    }
                },
                {
                    "component_id": "notification",
                    "component_type": "email_output",
                    "config": {
                        "template": "workflow_notification",
                        "recipient_field": "assigned_user"
                    }
                }
            ],
            "connections": [
                {"from": "workflow_trigger", "to": "condition_check"},
                {"from": "condition_check", "to": "task_router"},
                {"from": "task_router", "to": "notification"}
            ],
            "configuration": {
                "persona": {
                    "role": "Workflow Coordinator",
                    "expertise": ["process automation", "task routing", "decision making"],
                    "personality": "systematic"
                },
                "reasoningPattern": "planning"
            },
            "parameters": {
                "workflow_types": {
                    "type": "array",
                    "description": "Supported workflow types",
                    "default": ["approval", "review", "processing"]
                },
                "escalation_rules": {
                    "type": "object",
                    "description": "Rules for workflow escalation",
                    "default": {
                        "high_priority_deadline": "4 hours",
                        "critical_priority_deadline": "1 hour"
                    }
                }
            }
        }

    def store_template(self, template_data: Dict[str, Any], created_by: str) -> str:
        """Store a template in the system"""
        template_id = template_data['template_id']

        # Check if template already exists
        existing = self.db.query(AgentTemplate).filter_by(template_id=template_id).first()
        if existing:
            # Update existing template
            existing.name = template_data['name']
            existing.domain = template_data['domain']
            existing.description = template_data['description']
            existing.template_data = template_data
            existing.updated_at = datetime.utcnow()
        else:
            # Create new template
            template = AgentTemplate(
                template_id=template_id,
                name=template_data['name'],
                domain=template_data['domain'],
                description=template_data['description'],
                template_data=template_data,
                category=template_data.get('category', 'general'),
                tags=template_data.get('tags', []),
                is_public=template_data.get('is_public', True),
                created_by=created_by
            )
            self.db.add(template)

        # Store template file
        template_path = self.storage_path / f"{template_id}.json"
        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2, default=str)

        # Create initial version
        version = TemplateVersion(
            template_id=template_id,
            version=template_data.get('version', '1.0.0'),
            version_type='major',
            created_by=created_by,
            changelog='Initial template creation'
        )
        self.db.add(version)

        self.db.commit()

        # Cache template
        if Config.TEMPLATE_CACHE_ENABLED:
            cache_key = f"template:{template_id}"
            self.redis.setex(cache_key, Config.TEMPLATE_CACHE_TTL_SECONDS, json.dumps(template_data))

        return template_id

    def get_template(self, template_id: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a template by ID"""
        # Check cache first
        if Config.TEMPLATE_CACHE_ENABLED:
            cache_key = f"template:{template_id}"
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        # Get from database
        template = self.db.query(AgentTemplate).filter_by(template_id=template_id).first()
        if not template:
            return None

        # Update usage count
        template.usage_count += 1
        self.db.commit()

        # Cache template
        if Config.TEMPLATE_CACHE_ENABLED:
            cache_key = f"template:{template_id}"
            self.redis.setex(cache_key, Config.TEMPLATE_CACHE_TTL_SECONDS, json.dumps(template.template_data))

        return template.template_data

    def instantiate_template(self, request: TemplateInstantiationRequest) -> Dict[str, Any]:
        """Instantiate a template with custom parameters"""
        template_data = self.get_template(request.template_id, request.version)
        if not template_data:
            raise HTTPException(status_code=404, detail=f"Template {request.template_id} not found")

        # Create a copy of the template
        instantiated = json.loads(json.dumps(template_data))

        # Apply parameter substitution
        parameters = {**template_data.get('parameters', {}), **request.parameters}

        def substitute_parameters(obj, params):
            if isinstance(obj, dict):
                return {k: substitute_parameters(v, params) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_parameters(item, params) for item in obj]
            elif isinstance(obj, str):
                # Simple parameter substitution
                for param_name, param_value in params.items():
                    placeholder = f"{{{param_name}}}"
                    if placeholder in obj:
                        obj = obj.replace(placeholder, str(param_value))
                return obj
            else:
                return obj

        # Apply substitutions
        instantiated = substitute_parameters(instantiated, parameters)

        # Apply customizations
        if request.customizations:
            # Merge customizations into the instantiated template
            def merge_customizations(base, custom):
                if isinstance(base, dict) and isinstance(custom, dict):
                    result = base.copy()
                    for key, value in custom.items():
                        if key in result and isinstance(result[key], (dict, list)):
                            result[key] = merge_customizations(result[key], value)
                        else:
                            result[key] = value
                    return result
                elif isinstance(base, list) and isinstance(custom, list):
                    return base + custom
                else:
                    return custom

            instantiated = merge_customizations(instantiated, request.customizations)

        # Generate unique IDs for the instantiated template
        if request.target_agent_id:
            instantiated['template_id'] = f"{request.target_agent_id}_instance"
            instantiated['name'] = f"{instantiated.get('name', 'Agent')} (Instance)"

        # Track instantiation
        usage = TemplateUsage(
            template_id=request.template_id,
            usage_type='instantiate',
            usage_context={
                'parameters': request.parameters,
                'customizations': request.customizations,
                'target_agent_id': request.target_agent_id
            }
        )
        self.db.add(usage)
        self.db.commit()

        return instantiated

    def search_templates(self, search_request: TemplateSearchRequest) -> Dict[str, Any]:
        """Search templates with filtering and sorting"""
        query = self.db.query(AgentTemplate)

        # Apply filters
        if search_request.query:
            query = query.filter(
                AgentTemplate.name.ilike(f"%{search_request.query}%") |
                AgentTemplate.description.ilike(f"%{search_request.query}%")
            )

        if search_request.domain:
            query = query.filter_by(domain=search_request.domain)

        if search_request.category:
            query = query.filter_by(category=search_request.category)

        if search_request.tags:
            # Filter by tags (array contains)
            for tag in search_request.tags:
                query = query.filter(AgentTemplate.tags.contains([tag]))

        if search_request.author:
            query = query.filter_by(created_by=search_request.author)

        if search_request.is_public is not None:
            query = query.filter_by(is_public=search_request.is_public)

        if search_request.is_featured is not None:
            query = query.filter_by(is_featured=search_request.is_featured)

        if search_request.min_rating:
            query = query.filter(AgentTemplate.rating >= search_request.min_rating)

        # Apply sorting
        sort_column = getattr(AgentTemplate, search_request.sort_by, AgentTemplate.usage_count)
        if search_request.sort_order == 'desc':
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Apply pagination
        total_count = query.count()
        templates = query.offset(search_request.offset).limit(search_request.limit).all()

        # Convert to dict format
        results = []
        for template in templates:
            template_dict = {
                'template_id': template.template_id,
                'name': template.name,
                'domain': template.domain,
                'description': template.description,
                'category': template.category,
                'tags': template.tags,
                'rating': template.rating,
                'usage_count': template.usage_count,
                'created_by': template.created_by,
                'created_at': template.created_at.isoformat(),
                'updated_at': template.updated_at.isoformat()
            }
            results.append(template_dict)

        return {
            'templates': results,
            'total_count': total_count,
            'limit': search_request.limit,
            'offset': search_request.offset
        }

    def validate_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template structure and content"""
        errors = []
        warnings = []

        # Required fields validation
        required_fields = ['template_id', 'name', 'domain', 'components', 'connections']
        for field in required_fields:
            if field not in template_data:
                errors.append(f"Missing required field: {field}")

        # Template ID validation
        if 'template_id' in template_data:
            template_id = template_data['template_id']
            if not template_id.replace('_', '').replace('-', '').isalnum():
                errors.append("Template ID must contain only alphanumeric characters, underscores, and hyphens")

        # Components validation
        if 'components' in template_data:
            components = template_data['components']
            if not isinstance(components, list):
                errors.append("Components must be a list")
            else:
                component_ids = set()
                for i, component in enumerate(components):
                    if not isinstance(component, dict):
                        errors.append(f"Component {i} must be a dictionary")
                        continue

                    if 'component_id' not in component:
                        errors.append(f"Component {i} missing component_id")
                    elif component['component_id'] in component_ids:
                        errors.append(f"Duplicate component_id: {component['component_id']}")
                    else:
                        component_ids.add(component['component_id'])

                    if 'component_type' not in component:
                        errors.append(f"Component {i} missing component_type")

        # Connections validation
        if 'connections' in template_data and 'components' in template_data:
            connections = template_data['connections']
            component_ids = {c['component_id'] for c in template_data['components']}

            if not isinstance(connections, list):
                errors.append("Connections must be a list")
            else:
                for i, connection in enumerate(connections):
                    if not isinstance(connection, dict):
                        errors.append(f"Connection {i} must be a dictionary")
                        continue

                    if 'from' not in connection or 'to' not in connection:
                        errors.append(f"Connection {i} missing 'from' or 'to' field")
                        continue

                    if connection['from'] not in component_ids:
                        errors.append(f"Connection {i} 'from' component not found: {connection['from']}")
                    if connection['to'] not in component_ids:
                        errors.append(f"Connection {i} 'to' component not found: {connection['to']}")

        # Warnings for best practices
        if 'description' not in template_data or len(template_data.get('description', '')) < 10:
            warnings.append("Consider adding a more detailed description")

        if 'tags' not in template_data or len(template_data.get('tags', [])) == 0:
            warnings.append("Consider adding tags for better discoverability")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'validation_type': 'structure'
        }

    def export_template(self, template_id: str, format: str = 'json') -> bytes:
        """Export template in specified format"""
        template_data = self.get_template(template_id)
        if not template_data:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

        if format == 'json':
            return json.dumps(template_data, indent=2, default=str).encode('utf-8')
        elif format == 'zip':
            # Create a zip file with template and metadata
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                with zipfile.ZipFile(temp_file, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add template JSON
                    zip_file.writestr(f"{template_id}.json", json.dumps(template_data, indent=2, default=str))

                    # Add metadata
                    metadata = {
                        'template_id': template_id,
                        'exported_at': datetime.utcnow().isoformat(),
                        'version': template_data.get('version', '1.0.0'),
                        'format_version': '1.0'
                    }
                    zip_file.writestr('metadata.json', json.dumps(metadata, indent=2))

                temp_file.seek(0)
                return temp_file.read()
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")

# =============================================================================
# MONITORING & METRICS
# =============================================================================

class MetricsCollector:
    """Collects and exposes Prometheus metrics"""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Template metrics
        self.total_templates = Gauge('template_store_total_templates', 'Total number of templates', registry=self.registry)
        self.template_usage = Counter('template_store_usage_total', 'Total template usage', ['template_id', 'usage_type'], registry=self.registry)
        self.template_instantiations = Counter('template_store_instantiations_total', 'Total template instantiations', registry=self.registry)
        self.template_search_requests = Counter('template_store_search_requests_total', 'Total search requests', registry=self.registry)
        self.template_validation_requests = Counter('template_store_validation_requests_total', 'Total validation requests', registry=self.registry)

        # Performance metrics
        self.request_count = Counter('template_store_requests_total', 'Total number of requests', ['method', 'endpoint'], registry=self.registry)
        self.request_duration = Histogram('template_store_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'], registry=self.registry)
        self.error_count = Counter('template_store_errors_total', 'Total number of errors', ['type'], registry=self.registry)

    def update_template_metrics(self, template_manager: TemplateManager):
        """Update template-related metrics"""
        try:
            # This would typically query the database for template counts
            # For now, use a simple implementation
            pass
        except Exception as e:
            logger.error(f"Failed to update template metrics: {str(e)}")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Template Store Service",
    description="Manages prebuilt agent templates for no-code agent creation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db_engine = create_engine(Config.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=Config.REDIS_DB, decode_responses=True)
metrics_collector = MetricsCollector()

# Dependency injection
def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_template_manager(db: Session = Depends(get_db)):
    """Template manager dependency"""
    return TemplateManager(db, redis_client)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "template-store",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    if not Config.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    return generate_latest(metrics_collector.registry)

@app.get("/templates")
async def list_templates(
    domain: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    featured_only: bool = False,
    limit: int = 20,
    offset: int = 0,
    template_manager: TemplateManager = Depends(get_template_manager)
):
    """List available templates with filtering"""
    metrics_collector.request_count.labels(method='GET', endpoint='/templates').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/templates').time():
        search_request = TemplateSearchRequest(
            domain=domain,
            category=category,
            tags=tags.split(',') if tags else [],
            is_featured=featured_only if featured_only else None,
            limit=limit,
            offset=offset
        )

        result = template_manager.search_templates(search_request)

    return result

@app.get("/templates/{template_id}")
async def get_template(
    template_id: str,
    version: Optional[str] = None,
    template_manager: TemplateManager = Depends(get_template_manager)
):
    """Get a specific template"""
    metrics_collector.request_count.labels(method='GET', endpoint='/templates/{template_id}').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/templates/{template_id}').time():
        template = template_manager.get_template(template_id, version)

        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

    return template

@app.post("/templates/instantiate")
async def instantiate_template(
    request: TemplateInstantiationRequest,
    template_manager: TemplateManager = Depends(get_template_manager)
):
    """Instantiate a template with custom parameters"""
    metrics_collector.request_count.labels(method='POST', endpoint='/templates/instantiate').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/templates/instantiate').time():
        instantiated_template = template_manager.instantiate_template(request)

    return {
        "instantiated_template": instantiated_template,
        "message": "Template instantiated successfully"
    }

@app.post("/templates/validate")
async def validate_template(
    template_data: Dict[str, Any],
    template_manager: TemplateManager = Depends(get_template_manager)
):
    """Validate template structure"""
    metrics_collector.request_count.labels(method='POST', endpoint='/templates/validate').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/templates/validate').time():
        validation_result = template_manager.validate_template(template_data)

    return validation_result

@app.get("/templates/{template_id}/export")
async def export_template(
    template_id: str,
    format: str = "json",
    template_manager: TemplateManager = Depends(get_template_manager)
):
    """Export template in specified format"""
    metrics_collector.request_count.labels(method='GET', endpoint='/templates/{template_id}/export').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/templates/{template_id}/export').time():
        export_data = template_manager.export_template(template_id, format)

        if format == 'json':
            return JSONResponse(content=json.loads(export_data))
        elif format == 'zip':
            # Return file response for zip
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                temp_file.write(export_data)
                temp_file.flush()

            return FileResponse(
                temp_file.name,
                media_type='application/zip',
                filename=f"{template_id}_template.zip"
            )

@app.get("/templates/categories")
async def get_categories():
    """Get available template categories"""
    categories = [
        {"id": "business_process", "name": "Business Process", "description": "Templates for business process automation"},
        {"id": "customer_support", "name": "Customer Support", "description": "Templates for customer service automation"},
        {"id": "security", "name": "Security", "description": "Templates for security and compliance"},
        {"id": "data_processing", "name": "Data Processing", "description": "Templates for data processing and analytics"},
        {"id": "general", "name": "General", "description": "General-purpose templates"}
    ]

    return {"categories": categories}

@app.get("/templates/domains")
async def get_domains():
    """Get available business domains"""
    domains = [
        {"id": "underwriting", "name": "Underwriting", "description": "Insurance underwriting processes"},
        {"id": "claims", "name": "Claims", "description": "Insurance claims processing"},
        {"id": "fraud", "name": "Fraud Detection", "description": "Fraud detection and prevention"},
        {"id": "customer_service", "name": "Customer Service", "description": "Customer support and service"},
        {"id": "compliance", "name": "Compliance", "description": "Regulatory compliance"},
        {"id": "finance", "name": "Finance", "description": "Financial services and processing"},
        {"id": "healthcare", "name": "Healthcare", "description": "Healthcare and medical services"}
    ]

    return {"domains": domains}

@app.get("/templates/stats")
async def get_template_stats(template_manager: TemplateManager = Depends(get_template_manager)):
    """Get template usage statistics"""
    metrics_collector.request_count.labels(method='GET', endpoint='/templates/stats').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/templates/stats').time():
        # This would typically aggregate statistics from the database
        stats = {
            "total_templates": 4,  # Built-in templates
            "total_usage": 0,
            "popular_templates": [],
            "usage_by_domain": {},
            "usage_by_category": {}
        }

    return stats

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    metrics_collector.error_count.labels(type='validation').inc()

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "message": "Invalid request data"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    metrics_collector.error_count.labels(type='http').inc()

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    metrics_collector.error_count.labels(type='general').inc()

    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": str(uuid.uuid4())
        }
    )

# =============================================================================
# LIFECYCLE MANAGEMENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Starting Template Store Service...")

    # Create database tables
    try:
        Base.metadata.create_all(bind=db_engine)
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise

    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        raise

    logger.info(f"Template Store Service started on {Config.SERVICE_HOST}:{Config.SERVICE_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down Template Store Service...")

    # Close Redis connection
    try:
        redis_client.close()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {str(e)}")

    logger.info("Template Store Service shutdown complete")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.SERVICE_HOST,
        port=Config.SERVICE_PORT,
        reload=False,
        log_level="info",
        access_log=True
    )
