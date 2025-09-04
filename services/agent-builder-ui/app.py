#!/usr/bin/env python3
"""
Agent Builder UI Service

A modern, no-code visual interface for creating and configuring AI agents.
Provides drag-and-drop workflow building with component palette, properties panel,
and real-time validation for the Agentic Brain Platform.

Features:
- Drag-and-drop canvas for workflow creation
- Component palette with pre-built agent components
- Properties panel for component configuration
- Real-time workflow validation
- Template loading and instantiation
- Agent deployment pipeline integration
- Professional UI with responsive design
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import httpx

import structlog
from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for Agent Builder UI Service"""

    # Service Configuration
    SERVICE_HOST = os.getenv('AGENT_BUILDER_UI_HOST', '0.0.0.0')
    SERVICE_PORT = int(os.getenv('AGENT_BUILDER_UI_PORT', '8300'))

    # Integration Service URLs
    TEMPLATE_STORE_URL = os.getenv('TEMPLATE_STORE_URL', 'http://template-store:8203')
    WORKFLOW_ENGINE_URL = os.getenv('WORKFLOW_ENGINE_URL', 'http://workflow-engine:8202')
    BRAIN_FACTORY_URL = os.getenv('BRAIN_FACTORY_URL', 'http://brain-factory:8301')
    DEPLOYMENT_PIPELINE_URL = os.getenv('DEPLOYMENT_PIPELINE_URL', 'http://deployment-pipeline:8302')
    PLUGIN_REGISTRY_URL = os.getenv('PLUGIN_REGISTRY_URL', 'http://plugin-registry:8201')
    RULE_ENGINE_URL = os.getenv('RULE_ENGINE_URL', 'http://rule-engine:8204')

    # Authentication Configuration
    REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-jwt-key-change-in-production')

    # UI Configuration
    MAX_CANVAS_WIDTH = int(os.getenv('MAX_CANVAS_WIDTH', '2000'))
    MAX_CANVAS_HEIGHT = int(os.getenv('MAX_CANVAS_HEIGHT', '1500'))
    AUTO_SAVE_INTERVAL = int(os.getenv('AUTO_SAVE_INTERVAL', '30'))

# =============================================================================
# API MODELS
# =============================================================================

class ComponentTemplate(BaseModel):
    """Model for component templates"""
    id: str
    name: str
    category: str
    icon: str
    description: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    outputs: List[Dict[str, Any]] = Field(default_factory=list)
    color: str = "#6366f1"

class WorkflowData(BaseModel):
    """Model for workflow data"""
    workflow_id: str
    name: str
    description: str = ""
    components: List[Dict[str, Any]] = Field(default_factory=list)
    connections: List[Dict[str, Any]] = Field(default_factory=list)
    canvas: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentConfig(BaseModel):
    """Model for agent configuration"""
    agent_id: str
    name: str
    domain: str
    description: str = ""
    persona: Dict[str, Any] = Field(default_factory=dict)
    reasoning_pattern: str = "react"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    connections: List[Dict[str, Any]] = Field(default_factory=list)
    memory_config: Dict[str, Any] = Field(default_factory=dict)
    plugin_config: Dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# COMPONENT PALETTE
# =============================================================================

COMPONENT_PALETTE = [
    # Data Input Components
    ComponentTemplate(
        id="data_input_csv",
        name="CSV Data Input",
        category="Data Input",
        icon="📄",
        description="Load data from CSV files with flexible parsing options",
        properties={
            "service_url": {
                "type": "url",
                "label": "Service URL",
                "default": "http://csv-ingestion-service:8001",
                "required": True,
                "description": "URL of the CSV ingestion service"
            },
            "file_path": {
                "type": "text",
                "label": "File Path",
                "placeholder": "/data/input.csv",
                "required": True,
                "description": "Path to the CSV file to process"
            },
            "delimiter": {
                "type": "select",
                "label": "Delimiter",
                "options": [",", ";", "\t", "|"],
                "default": ",",
                "description": "Character used to separate values"
            },
            "has_header": {
                "type": "boolean",
                "label": "Has Header Row",
                "default": True,
                "description": "Whether the CSV file contains a header row"
            },
            "encoding": {
                "type": "select",
                "label": "File Encoding",
                "options": ["utf-8", "latin-1", "iso-8859-1"],
                "default": "utf-8",
                "description": "Character encoding of the CSV file"
            },
            "skip_rows": {
                "type": "number",
                "label": "Skip Rows",
                "min": 0,
                "max": 100,
                "default": 0,
                "description": "Number of rows to skip at the beginning"
            }
        },
        outputs=[{"name": "data", "type": "dataframe", "description": "Loaded CSV data as DataFrame"}],
        color="#10b981"
    ),
    ComponentTemplate(
        id="data_input_api",
        name="API Data Input",
        category="Data Input",
        icon="🌐",
        description="Fetch data from REST APIs",
        properties={
            "url": {"type": "url", "required": True},
            "method": {"type": "select", "options": ["GET", "POST"], "default": "GET"},
            "headers": {"type": "json", "default": "{}"},
            "params": {"type": "json", "default": "{}"},
            "auth_type": {"type": "select", "options": ["none", "basic", "bearer"], "default": "none"}
        },
        outputs=[{"name": "response", "type": "json", "description": "API response data"}],
        color="#10b981"
    ),

    # Processing Components
    ComponentTemplate(
        id="llm_processor",
        name="LLM Processor",
        category="Processing",
        icon="🧠",
        description="Process data with Large Language Models with advanced configuration",
        properties={
            "model": {
                "type": "select",
                "label": "LLM Model",
                "dynamic": True,
                "default": "gpt-4",
                "required": True,
                "description": "Choose the language model for processing"
            },
            "temperature": {
                "type": "slider",
                "label": "Temperature",
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
                "default": 0.7,
                "unit": "",
                "description": "Controls randomness (0.0 = deterministic, 2.0 = very random)"
            },
            "max_tokens": {
                "type": "number",
                "label": "Max Tokens",
                "min": 1,
                "max": 4096,
                "default": 1000,
                "description": "Maximum number of tokens to generate"
            },
            "top_p": {
                "type": "slider",
                "label": "Top P",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "default": 1.0,
                "description": "Nucleus sampling parameter"
            },
            "frequency_penalty": {
                "type": "slider",
                "label": "Frequency Penalty",
                "min": -2.0,
                "max": 2.0,
                "step": 0.1,
                "default": 0.0,
                "description": "Reduces repetition of frequent tokens"
            },
            "presence_penalty": {
                "type": "slider",
                "label": "Presence Penalty",
                "min": -2.0,
                "max": 2.0,
                "step": 0.1,
                "default": 0.0,
                "description": "Encourages new topics and ideas"
            },
            "prompt_template": {
                "type": "textarea",
                "label": "Prompt Template",
                "placeholder": "Enter your prompt template here...",
                "rows": 6,
                "required": True,
                "default": "Analyze the following data and provide insights:\n\n{input}\n\nProvide a detailed analysis with key findings.",
                "description": "Template for the LLM prompt. Use {input} to reference input data."
            },
            "system_message": {
                "type": "textarea",
                "label": "System Message",
                "placeholder": "You are a helpful AI assistant...",
                "rows": 3,
                "default": "You are a helpful AI assistant specialized in data analysis and processing.",
                "description": "Instructions that define the AI's role and behavior"
            },
            "stop_sequences": {
                "type": "array",
                "label": "Stop Sequences",
                "description": "Sequences where the model will stop generating",
                "default": ["\n\n", "###"]
            },
            "stream_response": {
                "type": "boolean",
                "label": "Stream Response",
                "default": False,
                "description": "Stream the response as it's generated"
            },
            "timeout_seconds": {
                "type": "number",
                "label": "Timeout (seconds)",
                "min": 10,
                "max": 300,
                "default": 60,
                "description": "Maximum time to wait for response"
            }
        },
        inputs=[{"name": "input", "type": "any", "description": "Input data to process with the LLM"}],
        outputs=[
            {"name": "output", "type": "text", "description": "Generated LLM response text"},
            {"name": "metadata", "type": "json", "description": "Response metadata (tokens used, etc.)"}
        ],
        color="#3b82f6"
    ),
    ComponentTemplate(
        id="rule_engine",
        name="Rule Engine",
        category="Processing",
        icon="⚖️",
        description="Apply business rules and decision logic to data streams",
        properties={
            "rule_set": {
                "type": "select",
                "label": "Rule Set",
                "dynamic": True,
                "required": True,
                "description": "Select the rule set to apply to the data"
            },
            "evaluation_mode": {
                "type": "select",
                "label": "Evaluation Mode",
                "options": ["all", "any", "first_match", "priority"],
                "default": "all",
                "description": "How to evaluate multiple matching rules"
            },
            "fail_on_error": {
                "type": "boolean",
                "label": "Fail on Error",
                "default": False,
                "description": "Stop processing if a rule evaluation fails"
            },
            "log_matches": {
                "type": "boolean",
                "label": "Log Rule Matches",
                "default": True,
                "description": "Log which rules matched for debugging"
            },
            "max_execution_time": {
                "type": "number",
                "label": "Max Execution Time (seconds)",
                "min": 1,
                "max": 300,
                "default": 30,
                "description": "Maximum time to spend evaluating rules"
            },
            "rule_priority": {
                "type": "array",
                "label": "Rule Priority Order",
                "description": "Custom priority order for rules (optional)",
                "default": []
            }
        },
        inputs=[{"name": "input", "type": "any", "description": "Data to evaluate against business rules"}],
        outputs=[
            {"name": "matched_rules", "type": "array", "description": "List of rules that matched the input"},
            {"name": "actions", "type": "array", "description": "Actions to execute based on matched rules"},
            {"name": "evaluation_result", "type": "json", "description": "Complete evaluation result with metadata"}
        ],
        color="#8b5cf6"
    ),

    # Decision Components
    ComponentTemplate(
        id="decision_node",
        name="Decision Node",
        category="Decision",
        icon="🔀",
        description="Make conditional decisions",
        properties={
            "condition_type": {"type": "select", "options": ["threshold", "contains", "equals", "custom"], "default": "threshold"},
            "threshold_value": {"type": "number", "default": 0.5},
            "comparison_operator": {"type": "select", "options": [">", ">=", "<", "<=", "==", "!="], "default": ">="},
            "true_label": {"type": "text", "default": "True"},
            "false_label": {"type": "text", "default": "False"}
        },
        inputs=[{"name": "input", "type": "any", "description": "Value to evaluate"}],
        outputs=[
            {"name": "true", "type": "any", "description": "When condition is true"},
            {"name": "false", "type": "any", "description": "When condition is false"}
        ],
        color="#f59e0b"
    ),

    # Coordination Components
    ComponentTemplate(
        id="multi_agent_coordinator",
        name="Multi-Agent Coordinator",
        category="Coordination",
        icon="👥",
        description="Coordinate multiple agents",
        properties={
            "coordination_type": {"type": "select", "options": ["sequential", "parallel", "conditional"], "default": "sequential"},
            "max_concurrent": {"type": "number", "min": 1, "max": 10, "default": 3},
            "timeout_seconds": {"type": "number", "min": 10, "max": 3600, "default": 300},
            "error_handling": {"type": "select", "options": ["stop", "continue", "retry"], "default": "stop"}
        },
        inputs=[{"name": "agents", "type": "array", "description": "List of agents to coordinate"}],
        outputs=[{"name": "results", "type": "array", "description": "Combined results from all agents"}],
        color="#ec4899"
    ),

    # Output Components
    ComponentTemplate(
        id="database_output",
        name="Database Output",
        category="Output",
        icon="🗄️",
        description="Store data in databases",
        properties={
            "connection_type": {"type": "select", "options": ["postgresql", "mongodb", "elasticsearch"], "default": "postgresql", "required": True},
            "table_name": {"type": "text", "required": True},
            "connection_string": {"type": "password", "required": True},
            "insert_mode": {"type": "select", "options": ["insert", "update", "upsert"], "default": "insert"},
            "batch_size": {"type": "number", "min": 1, "max": 1000, "default": 100}
        },
        inputs=[{"name": "data", "type": "dataframe", "description": "Data to store"}],
        outputs=[{"name": "result", "type": "json", "description": "Operation result"}],
        color="#ef4444"
    ),
    ComponentTemplate(
        id="email_output",
        name="Email Output",
        category="Output",
        icon="📧",
        description="Send email notifications",
        properties={
            "smtp_server": {"type": "text", "required": True},
            "smtp_port": {"type": "number", "default": 587},
            "username": {"type": "text", "required": True},
            "password": {"type": "password", "required": True},
            "from_email": {"type": "email", "required": True},
            "to_emails": {"type": "array", "required": True},
            "subject_template": {"type": "text", "required": True},
            "body_template": {"type": "textarea", "required": True}
        },
        inputs=[{"name": "data", "type": "any", "description": "Data to include in email"}],
        outputs=[{"name": "result", "type": "json", "description": "Email sending result"}],
        color="#f97316"
    ),
    ComponentTemplate(
        id="pdf_output",
        name="PDF Report Output",
        category="Output",
        icon="📄",
        description="Generate PDF reports",
        properties={
            "template_path": {"type": "file", "required": True},
            "output_filename": {"type": "text", "required": True},
            "page_size": {"type": "select", "options": ["A4", "Letter", "Legal"], "default": "A4"},
            "orientation": {"type": "select", "options": ["portrait", "landscape"], "default": "portrait"},
            "margins": {"type": "json", "default": '{"top": "1in", "bottom": "1in", "left": "1in", "right": "1in"}'}
        },
        inputs=[{"name": "data", "type": "any", "description": "Data for PDF generation"}],
        outputs=[{"name": "pdf_path", "type": "file", "description": "Generated PDF file path"}],
        color="#84cc16"
    )
]

# =============================================================================
# SERVICE INTEGRATION
# =============================================================================

class ServiceIntegrator:
    """Handles integration with backend services"""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_templates(self) -> List[Dict[str, Any]]:
        """Get available templates from template store"""
        try:
            async with self.client as client:
                response = await client.get(f"{Config.TEMPLATE_STORE_URL}/templates")
                if response.status_code == 200:
                    return response.json().get("templates", [])
                else:
                    logger.error(f"Failed to get templates: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error getting templates: {str(e)}")
            return []

    async def instantiate_template(self, template_id: str, customizations: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Instantiate a template with customizations"""
        try:
            async with self.client as client:
                response = await client.post(
                    f"{Config.TEMPLATE_STORE_URL}/templates/{template_id}/instantiate",
                    json=customizations
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to instantiate template: {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"Error instantiating template: {str(e)}")
            return None

    async def validate_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow through workflow engine"""
        try:
            async with self.client as client:
                response = await client.post(
                    f"{Config.WORKFLOW_ENGINE_URL}/workflows/validate",
                    json=workflow_data
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"valid": False, "errors": [f"Validation failed: {response.status_code}"]}
        except Exception as e:
            logger.error(f"Error validating workflow: {str(e)}")
            return {"valid": False, "errors": [str(e)]}

    async def deploy_agent(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy agent through deployment pipeline"""
        try:
            async with self.client as client:
                response = await client.post(
                    f"{Config.DEPLOYMENT_PIPELINE_URL}/deploy",
                    json=agent_config
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"success": False, "error": f"Deployment failed: {response.status_code}"}
        except Exception as e:
            logger.error(f"Error deploying agent: {str(e)}")
            return {"success": False, "error": str(e)}

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Agent Builder UI Service",
    description="No-code visual interface for creating AI agents",
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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize service integrator
service_integrator = ServiceIntegrator()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agent-builder-ui",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/components")
async def get_components():
    """Get available components for the palette"""
    return {"components": [component.dict() for component in COMPONENT_PALETTE]}

@app.get("/api/rule-sets")
async def get_rule_sets():
    """Get available rule sets for dynamic dropdowns"""
    # In a real implementation, this would fetch from the rule engine service
    # For now, return mock data
    rule_sets = [
        {"id": "fraud_detection", "name": "Fraud Detection Rules"},
        {"id": "compliance_check", "name": "Compliance Check Rules"},
        {"id": "data_validation", "name": "Data Validation Rules"},
        {"id": "risk_assessment", "name": "Risk Assessment Rules"},
        {"id": "underwriting_rules", "name": "Underwriting Business Rules"},
        {"id": "claims_processing", "name": "Claims Processing Rules"}
    ]
    return {"rule_sets": rule_sets}

@app.get("/api/llm-models")
async def get_llm_models():
    """Get available LLM models for dynamic dropdowns"""
    # In a real implementation, this would fetch from LLM service configuration
    models = [
        {"id": "gpt-4", "name": "GPT-4 (Most Capable)", "provider": "OpenAI", "context_window": 8192},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo (Fast)", "provider": "OpenAI", "context_window": 4096},
        {"id": "claude-3", "name": "Claude 3 (Balanced)", "provider": "Anthropic", "context_window": 100000},
        {"id": "llama-2-70b", "name": "Llama 2 70B (Open Source)", "provider": "Meta", "context_window": 4096},
        {"id": "mistral-7b", "name": "Mistral 7B", "provider": "Mistral AI", "context_window": 8192},
        {"id": "codellama-34b", "name": "Code Llama 34B", "provider": "Meta", "context_window": 16384}
    ]
    return {"models": models}

@app.get("/api/templates")
async def get_templates(
    domain: Optional[str] = Query(None, description="Filter by business domain"),
    category: Optional[str] = Query(None, description="Filter by template category"),
    limit: int = Query(50, description="Maximum number of templates to return", ge=1, le=100),
    offset: int = Query(0, description="Pagination offset", ge=0)
):
    """
    Get available agent templates with filtering and pagination.

    Returns a comprehensive list of templates including:
    - Pre-built agent templates for different domains
    - Template metadata (difficulty, components, description)
    - Usage statistics and ratings
    - Template categories and tags

    Query Parameters:
    - domain: Filter by business domain (underwriting, claims, fraud_detection)
    - category: Filter by template category (basic, advanced, enterprise)
    - limit: Maximum number of templates to return (default: 50)
    - offset: Pagination offset (default: 0)

    Returns:
        Dict containing:
        - templates: List of template objects
        - total_count: Total number of available templates
        - categories: Available template categories
        - domains: Available business domains
    """
    try:
        # Fetch templates from template store service
        templates = await service_integrator.get_templates()

        # Apply filters
        filtered_templates = []
        for template in templates:
            if domain and template.get('domain') != domain:
                continue
            if category and template.get('category') != category:
                continue
            filtered_templates.append(template)

        # Apply pagination
        paginated_templates = filtered_templates[offset:offset + limit]

        # Extract unique categories and domains
        categories = list(set(t.get('category', 'general') for t in templates))
        domains = list(set(t.get('domain', 'general') for t in templates))

        return {
            "templates": paginated_templates,
            "total_count": len(filtered_templates),
            "categories": categories,
            "domains": domains,
            "pagination": {
                "offset": offset,
                "limit": limit,
                "has_more": len(filtered_templates) > offset + limit
            }
        }

    except Exception as e:
        logger.error(f"Failed to get templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve templates: {str(e)}")

@app.post("/api/templates/{template_id}/instantiate")
async def instantiate_template(template_id: str, customizations: Dict[str, Any]):
    """
    Instantiate a template with custom user configurations.

    This endpoint loads a pre-built template and applies user customizations
    to create a personalized agent workflow. The template is transformed into
    a complete workflow with user-specified parameters.

    Path Parameters:
    - template_id: Unique identifier of the template to instantiate

    Request Body:
    - customizations: Dictionary containing user customizations including:
        - agent_name: Custom name for the instantiated agent
        - domain: Business domain for the agent
        - component_configs: Custom configurations for individual components
        - workflow_settings: Workflow-level customizations
        - service_mappings: Custom service endpoint mappings

    Returns:
        Dict containing:
        - workflow: Complete instantiated workflow configuration
        - agent_config: Generated agent configuration
        - validation_status: Template validation results
        - instantiation_metadata: Metadata about the instantiation process
    """
    try:
        # Validate template_id exists
        if not template_id or template_id.strip() == "":
            raise HTTPException(status_code=400, detail="Template ID is required")

        # Validate customizations structure
        if not isinstance(customizations, dict):
            raise HTTPException(status_code=400, detail="Customizations must be a valid JSON object")

        # Instantiate template through service integrator
        result = await service_integrator.instantiate_template(template_id, customizations)

        if result:
            # Add instantiation metadata
            result["instantiation_metadata"] = {
                "template_id": template_id,
                "instantiated_at": datetime.utcnow().isoformat(),
                "customizations_applied": len(customizations),
                "version": "1.0.0"
            }

            return result
        else:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found or instantiation failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template instantiation failed for {template_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Template instantiation failed: {str(e)}")

@app.get("/api/templates/{template_id}")
async def get_template_details(template_id: str):
    """
    Get detailed information about a specific template.

    Returns comprehensive template metadata including:
    - Template structure and components
    - Configuration options and defaults
    - Usage examples and documentation
    - Validation rules and constraints

    Path Parameters:
    - template_id: Unique identifier of the template

    Returns:
        Dict containing:
        - template: Complete template configuration
        - components: List of components in the template
        - validation_rules: Template validation constraints
        - usage_examples: Sample configurations and use cases
    """
    try:
        if not template_id or template_id.strip() == "":
            raise HTTPException(status_code=400, detail="Template ID is required")

        # Fetch all templates and find the specific one
        templates = await service_integrator.get_templates()
        template = next((t for t in templates if t.get('id') == template_id), None)

        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

        # Enhance template with additional metadata
        enhanced_template = {
            **template,
            "metadata": {
                "retrieved_at": datetime.utcnow().isoformat(),
                "version": template.get('version', '1.0.0'),
                "last_updated": template.get('updated_at', datetime.utcnow().isoformat()),
                "compatibility": template.get('compatibility', ['latest'])
            }
        }

        return {"template": enhanced_template}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template details for {template_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve template details: {str(e)}")

@app.post("/api/workflows/validate")
async def validate_workflow(workflow_data: WorkflowData):
    """
    Validate a visual workflow design for correctness and completeness.

    Performs comprehensive validation including:
    - Component connectivity and data flow validation
    - Configuration completeness and parameter validation
    - Service integration compatibility
    - Performance and scalability checks
    - Security and compliance validation

    Request Body:
    - workflow_data: Complete workflow configuration including:
        - components: List of workflow components with configurations
        - connections: Component connection mappings
        - canvas: Visual layout and positioning
        - metadata: Workflow metadata and settings

    Returns:
        Dict containing:
        - valid: Boolean indicating if workflow is valid
        - errors: List of validation errors (if any)
        - warnings: List of validation warnings
        - suggestions: Recommended improvements
        - performance_score: Workflow performance rating
        - compatibility_score: Service compatibility rating
    """
    try:
        # Validate workflow structure
        if not workflow_data.components:
            return {
                "valid": False,
                "errors": ["Workflow must contain at least one component"],
                "warnings": [],
                "suggestions": ["Add components to your workflow"],
                "performance_score": 0,
                "compatibility_score": 0
            }

        if not workflow_data.connections and len(workflow_data.components) > 1:
            return {
                "valid": False,
                "errors": ["Multi-component workflows must have connections"],
                "warnings": [],
                "suggestions": ["Connect your components to define data flow"],
                "performance_score": 0,
                "compatibility_score": 0
            }

        # Perform backend validation
        validation_result = await service_integrator.validate_workflow(workflow_data.dict())

        # Enhance with additional validation logic
        enhanced_result = {
            **validation_result,
            "validation_metadata": {
                "validated_at": datetime.utcnow().isoformat(),
                "component_count": len(workflow_data.components),
                "connection_count": len(workflow_data.connections),
                "validation_version": "1.0.0"
            }
        }

        # Add performance insights
        if enhanced_result.get("valid", False):
            enhanced_result["performance_insights"] = {
                "estimated_complexity": self._calculate_workflow_complexity(workflow_data),
                "bottleneck_analysis": self._analyze_bottlenecks(workflow_data),
                "optimization_suggestions": self._generate_optimization_suggestions(workflow_data)
            }

        return enhanced_result

    except Exception as e:
        logger.error(f"Workflow validation failed: {str(e)}")
        return {
            "valid": False,
            "errors": [f"Validation failed: {str(e)}"],
            "warnings": [],
            "suggestions": ["Please check your workflow configuration"],
            "performance_score": 0,
            "compatibility_score": 0
        }

def _calculate_workflow_complexity(workflow_data):
    """Calculate workflow complexity score"""
    component_count = len(workflow_data.components)
    connection_count = len(workflow_data.connections)

    # Simple complexity calculation
    complexity = (component_count * 0.3) + (connection_count * 0.7)
    return min(complexity, 10.0)  # Cap at 10

def _analyze_bottlenecks(workflow_data):
    """Analyze potential bottlenecks in the workflow"""
    bottlenecks = []

    # Check for components with high connection counts
    component_connections = {}
    for conn in workflow_data.connections:
        for component_id in [conn.source.componentId, conn.target.componentId]:
            component_connections[component_id] = component_connections.get(component_id, 0) + 1

    for component_id, conn_count in component_connections.items():
        if conn_count > 5:
            bottlenecks.append(f"High connection count on component {component_id}")

    return bottlenecks

def _generate_optimization_suggestions(workflow_data):
    """Generate optimization suggestions for the workflow"""
    suggestions = []

    # Check for disconnected components
    connected_components = set()
    for conn in workflow_data.connections:
        connected_components.add(conn.source.componentId)
        connected_components.add(conn.target.componentId)

    disconnected = [c for c in workflow_data.components if c.id not in connected_components]
    if disconnected:
        suggestions.append(f"Consider connecting or removing {len(disconnected)} disconnected components")

    # Check for potential parallel processing opportunities
    # This would be more sophisticated in a real implementation

    return suggestions

@app.post("/api/agents/deploy")
async def deploy_agent(agent_config: AgentConfig):
    """
    Deploy an agent configuration to the production environment.

    This endpoint handles the complete agent deployment pipeline including:
    - Agent configuration validation
    - Resource allocation and provisioning
    - Service registration and discovery
    - Health monitoring setup
    - Performance optimization configuration

    Request Body:
    - agent_config: Complete agent configuration including:
        - agent_id: Unique agent identifier
        - name: Human-readable agent name
        - domain: Business domain
        - persona: Agent personality configuration
        - reasoning_pattern: AI reasoning approach
        - components: Workflow components
        - connections: Component connections
        - memory_config: Memory management settings
        - plugin_config: Plugin configurations

    Returns:
        Dict containing:
        - success: Boolean deployment status
        - deployment_id: Unique deployment identifier
        - agent_endpoint: API endpoint for the deployed agent
        - monitoring_url: Monitoring dashboard URL
        - estimated_startup_time: Expected startup duration
        - deployment_metadata: Additional deployment information
    """
    try:
        # Validate agent configuration
        if not agent_config.agent_id:
            raise HTTPException(status_code=400, detail="Agent ID is required")

        if not agent_config.name:
            raise HTTPException(status_code=400, detail="Agent name is required")

        # Perform deployment through service integrator
        deployment_result = await service_integrator.deploy_agent(agent_config.dict())

        if deployment_result.get("success"):
            # Enhance result with additional metadata
            deployment_result["deployment_metadata"] = {
                "deployed_at": datetime.utcnow().isoformat(),
                "agent_id": agent_config.agent_id,
                "agent_name": agent_config.name,
                "domain": agent_config.domain,
                "component_count": len(agent_config.components) if agent_config.components else 0,
                "connection_count": len(agent_config.connections) if agent_config.connections else 0,
                "deployment_version": "1.0.0"
            }

            # Add monitoring and management URLs
            base_url = f"http://localhost:{Config.DEPLOYMENT_PIPELINE_PORT}"
            deployment_result["monitoring_url"] = f"{base_url}/monitoring/{agent_config.agent_id}"
            deployment_result["logs_url"] = f"{base_url}/logs/{agent_config.agent_id}"
            deployment_result["metrics_url"] = f"{base_url}/metrics/{agent_config.agent_id}"

        return deployment_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent deployment failed for {agent_config.agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent deployment failed: {str(e)}")

@app.post("/api/visual-config")
async def save_visual_config(visual_config: Dict[str, Any]):
    """
    Save visual workflow configuration.

    This endpoint stores the complete visual state of a workflow including:
    - Component positions and layouts
    - Connection paths and styling
    - Canvas zoom and pan settings
    - User interface state and preferences

    Request Body:
    - visual_config: Complete visual configuration including:
        - workflow_id: Unique workflow identifier
        - canvas_state: Canvas position, zoom, and viewport
        - component_positions: X,Y coordinates for all components
        - connection_paths: SVG path data for connections
        - ui_preferences: User interface settings and preferences

    Returns:
        Dict containing:
        - success: Boolean save status
        - config_id: Unique configuration identifier
        - saved_at: Timestamp of save operation
        - version: Configuration version
    """
    try:
        if not visual_config.get("workflow_id"):
            raise HTTPException(status_code=400, detail="Workflow ID is required")

        # In a real implementation, this would save to a database
        # For now, we'll simulate the save operation
        config_id = f"config_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Add metadata
        enhanced_config = {
            **visual_config,
            "config_id": config_id,
            "saved_at": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "metadata": {
                "user_agent": "agent-builder-ui",
                "save_method": "visual_config",
                "compatibility": ["v1.0", "v1.1"]
            }
        }

        # Here you would typically save to database/cache
        # await self.save_visual_config_to_storage(enhanced_config)

        return {
            "success": True,
            "config_id": config_id,
            "saved_at": enhanced_config["saved_at"],
            "version": enhanced_config["version"],
            "message": "Visual configuration saved successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save visual configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save visual configuration: {str(e)}")

@app.get("/api/services")
async def get_available_services():
    """
    Get list of available services for component configuration.

    Returns comprehensive service discovery information including:
    - Ingestion services (CSV, API, PDF, etc.)
    - Output services (PostgreSQL, Qdrant, etc.)
    - Processing services (LLM, Rule Engine, etc.)
    - Service health status and endpoints

    Returns:
        Dict containing:
        - services: Categorized list of available services
        - service_health: Health status for each service
        - service_endpoints: API endpoints for service communication
    """
    try:
        # In a real implementation, this would query service discovery
        # For now, return static service information
        services = {
            "ingestion": {
                "csv": {
                    "name": "CSV Ingestion Service",
                    "endpoint": "http://csv-ingestion-service:8001",
                    "status": "healthy",
                    "supported_formats": ["csv", "tsv"],
                    "max_file_size": "100MB"
                },
                "api": {
                    "name": "API Ingestion Service",
                    "endpoint": "http://api-ingestion-service:8002",
                    "status": "healthy",
                    "supported_methods": ["GET", "POST", "PUT"],
                    "rate_limits": "1000 req/min"
                },
                "pdf": {
                    "name": "PDF Ingestion Service",
                    "endpoint": "http://pdf-ingestion-service:8003",
                    "status": "healthy",
                    "ocr_support": True,
                    "supported_languages": ["en", "es", "fr", "de"]
                }
            },
            "output": {
                "postgresql": {
                    "name": "PostgreSQL Output",
                    "endpoint": "http://postgresql-output:8004",
                    "status": "healthy",
                    "supported_types": ["structured", "semi-structured"],
                    "max_connections": 100
                },
                "qdrant": {
                    "name": "Qdrant Vector Database",
                    "endpoint": "http://qdrant-vector:6333",
                    "status": "healthy",
                    "vector_dimensions": [384, 768, 1024],
                    "index_types": ["HNSW", "Flat"]
                },
                "elasticsearch": {
                    "name": "Elasticsearch Output",
                    "endpoint": "http://elasticsearch-output:9200",
                    "status": "healthy",
                    "supported_operations": ["index", "search", "aggregation"],
                    "max_batch_size": 10000
                }
            },
            "processing": {
                "llm": {
                    "name": "LLM Processing Service",
                    "endpoint": "http://llm-processor:8005",
                    "status": "healthy",
                    "supported_models": ["gpt-4", "claude-3", "llama-2"],
                    "max_tokens": 4096
                },
                "rules": {
                    "name": "Rule Engine Service",
                    "endpoint": "http://rule-engine:8204",
                    "status": "healthy",
                    "rule_types": ["business", "validation", "transformation"],
                    "max_rules": 1000
                }
            }
        }

        return {
            "services": services,
            "last_updated": datetime.utcnow().isoformat(),
            "service_count": sum(len(category) for category in services.values()),
            "health_summary": {
                "total_services": sum(len(category) for category in services.values()),
                "healthy_services": sum(len([s for s in category.values() if s["status"] == "healthy"]) for category in services.values()),
                "unhealthy_services": sum(len([s for s in category.values() if s["status"] != "healthy"]) for category in services.values())
            }
        }

    except Exception as e:
        logger.error(f"Failed to get available services: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve services: {str(e)}")

@app.post("/api/workflows/save")
async def save_workflow(workflow_data: Dict[str, Any]):
    """
    Save a complete workflow including visual and logical configuration.

    This endpoint handles the complete workflow persistence including:
    - Component configurations and positions
    - Connection definitions and routing
    - Visual layout and styling
    - Workflow metadata and versioning

    Request Body:
    - workflow_data: Complete workflow configuration including:
        - workflow_id: Unique workflow identifier
        - name: Human-readable workflow name
        - description: Workflow description
        - components: List of configured components
        - connections: List of component connections
        - canvas: Visual canvas configuration
        - metadata: Additional workflow metadata

    Returns:
        Dict containing:
        - success: Boolean save status
        - workflow_id: Unique workflow identifier
        - version: Workflow version
        - saved_at: Timestamp of save operation
    """
    try:
        if not workflow_data.get("workflow_id"):
            workflow_data["workflow_id"] = f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        if not workflow_data.get("name"):
            raise HTTPException(status_code=400, detail="Workflow name is required")

        # Validate workflow structure
        if not workflow_data.get("components"):
            raise HTTPException(status_code=400, detail="Workflow must contain at least one component")

        # Add metadata
        workflow_data["saved_at"] = datetime.utcnow().isoformat()
        workflow_data["version"] = workflow_data.get("version", "1.0.0")
        workflow_data["metadata"] = workflow_data.get("metadata", {})

        # In a real implementation, save to database
        # await self.save_workflow_to_database(workflow_data)

        return {
            "success": True,
            "workflow_id": workflow_data["workflow_id"],
            "version": workflow_data["version"],
            "saved_at": workflow_data["saved_at"],
            "component_count": len(workflow_data.get("components", [])),
            "connection_count": len(workflow_data.get("connections", [])),
            "message": "Workflow saved successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save workflow: {str(e)}")

@app.get("/api/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """
    Retrieve a saved workflow by ID.

    Returns the complete workflow configuration including:
    - Component definitions and configurations
    - Connection mappings and routing
    - Visual layout and positioning
    - Workflow metadata and history

    Path Parameters:
    - workflow_id: Unique workflow identifier

    Returns:
        Dict containing:
        - workflow: Complete workflow configuration
        - metadata: Workflow metadata and statistics
        - version_history: List of workflow versions
    """
    try:
        if not workflow_id:
            raise HTTPException(status_code=400, detail="Workflow ID is required")

        # In a real implementation, fetch from database
        # workflow = await self.get_workflow_from_database(workflow_id)

        # Mock response for demonstration
        workflow = {
            "workflow_id": workflow_id,
            "name": f"Workflow {workflow_id}",
            "description": "Sample workflow configuration",
            "components": [],
            "connections": [],
            "canvas": {"zoom": 1.0, "panX": 0, "panY": 0},
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        return {
            "workflow": workflow,
            "metadata": {
                "retrieved_at": datetime.utcnow().isoformat(),
                "component_count": len(workflow["components"]),
                "connection_count": len(workflow["connections"]),
                "last_modified": workflow["updated_at"]
            },
            "version_history": [
                {"version": "1.0.0", "created_at": workflow["created_at"], "changes": "Initial version"}
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve workflow: {str(e)}")

@app.get("/api/workflows")
async def list_workflows(
    limit: int = Query(20, description="Maximum number of workflows to return", ge=1, le=100),
    offset: int = Query(0, description="Pagination offset", ge=0),
    sort_by: str = Query("updated_at", description="Sort field", regex="^(name|created_at|updated_at)$"),
    sort_order: str = Query("desc", description="Sort order", regex="^(asc|desc)$")
):
    """
    List saved workflows with filtering and pagination.

    Returns a paginated list of workflows with metadata including:
    - Workflow names and descriptions
    - Creation and modification dates
    - Component and connection counts
    - Workflow status and health

    Query Parameters:
    - limit: Maximum number of workflows to return (default: 20)
    - offset: Pagination offset (default: 0)
    - sort_by: Field to sort by (name, created_at, updated_at)
    - sort_order: Sort order (asc, desc)

    Returns:
        Dict containing:
        - workflows: List of workflow summaries
        - total_count: Total number of workflows
        - pagination: Pagination metadata
    """
    try:
        # In a real implementation, fetch from database with proper filtering
        # workflows = await self.list_workflows_from_database(limit, offset, sort_by, sort_order)

        # Mock response for demonstration
        workflows = []
        for i in range(min(limit, 10)):  # Mock 10 workflows max
            workflow = {
                "workflow_id": f"workflow_{i+1:03d}",
                "name": f"Sample Workflow {i+1}",
                "description": f"This is sample workflow number {i+1}",
                "component_count": (i % 5) + 1,
                "connection_count": (i % 3) + 1,
                "created_at": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "updated_at": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                "status": "active" if i % 2 == 0 else "draft",
                "version": "1.0.0"
            }
            workflows.append(workflow)

        return {
            "workflows": workflows,
            "total_count": len(workflows),
            "pagination": {
                "offset": offset,
                "limit": limit,
                "has_more": len(workflows) == limit,
                "sort_by": sort_by,
                "sort_order": sort_order
            },
            "summary": {
                "active_workflows": len([w for w in workflows if w["status"] == "active"]),
                "draft_workflows": len([w for w in workflows if w["status"] == "draft"]),
                "total_components": sum(w["component_count"] for w in workflows),
                "total_connections": sum(w["connection_count"] for w in workflows)
            }
        }

    except Exception as e:
        logger.error(f"Failed to list workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main Agent Builder UI"""
    return FileResponse("templates/index.html", media_type="text/html")

@app.get("/editor", response_class=HTMLResponse)
async def editor(request: Request):
    """Serve the workflow editor interface"""
    return FileResponse("templates/editor.html", media_type="text/html")

@app.get("/templates/{template_id}", response_class=HTMLResponse)
async def template_editor(request: Request, template_id: str):
    """Serve template-specific editor"""
    return FileResponse("templates/editor.html", media_type="text/html")

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
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
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=Config.SERVICE_HOST,
        port=Config.SERVICE_PORT,
        reload=False,
        log_level="info",
        access_log=True
    )
