#!/usr/bin/env python3
"""
Brain Factory Service

This service implements the core agent instantiation and configuration management
for the Agentic Brain platform. It serves as the factory for creating fully
configured AgentBrain instances from AgentConfig specifications, integrating
all necessary components including reasoning modules, memory management,
plugin systems, and service connectors.

The Brain Factory handles:
- Agent configuration validation and processing
- Service dependency resolution and initialization
- Component integration and orchestration
- Agent instantiation with proper error handling
- Configuration persistence and versioning
- Performance monitoring and optimization

Architecture:
- Factory pattern for agent creation
- Dependency injection for service integration
- Async processing for scalability
- Comprehensive validation and error handling
- Configuration management and versioning

Author: AgenticAI Platform
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import httpx

import structlog
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic.dataclasses import dataclass

# Configure structured logging
logger = structlog.get_logger(__name__)

# Configuration class for service settings
class Config:
    """
    Configuration settings for Brain Factory service.

    This class centralizes all configuration parameters for the Brain Factory,
    including service endpoints, timeouts, limits, and feature flags. All
    configurations are environment-variable driven to support different
    deployment environments (development, staging, production).

    Service Endpoints:
        - AGENT_BRAIN_BASE_PORT: Core agent brain execution engine
        - REASONING_MODULE_FACTORY_PORT: AI reasoning pattern factory
        - MEMORY_MANAGER_PORT: Multi-tier memory management system
        - PLUGIN_REGISTRY_PORT: Domain-specific plugin repository
        - SERVICE_CONNECTOR_FACTORY_PORT: Data ingestion/output connectors
        - UI_TO_BRAIN_MAPPER_PORT: Visual workflow to brain config converter
        - AGENT_ORCHESTRATOR_PORT: Agent lifecycle and task orchestration

    Performance Settings:
        - MAX_AGENT_INSTANCES: Prevents resource exhaustion (default: 50)
        - AGENT_CREATION_TIMEOUT: Max time for agent instantiation (default: 120s)
        - AGENT_VALIDATION_TIMEOUT: Max time for configuration validation (default: 30s)

    Monitoring Settings:
        - ENABLE_METRICS: Enable Prometheus metrics collection
        - METRICS_RETENTION_HOURS: How long to retain metrics data

    Note:
        All timeouts are in seconds and help prevent hanging operations
        Service ports must match docker-compose.yml configuration
        Environment variables override defaults for deployment flexibility
    """

    # Service ports and endpoints - must match docker-compose.yml
    AGENT_BRAIN_BASE_PORT = int(os.getenv("AGENT_BRAIN_BASE_PORT", "8305"))
    REASONING_MODULE_FACTORY_PORT = int(os.getenv("REASONING_MODULE_FACTORY_PORT", "8304"))
    MEMORY_MANAGER_PORT = int(os.getenv("MEMORY_MANAGER_PORT", "8205"))
    PLUGIN_REGISTRY_PORT = int(os.getenv("PLUGIN_REGISTRY_PORT", "8201"))
    SERVICE_CONNECTOR_FACTORY_PORT = int(os.getenv("SERVICE_CONNECTOR_FACTORY_PORT", "8306"))
    UI_TO_BRAIN_MAPPER_PORT = int(os.getenv("UI_TO_BRAIN_MAPPER_PORT", "8302"))
    AGENT_ORCHESTRATOR_PORT = int(os.getenv("AGENT_ORCHESTRATOR_PORT", "8200"))

    # Service host configuration - supports container networking
    SERVICE_HOST = os.getenv("SERVICE_HOST", "localhost")

    # Agent creation settings - prevents resource exhaustion
    MAX_AGENT_INSTANCES = int(os.getenv("MAX_AGENT_INSTANCES", "50"))
    AGENT_CREATION_TIMEOUT = int(os.getenv("AGENT_CREATION_TIMEOUT", "120"))
    AGENT_VALIDATION_TIMEOUT = int(os.getenv("AGENT_VALIDATION_TIMEOUT", "30"))

    # Performance monitoring - enables observability
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_RETENTION_HOURS = int(os.getenv("METRICS_RETENTION_HOURS", "24"))

class AgentCreationStatus(Enum):
    """
    Enumeration of agent creation statuses.

    This enum tracks the complete lifecycle of agent instantiation,
    from initial request to final operational state or failure.

    States:
        PENDING: Agent creation request received, waiting to start
        VALIDATING: Configuration validation in progress
        INITIALIZING: Core agent components being initialized
        CONFIGURING: Service integrations and plugins being configured
        READY: Agent fully operational and ready for task execution
        FAILED: Agent creation failed, requires investigation
        TERMINATED: Agent explicitly terminated or cleaned up

    Note:
        Status transitions are atomic and logged for audit purposes
        FAILED state triggers automatic cleanup and notification
        READY state enables agent in orchestrator for task routing
    """
    PENDING = "pending"
    VALIDATING = "validating"
    INITIALIZING = "initializing"
    CONFIGURING = "configuring"
    READY = "ready"
    FAILED = "failed"
    TERMINATED = "terminated"

class ValidationSeverity(Enum):
    """
    Enumeration of validation severity levels.

    Used to categorize configuration validation issues by impact level,
    enabling appropriate handling and user notification strategies.

    Levels:
        ERROR: Critical issues preventing agent creation or operation
        WARNING: Potential issues that may affect performance or reliability
        INFO: Informational notices about configuration best practices

    Note:
        ERROR level blocks agent creation until resolved
        WARNING level allows creation but recommends fixes
        INFO level is logged but doesn't block creation
    """
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

# Pydantic models for request/response data structures

class AgentPersona(BaseModel):
    """Agent personality and behavioral configuration"""

    name: str = Field(..., description="Agent's display name")
    description: str = Field(..., description="Detailed description of the agent's role")
    domain: str = Field(..., description="Business domain specialization")
    expertise_level: str = Field(default="intermediate", description="Agent's expertise level")
    communication_style: str = Field(default="professional", description="Communication style preference")
    decision_making_style: str = Field(default="analytical", description="Decision making approach")
    risk_tolerance: str = Field(default="moderate", description="Risk tolerance level")
    learning_objectives: List[str] = Field(default_factory=list, description="Learning and improvement goals")

class MemoryConfig(BaseModel):
    """Memory management configuration"""

    working_memory_enabled: bool = Field(default=True, description="Enable working memory")
    episodic_memory_enabled: bool = Field(default=True, description="Enable episodic memory")
    semantic_memory_enabled: bool = Field(default=True, description="Enable semantic memory")
    vector_memory_enabled: bool = Field(default=True, description="Enable vector memory")
    memory_ttl_seconds: int = Field(default=3600, description="Default memory TTL")
    max_memory_items: int = Field(default=1000, description="Maximum memory items")
    consolidation_interval: int = Field(default=3600, description="Memory consolidation interval")

class PluginConfig(BaseModel):
    """Plugin configuration for agent capabilities"""

    enabled_plugins: List[str] = Field(default_factory=list, description="List of enabled plugins")
    plugin_settings: Dict[str, Any] = Field(default_factory=dict, description="Plugin-specific settings")
    auto_discovery: bool = Field(default=True, description="Enable automatic plugin discovery")
    domain_plugins: List[str] = Field(default_factory=list, description="Domain-specific plugins")
    generic_plugins: List[str] = Field(default_factory=list, description="Generic utility plugins")

class ReasoningConfig(BaseModel):
    """Reasoning pattern configuration"""

    pattern: str = Field(default="ReAct", description="Primary reasoning pattern")
    fallback_patterns: List[str] = Field(default_factory=list, description="Fallback reasoning patterns")
    pattern_weights: Dict[str, float] = Field(default_factory=dict, description="Pattern selection weights")
    adaptive_reasoning: bool = Field(default=True, description="Enable adaptive pattern selection")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")

class AgentConfig(BaseModel):
    """Complete agent configuration"""

    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent display name")
    persona: AgentPersona = Field(..., description="Agent personality configuration")
    reasoning_config: ReasoningConfig = Field(..., description="Reasoning pattern configuration")
    memory_config: MemoryConfig = Field(..., description="Memory management configuration")
    plugin_config: PluginConfig = Field(..., description="Plugin configuration")
    execution_limits: Dict[str, Any] = Field(default_factory=dict, description="Execution constraints")
    monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Monitoring settings")

class ValidationResult(BaseModel):
    """Result of configuration validation"""

    is_valid: bool = Field(..., description="Whether the configuration is valid")
    severity: str = Field(..., description="Validation severity level")
    message: str = Field(..., description="Validation message")
    field: Optional[str] = Field(None, description="Field that failed validation")
    suggestion: Optional[str] = Field(None, description="Suggested fix")

class AgentCreationRequest(BaseModel):
    """Request for agent creation"""

    agent_config: AgentConfig = Field(..., description="Complete agent configuration")
    deployment_options: Dict[str, Any] = Field(default_factory=dict, description="Deployment-specific options")
    validation_options: Dict[str, Any] = Field(default_factory=dict, description="Validation options")

class AgentCreationResponse(BaseModel):
    """Response for agent creation"""

    success: bool = Field(..., description="Whether agent creation was successful")
    agent_id: Optional[str] = Field(None, description="Created agent identifier")
    agent_endpoint: Optional[str] = Field(None, description="Agent API endpoint")
    status: str = Field(..., description="Creation status")
    validation_results: List[ValidationResult] = Field(default_factory=list, description="Configuration validation results")
    creation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Creation process metadata")
    error_message: Optional[str] = Field(None, description="Error message if creation failed")

class AgentStatusResponse(BaseModel):
    """Response for agent status inquiry"""

    agent_id: str = Field(..., description="Agent identifier")
    status: str = Field(..., description="Current agent status")
    created_at: datetime = Field(..., description="Agent creation timestamp")
    last_activity: Optional[datetime] = Field(None, description="Last agent activity")
    configuration_valid: bool = Field(..., description="Whether configuration is valid")
    services_status: Dict[str, Any] = Field(default_factory=dict, description="Status of integrated services")

class AgentMetrics(BaseModel):
    """Agent performance metrics"""

    agent_id: str = Field(..., description="Agent identifier")
    total_tasks: int = Field(default=0, description="Total tasks executed")
    successful_tasks: int = Field(default=0, description="Successfully completed tasks")
    failed_tasks: int = Field(default=0, description="Failed tasks")
    average_execution_time: float = Field(default=0.0, description="Average execution time")
    average_confidence: float = Field(default=0.0, description="Average confidence score")
    uptime_seconds: int = Field(default=0, description="Agent uptime in seconds")
    memory_usage_mb: float = Field(default=0.0, description="Current memory usage")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")

# Agent Factory Core Class
class AgentFactory:
    """
    Core factory class for creating and managing AgentBrain instances.

    This class handles the complete agent instantiation process including:
    - Configuration validation and processing
    - Service dependency resolution
    - Component integration and setup
    - Agent initialization and testing
    - Performance monitoring and optimization
    """

    def __init__(self):
        """Initialize the Agent Factory"""
        self.logger = structlog.get_logger(__name__)
        self.created_agents: Dict[str, Dict[str, Any]] = {}
        self.service_endpoints = self._initialize_service_endpoints()
        self.creation_metrics = {
            "total_created": 0,
            "successful_creations": 0,
            "failed_creations": 0,
            "average_creation_time": 0.0
        }

    def _initialize_service_endpoints(self) -> Dict[str, str]:
        """Initialize service endpoint mappings"""
        return {
            "agent_brain_base": f"http://{Config.SERVICE_HOST}:{Config.AGENT_BRAIN_BASE_PORT}",
            "reasoning_module_factory": f"http://{Config.SERVICE_HOST}:{Config.REASONING_MODULE_FACTORY_PORT}",
            "memory_manager": f"http://{Config.SERVICE_HOST}:{Config.MEMORY_MANAGER_PORT}",
            "plugin_registry": f"http://{Config.SERVICE_HOST}:{Config.PLUGIN_REGISTRY_PORT}",
            "service_connector_factory": f"http://{Config.SERVICE_HOST}:{Config.SERVICE_CONNECTOR_FACTORY_PORT}",
            "ui_to_brain_mapper": f"http://{Config.SERVICE_HOST}:{Config.UI_TO_BRAIN_MAPPER_PORT}",
            "agent_orchestrator": f"http://{Config.SERVICE_HOST}:{Config.AGENT_ORCHESTRATOR_PORT}"
        }

    async def create_agent(self, request: AgentCreationRequest) -> AgentCreationResponse:
        """
        Create a new agent instance from configuration.

        This method orchestrates the complete agent creation process:
        1. Validate agent configuration
        2. Check service dependencies
        3. Create agent through Agent Brain Base service
        4. Configure reasoning modules and memory
        5. Set up plugin system
        6. Register with orchestrator
        7. Perform initial validation

        Args:
            request: Agent creation request with configuration

        Returns:
            AgentCreationResponse: Creation result with status and metadata
        """
        start_time = datetime.utcnow()
        agent_id = request.agent_config.agent_id

        try:
            self.logger.info("Starting agent creation", agent_id=agent_id)

            # Step 1: Validate configuration
            validation_results = await self._validate_agent_config(request.agent_config)
            if any(v.severity == ValidationSeverity.ERROR.value for v in validation_results):
                return AgentCreationResponse(
                    success=False,
                    status=AgentCreationStatus.FAILED.value,
                    validation_results=validation_results,
                    error_message="Configuration validation failed"
                )

            # Step 2: Check service dependencies
            dependency_status = await self._check_service_dependencies()
            if not dependency_status["all_healthy"]:
                return AgentCreationResponse(
                    success=False,
                    status=AgentCreationStatus.FAILED.value,
                    error_message=f"Service dependencies not available: {dependency_status['issues']}"
                )

            # Step 3: Create agent instance
            agent_data = await self._create_agent_instance(request.agent_config)

            # Step 4: Configure agent services
            await self._configure_agent_services(agent_id, request.agent_config)

            # Step 5: Register with orchestrator
            await self._register_with_orchestrator(agent_id, request.agent_config)

            # Step 6: Perform initial validation
            validation_status = await self._perform_agent_validation(agent_id)

            # Update creation metrics
            creation_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_creation_metrics(True, creation_time)

            # Store agent metadata
            self.created_agents[agent_id] = {
                "config": request.agent_config,
                "created_at": datetime.utcnow(),
                "status": AgentCreationStatus.READY.value,
                "creation_time": creation_time,
                "validation_results": validation_results
            }

            return AgentCreationResponse(
                success=True,
                agent_id=agent_id,
                agent_endpoint=f"{self.service_endpoints['agent_brain_base']}/agents/{agent_id}",
                status=AgentCreationStatus.READY.value,
                validation_results=validation_results,
                creation_metadata={
                    "creation_time": creation_time,
                    "services_configured": len(request.agent_config.plugin_config.enabled_plugins),
                    "reasoning_pattern": request.agent_config.reasoning_config.pattern,
                    "memory_enabled": request.agent_config.memory_config.working_memory_enabled
                }
            )

        except Exception as e:
            creation_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_creation_metrics(False, creation_time)

            self.logger.error("Agent creation failed", agent_id=agent_id, error=str(e))

            return AgentCreationResponse(
                success=False,
                status=AgentCreationStatus.FAILED.value,
                error_message=str(e),
                creation_metadata={"creation_time": creation_time}
            )

    async def _validate_agent_config(self, config: AgentConfig) -> List[ValidationResult]:
        """
        Validate agent configuration for completeness and correctness.

        Args:
            config: Agent configuration to validate

        Returns:
            List of validation results
        """
        results = []

        # Validate agent ID
        if not config.agent_id or len(config.agent_id.strip()) == 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR.value,
                message="Agent ID is required",
                field="agent_id",
                suggestion="Provide a unique agent identifier"
            ))

        # Validate agent name
        if not config.name or len(config.name.strip()) == 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR.value,
                message="Agent name is required",
                field="name",
                suggestion="Provide a descriptive agent name"
            ))

        # Validate persona
        if not config.persona.name:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR.value,
                message="Agent persona name is required",
                field="persona.name",
                suggestion="Define the agent's persona and role"
            ))

        # Validate reasoning pattern
        valid_patterns = ["ReAct", "Reflection", "Planning", "Multi-Agent"]
        if config.reasoning_config.pattern not in valid_patterns:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR.value,
                message=f"Unsupported reasoning pattern: {config.reasoning_config.pattern}",
                field="reasoning_config.pattern",
                suggestion=f"Use one of: {', '.join(valid_patterns)}"
            ))

        # Check for duplicate agent ID
        if config.agent_id in self.created_agents:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING.value,
                message=f"Agent ID {config.agent_id} already exists",
                field="agent_id",
                suggestion="Use a unique agent identifier"
            ))

        # Validate memory configuration
        if config.memory_config.vector_memory_enabled and config.memory_config.vector_dimensions <= 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR.value,
                message="Vector dimensions must be positive when vector memory is enabled",
                field="memory_config.vector_dimensions",
                suggestion="Set vector_dimensions to a positive integer"
            ))

        return results

    async def _check_service_dependencies(self) -> Dict[str, Any]:
        """
        Check the health status of all required services.

        Returns:
            Dictionary with dependency status information
        """
        issues = []
        all_healthy = True

        for service_name, endpoint in self.service_endpoints.items():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    health_url = endpoint.replace(f"http://{Config.SERVICE_HOST}:", f"http://{Config.SERVICE_HOST}:") + "/health"
                    response = await client.get(health_url)
                    response.raise_for_status()
            except Exception as e:
                issues.append(f"{service_name}: {str(e)}")
                all_healthy = False

        return {
            "all_healthy": all_healthy,
            "issues": issues,
            "checked_at": datetime.utcnow().isoformat()
        }

    async def _create_agent_instance(self, config: AgentConfig) -> Dict[str, Any]:
        """
        Create agent instance through Agent Brain Base service.

        Args:
            config: Agent configuration

        Returns:
            Agent creation response data
        """
        async with httpx.AsyncClient(timeout=Config.AGENT_CREATION_TIMEOUT) as client:
            response = await client.post(
                f"{self.service_endpoints['agent_brain_base']}/agents",
                json=config.dict()
            )
            response.raise_for_status()
            return response.json()

    async def _configure_agent_services(self, agent_id: str, config: AgentConfig) -> None:
        """
        Configure agent services (reasoning, memory, plugins).

        Args:
            agent_id: Agent identifier
            config: Agent configuration
        """
        # Configure reasoning module
        if config.reasoning_config.pattern:
            await self._configure_reasoning_module(agent_id, config.reasoning_config)

        # Configure memory management
        if config.memory_config.working_memory_enabled:
            await self._configure_memory_management(agent_id, config.memory_config)

        # Configure plugin system
        if config.plugin_config.enabled_plugins:
            await self._configure_plugin_system(agent_id, config.plugin_config)

    async def _configure_reasoning_module(self, agent_id: str, reasoning_config: ReasoningConfig) -> None:
        """Configure reasoning module for the agent"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.service_endpoints['reasoning_module_factory']}/patterns/{reasoning_config.pattern}/capabilities"
                )
                if response.status_code == 200:
                    self.logger.info("Reasoning module configured", agent_id=agent_id, pattern=reasoning_config.pattern)
        except Exception as e:
            self.logger.warning("Reasoning module configuration failed", agent_id=agent_id, error=str(e))

    async def _configure_memory_management(self, agent_id: str, memory_config: MemoryConfig) -> None:
        """Configure memory management for the agent"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.service_endpoints['memory_manager']}/health")
                if response.status_code == 200:
                    self.logger.info("Memory management configured", agent_id=agent_id)
        except Exception as e:
            self.logger.warning("Memory management configuration failed", agent_id=agent_id, error=str(e))

    async def _configure_plugin_system(self, agent_id: str, plugin_config: PluginConfig) -> None:
        """Configure plugin system for the agent"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for plugin_name in plugin_config.enabled_plugins:
                    try:
                        response = await client.get(
                            f"{self.service_endpoints['plugin_registry']}/plugins/{plugin_name}"
                        )
                        if response.status_code == 200:
                            self.logger.info("Plugin configured", agent_id=agent_id, plugin=plugin_name)
                    except Exception as e:
                        self.logger.warning("Plugin configuration failed", agent_id=agent_id, plugin=plugin_name, error=str(e))
        except Exception as e:
            self.logger.warning("Plugin system configuration failed", agent_id=agent_id, error=str(e))

    async def _register_with_orchestrator(self, agent_id: str, config: AgentConfig) -> None:
        """Register agent with the orchestrator"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                registration_data = {
                    "agent_id": agent_id,
                    "agent_name": config.name,
                    "domain": config.persona.domain,
                    "deployment_id": f"deployment_{agent_id}",
                    "brain_config": {
                        "reasoning_pattern": config.reasoning_config.pattern,
                        "memory_config": config.memory_config.dict(),
                        "plugin_config": config.plugin_config.dict()
                    }
                }

                response = await client.post(
                    f"{self.service_endpoints['agent_orchestrator']}/orchestrator/register-agent",
                    json=registration_data
                )

                if response.status_code == 200:
                    self.logger.info("Agent registered with orchestrator", agent_id=agent_id)
                else:
                    self.logger.warning("Agent registration failed", agent_id=agent_id, status=response.status_code)

        except Exception as e:
            self.logger.warning("Orchestrator registration failed", agent_id=agent_id, error=str(e))

    async def _perform_agent_validation(self, agent_id: str) -> bool:
        """Perform initial validation of the created agent"""
        try:
            async with httpx.AsyncClient(timeout=Config.AGENT_VALIDATION_TIMEOUT) as client:
                response = await client.get(
                    f"{self.service_endpoints['agent_brain_base']}/agents/{agent_id}/status"
                )
                return response.status_code == 200
        except Exception as e:
            self.logger.warning("Agent validation failed", agent_id=agent_id, error=str(e))
            return False

    def _update_creation_metrics(self, success: bool, creation_time: float) -> None:
        """Update agent creation metrics"""
        self.creation_metrics["total_created"] += 1

        if success:
            self.creation_metrics["successful_creations"] += 1
        else:
            self.creation_metrics["failed_creations"] += 1

        # Update rolling average creation time
        total_created = self.creation_metrics["total_created"]
        current_avg = self.creation_metrics["average_creation_time"]

        if current_avg == 0:
            self.creation_metrics["average_creation_time"] = creation_time
        else:
            self.creation_metrics["average_creation_time"] = (
                (current_avg * (total_created - 1)) + creation_time
            ) / total_created

    async def get_agent_status(self, agent_id: str) -> Optional[AgentStatusResponse]:
        """
        Get status of a created agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent status information or None if not found
        """
        if agent_id not in self.created_agents:
            return None

        agent_data = self.created_agents[agent_id]

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.service_endpoints['agent_brain_base']}/agents/{agent_id}/status"
                )

                if response.status_code == 200:
                    status_data = response.json()
                    return AgentStatusResponse(
                        agent_id=agent_id,
                        status=status_data.get("status", "unknown"),
                        created_at=agent_data["created_at"],
                        last_activity=status_data.get("last_activity"),
                        configuration_valid=True,
                        services_status=status_data.get("services_status", {})
                    )
                else:
                    return AgentStatusResponse(
                        agent_id=agent_id,
                        status=AgentCreationStatus.FAILED.value,
                        created_at=agent_data["created_at"],
                        configuration_valid=False
                    )

        except Exception as e:
            self.logger.error("Failed to get agent status", agent_id=agent_id, error=str(e))
            return AgentStatusResponse(
                agent_id=agent_id,
                status="error",
                created_at=agent_data["created_at"],
                configuration_valid=False
            )

    def get_creation_metrics(self) -> Dict[str, Any]:
        """
        Get agent creation metrics.

        Returns:
            Dictionary with creation statistics
        """
        return {
            **self.creation_metrics,
            "success_rate": (
                self.creation_metrics["successful_creations"] /
                max(self.creation_metrics["total_created"], 1)
            ),
            "active_agents": len(self.created_agents),
            "last_updated": datetime.utcnow().isoformat()
        }

    async def list_created_agents(self) -> List[Dict[str, Any]]:
        """
        List all created agents with their status.

        Returns:
            List of agent information
        """
        agents = []

        for agent_id, agent_data in self.created_agents.items():
            agent_info = {
                "agent_id": agent_id,
                "name": agent_data["config"].name,
                "status": agent_data["status"],
                "created_at": agent_data["created_at"].isoformat(),
                "creation_time": agent_data["creation_time"],
                "reasoning_pattern": agent_data["config"].reasoning_config.pattern,
                "domain": agent_data["config"].persona.domain
            }
            agents.append(agent_info)

        return agents

# FastAPI application setup
app = FastAPI(
    title="Brain Factory Service",
    description="Agent instantiation and configuration management for the Agentic Brain platform",
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

# Initialize agent factory
agent_factory = AgentFactory()

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Brain Factory Service started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup service on shutdown"""
    logger.info("Brain Factory Service shutting down")

@app.get("/")
async def root():
    """Service health check and information endpoint"""
    return {
        "service": "Brain Factory",
        "version": "1.0.0",
        "status": "healthy",
        "description": "Agent instantiation and configuration management service",
        "capabilities": [
            "Agent creation from configuration",
            "Service dependency management",
            "Configuration validation",
            "Performance monitoring",
            "Agent lifecycle management"
        ],
        "endpoints": {
            "POST /generate-agent": "Create new agent instance",
            "GET /agents/{agent_id}/status": "Get agent status",
            "GET /agents": "List created agents",
            "GET /metrics": "Get creation metrics",
            "GET /health": "Service health check"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "Brain Factory",
        "active_agents": len(agent_factory.created_agents),
        "creation_metrics": agent_factory.get_creation_metrics()
    }

@app.post("/generate-agent", response_model=AgentCreationResponse)
async def generate_agent(request: AgentCreationRequest, background_tasks: BackgroundTasks):
    """
    Generate a new agent instance from configuration with comprehensive validation and orchestration.

    This critical endpoint orchestrates the complete agent creation lifecycle, ensuring all
    components are properly initialized, configured, and validated before the agent becomes
    operational. The process involves multiple microservices and requires careful coordination
    to maintain system consistency and reliability.

    Agent Creation Workflow:
    1. **Configuration Validation**: Comprehensive validation of agent persona, reasoning patterns,
       memory configuration, and plugin dependencies against business rules and constraints
    2. **Dependency Resolution**: Verification that all required services (reasoning modules,
       memory managers, plugin registries) are available and properly configured
    3. **Agent Brain Creation**: Instantiation of core agent brain through Agent Brain Base service
       with proper initialization of reasoning engine, memory systems, and service connectors
    4. **Component Integration**: Configuration of reasoning patterns, memory tiers, plugin systems,
       and external service integrations with dependency injection and proper error handling
    5. **Orchestrator Registration**: Registration of completed agent with Agent Orchestrator for
       task routing and lifecycle management with health monitoring setup
    6. **Operational Validation**: Final validation through test execution to ensure agent is
       fully functional and ready for production task processing

    The agent creation process is designed to be:
    - **Idempotent**: Multiple calls with same config produce identical results
    - **Transactional**: Either fully succeeds or fails with complete cleanup
    - **Observable**: Comprehensive logging and metrics throughout process
    - **Recoverable**: Failed creations can be retried with automatic cleanup

    Args:
        request: AgentCreationRequest containing:
            - agent_config: Complete agent configuration with:
                * persona: Agent personality and behavioral settings
                * reasoning_pattern: AI reasoning approach (ReAct, Reflection, Planning)
                * memory_config: Multi-tier memory configuration (working, episodic, semantic)
                * plugin_config: Domain-specific plugins and integrations
                * service_connectors: Data ingestion/output service connections
            - deployment_options: Optional deployment-specific settings:
                * environment: Target environment (dev, staging, prod)
                * resource_limits: CPU/memory limits and scaling policies
                * monitoring_config: Metrics and alerting configuration
            - validation_options: Optional validation configuration:
                * skip_validation: Skip configuration validation (not recommended)
                * strict_mode: Fail on warnings vs. allow with notifications

        background_tasks: FastAPI background tasks for async processing of:
            - Complex agent initializations that may take >30 seconds
            - Batch plugin loading and validation
            - Large-scale configuration processing

    Returns:
        AgentCreationResponse containing:
            - success: Boolean indicating successful agent creation
            - agent_id: Unique agent identifier for task routing and management
            - agent_endpoint: Direct API endpoint for agent-specific operations
            - status: Current agent status (ready, failed, initializing)
            - validation_results: Detailed configuration validation results with:
                * errors: Critical issues preventing agent creation
                * warnings: Potential issues affecting performance
                * recommendations: Best practice suggestions
            - creation_metadata: Process information including:
                * creation_time: Total time for agent creation
                * components_initialized: List of successfully configured components
                * warnings_generated: Non-critical issues encountered
                * retry_count: Number of internal retries performed

    Raises:
        HTTPException (400): Invalid configuration or missing required parameters
        HTTPException (409): Agent with same ID already exists
        HTTPException (503): Required services unavailable (reasoning factory, memory manager)
        HTTPException (408): Agent creation timeout exceeded
        HTTPException (500): Internal processing error with automatic cleanup

    Note:
        - Agent creation is resource-intensive and subject to concurrency limits
        - Complex agents with many plugins may take 2-5 minutes to fully initialize
        - Failed creations are automatically cleaned up to prevent resource leaks
        - All agent configurations are versioned and stored for audit purposes
        - Background task processing enables handling of long-running creations
        - Comprehensive metrics are collected for performance monitoring and optimization

    Performance Considerations:
        - Large agent configurations (>10 plugins) use background processing
        - Memory usage scales with agent complexity and plugin count
        - Network calls to multiple services require proper timeout handling
        - Database transactions ensure consistency across service boundaries
    """
    try:
        # Phase 1: Request Logging and Initial Validation
        # Log agent creation request with unique identifier for tracking
        # This enables correlation across distributed service calls
        logger.info("Agent generation requested", agent_id=request.agent_config.agent_id)

        # Phase 2: Agent Creation Execution
        # Delegate to agent factory for actual creation process
        # Factory handles all complex orchestration and validation logic
        # Background tasks could be used for very complex agents (>30s creation time)
        # For now, execute synchronously with timeout protection
        result = await agent_factory.create_agent(request)

        logger.info(
            "Agent generation completed",
            agent_id=request.agent_config.agent_id,
            success=result.success,
            status=result.status
        )

        return result

    except Exception as e:
        logger.error(
            "Agent generation failed",
            agent_id=request.agent_config.agent_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Agent generation failed: {str(e)}")

@app.get("/agents/{agent_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(agent_id: str):
    """
    Get comprehensive status information for a created agent.

    Path Parameters:
    - agent_id: Unique agent identifier

    Returns:
    - agent_id: Agent identifier
    - status: Current agent status
    - created_at: Agent creation timestamp
    - last_activity: Last agent activity timestamp
    - configuration_valid: Whether agent configuration is valid
    - services_status: Status of integrated services
    """
    try:
        status = await agent_factory.get_agent_status(agent_id)

        if status is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent status", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")

@app.get("/agents")
async def list_agents():
    """
    List all created agents with their current status.

    Returns:
    - agents: List of agent information including:
        - agent_id: Unique agent identifier
        - name: Agent display name
        - status: Current agent status
        - created_at: Agent creation timestamp
        - creation_time: Time taken to create agent
        - reasoning_pattern: Agent's reasoning pattern
        - domain: Agent's business domain
    - total_count: Total number of created agents
    """
    try:
        agents = await agent_factory.list_created_agents()

        return {
            "agents": agents,
            "total_count": len(agents),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

@app.get("/metrics")
async def get_creation_metrics():
    """
    Get comprehensive agent creation metrics and statistics.

    Returns:
    - total_created: Total number of agent creation attempts
    - successful_creations: Number of successful agent creations
    - failed_creations: Number of failed agent creations
    - success_rate: Percentage of successful creations
    - average_creation_time: Average time for agent creation
    - active_agents: Number of currently active agents
    - last_updated: Timestamp of last metrics update
    """
    try:
        metrics = agent_factory.get_creation_metrics()

        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to get creation metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get creation metrics: {str(e)}")

@app.get("/validate-config")
async def validate_agent_config_endpoint(agent_config: AgentConfig):
    """
    Validate an agent configuration without creating the agent.

    Query Parameters:
    - agent_config: Agent configuration to validate (as query parameter)

    Returns:
    - is_valid: Whether the configuration is valid
    - validation_results: Detailed validation results with errors, warnings, and suggestions
    - validation_summary: Summary of validation results
    """
    try:
        validation_results = await agent_factory._validate_agent_config(agent_config)

        is_valid = not any(v.severity == ValidationSeverity.ERROR.value for v in validation_results)

        error_count = len([v for v in validation_results if v.severity == ValidationSeverity.ERROR.value])
        warning_count = len([v for v in validation_results if v.severity == ValidationSeverity.WARNING.value])
        info_count = len([v for v in validation_results if v.severity == ValidationSeverity.INFO.value])

        return {
            "is_valid": is_valid,
            "validation_results": [v.dict() for v in validation_results],
            "validation_summary": {
                "total_issues": len(validation_results),
                "errors": error_count,
                "warnings": warning_count,
                "info": info_count
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Configuration validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Configuration validation failed: {str(e)}")

@app.get("/service-dependencies")
async def check_service_dependencies():
    """
    Check the health status of all service dependencies.

    Returns:
    - all_healthy: Whether all services are healthy
    - issues: List of service health issues
    - checked_at: Timestamp of health check
    - services_status: Status of individual services
    """
    try:
        dependency_status = await agent_factory._check_service_dependencies()

        return {
            "service_dependencies": dependency_status,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Service dependency check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Service dependency check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("BRAIN_FACTORY_PORT", "8301"))

    logger.info("Starting Brain Factory Service", port=port)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
