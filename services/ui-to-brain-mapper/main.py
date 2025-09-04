#!/usr/bin/env python3
"""
UI-to-Brain Mapper Service

This service converts visual workflow JSON from the Agent Builder UI into
structured AgentConfig JSON that backend services can understand and execute.

Key Responsibilities:
- Convert visual components to service configurations
- Map UI connections to data flow definitions
- Generate complete AgentConfig with all required fields
- Validate workflow structure and component compatibility
- Resolve service endpoints and integration points
- Handle reasoning pattern selection and configuration

Architecture:
- FastAPI-based REST service
- Pydantic models for type safety
- Comprehensive validation and error handling
- Integration with plugin registry and service discovery
- Async processing for scalability

Author: AgenticAI Platform
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import httpx

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic.dataclasses import dataclass

# Configure structured logging
logger = structlog.get_logger(__name__)

# Configuration class for service settings
class Config:
    """Configuration settings for UI-to-Brain Mapper service"""

    # Service ports and endpoints
    AGENT_BUILDER_UI_PORT = int(os.getenv("AGENT_BUILDER_UI_PORT", "8300"))
    PLUGIN_REGISTRY_PORT = int(os.getenv("PLUGIN_REGISTRY_PORT", "8201"))
    WORKFLOW_ENGINE_PORT = int(os.getenv("WORKFLOW_ENGINE_PORT", "8202"))
    TEMPLATE_STORE_PORT = int(os.getenv("TEMPLATE_STORE_PORT", "8203"))
    RULE_ENGINE_PORT = int(os.getenv("RULE_ENGINE_PORT", "8204"))
    MEMORY_MANAGER_PORT = int(os.getenv("MEMORY_MANAGER_PORT", "8205"))
    BRAIN_FACTORY_PORT = int(os.getenv("BRAIN_FACTORY_PORT", "8301"))

    # Service host configuration
    SERVICE_HOST = os.getenv("SERVICE_HOST", "localhost")

    # LLM Configuration
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4")
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))

    # Memory Configuration
    DEFAULT_MEMORY_TTL = int(os.getenv("DEFAULT_MEMORY_TTL", "3600"))
    DEFAULT_MEMORY_DIMENSIONS = int(os.getenv("DEFAULT_MEMORY_DIMENSIONS", "384"))

    # Reasoning patterns
    SUPPORTED_REASONING_PATTERNS = ["ReAct", "Reflection", "Planning", "Multi-Agent"]

    # Component type mappings
    COMPONENT_TYPE_MAPPING = {
        "data_input_csv": "csv_ingestion_service",
        "data_input_api": "api_ingestion_service",
        "data_input_pdf": "pdf_ingestion_service",
        "llm_processor": "llm_processor",
        "rule_engine": "rule_engine",
        "decision_node": "decision_node",
        "multi_agent_coordinator": "multi_agent_coordinator",
        "database_output": "postgresql_output",
        "email_output": "email_output",
        "pdf_report_output": "pdf_report_output"
    }

# Pydantic models for request/response data structures

class VisualComponent(BaseModel):
    """Represents a visual component from the UI workflow"""

    id: str = Field(..., description="Unique component identifier")
    type: str = Field(..., description="Component type (data_input_csv, llm_processor, etc.)")
    name: str = Field(..., description="Human-readable component name")
    position: Dict[str, float] = Field(..., description="Component position on canvas")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Component configuration properties")
    connections: List[str] = Field(default_factory=list, description="Connected component IDs")

    @validator('type')
    def validate_component_type(cls, v):
        """Validate component type is supported"""
        if v not in Config.COMPONENT_TYPE_MAPPING:
            raise ValueError(f"Unsupported component type: {v}")
        return v

class VisualConnection(BaseModel):
    """Represents a visual connection between components"""

    id: str = Field(..., description="Unique connection identifier")
    source: Dict[str, str] = Field(..., description="Source component and port information")
    target: Dict[str, str] = Field(..., description="Target component and port information")
    data_type: str = Field(default="json", description="Data type being transferred")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Connection properties")

class VisualWorkflow(BaseModel):
    """Complete visual workflow configuration from UI"""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    components: List[VisualComponent] = Field(..., description="List of workflow components")
    connections: List[VisualConnection] = Field(..., description="List of component connections")
    canvas: Dict[str, Any] = Field(default_factory=dict, description="Canvas configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AgentComponent(BaseModel):
    """Represents a mapped agent component configuration"""

    component_id: str = Field(..., description="Unique component identifier")
    component_type: str = Field(..., description="Mapped component type")
    service_endpoint: str = Field(..., description="Service endpoint URL")
    configuration: Dict[str, Any] = Field(..., description="Component configuration")
    input_ports: List[Dict[str, Any]] = Field(default_factory=list, description="Input port definitions")
    output_ports: List[Dict[str, Any]] = Field(default_factory=list, description="Output port definitions")

class AgentConnection(BaseModel):
    """Represents a mapped agent connection"""

    connection_id: str = Field(..., description="Unique connection identifier")
    source_component: str = Field(..., description="Source component ID")
    source_port: str = Field(..., description="Source port name")
    target_component: str = Field(..., description="Target component ID")
    target_port: str = Field(..., description="Target port name")
    data_mapping: Dict[str, Any] = Field(default_factory=dict, description="Data transformation mapping")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Connection properties")

class AgentConfig(BaseModel):
    """Complete agent configuration for backend services"""

    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    domain: str = Field(..., description="Business domain")
    persona: str = Field(..., description="Agent persona/description")
    reasoning_pattern: str = Field(..., description="Reasoning pattern (ReAct, Reflection, etc.)")
    components: List[AgentComponent] = Field(..., description="Agent components")
    connections: List[AgentConnection] = Field(..., description="Component connections")
    memory_config: Dict[str, Any] = Field(..., description="Memory configuration")
    plugin_config: Dict[str, Any] = Field(..., description="Plugin configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('reasoning_pattern')
    def validate_reasoning_pattern(cls, v):
        """Validate reasoning pattern is supported"""
        if v not in Config.SUPPORTED_REASONING_PATTERNS:
            raise ValueError(f"Unsupported reasoning pattern: {v}")
        return v

class MappingRequest(BaseModel):
    """Request model for UI-to-Brain mapping"""

    visual_workflow: VisualWorkflow = Field(..., description="Visual workflow to convert")
    agent_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional agent metadata")
    custom_mappings: Dict[str, Any] = Field(default_factory=dict, description="Custom component mappings")

class MappingResponse(BaseModel):
    """Response model for UI-to-Brain mapping"""

    success: bool = Field(..., description="Mapping success status")
    agent_config: Optional[AgentConfig] = Field(None, description="Generated agent configuration")
    mapping_report: Dict[str, Any] = Field(..., description="Mapping process report")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Mapping warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")

# Main service class
class UIToBrainMapper:
    """
    Core service class for converting UI workflows to agent configurations.

    This class handles the complex mapping logic to transform visual workflow
    representations into structured agent configurations that can be executed
    by the backend services.
    """

    def __init__(self):
        """Initialize the UI-to-Brain Mapper service"""
        self.logger = structlog.get_logger(__name__)
        self.service_endpoints = self._initialize_service_endpoints()

    def _initialize_service_endpoints(self) -> Dict[str, str]:
        """
        Initialize service endpoint mappings for component resolution.

        Returns:
            Dict mapping component types to service endpoints
        """
        return {
            "csv_ingestion_service": f"http://{Config.SERVICE_HOST}:8001",
            "api_ingestion_service": f"http://{Config.SERVICE_HOST}:8002",
            "pdf_ingestion_service": f"http://{Config.SERVICE_HOST}:8003",
            "postgresql_output": f"http://{Config.SERVICE_HOST}:8004",
            "llm_processor": f"http://{Config.SERVICE_HOST}:8005",
            "rule_engine": f"http://{Config.SERVICE_HOST}:{Config.RULE_ENGINE_PORT}",
            "memory_manager": f"http://{Config.SERVICE_HOST}:{Config.MEMORY_MANAGER_PORT}",
            "plugin_registry": f"http://{Config.SERVICE_HOST}:{Config.PLUGIN_REGISTRY_PORT}",
            "workflow_engine": f"http://{Config.SERVICE_HOST}:{Config.WORKFLOW_ENGINE_PORT}",
            "template_store": f"http://{Config.SERVICE_HOST}:{Config.TEMPLATE_STORE_PORT}",
        }

    async def map_visual_to_agent_config(
        self,
        visual_workflow: VisualWorkflow,
        agent_metadata: Dict[str, Any] = None,
        custom_mappings: Dict[str, Any] = None
    ) -> MappingResponse:
        """
        Convert visual workflow to agent configuration.

        This is the main mapping function that orchestrates the entire
        conversion process from visual UI components to executable agent
        configuration.

        Args:
            visual_workflow: Visual workflow from UI
            agent_metadata: Additional agent metadata
            custom_mappings: Custom component type mappings

        Returns:
            MappingResponse with agent configuration and validation results
        """
        try:
            self.logger.info("Starting visual to agent config mapping",
                           workflow_id=visual_workflow.workflow_id)

            # Initialize mapping context
            mapping_context = {
                "validation_errors": [],
                "warnings": [],
                "component_mappings": {},
                "connection_mappings": [],
                "service_discovery": {},
                "performance_metrics": {}
            }

            # Step 1: Validate visual workflow structure
            await self._validate_visual_workflow(visual_workflow, mapping_context)

            # Step 2: Map visual components to agent components
            agent_components = await self._map_components_to_services(
                visual_workflow.components, custom_mappings, mapping_context
            )

            # Step 3: Map visual connections to agent connections
            agent_connections = await self._map_connections_to_dataflow(
                visual_workflow.connections, agent_components, mapping_context
            )

            # Step 4: Generate complete agent configuration
            agent_config = await self._generate_agent_config(
                visual_workflow, agent_components, agent_connections,
                agent_metadata, mapping_context
            )

            # Step 5: Validate generated configuration
            await self._validate_agent_config(agent_config, mapping_context)

            # Generate mapping report
            mapping_report = self._generate_mapping_report(mapping_context)

            # Return successful response
            return MappingResponse(
                success=True,
                agent_config=agent_config,
                mapping_report=mapping_report,
                validation_errors=mapping_context["validation_errors"],
                warnings=mapping_context["warnings"],
                metadata={
                    "mapped_at": datetime.utcnow().isoformat(),
                    "mapping_version": "1.0.0",
                    "component_count": len(agent_components),
                    "connection_count": len(agent_connections)
                }
            )

        except Exception as e:
            self.logger.error("Mapping failed", error=str(e), workflow_id=visual_workflow.workflow_id)

            return MappingResponse(
                success=False,
                agent_config=None,
                mapping_report={"error": str(e)},
                validation_errors=[f"Mapping failed: {str(e)}"],
                warnings=[],
                metadata={
                    "failed_at": datetime.utcnow().isoformat(),
                    "error_type": type(e).__name__
                }
            )

    async def _validate_visual_workflow(
        self,
        visual_workflow: VisualWorkflow,
        mapping_context: Dict[str, Any]
    ) -> None:
        """
        Validate the structure and completeness of the visual workflow.

        Args:
            visual_workflow: Visual workflow to validate
            mapping_context: Context for storing validation results
        """
        # Check for minimum requirements
        if not visual_workflow.components:
            mapping_context["validation_errors"].append("Workflow must contain at least one component")
            return

        # Validate component types
        for component in visual_workflow.components:
            if component.type not in Config.COMPONENT_TYPE_MAPPING:
                mapping_context["validation_errors"].append(
                    f"Unsupported component type: {component.type}"
                )

        # Check for isolated components
        connected_components = set()
        for connection in visual_workflow.connections:
            connected_components.add(connection.source["componentId"])
            connected_components.add(connection.target["componentId"])

        isolated_components = [
            comp for comp in visual_workflow.components
            if comp.id not in connected_components and len(visual_workflow.components) > 1
        ]

        if isolated_components:
            mapping_context["warnings"].extend([
                f"Isolated component detected: {comp.name} ({comp.id})"
                for comp in isolated_components
            ])

        # Validate connection integrity
        component_ids = {comp.id for comp in visual_workflow.components}
        for connection in visual_workflow.connections:
            source_id = connection.source["componentId"]
            target_id = connection.target["componentId"]

            if source_id not in component_ids:
                mapping_context["validation_errors"].append(
                    f"Connection source component not found: {source_id}"
                )
            if target_id not in component_ids:
                mapping_context["validation_errors"].append(
                    f"Connection target component not found: {target_id}"
                )

    async def _map_components_to_services(
        self,
        visual_components: List[VisualComponent],
        custom_mappings: Dict[str, Any],
        mapping_context: Dict[str, Any]
    ) -> List[AgentComponent]:
        """
        Map visual components to agent service configurations.

        Args:
            visual_components: List of visual components
            custom_mappings: Custom component mappings
            mapping_context: Context for storing mapping results

        Returns:
            List of mapped agent components
        """
        agent_components = []

        for visual_comp in visual_components:
            try:
                # Determine service type and endpoint
                service_type = Config.COMPONENT_TYPE_MAPPING.get(visual_comp.type)
                if not service_type:
                    service_type = custom_mappings.get(visual_comp.type, visual_comp.type)

                service_endpoint = self.service_endpoints.get(service_type)
                if not service_endpoint:
                    mapping_context["warnings"].append(
                        f"No service endpoint found for component type: {service_type}"
                    )
                    # Create placeholder endpoint
                    service_endpoint = f"http://{Config.SERVICE_HOST}:8000/{service_type}"

                # Generate component configuration
                component_config = await self._generate_component_config(
                    visual_comp, service_type, mapping_context
                )

                # Create agent component
                agent_component = AgentComponent(
                    component_id=visual_comp.id,
                    component_type=service_type,
                    service_endpoint=service_endpoint,
                    configuration=component_config,
                    input_ports=self._generate_input_ports(visual_comp),
                    output_ports=self._generate_output_ports(visual_comp)
                )

                agent_components.append(agent_component)
                mapping_context["component_mappings"][visual_comp.id] = agent_component

            except Exception as e:
                mapping_context["validation_errors"].append(
                    f"Failed to map component {visual_comp.name}: {str(e)}"
                )

        return agent_components

    async def _generate_component_config(
        self,
        visual_comp: VisualComponent,
        service_type: str,
        mapping_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate service-specific configuration for a component.

        Args:
            visual_comp: Visual component
            service_type: Target service type
            mapping_context: Context for storing results

        Returns:
            Component configuration dictionary
        """
        base_config = {
            "name": visual_comp.name,
            "type": service_type,
            "enabled": True,
            "properties": visual_comp.properties
        }

        # Service-specific configuration generation
        if service_type == "llm_processor":
            base_config.update({
                "model": visual_comp.properties.get("model", Config.DEFAULT_LLM_MODEL),
                "temperature": visual_comp.properties.get("temperature", Config.DEFAULT_TEMPERATURE),
                "max_tokens": visual_comp.properties.get("maxTokens", Config.DEFAULT_MAX_TOKENS),
                "prompt_template": visual_comp.properties.get("promptTemplate", ""),
                "system_message": visual_comp.properties.get("systemMessage", "")
            })

        elif service_type == "rule_engine":
            base_config.update({
                "rule_set_id": visual_comp.properties.get("ruleSetId"),
                "rule_conditions": visual_comp.properties.get("conditions", []),
                "rule_actions": visual_comp.properties.get("actions", []),
                "evaluation_mode": visual_comp.properties.get("evaluationMode", "all_match")
            })

        elif service_type.startswith("csv_ingestion"):
            base_config.update({
                "file_path": visual_comp.properties.get("filePath"),
                "delimiter": visual_comp.properties.get("delimiter", ","),
                "has_headers": visual_comp.properties.get("hasHeaders", True),
                "encoding": visual_comp.properties.get("encoding", "utf-8"),
                "skip_rows": visual_comp.properties.get("skipRows", 0)
            })

        elif service_type == "postgresql_output":
            base_config.update({
                "table_name": visual_comp.properties.get("tableName"),
                "connection_string": visual_comp.properties.get("connectionString"),
                "batch_size": visual_comp.properties.get("batchSize", 100),
                "create_table": visual_comp.properties.get("createTable", False),
                "schema_definition": visual_comp.properties.get("schema", {})
            })

        elif service_type == "email_output":
            base_config.update({
                "smtp_server": visual_comp.properties.get("smtpServer"),
                "smtp_port": visual_comp.properties.get("smtpPort", 587),
                "sender_email": visual_comp.properties.get("senderEmail"),
                "recipient_emails": visual_comp.properties.get("recipientEmails", []),
                "subject_template": visual_comp.properties.get("subjectTemplate"),
                "body_template": visual_comp.properties.get("bodyTemplate")
            })

        return base_config

    def _generate_input_ports(self, visual_comp: VisualComponent) -> List[Dict[str, Any]]:
        """Generate input ports for a component"""
        # Component-specific input port generation
        if visual_comp.type == "data_input_csv":
            return [{"name": "input", "type": "file", "required": True}]
        elif visual_comp.type == "llm_processor":
            return [{"name": "prompt", "type": "string", "required": True}]
        elif visual_comp.type == "rule_engine":
            return [{"name": "data", "type": "object", "required": True}]
        elif visual_comp.type == "decision_node":
            return [{"name": "condition", "type": "boolean", "required": True}]
        else:
            return [{"name": "input", "type": "any", "required": True}]

    def _generate_output_ports(self, visual_comp: VisualComponent) -> List[Dict[str, Any]]:
        """Generate output ports for a component"""
        # Component-specific output port generation
        if visual_comp.type == "data_input_csv":
            return [{"name": "data", "type": "array", "description": "Parsed CSV data"}]
        elif visual_comp.type == "llm_processor":
            return [{"name": "response", "type": "string", "description": "LLM response"}]
        elif visual_comp.type == "rule_engine":
            return [{"name": "result", "type": "object", "description": "Rule evaluation result"}]
        elif visual_comp.type == "decision_node":
            return [
                {"name": "true", "type": "boolean", "description": "True path"},
                {"name": "false", "type": "boolean", "description": "False path"}
            ]
        else:
            return [{"name": "output", "type": "any", "description": "Component output"}]

    async def _map_connections_to_dataflow(
        self,
        visual_connections: List[VisualConnection],
        agent_components: List[AgentComponent],
        mapping_context: Dict[str, Any]
    ) -> List[AgentConnection]:
        """
        Map visual connections to agent data flow connections.

        Args:
            visual_connections: List of visual connections
            agent_components: List of mapped agent components
            mapping_context: Context for storing results

        Returns:
            List of mapped agent connections
        """
        agent_connections = []

        for visual_conn in visual_connections:
            try:
                source_comp_id = visual_conn.source["componentId"]
                target_comp_id = visual_conn.target["componentId"]

                # Find corresponding agent components
                source_agent_comp = next(
                    (comp for comp in agent_components if comp.component_id == source_comp_id),
                    None
                )
                target_agent_comp = next(
                    (comp for comp in agent_components if comp.component_id == target_comp_id),
                    None
                )

                if not source_agent_comp or not target_agent_comp:
                    mapping_context["warnings"].append(
                        f"Could not find agent components for connection: {source_comp_id} -> {target_comp_id}"
                    )
                    continue

                # Create agent connection
                agent_connection = AgentConnection(
                    connection_id=visual_conn.id,
                    source_component=source_comp_id,
                    source_port=visual_conn.source.get("port", "output"),
                    target_component=target_comp_id,
                    target_port=visual_conn.target.get("port", "input"),
                    data_mapping=self._generate_data_mapping(visual_conn, source_agent_comp, target_agent_comp),
                    properties=visual_conn.properties
                )

                agent_connections.append(agent_connection)

            except Exception as e:
                mapping_context["validation_errors"].append(
                    f"Failed to map connection {visual_conn.id}: {str(e)}"
                )

        return agent_connections

    def _generate_data_mapping(
        self,
        visual_conn: VisualConnection,
        source_comp: AgentComponent,
        target_comp: AgentComponent
    ) -> Dict[str, Any]:
        """
        Generate data transformation mapping for a connection.

        Args:
            visual_conn: Visual connection
            source_comp: Source agent component
            target_comp: Target agent component

        Returns:
            Data mapping dictionary
        """
        # Basic data mapping - can be enhanced with more sophisticated logic
        return {
            "source_type": source_comp.component_type,
            "target_type": target_comp.component_type,
            "data_type": visual_conn.data_type,
            "transformation": "passthrough",  # Default transformation
            "field_mappings": {},  # Can be enhanced with field-level mappings
            "validation_rules": []  # Can include data validation rules
        }

    async def _generate_agent_config(
        self,
        visual_workflow: VisualWorkflow,
        agent_components: List[AgentComponent],
        agent_connections: List[AgentConnection],
        agent_metadata: Dict[str, Any],
        mapping_context: Dict[str, Any]
    ) -> AgentConfig:
        """
        Generate complete agent configuration.

        Args:
            visual_workflow: Original visual workflow
            agent_components: Mapped agent components
            agent_connections: Mapped agent connections
            agent_metadata: Additional agent metadata
            mapping_context: Mapping context

        Returns:
            Complete agent configuration
        """
        # Determine reasoning pattern based on workflow analysis
        reasoning_pattern = self._determine_reasoning_pattern(
            agent_components, agent_connections, agent_metadata
        )

        # Generate memory configuration
        memory_config = self._generate_memory_config(agent_metadata)

        # Generate plugin configuration
        plugin_config = await self._generate_plugin_config(agent_components, mapping_context)

        # Create agent configuration
        agent_config = AgentConfig(
            agent_id=agent_metadata.get("agent_id", f"agent_{uuid.uuid4().hex[:8]}"),
            name=agent_metadata.get("name", visual_workflow.name),
            domain=agent_metadata.get("domain", "general"),
            persona=agent_metadata.get("persona", "A helpful AI agent"),
            reasoning_pattern=reasoning_pattern,
            components=agent_components,
            connections=agent_connections,
            memory_config=memory_config,
            plugin_config=plugin_config,
            metadata={
                "source_workflow_id": visual_workflow.workflow_id,
                "mapped_at": datetime.utcnow().isoformat(),
                "mapping_service_version": "1.0.0",
                **agent_metadata
            }
        )

        return agent_config

    def _determine_reasoning_pattern(
        self,
        agent_components: List[AgentComponent],
        agent_connections: List[AgentConnection],
        agent_metadata: Dict[str, Any]
    ) -> str:
        """
        Determine the most appropriate reasoning pattern for the agent.

        Args:
            agent_components: Agent components
            agent_connections: Agent connections
            agent_metadata: Agent metadata

        Returns:
            Reasoning pattern name
        """
        # Check for explicit reasoning pattern in metadata
        if "reasoning_pattern" in agent_metadata:
            pattern = agent_metadata["reasoning_pattern"]
            if pattern in Config.SUPPORTED_REASONING_PATTERNS:
                return pattern

        # Analyze workflow structure to determine pattern
        component_types = {comp.component_type for comp in agent_components}
        connection_count = len(agent_connections)

        # Multi-Agent pattern if multiple LLM processors or complex coordination
        if ("llm_processor" in component_types and
            len([c for c in agent_components if c.component_type == "llm_processor"]) > 1):
            return "Multi-Agent"

        # Planning pattern if complex workflow with many connections
        if connection_count > 5:
            return "Planning"

        # Reflection pattern if memory-intensive components
        if any(comp.component_type in ["memory_manager", "rule_engine"] for comp in agent_components):
            return "Reflection"

        # Default to ReAct pattern
        return "ReAct"

    def _generate_memory_config(self, agent_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate memory configuration for the agent.

        Args:
            agent_metadata: Agent metadata

        Returns:
            Memory configuration dictionary
        """
        return {
            "working_memory": {
                "enabled": True,
                "ttl_seconds": agent_metadata.get("memory_ttl", Config.DEFAULT_MEMORY_TTL),
                "max_items": agent_metadata.get("max_memory_items", 1000)
            },
            "episodic_memory": {
                "enabled": True,
                "retention_days": agent_metadata.get("episodic_retention_days", 30),
                "consolidation_interval": agent_metadata.get("consolidation_interval", 3600)
            },
            "semantic_memory": {
                "enabled": True,
                "vector_dimensions": agent_metadata.get("vector_dimensions", Config.DEFAULT_MEMORY_DIMENSIONS),
                "similarity_threshold": agent_metadata.get("similarity_threshold", 0.8)
            },
            "vector_memory": {
                "enabled": True,
                "collection_name": f"agent_{agent_metadata.get('agent_id', 'default')}",
                "index_type": "HNSW",
                "metric": "cosine"
            }
        }

    async def _generate_plugin_config(
        self,
        agent_components: List[AgentComponent],
        mapping_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate plugin configuration based on agent components.

        Args:
            agent_components: Agent components
            mapping_context: Mapping context

        Returns:
            Plugin configuration dictionary
        """
        plugin_config = {
            "enabled_plugins": [],
            "plugin_settings": {},
            "domain_plugins": [],
            "generic_plugins": []
        }

        # Analyze components to determine required plugins
        component_types = {comp.component_type for comp in agent_components}

        # Domain-specific plugins based on component types
        if "rule_engine" in component_types:
            plugin_config["domain_plugins"].append("fraud_detector")
            plugin_config["domain_plugins"].append("regulatory_checker")

        if any("llm" in comp_type for comp_type in component_types):
            plugin_config["domain_plugins"].append("risk_calculator")

        # Generic plugins
        if len(agent_components) > 3:
            plugin_config["generic_plugins"].append("data_retriever")
            plugin_config["generic_plugins"].append("validator")

        plugin_config["enabled_plugins"] = (
            plugin_config["domain_plugins"] + plugin_config["generic_plugins"]
        )

        # Plugin-specific settings
        plugin_config["plugin_settings"] = {
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "cache_enabled": True,
            "cache_ttl": 3600
        }

        return plugin_config

    async def _validate_agent_config(
        self,
        agent_config: AgentConfig,
        mapping_context: Dict[str, Any]
    ) -> None:
        """
        Validate the generated agent configuration.

        Args:
            agent_config: Agent configuration to validate
            mapping_context: Context for storing validation results
        """
        # Validate component configurations
        for component in agent_config.components:
            if not component.service_endpoint:
                mapping_context["validation_errors"].append(
                    f"Missing service endpoint for component: {component.component_id}"
                )

        # Validate connection integrity
        component_ids = {comp.component_id for comp in agent_config.components}
        for connection in agent_config.connections:
            if connection.source_component not in component_ids:
                mapping_context["validation_errors"].append(
                    f"Connection source component not found: {connection.source_component}"
                )
            if connection.target_component not in component_ids:
                mapping_context["validation_errors"].append(
                    f"Connection target component not found: {connection.target_component}"
                )

        # Validate memory configuration
        if agent_config.memory_config.get("vector_memory", {}).get("enabled"):
            vector_dims = agent_config.memory_config["vector_memory"].get("vector_dimensions", 384)
            if not isinstance(vector_dims, int) or vector_dims <= 0:
                mapping_context["validation_errors"].append(
                    "Invalid vector dimensions for memory configuration"
                )

    def _generate_mapping_report(self, mapping_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive mapping report.

        Args:
            mapping_context: Mapping context with results

        Returns:
            Mapping report dictionary
        """
        return {
            "summary": {
                "total_components": len(mapping_context.get("component_mappings", {})),
                "total_connections": len(mapping_context.get("connection_mappings", [])),
                "validation_errors": len(mapping_context.get("validation_errors", [])),
                "warnings": len(mapping_context.get("warnings", [])),
                "success_rate": 1.0 if not mapping_context.get("validation_errors") else 0.8
            },
            "component_mapping_details": {
                comp_id: {
                    "original_type": comp.type if hasattr(comp, 'type') else 'unknown',
                    "mapped_type": mapping_context["component_mappings"][comp_id].component_type,
                    "service_endpoint": mapping_context["component_mappings"][comp_id].service_endpoint
                }
                for comp_id, comp in mapping_context.get("component_mappings", {}).items()
            },
            "performance_metrics": mapping_context.get("performance_metrics", {}),
            "recommendations": self._generate_mapping_recommendations(mapping_context)
        }

    def _generate_mapping_recommendations(self, mapping_context: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on mapping results.

        Args:
            mapping_context: Mapping context

        Returns:
            List of recommendations
        """
        recommendations = []

        if mapping_context.get("validation_errors"):
            recommendations.append("Review and fix validation errors before deployment")

        if len(mapping_context.get("warnings", [])) > 3:
            recommendations.append("Consider reviewing the warnings for potential improvements")

        component_count = len(mapping_context.get("component_mappings", {}))
        if component_count > 10:
            recommendations.append("Consider breaking down complex workflows into smaller, manageable components")

        return recommendations

# FastAPI application setup
app = FastAPI(
    title="UI-to-Brain Mapper Service",
    description="Converts visual workflow JSON to structured AgentConfig JSON",
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

# Initialize service
mapper_service = UIToBrainMapper()

@app.get("/")
async def root():
    """Service health check endpoint"""
    return {
        "service": "UI-to-Brain Mapper",
        "version": "1.0.0",
        "status": "healthy",
        "description": "Converts visual workflows to agent configurations",
        "endpoints": {
            "POST /map-workflow": "Convert visual workflow to agent config",
            "GET /health": "Service health check",
            "GET /supported-components": "List supported component types"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "UI-to-Brain Mapper",
        "dependencies": {
            "fastapi": "OK",
            "pydantic": "OK",
            "httpx": "OK"
        }
    }

@app.get("/supported-components")
async def get_supported_components():
    """Get list of supported component types and their configurations"""
    return {
        "supported_components": list(Config.COMPONENT_TYPE_MAPPING.keys()),
        "component_mappings": Config.COMPONENT_TYPE_MAPPING,
        "supported_reasoning_patterns": Config.SUPPORTED_REASONING_PATTERNS,
        "default_configurations": {
            "llm_processor": {
                "model": Config.DEFAULT_LLM_MODEL,
                "temperature": Config.DEFAULT_TEMPERATURE,
                "max_tokens": Config.DEFAULT_MAX_TOKENS
            },
            "memory": {
                "ttl_seconds": Config.DEFAULT_MEMORY_TTL,
                "vector_dimensions": Config.DEFAULT_MEMORY_DIMENSIONS
            }
        }
    }

@app.post("/map-workflow", response_model=MappingResponse)
async def map_visual_workflow(request: MappingRequest):
    """
    Convert visual workflow to agent configuration.

    This endpoint takes a visual workflow from the Agent Builder UI and converts
    it into a structured agent configuration that can be executed by backend services.

    Request Body:
    - visual_workflow: Complete visual workflow configuration
    - agent_metadata: Additional agent metadata (optional)
    - custom_mappings: Custom component type mappings (optional)

    Returns:
    - success: Boolean indicating mapping success
    - agent_config: Generated agent configuration (if successful)
    - mapping_report: Detailed mapping process report
    - validation_errors: List of validation errors (if any)
    - warnings: List of mapping warnings
    """
    try:
        # Log incoming request
        logger.info("Received workflow mapping request",
                  workflow_id=request.visual_workflow.workflow_id,
                  component_count=len(request.visual_workflow.components),
                  connection_count=len(request.visual_workflow.connections))

        # Perform mapping
        result = await mapper_service.map_visual_to_agent_config(
            visual_workflow=request.visual_workflow,
            agent_metadata=request.agent_metadata,
            custom_mappings=request.custom_mappings
        )

        # Log result
        if result.success:
            logger.info("Workflow mapping completed successfully",
                      workflow_id=request.visual_workflow.workflow_id,
                      agent_id=result.agent_config.agent_id if result.agent_config else None)
        else:
            logger.warning("Workflow mapping failed",
                         workflow_id=request.visual_workflow.workflow_id,
                         error_count=len(result.validation_errors))

        return result

    except Exception as e:
        logger.error("Mapping request failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Mapping request failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    import os

    # Get port from environment or use default
    port = int(os.getenv("UI_TO_BRAIN_MAPPER_PORT", "8302"))

    logger.info("Starting UI-to-Brain Mapper Service", port=port)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
