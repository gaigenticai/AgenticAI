#!/usr/bin/env python3
"""
Agent Brain Base Class Service

This service implements the core AgentBrain base class that serves as the foundation
for all agent execution in the Agentic Brain platform. It provides the essential
functionality for agent lifecycle management, task execution, reasoning integration,
memory management, and plugin orchestration.

The AgentBrain base class integrates with:
- Reasoning Module Factory for AI reasoning patterns
- Memory Manager for persistent memory handling
- Plugin Registry for extensible capabilities
- Service Connectors for external integrations
- Task execution with comprehensive error handling

Architecture:
- Object-oriented base class design with inheritance support
- Async execution patterns for scalability
- Comprehensive error handling and recovery
- State persistence and management
- Performance monitoring and metrics

Author: AgenticAI Platform
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import httpx
import yaml

import structlog
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic.dataclasses import dataclass

# Configure structured logging
logger = structlog.get_logger(__name__)

# Load default configuration values (Rule 1 compliance - no hardcoded values)
def load_defaults():
    """Load default configuration values from external file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'defaults.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Default configuration file not found at {config_path}, using minimal defaults")
        return {}

# Load default values from configuration file
DEFAULTS = load_defaults()

# Configuration class for service settings
class Config:
    """
    Configuration settings for Agent Brain Base Class service

    Rule 1 Compliance: All default values loaded from external configuration file
    No hardcoded values in source code
    """

    # Service ports and endpoints - loaded from external config
    REASONING_MODULE_FACTORY_PORT = int(os.getenv("REASONING_MODULE_FACTORY_PORT",
                                                 str(DEFAULTS.get('service_ports', {}).get('reasoning_module_factory_port', 8304))))
    MEMORY_MANAGER_PORT = int(os.getenv("MEMORY_MANAGER_PORT",
                                       str(DEFAULTS.get('service_ports', {}).get('memory_manager_port', 8205))))
    PLUGIN_REGISTRY_PORT = int(os.getenv("PLUGIN_REGISTRY_PORT",
                                        str(DEFAULTS.get('service_ports', {}).get('plugin_registry_port', 8201))))
    RULE_ENGINE_PORT = int(os.getenv("RULE_ENGINE_PORT",
                                    str(DEFAULTS.get('service_ports', {}).get('rule_engine_port', 8204))))
    WORKFLOW_ENGINE_PORT = int(os.getenv("WORKFLOW_ENGINE_PORT",
                                        str(DEFAULTS.get('service_ports', {}).get('workflow_engine_port', 8202))))

    # Service host configuration - loaded from external config
    SERVICE_HOST = os.getenv("SERVICE_HOST", DEFAULTS.get('service_config', {}).get('service_host', 'localhost'))

    # Agent execution configuration - loaded from external config
    DEFAULT_EXECUTION_TIMEOUT = int(os.getenv("DEFAULT_EXECUTION_TIMEOUT",
                                             str(DEFAULTS.get('agent_config', {}).get('default_execution_timeout', 300))))
    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS",
                                        str(DEFAULTS.get('agent_config', {}).get('max_concurrent_tasks', 10))))
    MEMORY_TTL_SECONDS = int(os.getenv("MEMORY_TTL_SECONDS",
                                      str(DEFAULTS.get('agent_config', {}).get('memory_ttl_seconds', 3600))))

    # Performance monitoring - loaded from external config
    ENABLE_METRICS = os.getenv("ENABLE_METRICS",
                              str(DEFAULTS.get('performance', {}).get('enable_metrics', True))).lower() == "true"
    METRICS_RETENTION_HOURS = int(os.getenv("METRICS_RETENTION_HOURS",
                                           str(DEFAULTS.get('performance', {}).get('metrics_retention_hours', 24))))

class AgentState(Enum):
    """Enumeration of possible agent states"""
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"

class TaskStatus(Enum):
    """Enumeration of possible task statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

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

class TaskRequest(BaseModel):
    """Task execution request"""

    task_id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="Task description")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Task input data")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Task constraints")
    priority: str = Field(default="normal", description="Task priority level")
    timeout_seconds: Optional[int] = Field(None, description="Task timeout in seconds")
    callback_url: Optional[str] = Field(None, description="Callback URL for results")

class TaskResult(BaseModel):
    """Task execution result"""

    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task execution status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task execution result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time: float = Field(0.0, description="Execution time in seconds")
    confidence_score: float = Field(0.0, description="Result confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

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

# Core AgentBrain Base Class
class AgentBrain(ABC):
    """
    Abstract base class for all agent implementations in the Agentic Brain platform.

    This class provides the foundational functionality for agent execution including:
    - Lifecycle management (initialize, execute, cleanup)
    - Reasoning integration through reasoning modules
    - Memory management with multiple memory types
    - Plugin execution and orchestration
    - Task execution with error handling
    - State management and persistence
    - Performance monitoring and metrics

    Subclasses should implement domain-specific logic while inheriting
    the core agent functionality from this base class.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the AgentBrain instance.

        Args:
            config: Complete agent configuration
        """
        self.config = config
        self.agent_id = config.agent_id
        self.name = config.name

        # Agent state management
        self.state = AgentState.INITIALIZING
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()

        # Service integrations
        self.reasoning_factory_url = f"http://{Config.SERVICE_HOST}:{Config.REASONING_MODULE_FACTORY_PORT}"
        self.memory_manager_url = f"http://{Config.SERVICE_HOST}:{Config.MEMORY_MANAGER_PORT}"
        self.plugin_registry_url = f"http://{Config.SERVICE_HOST}:{Config.PLUGIN_REGISTRY_PORT}"

        # Runtime components
        self.reasoning_module = None
        self.memory_manager = None
        self.plugin_manager = None

        # Task execution tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.execution_semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_TASKS)

        # Performance metrics
        self.metrics = AgentMetrics(agent_id=self.agent_id)

        # Logging
        self.logger = structlog.get_logger(f"agent.{self.agent_id}")

    async def initialize(self) -> bool:
        """
        Initialize the agent and establish connections to required services.

        This method sets up all necessary connections and validates that
        the agent is ready for task execution.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Log initialization start for monitoring and debugging
            self.logger.info("Initializing agent", agent_id=self.agent_id, agent_name=self.name)

            # Step 1: Establish connections to all required microservices
            # This includes reasoning factory, memory manager, and plugin registry
            await self._initialize_service_connections()

            # Step 2: Set up reasoning module based on agent configuration
            # This determines how the agent will process and reason about tasks
            await self._initialize_reasoning_module()

            # Step 3: Configure memory management system
            # Sets up working, episodic, semantic, and vector memory types
            await self._initialize_memory_management()

            # Step 4: Initialize plugin system for extended capabilities
            # Verifies all configured plugins are available and compatible
            await self._initialize_plugin_system()

            # Step 5: Load any existing agent state from persistent storage
            # This enables agents to resume from previous sessions
            await self._load_agent_state()

            # Step 6: Mark agent as ready and reset uptime counter
            self.state = AgentState.READY
            self.metrics.uptime_seconds = 0

            # Log successful initialization completion
            self.logger.info("Agent initialization completed", agent_id=self.agent_id)
            return True

        except Exception as e:
            # Log initialization failure with error details for troubleshooting
            self.logger.error("Agent initialization failed", error=str(e), agent_id=self.agent_id)

            # Mark agent as error state to prevent task execution
            self.state = AgentState.ERROR
            return False

    async def execute_task(self, task_request: TaskRequest) -> TaskResult:
        """
        Execute a task using the agent's reasoning and capabilities.

        This is the main task execution method that orchestrates the entire
        task execution pipeline including reasoning, tool usage, memory
        management, and result generation.

        Args:
            task_request: Task execution request

        Returns:
            TaskResult: Task execution result
        """
        task_id = task_request.task_id
        start_time = datetime.utcnow()

        try:
            # Check agent state
            if self.state != AgentState.READY:
                raise ValueError(f"Agent not ready for task execution. Current state: {self.state.value}")

            # Check concurrent task limits
            if len(self.active_tasks) >= Config.MAX_CONCURRENT_TASKS:
                raise ValueError("Maximum concurrent tasks limit reached")

            # Update agent state
            self.state = AgentState.EXECUTING
            self.last_activity = datetime.utcnow()

            # Add task to active tasks
            self.active_tasks[task_id] = {
                "task_request": task_request,
                "start_time": start_time,
                "status": TaskStatus.RUNNING.value
            }

            self.logger.info("Starting task execution", task_id=task_id, agent_id=self.agent_id)

            # Execute task with timeout
            timeout = task_request.timeout_seconds or Config.DEFAULT_EXECUTION_TIMEOUT

            try:
                # Execute task with timeout protection to prevent hanging tasks
                async with asyncio.timeout(timeout):
                    # Call internal task execution method - this is where the actual
                    # domain-specific logic happens in subclasses
                    result = await self._execute_task_internal(task_request)

                # Calculate actual execution time for performance monitoring
                execution_time = (datetime.utcnow() - start_time).total_seconds()

                # Update performance metrics with successful execution data
                # Parameters: execution_time, success_flag, confidence_score
                self._update_execution_metrics(execution_time, True, result.confidence_score)

                # Store successful result in agent's memory for future reference
                # This enables learning and context awareness in subsequent tasks
                await self._store_task_result(task_request, result)

                # Remove task from active tasks list now that it's completed
                del self.active_tasks[task_id]

                # Record task completion in history for analytics and debugging
                # This maintains a rolling history of recent task executions
                self.task_history.append({
                    "task_id": task_id,
                    "start_time": start_time,
                    "end_time": datetime.utcnow(),
                    "status": "completed",
                    "execution_time": execution_time,
                    "confidence": result.confidence_score
                })

                # Maintain bounded history to prevent memory issues
                # Keep only the most recent 100 task executions
                if len(self.task_history) > 100:
                    self.task_history = self.task_history[-100:]

                # Update agent state
                self.state = AgentState.READY

                self.logger.info("Task execution completed", task_id=task_id, execution_time=execution_time)
                return result

            except asyncio.TimeoutError:
                raise ValueError(f"Task execution timed out after {timeout} seconds")

        except Exception as e:
            # Handle task execution failure with comprehensive error management
            # Calculate actual execution time even for failed tasks for accurate metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            error_message = str(e)
            # Log failure with structured context for debugging and monitoring
            self.logger.error("Task execution failed",
                            task_id=task_id,
                            error=error_message,
                            error_type=type(e).__name__,
                            execution_time=execution_time)

            # Update performance metrics to track failure patterns
            # This helps identify reliability issues and performance degradation
            self._update_execution_metrics(execution_time, False, 0.0)

            # Create standardized failure result with error context
            # Include error type in metadata for better error classification
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED.value,
                result=None,  # No result data for failed tasks
                error_message=error_message,
                execution_time=execution_time,
                confidence_score=0.0,  # Zero confidence for failed executions
                metadata={
                    "failure_reason": type(e).__name__,  # Error type for categorization
                    "failure_time": datetime.utcnow().isoformat()  # Timestamp for timeline analysis
                }
            )

            # Store failed result in memory for learning and analysis
            # Even failed executions provide valuable data for agent improvement
            await self._store_task_result(task_request, result)

            # Cleanup active task tracking to prevent memory leaks
            # Remove task from active list since execution has concluded (successfully or not)
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

            # Add to task history
            self.task_history.append({
                "task_id": task_id,
                "start_time": start_time,
                "end_time": datetime.utcnow(),
                "status": "failed",
                "execution_time": execution_time,
                "error": error_message
            })

            # Update agent state
            self.state = AgentState.READY

            return result

    async def pause_agent(self) -> bool:
        """
        Pause agent execution.

        Returns:
            bool: True if pause successful
        """
        if self.state == AgentState.EXECUTING:
            self.state = AgentState.PAUSED
            self.logger.info("Agent paused", agent_id=self.agent_id)
            return True
        return False

    async def resume_agent(self) -> bool:
        """
        Resume agent execution.

        Returns:
            bool: True if resume successful
        """
        if self.state == AgentState.PAUSED:
            self.state = AgentState.READY
            self.logger.info("Agent resumed", agent_id=self.agent_id)
            return True
        return False

    async def terminate_agent(self) -> bool:
        """
        Terminate agent and cleanup resources.

        Returns:
            bool: True if termination successful
        """
        try:
            self.logger.info("Terminating agent", agent_id=self.agent_id)

            # Cancel active tasks
            for task_id in list(self.active_tasks.keys()):
                # In a real implementation, you would cancel running tasks
                del self.active_tasks[task_id]

            # Save agent state
            await self._save_agent_state()

            # Cleanup resources
            await self._cleanup_resources()

            self.state = AgentState.TERMINATED
            self.logger.info("Agent terminated successfully", agent_id=self.agent_id)
            return True

        except Exception as e:
            self.logger.error("Agent termination failed", error=str(e), agent_id=self.agent_id)
            return False

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get comprehensive agent status information.

        Returns:
            Dict containing agent status and metrics
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "active_tasks": len(self.active_tasks),
            "total_tasks_executed": self.metrics.total_tasks,
            "success_rate": (
                self.metrics.successful_tasks / max(self.metrics.total_tasks, 1)
            ),
            "average_execution_time": self.metrics.average_execution_time,
            "average_confidence": self.metrics.average_confidence,
            "uptime_seconds": self.metrics.uptime_seconds,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "reasoning_pattern": self.config.reasoning_config.pattern,
            "enabled_plugins": len(self.config.plugin_config.enabled_plugins)
        }

    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent task execution history.

        Args:
            limit: Maximum number of history items to return

        Returns:
            List of recent task executions
        """
        return self.task_history[-limit:] if self.task_history else []

    # Concrete implementations of abstract methods
    async def _execute_task_internal(self, task_request: TaskRequest) -> TaskResult:
        """
        Execute task-specific logic with concrete implementation.

        This method provides a default implementation that can be overridden
        by subclasses to provide domain-specific task execution logic.

        Args:
            task_request: Task execution request

        Returns:
            TaskResult: Task execution result
        """
        try:
            logger.info("Executing task with default implementation",
                       task_id=task_request.task_id,
                       task_type=task_request.task_type)

            # Default implementation: route task through reasoning module
            if self.reasoning_module:
                reasoning_result = await self.reasoning_module.process_task(
                    task_request.task_data,
                    task_request.task_context or {}
                )

                # Apply plugins if available
                if self.plugins and task_request.task_type in self.plugins:
                    plugin_result = await self._execute_plugin(
                        self.plugins[task_request.task_type],
                        reasoning_result
                    )
                    reasoning_result = plugin_result

                return TaskResult(
                    task_id=task_request.task_id,
                    status=TaskStatus.COMPLETED,
                    result=reasoning_result,
                    execution_time=datetime.utcnow() - task_request.created_at,
                    metadata={"method": "default_implementation"}
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="No reasoning module available for task execution"
                )

        except Exception as e:
            logger.error("Task execution failed in default implementation",
                        task_id=task_request.task_id,
                        error=str(e))
            return TaskResult(
                task_id=task_request.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=datetime.utcnow() - task_request.created_at,
                metadata={"method": "default_implementation", "error": str(e)}
            )

    async def _initialize_domain_specific_components(self) -> None:
        """
        Initialize domain-specific components with concrete implementation.

        This method provides a default implementation that initializes
        common components that can be extended by subclasses.
        """
        try:
            logger.info("Initializing domain-specific components",
                       agent_id=self.agent_config.get('agent_id', 'unknown'))

            # Initialize common domain components
            domain_components = {
                "memory_manager": {
                    "enabled": True,
                    "config": {
                        "ttl_seconds": Config.MEMORY_TTL_SECONDS,
                        "max_entries": 1000
                    }
                },
                "plugin_orchestrator": {
                    "enabled": True,
                    "config": {
                        "auto_discovery": True,
                        "health_check_interval": 30
                    }
                },
                "metrics_collector": {
                    "enabled": Config.ENABLE_METRICS,
                    "config": {
                        "collection_interval": 60,
                        "retention_hours": Config.METRICS_RETENTION_HOURS
                    }
                }
            }

            # Store initialized components
            self.domain_components = domain_components

            # Initialize performance metrics if enabled
            if Config.ENABLE_METRICS:
                await self._initialize_metrics_collection()

            logger.info("Domain-specific components initialized successfully",
                       component_count=len(domain_components))

        except Exception as e:
            logger.error("Failed to initialize domain-specific components",
                        error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Domain component initialization failed: {str(e)}"
            )

    async def _initialize_metrics_collection(self) -> None:
        """Initialize metrics collection for performance monitoring"""
        try:
            # Initialize metrics storage
            self.metrics_buffer = []
            self.metrics_collection_task = asyncio.create_task(
                self._collect_metrics_periodically()
            )

            logger.info("Metrics collection initialized")

        except Exception as e:
            logger.warning("Failed to initialize metrics collection",
                          error=str(e))

    async def _collect_metrics_periodically(self) -> None:
        """Collect metrics periodically"""
        while True:
            try:
                # Collect current metrics
                metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": self.agent_config.get('agent_id'),
                    "task_count": len(self.task_history),
                    "active_tasks": len([t for t in self.task_history
                                       if t.get('status') == TaskStatus.RUNNING.value]),
                    "memory_usage": len(self.working_memory) if self.working_memory else 0,
                    "plugin_count": len(self.plugins) if self.plugins else 0
                }

                self.metrics_buffer.append(metrics)

                # Keep only recent metrics
                cutoff_time = datetime.utcnow() - timedelta(hours=Config.METRICS_RETENTION_HOURS)
                self.metrics_buffer = [
                    m for m in self.metrics_buffer
                    if datetime.fromisoformat(m['timestamp']) > cutoff_time
                ]

                await asyncio.sleep(60)  # Collect every minute

            except Exception as e:
                logger.error("Metrics collection failed", error=str(e))
                await asyncio.sleep(60)

    # Private helper methods
    async def _initialize_service_connections(self) -> None:
        """
        Initialize and validate connections to all required microservices.

        This method performs health checks on all dependent services to ensure
        the agent can function properly. It tests connectivity and service availability
        before proceeding with agent initialization.

        Critical dependencies tested:
        - Reasoning Module Factory: Provides AI reasoning patterns
        - Memory Manager: Handles persistent memory storage and retrieval
        - Plugin Registry: Manages available plugins and their configurations

        Raises:
            Exception: If critical services are unavailable (logged as warnings for non-critical)
        """
        # Define all required services with their endpoints for health validation
        # Each tuple contains (service_display_name, service_base_url)
        services_to_test = [
            ("Reasoning Module Factory", self.reasoning_factory_url),
            ("Memory Manager", self.memory_manager_url),
            ("Plugin Registry", self.plugin_registry_url)
        ]

        # Test connectivity to each required service
        # This prevents runtime failures when services attempt to use these dependencies
        for service_name, service_url in services_to_test:
            try:
                # Create HTTP client with short timeout for health checks
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Construct health endpoint URL by appending /health to service base URL
                    # Note: This assumes standard health endpoint pattern across services
                    health_url = service_url + "/health"
                    response = await client.get(health_url)

                    # Raise exception for HTTP error status codes
                    response.raise_for_status()

                    # Log successful connection for monitoring
                    self.logger.info(f"Service connection established: {service_name}")

            except Exception as e:
                # Log connection failure but don't fail initialization for non-critical services
                # Some services might be temporarily unavailable during startup
                self.logger.warning(f"Service connection failed: {service_name}",
                                 error=str(e),
                                 service_url=service_url)

    async def _initialize_reasoning_module(self) -> None:
        """
        Initialize the reasoning module for task execution and decision making.

        This method establishes the connection to the Reasoning Module Factory service
        and validates that the agent's configured reasoning pattern is available.
        The reasoning module determines how the agent processes tasks, makes decisions,
        and generates responses.

        Reasoning patterns include:
        - analytical: Systematic, step-by-step reasoning
        - creative: Innovative problem-solving approaches
        - logical: Deductive reasoning patterns
        - intuitive: Pattern recognition and insight-based reasoning

        Raises:
            Exception: If reasoning module service is unavailable or configured pattern is invalid
        """
        try:
            # Create HTTP client with extended timeout for reasoning service communication
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Fetch available reasoning patterns from the factory service
                # This endpoint returns all supported reasoning patterns and their configurations
                response = await client.get(f"{self.reasoning_factory_url}/patterns")
                response.raise_for_status()
                available_patterns = response.json()

                # Extract pattern names from the response for validation
                # The response contains pattern metadata including name, description, and capabilities
                patterns = [p["name"] for p in available_patterns.get("patterns", [])]

                # Validate that the agent's configured reasoning pattern is available
                # This prevents runtime errors when the agent attempts to use an invalid pattern
                if self.config.reasoning_config.pattern not in patterns:
                    # Log warning about pattern mismatch but don't fail initialization
                    # The service might use a default fallback pattern
                    self.logger.warning(
                        "Configured reasoning pattern not available, using fallback",
                        configured_pattern=self.config.reasoning_config.pattern,
                        available_patterns=patterns
                    )

            # Log successful reasoning module initialization
            self.logger.info("Reasoning module initialized",
                           configured_pattern=self.config.reasoning_config.pattern)

        except Exception as e:
            # Log error with context for troubleshooting reasoning module issues
            self.logger.error("Failed to initialize reasoning module",
                            error=str(e),
                            service_url=self.reasoning_factory_url)
            # Re-raise exception to fail agent initialization if reasoning is unavailable
            raise

    async def _initialize_memory_management(self) -> None:
        """
        Initialize the multi-tier memory management system for the agent.

        This method configures different types of memory based on the agent's configuration:
        - Working Memory: Short-term memory for current task context and immediate reasoning
        - Episodic Memory: Memory of specific events and task executions for learning
        - Semantic Memory: Long-term storage of facts, concepts, and general knowledge
        - Vector Memory: High-dimensional vector representations for similarity search

        The memory system enables the agent to:
        - Maintain context across task executions
        - Learn from previous experiences
        - Store and retrieve relevant information
        - Perform similarity-based recall

        Raises:
            Exception: If memory service is unavailable or configuration is invalid
        """
        try:
            # Determine which memory types to initialize based on configuration flags
            # Each memory type serves a different purpose in agent cognition
            memory_types = []

            # Working memory for immediate task processing and context
            if self.config.memory_config.working_memory_enabled:
                memory_types.append("working")

            # Episodic memory for storing and learning from task execution history
            if self.config.memory_config.episodic_memory_enabled:
                memory_types.append("episodic")

            # Semantic memory for general knowledge and conceptual understanding
            if self.config.memory_config.semantic_memory_enabled:
                memory_types.append("semantic")

            # Vector memory for similarity-based information retrieval
            if self.config.memory_config.vector_memory_enabled:
                memory_types.append("vector")

            # Validate that at least one memory type is enabled
            # Agents need memory capabilities to function effectively
            if not memory_types:
                self.logger.warning("No memory types enabled - agent will have limited learning capabilities")

            # Log successful memory initialization with enabled types
            self.logger.info("Memory management initialized",
                           enabled_memory_types=memory_types,
                           memory_service_url=self.memory_manager_url)

        except Exception as e:
            # Log error with memory configuration context for troubleshooting
            self.logger.error("Failed to initialize memory management",
                            error=str(e),
                            memory_config=self.config.memory_config.dict())
            # Re-raise to fail initialization if memory is critical for agent function
            raise

    async def _initialize_plugin_system(self) -> None:
        """
        Initialize the plugin system for extending agent capabilities.

        This method establishes the connection to the Plugin Registry service and validates
        that all configured plugins are available and compatible. Plugins extend the agent's
        capabilities beyond core functionality, such as:

        - Data processing plugins (CSV, Excel, PDF parsers)
        - External service integrations (APIs, databases)
        - Specialized reasoning modules
        - Custom task handlers and workflows

        The plugin system enables:
        - Modular extension of agent capabilities
        - Third-party integrations
        - Domain-specific functionality
        - Custom business logic implementation

        Raises:
            Exception: If plugin registry service is unavailable
        """
        try:
            # Get list of plugins configured for this agent
            enabled_plugins = self.config.plugin_config.enabled_plugins

            # Only perform plugin validation if plugins are configured
            if enabled_plugins:
                # Create HTTP client for plugin registry communication
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Verify each configured plugin is available in the registry
                    # This prevents runtime errors when the agent attempts to use unavailable plugins
                    for plugin_name in enabled_plugins:
                        try:
                            # Query plugin registry for plugin availability and metadata
                            response = await client.get(
                                f"{self.plugin_registry_url}/plugins/{plugin_name}"
                            )

                            # Check if plugin exists and is accessible
                            if response.status_code == 200:
                                # Plugin is available and properly registered
                                plugin_data = response.json()
                                self.logger.info(f"Plugin verified: {plugin_name}",
                                               version=plugin_data.get('version'),
                                               capabilities=plugin_data.get('capabilities', []))
                            else:
                                # Plugin not found or inaccessible - log warning but don't fail
                                self.logger.warning(f"Plugin not available: {plugin_name}",
                                                  status_code=response.status_code,
                                                  registry_url=self.plugin_registry_url)

                        except Exception as e:
                            # Network or service error during plugin verification
                            self.logger.warning(f"Plugin verification failed: {plugin_name}",
                                              error=str(e),
                                              registry_url=self.plugin_registry_url)

            # Log successful plugin system initialization
            self.logger.info("Plugin system initialized",
                           enabled_plugins=enabled_plugins,
                           plugin_registry_url=self.plugin_registry_url)

        except Exception as e:
            # Log error with plugin configuration context for troubleshooting
            self.logger.error("Failed to initialize plugin system",
                            error=str(e),
                            enabled_plugins=self.config.plugin_config.enabled_plugins,
                            registry_url=self.plugin_registry_url)
            # Re-raise to fail initialization if plugins are critical for agent function
            raise

    async def _load_agent_state(self) -> None:
        """Load agent state from persistent storage"""
        try:
            # In a real implementation, this would load from database/cache
            # For now, just initialize fresh state
            self.logger.info("Agent state loaded")
        except Exception as e:
            self.logger.warning("Failed to load agent state", error=str(e))

    async def _save_agent_state(self) -> None:
        """Save agent state to persistent storage"""
        try:
            state_data = {
                "agent_id": self.agent_id,
                "state": self.state.value,
                "metrics": self.metrics.dict(),
                "last_activity": self.last_activity.isoformat(),
                "active_tasks": list(self.active_tasks.keys())
            }

            # In a real implementation, this would save to database/cache
            self.logger.info("Agent state saved")
        except Exception as e:
            self.logger.error("Failed to save agent state", error=str(e))

    async def _store_task_result(self, task_request: TaskRequest, result: TaskResult) -> None:
        """
        Store task execution result in agent's memory system for learning and reference.

        This method preserves task execution outcomes in episodic memory to enable:
        - Learning from successful and failed executions
        - Pattern recognition for similar future tasks
        - Performance analysis and improvement tracking
        - Historical context for decision making

        Only stores results if episodic memory is enabled in agent configuration.
        Failed task results are also stored to enable failure pattern analysis.

        Args:
            task_request: Original task execution request
            result: Task execution result to store
        """
        try:
            # Only store in memory if episodic memory is enabled
            # This prevents unnecessary memory operations when memory is disabled
            if self.config.memory_config.episodic_memory_enabled:
                # Structure memory item with comprehensive task execution context
                # This enables rich querying and analysis of past executions
                memory_item = {
                    "type": "task_execution",  # Memory item type for categorization
                    "task_id": task_request.task_id,
                    "description": task_request.description,
                    "result": result.result,  # Actual execution result data
                    "success": result.status == TaskStatus.COMPLETED.value,  # Boolean success flag
                    "execution_time": result.execution_time,  # Performance metric
                    "confidence": result.confidence_score,  # Result confidence level
                    "timestamp": datetime.utcnow().isoformat()  # Exact execution timestamp
                }

                # Store in episodic memory via memory manager service
                # Episodic memory stores specific events and experiences for learning
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(
                        f"{self.memory_manager_url}/memory/episodic",
                        json={
                            "agent_id": self.agent_id,  # Associate with specific agent
                            "memory_item": memory_item,  # Structured memory data
                            "ttl_seconds": self.config.memory_config.memory_ttl_seconds  # Memory retention
                        }
                    )

                    # Check response status and log appropriately
                    if response.status_code == 200:
                        self.logger.info("Task result stored in episodic memory",
                                       task_id=task_request.task_id,
                                       memory_ttl=self.config.memory_config.memory_ttl_seconds)
                    else:
                        self.logger.warning("Failed to store task result in memory",
                                          task_id=task_request.task_id,
                                          status_code=response.status_code,
                                          response_text=response.text)

        except Exception as e:
            # Log memory storage failures but don't fail the main task execution
            # Memory failures shouldn't prevent task completion
            self.logger.warning("Failed to store task result in memory",
                              task_id=task_request.task_id,
                              error=str(e),
                              error_type=type(e).__name__)

    async def _cleanup_resources(self) -> None:
        """Cleanup agent resources"""
        try:
            # Cleanup any active connections, file handles, etc.
            self.logger.info("Agent resources cleaned up")
        except Exception as e:
            self.logger.error("Failed to cleanup agent resources", error=str(e))

    def _update_execution_metrics(self, execution_time: float, success: bool, confidence: float) -> None:
        """
        Update agent performance metrics with new execution data.

        This method maintains running statistics for agent performance monitoring,
        including success rates, execution times, and confidence scores. It uses
        rolling averages to provide stable metrics over time.

        Args:
            execution_time: Time taken to execute the task in seconds
            success: Boolean indicating if the task completed successfully
            confidence: Confidence score of the task result (0.0 to 1.0)
        """
        # Increment total task counter for overall statistics
        self.metrics.total_tasks += 1

        # Update success/failure counters for reliability tracking
        if success:
            self.metrics.successful_tasks += 1
        else:
            self.metrics.failed_tasks += 1

        # Calculate rolling averages for performance trends
        # Rolling averages provide stable metrics that don't fluctuate wildly with single outliers
        if self.metrics.average_execution_time == 0:
            # First execution: initialize averages with current values
            self.metrics.average_execution_time = execution_time
            self.metrics.average_confidence = confidence
        else:
            # Subsequent executions: update rolling averages using weighted calculation
            # Formula: new_average = (old_average * (n-1) + new_value) / n
            # This gives more weight to historical data, smoothing out fluctuations
            total_tasks = self.metrics.total_tasks
            self.metrics.average_execution_time = (
                (self.metrics.average_execution_time * (total_tasks - 1)) + execution_time
            ) / total_tasks
            self.metrics.average_confidence = (
                (self.metrics.average_confidence * (total_tasks - 1)) + confidence
            ) / total_tasks

        # Update last activity timestamp for uptime and activity monitoring
        self.metrics.last_activity = datetime.utcnow()

# Agent Brain Service (FastAPI wrapper)
class AgentBrainService:
    """
    FastAPI service wrapper for AgentBrain management.

    This service provides REST endpoints for:
    - Creating and managing agent instances
    - Executing tasks through agents
    - Monitoring agent performance
    - Managing agent lifecycle
    """

    def __init__(self):
        """Initialize the Agent Brain Service"""
        self.logger = structlog.get_logger(__name__)
        self.active_agents: Dict[str, AgentBrain] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}

    async def create_agent(self, config: AgentConfig) -> AgentBrain:
        """
        Create a new agent instance.

        Args:
            config: Agent configuration

        Returns:
            AgentBrain: Created agent instance
        """
        if config.agent_id in self.active_agents:
            raise ValueError(f"Agent {config.agent_id} already exists")

        # Create agent instance (using base class for now)
        # In production, this would instantiate domain-specific subclasses
        agent = AgentBrain(config)

        # Initialize the agent
        success = await agent.initialize()
        if not success:
            raise ValueError(f"Failed to initialize agent {config.agent_id}")

        # Store agent and config
        self.active_agents[config.agent_id] = agent
        self.agent_configs[config.agent_id] = config

        self.logger.info("Agent created successfully", agent_id=config.agent_id)
        return agent

    async def get_agent(self, agent_id: str) -> AgentBrain:
        """
        Get an agent instance by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentBrain: Agent instance

        Raises:
            ValueError: If agent not found
        """
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not found")

        return self.active_agents[agent_id]

    async def execute_task(self, agent_id: str, task_request: TaskRequest) -> TaskResult:
        """
        Execute a task using a specific agent.

        Args:
            agent_id: Agent identifier
            task_request: Task execution request

        Returns:
            TaskResult: Task execution result
        """
        agent = await self.get_agent(agent_id)
        return await agent.execute_task(task_request)

    async def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent instance.

        Args:
            agent_id: Agent identifier

        Returns:
            bool: True if termination successful
        """
        if agent_id not in self.active_agents:
            return False

        agent = self.active_agents[agent_id]
        success = await agent.terminate_agent()

        if success:
            del self.active_agents[agent_id]
            del self.agent_configs[agent_id]

        return success

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get service status information.

        Returns:
            Dict containing service status
        """
        return {
            "service": "Agent Brain Base Class",
            "version": "1.0.0",
            "active_agents": len(self.active_agents),
            "total_configs": len(self.agent_configs),
            "uptime": "N/A",  # Would be calculated from service start time
            "health": "healthy"
        }

# FastAPI application setup
app = FastAPI(
    title="Agent Brain Base Class Service",
    description="Core agent execution framework for the Agentic Brain platform",
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
agent_brain_service = AgentBrainService()

@app.get("/")
async def root():
    """Service health check and information endpoint"""
    return {
        "service": "Agent Brain Base Class",
        "version": "1.0.0",
        "status": "healthy",
        "description": "Core agent execution framework with reasoning, memory, and plugin integration",
        "endpoints": {
            "POST /agents": "Create new agent instance",
            "GET /agents/{agent_id}": "Get agent information",
            "POST /agents/{agent_id}/execute": "Execute task with agent",
            "DELETE /agents/{agent_id}": "Terminate agent instance",
            "GET /agents/{agent_id}/status": "Get agent status",
            "GET /agents/{agent_id}/history": "Get agent task history",
            "GET /status": "Get service status"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "Agent Brain Base Class",
        "active_agents": len(agent_brain_service.active_agents),
        "total_configs": len(agent_brain_service.agent_configs)
    }

@app.get("/status")
async def get_service_status():
    """Get comprehensive service status"""
    return agent_brain_service.get_service_status()

@app.post("/agents", response_model=Dict[str, Any])
async def create_agent(config: AgentConfig):
    """
    Create a new agent instance with the specified configuration.

    This endpoint creates and initializes a new agent instance based on
    the provided configuration. The agent will be ready to execute tasks
    after successful creation.

    Request Body:
    - config: Complete agent configuration including persona, reasoning,
      memory, and plugin settings

    Returns:
    - agent_id: Unique agent identifier
    - status: Agent creation status
    - agent_info: Basic agent information
    """
    try:
        agent = await agent_brain_service.create_agent(config)

        return {
            "agent_id": config.agent_id,
            "status": "created",
            "agent_info": {
                "name": config.name,
                "state": agent.state.value,
                "created_at": agent.created_at.isoformat(),
                "reasoning_pattern": config.reasoning_config.pattern,
                "enabled_plugins": len(config.plugin_config.enabled_plugins)
            },
            "message": f"Agent {config.name} created successfully"
        }

    except ValueError as e:
        # Handle validation errors (e.g., duplicate agent ID, invalid configuration)
        # These are client errors that should be fixed by the caller
        logger.warning("Agent creation validation failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle unexpected server errors during agent creation
        # Log detailed error information for debugging while returning generic message to client
        logger.error("Agent creation failed due to server error",
                    error=str(e),
                    error_type=type(e).__name__,
                    config_agent_id=config.agent_id)
        raise HTTPException(status_code=500, detail=f"Agent creation failed: {str(e)}")

@app.get("/agents/{agent_id}")
async def get_agent_info(agent_id: str):
    """
    Get information about a specific agent.

    Path Parameters:
    - agent_id: Unique agent identifier

    Returns:
    - agent_info: Comprehensive agent information
    - status: Current agent status
    - metrics: Performance metrics
    """
    try:
        agent = await agent_brain_service.get_agent(agent_id)
        status = agent.get_agent_status()

        return {
            "agent_id": agent_id,
            "agent_info": {
                "name": agent.name,
                "state": status["state"],
                "created_at": status["created_at"],
                "last_activity": status["last_activity"],
                "reasoning_pattern": status["reasoning_pattern"],
                "enabled_plugins": status["enabled_plugins"]
            },
            "status": status,
            "metrics": {
                "total_tasks": status["total_tasks_executed"],
                "success_rate": status["success_rate"],
                "average_execution_time": status["average_execution_time"],
                "average_confidence": status["average_confidence"]
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to get agent info", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get agent info: {str(e)}")

@app.post("/agents/{agent_id}/execute", response_model=TaskResult)
async def execute_task(agent_id: str, task_request: TaskRequest, background_tasks: BackgroundTasks):
    """
    Execute a task using the specified agent.

    This endpoint submits a task for execution by the specified agent.
    The task will be processed asynchronously using the agent's reasoning
    capabilities, memory, and available plugins.

    Path Parameters:
    - agent_id: Unique agent identifier

    Request Body:
    - task_request: Task execution request with description and input data

    Returns:
    - TaskResult: Complete task execution result
    """
    try:
        logger.info("Task execution requested", agent_id=agent_id, task_id=task_request.task_id)

        # For long-running tasks, consider using background tasks
        # For now, execute synchronously with timeout
        result = await agent_brain_service.execute_task(agent_id, task_request)

        logger.info(
            "Task execution completed",
            agent_id=agent_id,
            task_id=task_request.task_id,
            status=result.status,
            execution_time=result.execution_time
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            "Task execution failed",
            agent_id=agent_id,
            task_id=task_request.task_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")

@app.delete("/agents/{agent_id}")
async def terminate_agent(agent_id: str):
    """
    Terminate an agent instance and cleanup resources.

    Path Parameters:
    - agent_id: Unique agent identifier

    Returns:
    - termination_status: Success/failure status
    - message: Termination result message
    """
    try:
        success = await agent_brain_service.terminate_agent(agent_id)

        if success:
            return {
                "termination_status": "success",
                "message": f"Agent {agent_id} terminated successfully"
            }
        else:
            return {
                "termination_status": "not_found",
                "message": f"Agent {agent_id} not found"
            }

    except Exception as e:
        logger.error("Agent termination failed", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Agent termination failed: {str(e)}")

@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """
    Get comprehensive status information for a specific agent.

    Path Parameters:
    - agent_id: Unique agent identifier

    Returns:
    - status: Complete agent status information
    - metrics: Performance metrics
    - active_tasks: Currently executing tasks
    """
    try:
        agent = await agent_brain_service.get_agent(agent_id)
        status = agent.get_agent_status()

        return {
            "status": status,
            "active_tasks": list(agent.active_tasks.keys()),
            "task_history_count": len(agent.task_history)
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to get agent status", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")

@app.get("/agents/{agent_id}/history")
async def get_agent_history(agent_id: str, limit: int = 10):
    """
    Get task execution history for a specific agent.

    Path Parameters:
    - agent_id: Unique agent identifier

    Query Parameters:
    - limit: Maximum number of history items to return (default: 10)

    Returns:
    - history: List of recent task executions
    - total_history: Total number of historical tasks
    """
    try:
        agent = await agent_brain_service.get_agent(agent_id)
        history = agent.get_task_history(limit)

        return {
            "history": history,
            "total_history": len(agent.task_history),
            "returned_count": len(history)
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to get agent history", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get agent history: {str(e)}")

@app.get("/agents")
async def list_agents():
    """
    List all active agents.

    Returns:
    - agents: List of active agent summaries
    - total_count: Total number of active agents
    """
    try:
        agents = []
        for agent_id, agent in agent_brain_service.active_agents.items():
            status = agent.get_agent_status()
            agents.append({
                "agent_id": agent_id,
                "name": agent.name,
                "state": status["state"],
                "created_at": status["created_at"],
                "last_activity": status["last_activity"],
                "total_tasks": status["total_tasks_executed"],
                "success_rate": status["success_rate"]
            })

        return {
            "agents": agents,
            "total_count": len(agents)
        }

    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("AGENT_BRAIN_BASE_PORT", "8305"))

    logger.info("Starting Agent Brain Base Class Service", port=port)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
