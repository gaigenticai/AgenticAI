#!/usr/bin/env python3
"""
Reasoning Module Factory Service

This service implements a comprehensive factory pattern for creating AI reasoning modules
that power the Agentic Brain platform. It provides multiple reasoning patterns including:

- ReAct: Reasoning + Acting pattern for general-purpose agent reasoning
- Reflection: Self-assessment and improvement through metacognition
- Planning: Multi-step task decomposition and execution planning
- Multi-Agent: Coordination and collaboration between multiple agents

The factory uses dependency injection to ensure modularity and testability,
allowing for easy extension with new reasoning patterns and seamless integration
with different LLM providers and external services.

Architecture:
- Factory pattern with dependency injection
- Modular reasoning pattern implementations
- Async processing for scalability
- Comprehensive error handling and logging
- Configuration-driven pattern selection

Author: AgenticAI Platform
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import httpx

import structlog
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic.dataclasses import dataclass

# Configure structured logging
logger = structlog.get_logger(__name__)

# Configuration class for service settings
class Config:
    """Configuration settings for Reasoning Module Factory service"""

    # Service ports and endpoints
    LLM_PROCESSOR_PORT = int(os.getenv("LLM_PROCESSOR_PORT", "8005"))
    MEMORY_MANAGER_PORT = int(os.getenv("MEMORY_MANAGER_PORT", "8205"))
    PLUGIN_REGISTRY_PORT = int(os.getenv("PLUGIN_REGISTRY_PORT", "8201"))
    RULE_ENGINE_PORT = int(os.getenv("RULE_ENGINE_PORT", "8204"))

    # Service host configuration
    SERVICE_HOST = os.getenv("SERVICE_HOST", "localhost")

    # LLM Configuration
    DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4")
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

    # Reasoning pattern configurations
    REACT_MAX_STEPS = int(os.getenv("REACT_MAX_STEPS", "10"))
    REFLECTION_MAX_ITERATIONS = int(os.getenv("REFLECTION_MAX_ITERATIONS", "5"))
    PLANNING_MAX_DEPTH = int(os.getenv("PLANNING_MAX_DEPTH", "5"))
    MULTI_AGENT_MAX_AGENTS = int(os.getenv("MULTI_AGENT_MAX_AGENTS", "5"))

    # Supported reasoning patterns
    SUPPORTED_PATTERNS = ["ReAct", "Reflection", "Planning", "Multi-Agent"]

# Pydantic models for request/response data structures

class ReasoningContext(BaseModel):
    """Context information for reasoning operations"""

    task_description: str = Field(..., description="Description of the task to be performed")
    agent_id: str = Field(..., description="Unique identifier of the agent")
    agent_name: str = Field(..., description="Human-readable agent name")
    domain: str = Field(..., description="Business domain context")
    persona: str = Field(..., description="Agent's personality and role")
    available_tools: List[Dict[str, Any]] = Field(default_factory=list, description="Available tools and functions")
    previous_actions: List[Dict[str, Any]] = Field(default_factory=list, description="History of previous actions")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Task constraints and limitations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context metadata")

class ReasoningResult(BaseModel):
    """Result of a reasoning operation"""

    reasoning_id: str = Field(..., description="Unique identifier for this reasoning operation")
    pattern_used: str = Field(..., description="Reasoning pattern that was applied")
    final_answer: Optional[str] = Field(None, description="Final answer or conclusion")
    reasoning_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Step-by-step reasoning process")
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list, description="Actions executed during reasoning")
    confidence_score: float = Field(0.0, description="Confidence score of the reasoning result")
    execution_time: float = Field(0.0, description="Time taken for reasoning in seconds")
    success: bool = Field(True, description="Whether the reasoning was successful")
    error_message: Optional[str] = Field(None, description="Error message if reasoning failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")

class ReasoningConfig(BaseModel):
    """Configuration for reasoning pattern instantiation"""

    pattern: str = Field(..., description="Reasoning pattern to use")
    model: str = Field(default=Config.DEFAULT_MODEL, description="LLM model to use")
    temperature: float = Field(default=Config.DEFAULT_TEMPERATURE, description="Temperature for LLM generation")
    max_tokens: int = Field(default=Config.MAX_TOKENS, description="Maximum tokens for LLM response")
    pattern_specific_config: Dict[str, Any] = Field(default_factory=dict, description="Pattern-specific configuration")

    @validator('pattern')
    def validate_pattern(cls, v):
        """Validate reasoning pattern is supported"""
        if v not in Config.SUPPORTED_PATTERNS:
            raise ValueError(f"Unsupported reasoning pattern: {v}")
        return v

class ReasoningRequest(BaseModel):
    """Request model for reasoning operations"""

    context: ReasoningContext = Field(..., description="Context for the reasoning operation")
    config: ReasoningConfig = Field(..., description="Configuration for reasoning pattern")

class ReasoningResponse(BaseModel):
    """Response model for reasoning operations"""

    success: bool = Field(..., description="Whether the reasoning operation was successful")
    result: Optional[ReasoningResult] = Field(None, description="Result of the reasoning operation")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    execution_metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")

# Abstract base class for reasoning patterns
class ReasoningPattern(ABC):
    """
    Abstract base class for all reasoning patterns.

    This class defines the interface that all reasoning pattern implementations
    must follow, ensuring consistency and interchangeability.
    """

    def __init__(self, config: ReasoningConfig, dependencies: Dict[str, Any]):
        """
        Initialize the reasoning pattern.

        Args:
            config: Configuration for this reasoning pattern
            dependencies: Injected dependencies (LLM service, memory manager, etc.)
        """
        self.config = config
        self.dependencies = dependencies
        self.logger = structlog.get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """
        Execute the reasoning pattern.

        Args:
            context: Context information for the reasoning operation

        Returns:
            ReasoningResult containing the reasoning output
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities and limitations of this reasoning pattern.

        Returns:
            Dictionary describing pattern capabilities
        """
        pass

# ReAct Pattern Implementation
class ReActPattern(ReasoningPattern):
    """
    Reasoning + Acting pattern implementation.

    This pattern alternates between reasoning about what to do and taking actions,
    using the results of actions to inform subsequent reasoning. It's particularly
    effective for tasks requiring exploration and tool usage.
    """

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """
        Execute ReAct reasoning pattern.

        The ReAct pattern follows this cycle:
        1. Observe current state
        2. Reason about what action to take
        3. Take the action
        4. Observe the result
        5. Repeat until goal is achieved or max steps reached
        """
        start_time = datetime.utcnow()
        reasoning_id = str(uuid.uuid4())
        reasoning_steps = []
        actions_taken = []

        try:
            # Initialize reasoning state
            current_state = {
                "observations": [context.task_description],
                "available_actions": context.available_tools,
                "goal": context.task_description,
                "agent_info": {
                    "name": context.agent_name,
                    "domain": context.domain,
                    "persona": context.persona
                }
            }

            max_steps = self.config.pattern_specific_config.get(
                'max_steps', Config.REACT_MAX_STEPS
            )

            for step in range(max_steps):
                # Step 1: Reason about next action
                reasoning_prompt = self._build_reasoning_prompt(current_state, step)
                thought = await self._call_llm(reasoning_prompt)

                # Step 2: Parse action from reasoning
                action = self._parse_action_from_reasoning(thought)

                if not action:
                    # No action needed, provide final answer
                    final_answer = self._extract_final_answer(thought)
                    break

                reasoning_steps.append({
                    "step": step + 1,
                    "thought": thought,
                    "action": action
                })

                # Step 3: Execute action
                action_result = await self._execute_action(action, context)

                actions_taken.append({
                    "step": step + 1,
                    "action": action,
                    "result": action_result
                })

                # Step 4: Update state with action result
                current_state["observations"].append(f"Action result: {action_result}")

                # Check if goal is achieved
                if self._is_goal_achieved(current_state, action_result):
                    final_answer = self._extract_final_answer_from_result(action_result)
                    break

            else:
                # Max steps reached without achieving goal
                final_answer = self._generate_fallback_answer(current_state)

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return ReasoningResult(
                reasoning_id=reasoning_id,
                pattern_used="ReAct",
                final_answer=final_answer,
                reasoning_steps=reasoning_steps,
                actions_taken=actions_taken,
                confidence_score=self._calculate_confidence(final_answer, reasoning_steps),
                execution_time=execution_time,
                success=True,
                metadata={
                    "steps_taken": len(reasoning_steps),
                    "actions_executed": len(actions_taken),
                    "max_steps": max_steps
                }
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error("ReAct reasoning failed", error=str(e), reasoning_id=reasoning_id)

            return ReasoningResult(
                reasoning_id=reasoning_id,
                pattern_used="ReAct",
                final_answer=None,
                reasoning_steps=reasoning_steps,
                actions_taken=actions_taken,
                confidence_score=0.0,
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    def _build_reasoning_prompt(self, state: Dict[str, Any], step: int) -> str:
        """Build the reasoning prompt for the current state"""
        observations = "\n".join(f"- {obs}" for obs in state["observations"])

        tools = "\n".join(
            f"- {tool['name']}: {tool.get('description', 'No description')}"
            for tool in state["available_actions"]
        )

        return f"""
You are {state['agent_info']['name']}, an AI agent specializing in {state['agent_info']['domain']}.
Your persona: {state['agent_info']['persona']}

Current observations:
{observations}

Available tools:
{tools}

Goal: {state['goal']}

This is step {step + 1} of your reasoning process.

Think step-by-step about what to do next. You can:
1. Use a tool to gather more information or perform an action
2. Provide a final answer if you have enough information

Format your response as:
Thought: [Your reasoning]
Action: [Tool name and parameters, or "Final Answer" if done]
"""

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM service for reasoning"""
        llm_service = self.dependencies.get("llm_service")
        if not llm_service:
            raise ValueError("LLM service not available")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{Config.SERVICE_HOST}:{Config.LLM_PROCESSOR_PORT}/generate",
                    json={
                        "prompt": prompt,
                        "model": self.config.model,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
        except Exception as e:
            self.logger.error("LLM call failed", error=str(e))
            raise

    def _parse_action_from_reasoning(self, thought: str) -> Optional[Dict[str, Any]]:
        """Parse action from LLM reasoning output"""
        # Simple parsing logic - in production this would be more sophisticated
        lines = thought.strip().split('\n')
        for line in lines:
            if line.lower().startswith('action:'):
                action_text = line[7:].strip()
                if action_text.lower() == 'final answer':
                    return None

                # Parse tool name and parameters
                # This is a simplified implementation
                return {"tool": action_text, "parameters": {}}

        return None

    async def _execute_action(self, action: Dict[str, Any], context: ReasoningContext) -> str:
        """Execute the specified action"""
        tool_name = action.get("tool", "")
        parameters = action.get("parameters", {})

        # Find the tool in available tools
        tool = next(
            (t for t in context.available_tools if t.get("name") == tool_name),
            None
        )

        if not tool:
            return f"Tool '{tool_name}' not found"

        # Execute the tool (simplified implementation)
        # In production, this would integrate with actual tool execution
        return f"Executed tool '{tool_name}' with parameters {parameters}"

    def _is_goal_achieved(self, state: Dict[str, Any], action_result: str) -> bool:
        """Check if the goal has been achieved"""
        # Simple heuristic - check if result contains success indicators
        success_indicators = ["completed", "finished", "achieved", "success"]
        return any(indicator in action_result.lower() for indicator in success_indicators)

    def _extract_final_answer(self, thought: str) -> str:
        """Extract final answer from reasoning"""
        # Simple extraction - look for "Final Answer:" in the thought
        lines = thought.split('\n')
        for line in lines:
            if "final answer:" in line.lower():
                return line.split(":", 1)[1].strip()
        return thought  # Fallback to entire thought

    def _extract_final_answer_from_result(self, action_result: str) -> str:
        """Extract final answer from action result"""
        return action_result

    def _generate_fallback_answer(self, state: Dict[str, Any]) -> str:
        """Generate fallback answer when max steps reached"""
        return "Unable to complete the task within the maximum number of reasoning steps."

    def _calculate_confidence(self, final_answer: str, reasoning_steps: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the reasoning result"""
        if not final_answer:
            return 0.0

        # Simple confidence calculation based on reasoning steps
        base_confidence = 0.5
        step_bonus = min(len(reasoning_steps) * 0.1, 0.4)
        return min(base_confidence + step_bonus, 1.0)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get ReAct pattern capabilities"""
        return {
            "name": "ReAct",
            "description": "Reasoning + Acting pattern for tool-using agents",
            "strengths": [
                "Effective for tasks requiring tool usage",
                "Handles exploration and information gathering",
                "Provides transparent reasoning steps"
            ],
            "limitations": [
                "Can be verbose for simple tasks",
                "May take many steps for complex problems",
                "Requires well-defined tools"
            ],
            "best_for": [
                "Problem-solving with external tools",
                "Information gathering tasks",
                "Multi-step decision making"
            ],
            "max_steps": Config.REACT_MAX_STEPS
        }

# Reflection Pattern Implementation
class ReflectionPattern(ReasoningPattern):
    """
    Reflection pattern implementation.

    This pattern involves self-assessment and iterative improvement,
    where the agent reflects on its previous actions and thoughts to
    improve future performance.
    """

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """
        Execute Reflection reasoning pattern.

        The Reflection pattern:
        1. Generate initial solution
        2. Reflect on the solution's quality
        3. Identify improvements
        4. Generate improved solution
        5. Repeat for multiple iterations
        """
        start_time = datetime.utcnow()
        reasoning_id = str(uuid.uuid4())
        reasoning_steps = []

        try:
            current_solution = ""
            max_iterations = self.config.pattern_specific_config.get(
                'max_iterations', Config.REFLECTION_MAX_ITERATIONS
            )

            for iteration in range(max_iterations):
                # Generate or improve solution
                if iteration == 0:
                    # First iteration - generate initial solution
                    solution = await self._generate_initial_solution(context)
                else:
                    # Subsequent iterations - reflect and improve
                    reflection = await self._reflect_on_solution(current_solution, context)
                    solution = await self._improve_solution(current_solution, reflection, context)

                current_solution = solution

                reasoning_steps.append({
                    "iteration": iteration + 1,
                    "solution": solution,
                    "reflection": await self._reflect_on_solution(solution, context) if iteration > 0 else None
                })

                # Check if solution is good enough
                quality_score = await self._assess_solution_quality(solution, context)
                if quality_score >= 0.8:  # Good enough threshold
                    break

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return ReasoningResult(
                reasoning_id=reasoning_id,
                pattern_used="Reflection",
                final_answer=current_solution,
                reasoning_steps=reasoning_steps,
                actions_taken=[],  # Reflection doesn't typically involve external actions
                confidence_score=await self._assess_solution_quality(current_solution, context),
                execution_time=execution_time,
                success=True,
                metadata={
                    "iterations_completed": len(reasoning_steps),
                    "max_iterations": max_iterations
                }
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error("Reflection reasoning failed", error=str(e), reasoning_id=reasoning_id)

            return ReasoningResult(
                reasoning_id=reasoning_id,
                pattern_used="Reflection",
                final_answer=None,
                reasoning_steps=reasoning_steps,
                actions_taken=[],
                confidence_score=0.0,
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    async def _generate_initial_solution(self, context: ReasoningContext) -> str:
        """Generate initial solution"""
        prompt = f"""
You are {context.agent_name}, an expert in {context.domain}.
Your persona: {context.persona}

Task: {context.task_description}

Generate an initial solution to this task. Be thorough and consider all aspects.
"""
        return await self._call_llm(prompt)

    async def _reflect_on_solution(self, solution: str, context: ReasoningContext) -> str:
        """Reflect on the quality of the current solution"""
        prompt = f"""
You are {context.agent_name}, an expert in {context.domain}.

Current solution:
{solution}

Task: {context.task_description}

Reflect on this solution:
1. What are its strengths?
2. What are its weaknesses?
3. What could be improved?
4. Are there any gaps or missing elements?

Provide constructive feedback for improvement.
"""
        return await self._call_llm(prompt)

    async def _improve_solution(self, current_solution: str, reflection: str, context: ReasoningContext) -> str:
        """Generate improved solution based on reflection"""
        prompt = f"""
You are {context.agent_name}, an expert in {context.domain}.

Current solution:
{current_solution}

Reflection on current solution:
{reflection}

Task: {context.task_description}

Based on the reflection, provide an improved solution that addresses the identified issues and gaps.
Make the solution more comprehensive, accurate, and effective.
"""
        return await self._call_llm(prompt)

    async def _assess_solution_quality(self, solution: str, context: ReasoningContext) -> float:
        """Assess the quality of the solution"""
        if not solution or len(solution.strip()) < 10:
            return 0.0

        # Simple quality assessment - count key indicators
        quality_indicators = [
            "therefore", "because", "however", "furthermore",
            "conclusion", "summary", "analysis", "recommendation"
        ]

        indicator_count = sum(1 for indicator in quality_indicators
                            if indicator in solution.lower())

        # Normalize to 0-1 scale
        return min(indicator_count / 5.0, 1.0)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM service for reasoning"""
        llm_service = self.dependencies.get("llm_service")
        if not llm_service:
            raise ValueError("LLM service not available")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{Config.SERVICE_HOST}:{Config.LLM_PROCESSOR_PORT}/generate",
                    json={
                        "prompt": prompt,
                        "model": self.config.model,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
        except Exception as e:
            self.logger.error("LLM call failed", error=str(e))
            raise

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Reflection pattern capabilities"""
        return {
            "name": "Reflection",
            "description": "Self-assessment and iterative improvement pattern",
            "strengths": [
                "Improves solution quality through iteration",
                "Identifies and corrects weaknesses",
                "Provides metacognitive insights"
            ],
            "limitations": [
                "Can be time-consuming for simple tasks",
                "May over-optimize for complex problems",
                "Requires good self-assessment capabilities"
            ],
            "best_for": [
                "Complex problem-solving requiring careful analysis",
                "Creative tasks needing iteration",
                "Quality-critical applications"
            ],
            "max_iterations": Config.REFLECTION_MAX_ITERATIONS
        }

# Planning Pattern Implementation
class PlanningPattern(ReasoningPattern):
    """
    Planning pattern implementation.

    This pattern involves breaking down complex tasks into manageable steps,
    creating a structured plan, and executing it systematically.
    """

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """
        Execute Planning reasoning pattern.

        The Planning pattern:
        1. Analyze the task and break it down
        2. Create a structured plan with steps
        3. Execute the plan step by step
        4. Monitor progress and adapt as needed
        5. Provide final result
        """
        start_time = datetime.utcnow()
        reasoning_id = str(uuid.uuid4())
        reasoning_steps = []
        actions_taken = []

        try:
            # Step 1: Create task breakdown
            task_breakdown = await self._break_down_task(context)

            # Step 2: Generate execution plan
            execution_plan = await self._create_execution_plan(task_breakdown, context)

            reasoning_steps.append({
                "phase": "planning",
                "task_breakdown": task_breakdown,
                "execution_plan": execution_plan
            })

            # Step 3: Execute the plan
            execution_results = []
            max_depth = self.config.pattern_specific_config.get(
                'max_depth', Config.PLANNING_MAX_DEPTH
            )

            for step_idx, step in enumerate(execution_plan):
                if step_idx >= max_depth:
                    break

                # Execute step
                step_result = await self._execute_plan_step(step, context)
                execution_results.append(step_result)

                actions_taken.append({
                    "step": step_idx + 1,
                    "plan_step": step,
                    "result": step_result
                })

                # Check if we can conclude
                if self._can_conclude_from_results(execution_results, context):
                    break

            # Step 4: Synthesize final result
            final_result = await self._synthesize_final_result(execution_results, context)

            reasoning_steps.append({
                "phase": "execution",
                "execution_results": execution_results,
                "final_synthesis": final_result
            })

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return ReasoningResult(
                reasoning_id=reasoning_id,
                pattern_used="Planning",
                final_answer=final_result,
                reasoning_steps=reasoning_steps,
                actions_taken=actions_taken,
                confidence_score=self._calculate_planning_confidence(execution_results),
                execution_time=execution_time,
                success=True,
                metadata={
                    "plan_steps": len(execution_plan),
                    "steps_executed": len(execution_results),
                    "task_breakdown_size": len(task_breakdown)
                }
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error("Planning reasoning failed", error=str(e), reasoning_id=reasoning_id)

            return ReasoningResult(
                reasoning_id=reasoning_id,
                pattern_used="Planning",
                final_answer=None,
                reasoning_steps=reasoning_steps,
                actions_taken=actions_taken,
                confidence_score=0.0,
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    async def _break_down_task(self, context: ReasoningContext) -> List[str]:
        """Break down the task into manageable components"""
        prompt = f"""
You are {context.agent_name}, an expert planner in {context.domain}.

Task: {context.task_description}

Break this task down into specific, actionable components or steps.
Consider:
- What information do I need?
- What analysis is required?
- What actions need to be taken?
- What are the dependencies between steps?

Provide a clear, logical breakdown of the task.
"""
        breakdown_text = await self._call_llm(prompt)

        # Parse breakdown into list (simplified)
        return [line.strip() for line in breakdown_text.split('\n')
                if line.strip() and not line.strip().startswith(('Task:', 'Consider:', '-'))]

    async def _create_execution_plan(self, task_breakdown: List[str], context: ReasoningContext) -> List[Dict[str, Any]]:
        """Create a structured execution plan"""
        plan_prompt = f"""
Based on this task breakdown:
{chr(10).join(f"- {item}" for item in task_breakdown)}

Create a detailed execution plan with specific steps.
For each step, include:
- Description of what to do
- Required inputs or prerequisites
- Expected outputs
- Success criteria

Task: {context.task_description}
"""
        plan_text = await self._call_llm(plan_prompt)

        # Parse plan into structured format (simplified)
        return [{"description": plan_text, "step_number": 1}]

    async def _execute_plan_step(self, step: Dict[str, Any], context: ReasoningContext) -> str:
        """Execute a single plan step"""
        execution_prompt = f"""
Execute this plan step:
{step['description']}

Context: {context.task_description}

Available tools: {[tool['name'] for tool in context.available_tools]}

Provide the result of executing this step.
"""
        return await self._call_llm(execution_prompt)

    def _can_conclude_from_results(self, execution_results: List[str], context: ReasoningContext) -> bool:
        """Check if we have enough results to conclude"""
        return len(execution_results) >= 3  # Simple threshold

    async def _synthesize_final_result(self, execution_results: List[str], context: ReasoningContext) -> str:
        """Synthesize final result from all execution results"""
        synthesis_prompt = f"""
Based on these execution results:
{chr(10).join(f"Step {i+1}: {result}" for i, result in enumerate(execution_results))}

Synthesize a comprehensive final answer for the task:
{context.task_description}

Provide a clear, complete response that addresses the original task.
"""
        return await self._call_llm(synthesis_prompt)

    def _calculate_planning_confidence(self, execution_results: List[str]) -> float:
        """Calculate confidence score for planning results"""
        if not execution_results:
            return 0.0

        # Simple confidence based on number of completed steps
        return min(len(execution_results) / 5.0, 1.0)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM service for reasoning"""
        llm_service = self.dependencies.get("llm_service")
        if not llm_service:
            raise ValueError("LLM service not available")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{Config.SERVICE_HOST}:{Config.LLM_PROCESSOR_PORT}/generate",
                    json={
                        "prompt": prompt,
                        "model": self.config.model,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
        except Exception as e:
            self.logger.error("LLM call failed", error=str(e))
            raise

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Planning pattern capabilities"""
        return {
            "name": "Planning",
            "description": "Multi-step task decomposition and structured execution",
            "strengths": [
                "Handles complex, multi-step tasks effectively",
                "Provides clear structure and progress tracking",
                "Systematic approach to problem-solving"
            ],
            "limitations": [
                "May be overkill for simple tasks",
                "Planning overhead for straightforward problems",
                "Requires good task decomposition skills"
            ],
            "best_for": [
                "Complex projects with multiple dependencies",
                "Tasks requiring systematic execution",
                "Problems needing careful planning"
            ],
            "max_depth": Config.PLANNING_MAX_DEPTH
        }

# Multi-Agent Pattern Implementation
class MultiAgentPattern(ReasoningPattern):
    """
    Multi-Agent coordination pattern implementation.

    This pattern involves multiple specialized agents working together
    to solve complex problems through coordination and collaboration.
    """

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """
        Execute Multi-Agent reasoning pattern.

        The Multi-Agent pattern:
        1. Analyze task and determine agent roles
        2. Create or assign specialized agents
        3. Coordinate agent activities
        4. Synthesize results from multiple agents
        5. Provide unified final answer
        """
        start_time = datetime.utcnow()
        reasoning_id = str(uuid.uuid4())
        reasoning_steps = []
        actions_taken = []

        try:
            # Step 1: Determine agent roles and responsibilities
            agent_roles = await self._determine_agent_roles(context)

            reasoning_steps.append({
                "phase": "role_assignment",
                "agent_roles": agent_roles
            })

            # Step 2: Execute agent activities
            agent_results = []
            max_agents = self.config.pattern_specific_config.get(
                'max_agents', Config.MULTI_AGENT_MAX_AGENTS
            )

            for i, role in enumerate(agent_roles[:max_agents]):
                agent_result = await self._execute_agent_role(role, context, i + 1)
                agent_results.append({
                    "role": role,
                    "result": agent_result,
                    "agent_number": i + 1
                })

                actions_taken.append({
                    "action_type": "agent_execution",
                    "role": role,
                    "result_summary": agent_result[:100] + "..." if len(agent_result) > 100 else agent_result
                })

            # Step 3: Synthesize results from all agents
            final_result = await self._synthesize_multi_agent_results(agent_results, context)

            reasoning_steps.append({
                "phase": "synthesis",
                "agent_results": agent_results,
                "final_synthesis": final_result
            })

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return ReasoningResult(
                reasoning_id=reasoning_id,
                pattern_used="Multi-Agent",
                final_answer=final_result,
                reasoning_steps=reasoning_steps,
                actions_taken=actions_taken,
                confidence_score=self._calculate_multi_agent_confidence(agent_results),
                execution_time=execution_time,
                success=True,
                metadata={
                    "agents_used": len(agent_results),
                    "max_agents": max_agents,
                    "roles_assigned": len(agent_roles)
                }
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error("Multi-Agent reasoning failed", error=str(e), reasoning_id=reasoning_id)

            return ReasoningResult(
                reasoning_id=reasoning_id,
                pattern_used="Multi-Agent",
                final_answer=None,
                reasoning_steps=reasoning_steps,
                actions_taken=actions_taken,
                confidence_score=0.0,
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    async def _determine_agent_roles(self, context: ReasoningContext) -> List[str]:
        """Determine what roles/agents are needed for this task"""
        prompt = f"""
You are {context.agent_name}, a coordinator of multiple AI agents in {context.domain}.

Task: {context.task_description}

Determine what specialized roles or agent types would be most effective for solving this task.
Consider different perspectives, expertise areas, and complementary skills.

List 2-4 agent roles that would work well together to solve this task.
For each role, briefly describe:
- What the agent's specialty is
- What specific contribution they would make
- How they complement other agents

Available context: {context.persona}
"""
        roles_text = await self._call_llm(prompt)

        # Parse roles (simplified)
        return [line.strip() for line in roles_text.split('\n')
                if line.strip() and len(line.strip()) > 10][:4]  # Limit to 4 roles

    async def _execute_agent_role(self, role: str, context: ReasoningContext, agent_number: int) -> str:
        """Execute a specific agent role"""
        role_prompt = f"""
You are Agent #{agent_number} with the role: {role}

Main Task: {context.task_description}

Your specific role and expertise: {role}

Provide your analysis, insights, or contribution to solving this task from your specialized perspective.
Be thorough but focused on your area of expertise.

Context about the overall agent team: {context.agent_name} in {context.domain}
"""
        return await self._call_llm(role_prompt)

    async def _synthesize_multi_agent_results(self, agent_results: List[Dict[str, Any]], context: ReasoningContext) -> str:
        """Synthesize results from multiple agents"""
        results_summary = "\n\n".join([
            f"Agent {result['agent_number']} ({result['role']}):\n{result['result']}"
            for result in agent_results
        ])

        synthesis_prompt = f"""
You are {context.agent_name}, synthesizing insights from multiple specialized agents.

Original Task: {context.task_description}

Agent Contributions:
{results_summary}

Synthesize these diverse perspectives into a comprehensive, unified solution.
Consider:
- Common themes and consensus points
- Complementary insights from different perspectives
- Resolving any conflicting viewpoints
- Creating a more complete and robust solution

Provide a final, integrated answer that leverages the strengths of all contributing agents.
"""
        return await self._call_llm(synthesis_prompt)

    def _calculate_multi_agent_confidence(self, agent_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for multi-agent results"""
        if not agent_results:
            return 0.0

        # Confidence based on number of agents and result diversity
        agent_count = len(agent_results)
        base_confidence = 0.6
        agent_bonus = min(agent_count * 0.1, 0.3)
        return min(base_confidence + agent_bonus, 1.0)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM service for reasoning"""
        llm_service = self.dependencies.get("llm_service")
        if not llm_service:
            raise ValueError("LLM service not available")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{Config.SERVICE_HOST}:{Config.LLM_PROCESSOR_PORT}/generate",
                    json={
                        "prompt": prompt,
                        "model": self.config.model,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
        except Exception as e:
            self.logger.error("LLM call failed", error=str(e))
            raise

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Multi-Agent pattern capabilities"""
        return {
            "name": "Multi-Agent",
            "description": "Coordination and collaboration between multiple specialized agents",
            "strengths": [
                "Leverages diverse expertise and perspectives",
                "Handles complex problems through specialization",
                "Provides robustness through redundancy"
            ],
            "limitations": [
                "Coordination overhead for simple tasks",
                "Potential for conflicting agent outputs",
                "Requires careful result synthesis"
            ],
            "best_for": [
                "Complex problems needing diverse expertise",
                "Tasks benefiting from multiple perspectives",
                "Problems requiring specialization and collaboration"
            ],
            "max_agents": Config.MULTI_AGENT_MAX_AGENTS
        }

# Reasoning Module Factory
class ReasoningModuleFactory:
    """
    Factory class for creating reasoning pattern instances.

    This factory implements the dependency injection pattern and provides
    a centralized way to create and configure reasoning modules.
    """

    def __init__(self):
        """Initialize the reasoning module factory"""
        self.logger = structlog.get_logger(__name__)
        self._pattern_classes = {
            "ReAct": ReActPattern,
            "Reflection": ReflectionPattern,
            "Planning": PlanningPattern,
            "Multi-Agent": MultiAgentPattern
        }
        self._service_dependencies = {}

    def register_dependency(self, name: str, dependency: Any):
        """
        Register a service dependency for injection.

        Args:
            name: Name of the dependency
            dependency: The dependency instance
        """
        self._service_dependencies[name] = dependency
        self.logger.info("Registered dependency", dependency_name=name)

    async def create_reasoning_module(
        self,
        config: ReasoningConfig
    ) -> ReasoningPattern:
        """
        Create a reasoning module instance based on configuration.

        Args:
            config: Configuration for the reasoning pattern

        Returns:
            Configured reasoning pattern instance

        Raises:
            ValueError: If the requested pattern is not supported
        """
        if config.pattern not in self._pattern_classes:
            raise ValueError(f"Unsupported reasoning pattern: {config.pattern}")

        pattern_class = self._pattern_classes[config.pattern]

        # Create pattern instance with dependencies
        pattern_instance = pattern_class(config, self._service_dependencies)

        self.logger.info(
            "Created reasoning module",
            pattern=config.pattern,
            model=config.model
        )

        return pattern_instance

    def get_supported_patterns(self) -> List[str]:
        """Get list of supported reasoning patterns"""
        return list(self._pattern_classes.keys())

    def get_pattern_capabilities(self, pattern: str) -> Dict[str, Any]:
        """
        Get capabilities of a specific reasoning pattern.

        Args:
            pattern: Name of the reasoning pattern

        Returns:
            Dictionary of pattern capabilities

        Raises:
            ValueError: If pattern is not supported
        """
        if pattern not in self._pattern_classes:
            raise ValueError(f"Unsupported reasoning pattern: {pattern}")

        # Create a temporary instance to get capabilities
        temp_config = ReasoningConfig(pattern=pattern)
        temp_instance = self._pattern_classes[pattern](temp_config, {})
        return temp_instance.get_capabilities()

    async def reason_with_pattern(
        self,
        pattern: str,
        context: ReasoningContext,
        config: Optional[ReasoningConfig] = None
    ) -> ReasoningResult:
        """
        Convenience method to reason with a specific pattern.

        Args:
            pattern: Name of the reasoning pattern to use
            context: Context for the reasoning operation
            config: Optional configuration (will use defaults if not provided)

        Returns:
            ReasoningResult from the pattern execution
        """
        if config is None:
            config = ReasoningConfig(pattern=pattern)

        reasoning_module = await self.create_reasoning_module(config)
        return await reasoning_module.reason(context)

# FastAPI application setup
app = FastAPI(
    title="Reasoning Module Factory Service",
    description="Factory service for creating and managing AI reasoning patterns",
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

# Initialize factory
reasoning_factory = ReasoningModuleFactory()

# Register service dependencies
@app.on_event("startup")
async def startup_event():
    """Initialize service dependencies on startup"""
    # Register LLM service dependency
    reasoning_factory.register_dependency("llm_service", {
        "endpoint": f"http://{Config.SERVICE_HOST}:{Config.LLM_PROCESSOR_PORT}",
        "timeout": 30.0
    })

    # Register memory manager dependency
    reasoning_factory.register_dependency("memory_manager", {
        "endpoint": f"http://{Config.SERVICE_HOST}:{Config.MEMORY_MANAGER_PORT}",
        "timeout": 30.0
    })

    # Register plugin registry dependency
    reasoning_factory.register_dependency("plugin_registry", {
        "endpoint": f"http://{Config.SERVICE_HOST}:{Config.PLUGIN_REGISTRY_PORT}",
        "timeout": 30.0
    })

    # Register rule engine dependency
    reasoning_factory.register_dependency("rule_engine", {
        "endpoint": f"http://{Config.SERVICE_HOST}:{Config.RULE_ENGINE_PORT}",
        "timeout": 30.0
    })

@app.get("/")
async def root():
    """Service health check and information endpoint"""
    return {
        "service": "Reasoning Module Factory",
        "version": "1.0.0",
        "status": "healthy",
        "description": "Factory for AI reasoning patterns (ReAct, Reflection, Planning, Multi-Agent)",
        "supported_patterns": reasoning_factory.get_supported_patterns(),
        "endpoints": {
            "POST /reason": "Execute reasoning with specified pattern",
            "GET /patterns": "List supported reasoning patterns",
            "GET /patterns/{pattern}/capabilities": "Get pattern capabilities",
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
        "service": "Reasoning Module Factory",
        "patterns_loaded": len(reasoning_factory.get_supported_patterns()),
        "dependencies": {
            "llm_service": "configured",
            "memory_manager": "configured",
            "plugin_registry": "configured",
            "rule_engine": "configured"
        }
    }

@app.get("/patterns")
async def get_supported_patterns():
    """Get list of supported reasoning patterns"""
    patterns = []
    for pattern_name in reasoning_factory.get_supported_patterns():
        try:
            capabilities = reasoning_factory.get_pattern_capabilities(pattern_name)
            patterns.append({
                "name": pattern_name,
                "description": capabilities.get("description", ""),
                "best_for": capabilities.get("best_for", [])
            })
        except Exception as e:
            patterns.append({
                "name": pattern_name,
                "description": "Pattern available",
                "error": str(e)
            })

    return {
        "patterns": patterns,
        "total_count": len(patterns)
    }

@app.get("/patterns/{pattern}/capabilities")
async def get_pattern_capabilities(pattern: str):
    """Get detailed capabilities of a specific reasoning pattern"""
    try:
        capabilities = reasoning_factory.get_pattern_capabilities(pattern)
        return {
            "pattern": pattern,
            "capabilities": capabilities
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving capabilities: {str(e)}")

@app.post("/reason", response_model=ReasoningResponse)
async def execute_reasoning(request: ReasoningRequest):
    """
    Execute reasoning with the specified pattern and context.

    This endpoint creates a reasoning module instance based on the requested
    pattern and executes it with the provided context.

    Request Body:
    - context: ReasoningContext with task description and available tools
    - config: ReasoningConfig specifying pattern and parameters

    Returns:
    - success: Boolean indicating if reasoning was successful
    - result: ReasoningResult with final answer and reasoning steps
    - error_message: Error message if reasoning failed
    """
    try:
        logger.info(
            "Starting reasoning execution",
            pattern=request.config.pattern,
            agent_id=request.context.agent_id,
            task_length=len(request.context.task_description)
        )

        # Execute reasoning with the factory
        result = await reasoning_factory.reason_with_pattern(
            pattern=request.config.pattern,
            context=request.context,
            config=request.config
        )

        logger.info(
            "Reasoning execution completed",
            pattern=request.config.pattern,
            success=result.success,
            execution_time=result.execution_time,
            reasoning_steps=len(result.reasoning_steps)
        )

        return ReasoningResponse(
            success=result.success,
            result=result,
            error_message=result.error_message if not result.success else None,
            execution_metadata={
                "pattern": request.config.pattern,
                "execution_time": result.execution_time,
                "reasoning_steps": len(result.reasoning_steps),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Reasoning execution failed",
            pattern=request.config.pattern,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning execution failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("REASONING_MODULE_FACTORY_PORT", "8304"))

    logger.info("Starting Reasoning Module Factory Service", port=port)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
