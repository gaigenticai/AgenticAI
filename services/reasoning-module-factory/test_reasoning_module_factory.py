#!/usr/bin/env python3
"""
Test Suite for Reasoning Module Factory Service

Comprehensive testing for the reasoning pattern implementations including:
- Factory pattern creation and dependency injection
- Individual reasoning pattern execution
- Integration with mock services
- Error handling and edge cases
- Performance and scalability testing

Author: AgenticAI Platform
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json

from main import (
    ReasoningModuleFactory,
    ReActPattern,
    ReflectionPattern,
    PlanningPattern,
    MultiAgentPattern,
    ReasoningContext,
    ReasoningConfig,
    ReasoningResult,
    Config
)


class TestReasoningModuleFactory:
    """Test suite for Reasoning Module Factory"""

    @pytest.fixture
    def factory(self):
        """Fixture for ReasoningModuleFactory instance"""
        return ReasoningModuleFactory()

    @pytest.fixture
    def mock_dependencies(self):
        """Fixture for mock service dependencies"""
        return {
            "llm_service": {
                "endpoint": "http://localhost:8005",
                "timeout": 30.0
            },
            "memory_manager": {
                "endpoint": "http://localhost:8205",
                "timeout": 30.0
            },
            "plugin_registry": {
                "endpoint": "http://localhost:8201",
                "timeout": 30.0
            },
            "rule_engine": {
                "endpoint": "http://localhost:8204",
                "timeout": 30.0
            }
        }

    @pytest.fixture
    def sample_context(self):
        """Fixture for sample reasoning context"""
        return ReasoningContext(
            task_description="Solve this math problem: 2x + 3 = 7",
            agent_id="math_agent_001",
            agent_name="Math Tutor Agent",
            domain="mathematics",
            persona="Expert mathematics tutor with step-by-step problem solving",
            available_tools=[
                {
                    "name": "calculator",
                    "description": "Perform mathematical calculations",
                    "parameters": {"expression": "string"}
                },
                {
                    "name": "graph_plotter",
                    "description": "Create mathematical graphs",
                    "parameters": {"equation": "string"}
                }
            ],
            previous_actions=[
                {
                    "action": "analyze_problem",
                    "result": "This is a linear equation",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ],
            constraints={
                "time_limit": 300,
                "complexity": "intermediate"
            }
        )

    def test_factory_initialization(self, factory):
        """Test factory initialization"""
        assert factory is not None
        assert len(factory.get_supported_patterns()) == 4
        assert "ReAct" in factory.get_supported_patterns()
        assert "Reflection" in factory.get_supported_patterns()
        assert "Planning" in factory.get_supported_patterns()
        assert "Multi-Agent" in factory.get_supported_patterns()

    def test_dependency_registration(self, factory, mock_dependencies):
        """Test dependency registration"""
        for name, dependency in mock_dependencies.items():
            factory.register_dependency(name, dependency)

        # Check that dependencies are registered (implementation detail)
        assert hasattr(factory, '_service_dependencies')

    @pytest.mark.asyncio
    async def test_create_reasoning_module_react(self, factory, mock_dependencies):
        """Test creating ReAct reasoning module"""
        for name, dependency in mock_dependencies.items():
            factory.register_dependency(name, dependency)

        config = ReasoningConfig(pattern="ReAct", model="gpt-4")
        module = await factory.create_reasoning_module(config)

        assert isinstance(module, ReActPattern)
        assert module.config.pattern == "ReAct"
        assert module.config.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_create_reasoning_module_reflection(self, factory, mock_dependencies):
        """Test creating Reflection reasoning module"""
        for name, dependency in mock_dependencies.items():
            factory.register_dependency(name, dependency)

        config = ReasoningConfig(pattern="Reflection", model="gpt-4")
        module = await factory.create_reasoning_module(config)

        assert isinstance(module, ReflectionPattern)
        assert module.config.pattern == "Reflection"

    @pytest.mark.asyncio
    async def test_create_reasoning_module_planning(self, factory, mock_dependencies):
        """Test creating Planning reasoning module"""
        for name, dependency in mock_dependencies.items():
            factory.register_dependency(name, dependency)

        config = ReasoningConfig(pattern="Planning", model="gpt-4")
        module = await factory.create_reasoning_module(config)

        assert isinstance(module, PlanningPattern)
        assert module.config.pattern == "Planning"

    @pytest.mark.asyncio
    async def test_create_reasoning_module_multi_agent(self, factory, mock_dependencies):
        """Test creating Multi-Agent reasoning module"""
        for name, dependency in mock_dependencies.items():
            factory.register_dependency(name, dependency)

        config = ReasoningConfig(pattern="Multi-Agent", model="gpt-4")
        module = await factory.create_reasoning_module(config)

        assert isinstance(module, MultiAgentPattern)
        assert module.config.pattern == "Multi-Agent"

    @pytest.mark.asyncio
    async def test_create_invalid_reasoning_module(self, factory):
        """Test creating invalid reasoning module"""
        config = ReasoningConfig(pattern="InvalidPattern", model="gpt-4")

        with pytest.raises(ValueError, match="Unsupported reasoning pattern"):
            await factory.create_reasoning_module(config)

    def test_get_pattern_capabilities_react(self, factory):
        """Test getting ReAct pattern capabilities"""
        capabilities = factory.get_pattern_capabilities("ReAct")

        assert capabilities["name"] == "ReAct"
        assert "description" in capabilities
        assert "strengths" in capabilities
        assert "limitations" in capabilities
        assert "best_for" in capabilities
        assert "max_steps" in capabilities

    def test_get_pattern_capabilities_reflection(self, factory):
        """Test getting Reflection pattern capabilities"""
        capabilities = factory.get_pattern_capabilities("Reflection")

        assert capabilities["name"] == "Reflection"
        assert "max_iterations" in capabilities

    def test_get_pattern_capabilities_planning(self, factory):
        """Test getting Planning pattern capabilities"""
        capabilities = factory.get_pattern_capabilities("Planning")

        assert capabilities["name"] == "Planning"
        assert "max_depth" in capabilities

    def test_get_pattern_capabilities_multi_agent(self, factory):
        """Test getting Multi-Agent pattern capabilities"""
        capabilities = factory.get_pattern_capabilities("Multi-Agent")

        assert capabilities["name"] == "Multi-Agent"
        assert "max_agents" in capabilities

    def test_get_invalid_pattern_capabilities(self, factory):
        """Test getting capabilities for invalid pattern"""
        with pytest.raises(ValueError, match="Unsupported reasoning pattern"):
            factory.get_pattern_capabilities("InvalidPattern")

    @pytest.mark.asyncio
    async def test_reason_with_pattern_react(self, factory, sample_context, mock_dependencies):
        """Test reasoning with ReAct pattern"""
        for name, dependency in mock_dependencies.items():
            factory.register_dependency(name, dependency)

        # Mock the LLM call
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "Thought: This is a simple equation. I can solve it directly.\nAction: Final Answer\nThe solution is x = 2"}
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await factory.reason_with_pattern("ReAct", sample_context)

            assert isinstance(result, ReasoningResult)
            assert result.pattern_used == "ReAct"
            assert result.success is True
            assert "x = 2" in result.final_answer or "solution" in result.final_answer.lower()

    @pytest.mark.asyncio
    async def test_reason_with_pattern_reflection(self, factory, sample_context, mock_dependencies):
        """Test reasoning with Reflection pattern"""
        for name, dependency in mock_dependencies.items():
            factory.register_dependency(name, dependency)

        # Mock the LLM calls for reflection pattern
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "This is a linear equation. The solution involves isolating x by subtracting 3 and dividing by 2."}
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await factory.reason_with_pattern("Reflection", sample_context)

            assert isinstance(result, ReasoningResult)
            assert result.pattern_used == "Reflection"
            assert result.success is True
            assert len(result.reasoning_steps) >= 1

    @pytest.mark.asyncio
    async def test_reason_with_pattern_planning(self, factory, sample_context, mock_dependencies):
        """Test reasoning with Planning pattern"""
        for name, dependency in mock_dependencies.items():
            factory.register_dependency(name, dependency)

        # Mock the LLM calls for planning pattern
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "1. Identify the equation type\n2. Isolate the variable\n3. Solve for x\n4. Verify the solution"}
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await factory.reason_with_pattern("Planning", sample_context)

            assert isinstance(result, ReasoningResult)
            assert result.pattern_used == "Planning"
            assert result.success is True
            assert "plan" in result.reasoning_steps[0] or "steps" in str(result.reasoning_steps[0]).lower()

    @pytest.mark.asyncio
    async def test_reason_with_pattern_multi_agent(self, factory, sample_context, mock_dependencies):
        """Test reasoning with Multi-Agent pattern"""
        for name, dependency in mock_dependencies.items():
            factory.register_dependency(name, dependency)

        # Mock the LLM calls for multi-agent pattern
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "Agent 1: Math Expert - This is a linear equation\nAgent 2: Logic Checker - The steps are mathematically sound\nAgent 3: Solution Verifier - x = 2 is correct"}
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await factory.reason_with_pattern("Multi-Agent", sample_context)

            assert isinstance(result, ReasoningResult)
            assert result.pattern_used == "Multi-Agent"
            assert result.success is True
            assert "Agent" in str(result.reasoning_steps[0]) or "agent" in str(result.reasoning_steps[0]).lower()


class TestReActPattern:
    """Test suite for ReAct pattern implementation"""

    @pytest.fixture
    def react_pattern(self, mock_dependencies):
        """Fixture for ReAct pattern instance"""
        config = ReasoningConfig(pattern="ReAct", model="gpt-4")
        return ReActPattern(config, mock_dependencies)

    @pytest.fixture
    def mock_dependencies(self):
        """Fixture for mock dependencies"""
        return {
            "llm_service": Mock(),
            "memory_manager": Mock(),
            "plugin_registry": Mock(),
            "rule_engine": Mock()
        }

    def test_react_initialization(self, react_pattern):
        """Test ReAct pattern initialization"""
        assert react_pattern.config.pattern == "ReAct"
        assert react_pattern.config.model == "gpt-4"

    def test_react_capabilities(self, react_pattern):
        """Test ReAct pattern capabilities"""
        capabilities = react_pattern.get_capabilities()

        assert capabilities["name"] == "ReAct"
        assert "max_steps" in capabilities
        assert isinstance(capabilities["strengths"], list)
        assert isinstance(capabilities["limitations"], list)

    @pytest.mark.asyncio
    async def test_react_reasoning_simple(self, react_pattern, sample_context):
        """Test simple ReAct reasoning execution"""
        # Mock LLM call
        with patch.object(react_pattern, '_call_llm') as mock_llm:
            mock_llm.return_value = "Thought: This is a simple equation I can solve directly.\nAction: Final Answer\nx = 2"

            result = await react_pattern.reason(sample_context)

            assert result.success is True
            assert result.pattern_used == "ReAct"
            assert "x = 2" in result.final_answer
            assert len(result.reasoning_steps) >= 1
            assert result.confidence_score > 0

    @pytest.mark.asyncio
    async def test_react_reasoning_with_tools(self, react_pattern, sample_context):
        """Test ReAct reasoning with tool usage"""
        with patch.object(react_pattern, '_call_llm') as mock_llm:
            # Mock multiple LLM calls for tool usage
            mock_llm.side_effect = [
                "Thought: I need to use the calculator to solve this.\nAction: calculator\nParameters: {\"expression\": \"2x + 3 = 7\"}",
                "Thought: The calculator gave me the solution.\nAction: Final Answer\nx = 2"
            ]

            with patch.object(react_pattern, '_execute_action') as mock_execute:
                mock_execute.return_value = "Calculator result: x = 2"

                result = await react_pattern.reason(sample_context)

                assert result.success is True
                assert len(result.actions_taken) >= 1
                assert "calculator" in str(result.actions_taken[0])

    def test_react_prompt_building(self, react_pattern, sample_context):
        """Test ReAct prompt building"""
        prompt = react_pattern._build_reasoning_prompt(
            {
                "observations": ["Current equation: 2x + 3 = 7"],
                "available_actions": sample_context.available_tools,
                "goal": sample_context.task_description,
                "agent_info": {
                    "name": sample_context.agent_name,
                    "domain": sample_context.domain,
                    "persona": sample_context.persona
                }
            },
            1
        )

        assert sample_context.task_description in prompt
        assert sample_context.agent_name in prompt
        assert "Thought:" in prompt
        assert "Action:" in prompt

    def test_react_action_parsing(self, react_pattern):
        """Test action parsing from reasoning"""
        # Test final answer
        thought = "Thought: I have the solution.\nAction: Final Answer\nx = 2"
        action = react_pattern._parse_action_from_reasoning(thought)
        assert action is None  # None means final answer

        # Test tool action
        thought = "Thought: I need to calculate.\nAction: calculator\nParameters: {\"expression\": \"2x+3=7\"}"
        action = react_pattern._parse_action_from_reasoning(thought)
        assert action is not None
        assert action["tool"] == "calculator"

    def test_react_confidence_calculation(self, react_pattern):
        """Test confidence score calculation"""
        # Test with reasoning steps
        reasoning_steps = [
            {"thought": "Step 1", "action": "calculator"},
            {"thought": "Step 2", "action": "verify"}
        ]

        confidence = react_pattern._calculate_confidence("x = 2", reasoning_steps)
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably high with steps

        # Test without reasoning steps
        confidence = react_pattern._calculate_confidence("x = 2", [])
        assert 0 <= confidence <= 1


class TestReflectionPattern:
    """Test suite for Reflection pattern implementation"""

    @pytest.fixture
    def reflection_pattern(self, mock_dependencies):
        """Fixture for Reflection pattern instance"""
        config = ReasoningConfig(pattern="Reflection", model="gpt-4")
        return ReflectionPattern(config, mock_dependencies)

    @pytest.fixture
    def mock_dependencies(self):
        """Fixture for mock dependencies"""
        return {
            "llm_service": Mock(),
            "memory_manager": Mock(),
            "plugin_registry": Mock(),
            "rule_engine": Mock()
        }

    def test_reflection_initialization(self, reflection_pattern):
        """Test Reflection pattern initialization"""
        assert reflection_pattern.config.pattern == "Reflection"

    def test_reflection_capabilities(self, reflection_pattern):
        """Test Reflection pattern capabilities"""
        capabilities = reflection_pattern.get_capabilities()

        assert capabilities["name"] == "Reflection"
        assert "max_iterations" in capabilities

    @pytest.mark.asyncio
    async def test_reflection_reasoning(self, reflection_pattern, sample_context):
        """Test Reflection reasoning execution"""
        with patch.object(reflection_pattern, '_call_llm') as mock_llm:
            mock_llm.side_effect = [
                "Initial solution: x = 2",  # Initial solution
                "The solution looks correct but could be more detailed",  # Reflection
                "Improved solution: x = 2 (subtract 3: 2x = 4, divide by 2: x = 2)"  # Improved solution
            ]

            result = await reflection_pattern.reason(sample_context)

            assert result.success is True
            assert result.pattern_used == "Reflection"
            assert len(result.reasoning_steps) >= 2  # Initial + at least one reflection

    def test_reflection_quality_assessment(self, reflection_pattern):
        """Test solution quality assessment"""
        # High quality solution
        quality = reflection_pattern._assess_solution_quality(
            "Based on algebraic principles, x = 2. This is verified by substitution.",
            sample_context
        )
        assert quality > 0.5

        # Low quality solution
        quality = reflection_pattern._assess_solution_quality("x = 2", sample_context)
        assert quality <= 0.5


class TestPlanningPattern:
    """Test suite for Planning pattern implementation"""

    @pytest.fixture
    def planning_pattern(self, mock_dependencies):
        """Fixture for Planning pattern instance"""
        config = ReasoningConfig(pattern="Planning", model="gpt-4")
        return PlanningPattern(config, mock_dependencies)

    @pytest.fixture
    def mock_dependencies(self):
        """Fixture for mock dependencies"""
        return {
            "llm_service": Mock(),
            "memory_manager": Mock(),
            "plugin_registry": Mock(),
            "rule_engine": Mock()
        }

    def test_planning_initialization(self, planning_pattern):
        """Test Planning pattern initialization"""
        assert planning_pattern.config.pattern == "Planning"

    def test_planning_capabilities(self, planning_pattern):
        """Test Planning pattern capabilities"""
        capabilities = planning_pattern.get_capabilities()

        assert capabilities["name"] == "Planning"
        assert "max_depth" in capabilities

    @pytest.mark.asyncio
    async def test_planning_reasoning(self, planning_pattern, sample_context):
        """Test Planning reasoning execution"""
        with patch.object(planning_pattern, '_call_llm') as mock_llm:
            mock_llm.side_effect = [
                "1. Identify equation type\n2. Isolate variable\n3. Solve for x\n4. Verify solution",  # Breakdown
                "Plan: Execute steps systematically",  # Plan creation
                "Step 1 result: Linear equation identified",  # Step execution
                "Final synthesis: x = 2"  # Synthesis
            ]

            result = await planning_pattern.reason(sample_context)

            assert result.success is True
            assert result.pattern_used == "Planning"
            assert len(result.reasoning_steps) >= 2  # Breakdown + execution

    def test_planning_confidence_calculation(self, planning_pattern):
        """Test planning confidence calculation"""
        # Test with multiple execution results
        execution_results = ["Step 1 completed", "Step 2 completed", "Step 3 completed"]
        confidence = planning_pattern._calculate_planning_confidence(execution_results)

        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably high with multiple steps


class TestMultiAgentPattern:
    """Test suite for Multi-Agent pattern implementation"""

    @pytest.fixture
    def multi_agent_pattern(self, mock_dependencies):
        """Fixture for Multi-Agent pattern instance"""
        config = ReasoningConfig(pattern="Multi-Agent", model="gpt-4")
        return MultiAgentPattern(config, mock_dependencies)

    @pytest.fixture
    def mock_dependencies(self):
        """Fixture for mock dependencies"""
        return {
            "llm_service": Mock(),
            "memory_manager": Mock(),
            "plugin_registry": Mock(),
            "rule_engine": Mock()
        }

    def test_multi_agent_initialization(self, multi_agent_pattern):
        """Test Multi-Agent pattern initialization"""
        assert multi_agent_pattern.config.pattern == "Multi-Agent"

    def test_multi_agent_capabilities(self, multi_agent_pattern):
        """Test Multi-Agent pattern capabilities"""
        capabilities = multi_agent_pattern.get_capabilities()

        assert capabilities["name"] == "Multi-Agent"
        assert "max_agents" in capabilities

    @pytest.mark.asyncio
    async def test_multi_agent_reasoning(self, multi_agent_pattern, sample_context):
        """Test Multi-Agent reasoning execution"""
        with patch.object(multi_agent_pattern, '_call_llm') as mock_llm:
            mock_llm.side_effect = [
                "Agent 1: Math Expert\nAgent 2: Logic Verifier\nAgent 3: Solution Checker",  # Role assignment
                "Math Expert: This is a linear equation. x = 2",  # Agent 1 execution
                "Logic Verifier: The algebraic steps are correct",  # Agent 2 execution
                "Solution Checker: x = 2 satisfies the equation",  # Agent 3 execution
                "Synthesis: All agents agree x = 2 is the correct solution"  # Synthesis
            ]

            result = await multi_agent_pattern.reason(sample_context)

            assert result.success is True
            assert result.pattern_used == "Multi-Agent"
            assert len(result.reasoning_steps) >= 2  # Role assignment + synthesis

    def test_multi_agent_confidence_calculation(self, multi_agent_pattern):
        """Test multi-agent confidence calculation"""
        # Test with multiple agent results
        agent_results = [
            {"role": "Math Expert", "result": "x = 2"},
            {"role": "Logic Verifier", "result": "Steps are correct"},
            {"role": "Solution Checker", "result": "Solution verified"}
        ]

        confidence = multi_agent_pattern._calculate_multi_agent_confidence(agent_results)

        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably high with multiple agents


class TestIntegration:
    """Integration tests for reasoning patterns"""

    @pytest.mark.asyncio
    async def test_factory_with_all_patterns(self, sample_context):
        """Test factory with all supported patterns"""
        factory = ReasoningModuleFactory()

        # Mock dependencies
        mock_deps = {
            "llm_service": Mock(),
            "memory_manager": Mock(),
            "plugin_registry": Mock(),
            "rule_engine": Mock()
        }

        for name, dep in mock_deps.items():
            factory.register_dependency(name, dep)

        patterns = factory.get_supported_patterns()

        for pattern in patterns:
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.json.return_value = {"response": f"Pattern {pattern} result"}
                mock_response.raise_for_status.return_value = None
                mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

                result = await factory.reason_with_pattern(pattern, sample_context)

                assert result.success is True
                assert result.pattern_used == pattern
                assert isinstance(result, ReasoningResult)

    @pytest.mark.asyncio
    async def test_error_handling(self, sample_context):
        """Test error handling across patterns"""
        factory = ReasoningModuleFactory()

        # Test with no dependencies registered
        with pytest.raises(Exception):
            await factory.reason_with_pattern("ReAct", sample_context)

        # Test with invalid pattern
        with pytest.raises(ValueError):
            await factory.reason_with_pattern("InvalidPattern", sample_context)


if __name__ == "__main__":
    pytest.main([__file__])
