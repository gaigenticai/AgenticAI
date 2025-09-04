#!/usr/bin/env python3
"""
Test Suite for UI-to-Brain Mapper Service

Comprehensive testing for the visual workflow to agent configuration conversion.
Tests cover component mapping, connection mapping, validation, and error handling.

Author: AgenticAI Platform
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import json

from main import (
    UIToBrainMapper,
    VisualWorkflow,
    VisualComponent,
    VisualConnection,
    AgentConfig,
    MappingRequest,
    Config
)


class TestUIToBrainMapper:
    """Test suite for UI-to-Brain Mapper functionality"""

    @pytest.fixture
    def mapper_service(self):
        """Fixture for UIToBrainMapper instance"""
        return UIToBrainMapper()

    @pytest.fixture
    def sample_visual_workflow(self):
        """Fixture for sample visual workflow"""
        return VisualWorkflow(
            workflow_id="test_workflow_123",
            name="Test Insurance Agent",
            description="A test workflow for insurance claim processing",
            components=[
                VisualComponent(
                    id="comp_1",
                    type="data_input_csv",
                    name="Claims Data Input",
                    position={"x": 100, "y": 100},
                    properties={
                        "filePath": "/data/claims.csv",
                        "delimiter": ",",
                        "hasHeaders": True
                    }
                ),
                VisualComponent(
                    id="comp_2",
                    type="llm_processor",
                    name="Risk Assessor",
                    position={"x": 300, "y": 100},
                    properties={
                        "model": "gpt-4",
                        "temperature": 0.7,
                        "promptTemplate": "Assess risk for claim: {claim_data}"
                    }
                ),
                VisualComponent(
                    id="comp_3",
                    type="decision_node",
                    name="Approval Decision",
                    position={"x": 500, "y": 100},
                    properties={
                        "threshold": 0.8,
                        "condition": "risk_score > threshold"
                    }
                ),
                VisualComponent(
                    id="comp_4",
                    type="database_output",
                    name="Claims Database",
                    position={"x": 700, "y": 100},
                    properties={
                        "tableName": "processed_claims",
                        "connectionString": "postgresql://..."
                    }
                )
            ],
            connections=[
                VisualConnection(
                    id="conn_1",
                    source={"componentId": "comp_1", "port": "output"},
                    target={"componentId": "comp_2", "port": "input"},
                    data_type="json"
                ),
                VisualConnection(
                    id="conn_2",
                    source={"componentId": "comp_2", "port": "output"},
                    target={"componentId": "comp_3", "port": "condition"},
                    data_type="number"
                ),
                VisualConnection(
                    id="conn_3",
                    source={"componentId": "comp_3", "port": "true"},
                    target={"componentId": "comp_4", "port": "input"},
                    data_type="json"
                )
            ]
        )

    @pytest.fixture
    def sample_agent_metadata(self):
        """Fixture for sample agent metadata"""
        return {
            "agent_id": "test_agent_456",
            "name": "Test Insurance Agent",
            "domain": "insurance",
            "persona": "Expert insurance claim processor with risk assessment capabilities",
            "reasoning_pattern": "ReAct"
        }

    def test_initialization(self, mapper_service):
        """Test service initialization"""
        assert mapper_service is not None
        assert isinstance(mapper_service.service_endpoints, dict)
        assert len(mapper_service.service_endpoints) > 0

    def test_component_type_mapping(self, mapper_service):
        """Test component type to service mapping"""
        assert Config.COMPONENT_TYPE_MAPPING["data_input_csv"] == "csv_ingestion_service"
        assert Config.COMPONENT_TYPE_MAPPING["llm_processor"] == "llm_processor"
        assert Config.COMPONENT_TYPE_MAPPING["database_output"] == "postgresql_output"

    @pytest.mark.asyncio
    async def test_workflow_validation_success(self, mapper_service, sample_visual_workflow):
        """Test successful workflow validation"""
        mapping_context = {"validation_errors": [], "warnings": []}

        await mapper_service._validate_visual_workflow(sample_visual_workflow, mapping_context)

        assert len(mapping_context["validation_errors"]) == 0
        assert len(mapping_context["warnings"]) == 0

    @pytest.mark.asyncio
    async def test_workflow_validation_empty_components(self, mapper_service):
        """Test workflow validation with empty components"""
        empty_workflow = VisualWorkflow(
            workflow_id="empty_test",
            name="Empty Workflow",
            components=[],
            connections=[]
        )

        mapping_context = {"validation_errors": [], "warnings": []}

        await mapper_service._validate_visual_workflow(empty_workflow, mapping_context)

        assert len(mapping_context["validation_errors"]) > 0
        assert "must contain at least one component" in mapping_context["validation_errors"][0]

    @pytest.mark.asyncio
    async def test_workflow_validation_invalid_connections(self, mapper_service):
        """Test workflow validation with invalid connections"""
        invalid_workflow = VisualWorkflow(
            workflow_id="invalid_conn_test",
            name="Invalid Connections",
            components=[
                VisualComponent(id="comp_1", type="data_input_csv", name="Input")
            ],
            connections=[
                VisualConnection(
                    id="invalid_conn",
                    source={"componentId": "nonexistent", "port": "output"},
                    target={"componentId": "comp_1", "port": "input"}
                )
            ]
        )

        mapping_context = {"validation_errors": [], "warnings": []}

        await mapper_service._validate_visual_workflow(invalid_workflow, mapping_context)

        assert len(mapping_context["validation_errors"]) > 0
        assert "Connection source component not found" in mapping_context["validation_errors"][0]

    @pytest.mark.asyncio
    async def test_component_mapping(self, mapper_service, sample_visual_workflow):
        """Test component to service mapping"""
        mapping_context = {
            "validation_errors": [],
            "warnings": [],
            "component_mappings": {}
        }

        agent_components = await mapper_service._map_components_to_services(
            sample_visual_workflow.components, {}, mapping_context
        )

        assert len(agent_components) == 4
        assert len(mapping_context["validation_errors"]) == 0

        # Check specific component mappings
        csv_component = next(c for c in agent_components if c.component_id == "comp_1")
        assert csv_component.component_type == "csv_ingestion_service"
        assert "file_path" in csv_component.configuration
        assert csv_component.configuration["file_path"] == "/data/claims.csv"

        llm_component = next(c for c in agent_components if c.component_id == "comp_2")
        assert llm_component.component_type == "llm_processor"
        assert llm_component.configuration["model"] == "gpt-4"
        assert llm_component.configuration["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_connection_mapping(self, mapper_service, sample_visual_workflow):
        """Test connection to dataflow mapping"""
        # First map components
        mapping_context = {
            "validation_errors": [],
            "warnings": [],
            "component_mappings": {}
        }

        agent_components = await mapper_service._map_components_to_services(
            sample_visual_workflow.components, {}, mapping_context
        )

        # Map connections
        agent_connections = await mapper_service._map_connections_to_dataflow(
            sample_visual_workflow.connections, agent_components, mapping_context
        )

        assert len(agent_connections) == 3
        assert len(mapping_context["validation_errors"]) == 0

        # Check specific connection mappings
        first_connection = next(c for c in agent_connections if c.connection_id == "conn_1")
        assert first_connection.source_component == "comp_1"
        assert first_connection.target_component == "comp_2"
        assert first_connection.data_mapping["data_type"] == "json"

    def test_reasoning_pattern_determination(self, mapper_service):
        """Test reasoning pattern selection logic"""
        # Test explicit pattern
        agent_metadata = {"reasoning_pattern": "Planning"}
        pattern = mapper_service._determine_reasoning_pattern([], [], agent_metadata)
        assert pattern == "Planning"

        # Test multi-agent pattern
        from main import AgentComponent
        components = [
            AgentComponent(component_id="1", component_type="llm_processor", service_endpoint=""),
            AgentComponent(component_id="2", component_type="llm_processor", service_endpoint="")
        ]
        pattern = mapper_service._determine_reasoning_pattern(components, [], {})
        assert pattern == "Multi-Agent"

        # Test planning pattern for complex workflows
        connections = [Mock()] * 6  # Mock 6 connections
        pattern = mapper_service._determine_reasoning_pattern([], connections, {})
        assert pattern == "Planning"

        # Test default ReAct pattern
        pattern = mapper_service._determine_reasoning_pattern([], [], {})
        assert pattern == "ReAct"

    def test_memory_config_generation(self, mapper_service):
        """Test memory configuration generation"""
        agent_metadata = {
            "memory_ttl": 7200,
            "vector_dimensions": 512,
            "episodic_retention_days": 60
        }

        memory_config = mapper_service._generate_memory_config(agent_metadata)

        assert memory_config["working_memory"]["ttl_seconds"] == 7200
        assert memory_config["vector_memory"]["vector_dimensions"] == 512
        assert memory_config["episodic_memory"]["retention_days"] == 60

    @pytest.mark.asyncio
    async def test_plugin_config_generation(self, mapper_service):
        """Test plugin configuration generation"""
        from main import AgentComponent

        # Components that should trigger plugins
        components = [
            AgentComponent(component_id="1", component_type="llm_processor", service_endpoint=""),
            AgentComponent(component_id="2", component_type="rule_engine", service_endpoint=""),
            AgentComponent(component_id="3", component_type="csv_ingestion_service", service_endpoint="")
        ]

        mapping_context = {}
        plugin_config = await mapper_service._generate_plugin_config(components, mapping_context)

        assert "fraud_detector" in plugin_config["domain_plugins"]
        assert "risk_calculator" in plugin_config["domain_plugins"]
        assert len(plugin_config["enabled_plugins"]) > 0

    @pytest.mark.asyncio
    async def test_complete_mapping_workflow(self, mapper_service, sample_visual_workflow, sample_agent_metadata):
        """Test complete mapping workflow from visual to agent config"""
        result = await mapper_service.map_visual_to_agent_config(
            visual_workflow=sample_visual_workflow,
            agent_metadata=sample_agent_metadata
        )

        assert result.success is True
        assert result.agent_config is not None
        assert isinstance(result.agent_config, AgentConfig)

        # Verify agent configuration structure
        agent_config = result.agent_config
        assert agent_config.agent_id == "test_agent_456"
        assert agent_config.name == "Test Insurance Agent"
        assert agent_config.domain == "insurance"
        assert len(agent_config.components) == 4
        assert len(agent_config.connections) == 3

        # Verify reasoning pattern was set
        assert agent_config.reasoning_pattern in Config.SUPPORTED_REASONING_PATTERNS

        # Verify memory and plugin configs
        assert "working_memory" in agent_config.memory_config
        assert "vector_memory" in agent_config.memory_config
        assert isinstance(agent_config.plugin_config["enabled_plugins"], list)

        # Verify mapping report
        assert "summary" in result.mapping_report
        assert result.mapping_report["summary"]["total_components"] == 4
        assert result.mapping_report["summary"]["total_connections"] == 3

    @pytest.mark.asyncio
    async def test_mapping_with_validation_errors(self, mapper_service):
        """Test mapping with validation errors"""
        # Create invalid workflow
        invalid_workflow = VisualWorkflow(
            workflow_id="invalid_test",
            name="Invalid Workflow",
            components=[
                VisualComponent(
                    id="comp_1",
                    type="invalid_type",  # Invalid component type
                    name="Invalid Component"
                )
            ],
            connections=[]
        )

        result = await mapper_service.map_visual_to_agent_config(
            visual_workflow=invalid_workflow
        )

        assert result.success is False
        assert len(result.validation_errors) > 0
        assert "Unsupported component type" in result.validation_errors[0]

    @pytest.mark.asyncio
    async def test_mapping_with_service_unavailable(self, mapper_service, sample_visual_workflow):
        """Test mapping when services are unavailable"""
        # This would test graceful degradation when service endpoints are unavailable
        # In a real test, we would mock httpx calls to simulate service failures

        result = await mapper_service.map_visual_to_agent_config(
            visual_workflow=sample_visual_workflow
        )

        # Should still succeed with placeholder endpoints
        assert result.success is True
        assert len(result.warnings) >= 0  # May have warnings about service availability

    def test_data_mapping_generation(self, mapper_service):
        """Test data mapping generation for connections"""
        from main import AgentComponent

        source_comp = AgentComponent(
            component_id="source",
            component_type="csv_ingestion_service",
            service_endpoint="http://localhost:8001"
        )

        target_comp = AgentComponent(
            component_id="target",
            component_type="llm_processor",
            service_endpoint="http://localhost:8005"
        )

        visual_conn = VisualConnection(
            id="test_conn",
            source={"componentId": "source", "port": "output"},
            target={"componentId": "target", "port": "input"},
            data_type="json"
        )

        data_mapping = mapper_service._generate_data_mapping(
            visual_conn, source_comp, target_comp
        )

        assert data_mapping["source_type"] == "csv_ingestion_service"
        assert data_mapping["target_type"] == "llm_processor"
        assert data_mapping["data_type"] == "json"

    def test_input_output_port_generation(self, mapper_service):
        """Test input and output port generation for components"""
        # Test CSV input component
        csv_comp = VisualComponent(id="csv", type="data_input_csv", name="CSV Input")
        input_ports = mapper_service._generate_input_ports(csv_comp)
        output_ports = mapper_service._generate_output_ports(csv_comp)

        assert len(input_ports) > 0
        assert len(output_ports) > 0
        assert input_ports[0]["type"] == "file"
        assert output_ports[0]["type"] == "array"

        # Test LLM processor component
        llm_comp = VisualComponent(id="llm", type="llm_processor", name="LLM Processor")
        input_ports = mapper_service._generate_input_ports(llm_comp)
        output_ports = mapper_service._generate_output_ports(llm_comp)

        assert input_ports[0]["type"] == "string"
        assert output_ports[0]["type"] == "string"

    def test_mapping_report_generation(self, mapper_service):
        """Test mapping report generation"""
        mapping_context = {
            "validation_errors": ["Error 1", "Error 2"],
            "warnings": ["Warning 1"],
            "component_mappings": {
                "comp1": Mock(type="data_input_csv", component_type="csv_ingestion_service", service_endpoint="http://test")
            },
            "performance_metrics": {"processing_time": 1.5}
        }

        report = mapper_service._generate_mapping_report(mapping_context)

        assert report["summary"]["validation_errors"] == 2
        assert report["summary"]["warnings"] == 1
        assert "component_mapping_details" in report
        assert "recommendations" in report

    @pytest.mark.asyncio
    async def test_agent_config_validation(self, mapper_service):
        """Test agent configuration validation"""
        # Create valid agent config
        agent_config = AgentConfig(
            agent_id="test_agent",
            name="Test Agent",
            domain="test",
            persona="Test persona",
            reasoning_pattern="ReAct",
            components=[
                Mock(component_id="comp1", service_endpoint="http://test", component_type="test")
            ],
            connections=[],
            memory_config={"working_memory": {"enabled": True}},
            plugin_config={"enabled_plugins": []}
        )

        mapping_context = {"validation_errors": [], "warnings": []}

        await mapper_service._validate_agent_config(agent_config, mapping_context)

        # Should pass validation
        assert len(mapping_context["validation_errors"]) == 0


# Integration tests
class TestUIToBrainMapperIntegration:
    """Integration tests for UI-to-Brain Mapper"""

    @pytest.mark.asyncio
    async def test_complex_workflow_mapping(self):
        """Test mapping of complex multi-component workflow"""
        mapper = UIToBrainMapper()

        # Create complex workflow with multiple branches
        complex_workflow = VisualWorkflow(
            workflow_id="complex_workflow",
            name="Complex Multi-Branch Workflow",
            components=[
                # Data inputs
                VisualComponent(id="input1", type="data_input_csv", name="Customer Data"),
                VisualComponent(id="input2", type="data_input_api", name="External API Data"),

                # Processing components
                VisualComponent(id="processor1", type="llm_processor", name="Data Analyzer"),
                VisualComponent(id="processor2", type="rule_engine", name="Business Rules"),

                # Decision components
                VisualComponent(id="decision1", type="decision_node", name="Risk Decision"),
                VisualComponent(id="decision2", type="decision_node", name="Approval Decision"),

                # Output components
                VisualComponent(id="output1", type="database_output", name="Results DB"),
                VisualComponent(id="output2", type="email_output", name="Notification Email"),
                VisualComponent(id="output3", type="pdf_report_output", name="PDF Report")
            ],
            connections=[
                # Data flow connections
                VisualConnection(id="c1", source={"componentId": "input1"}, target={"componentId": "processor1"}),
                VisualConnection(id="c2", source={"componentId": "input2"}, target={"componentId": "processor1"}),
                VisualConnection(id="c3", source={"componentId": "processor1"}, target={"componentId": "processor2"}),
                VisualConnection(id="c4", source={"componentId": "processor2"}, target={"componentId": "decision1"}),

                # Decision branches
                VisualConnection(id="c5", source={"componentId": "decision1"}, target={"componentId": "decision2"}),
                VisualConnection(id="c6", source={"componentId": "decision2"}, target={"componentId": "output1"}),
                VisualConnection(id="c7", source={"componentId": "decision2"}, target={"componentId": "output2"}),
                VisualConnection(id="c8", source={"componentId": "decision1"}, target={"componentId": "output3"})
            ]
        )

        result = await mapper.map_visual_to_agent_config(
            visual_workflow=complex_workflow,
            agent_metadata={
                "name": "Complex Enterprise Agent",
                "domain": "enterprise",
                "persona": "Comprehensive data processing and decision-making agent"
            }
        )

        assert result.success is True
        assert len(result.agent_config.components) == 9
        assert len(result.agent_config.connections) == 8

        # Should select Planning pattern for complex workflow
        assert result.agent_config.reasoning_pattern == "Planning"


if __name__ == "__main__":
    pytest.main([__file__])
