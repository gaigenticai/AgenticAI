#!/usr/bin/env python3
"""
Docker-Based Automated Testing Framework for Agentic Platform

This module implements Rule 18 compliance by providing Docker-based automated testing
for all platform services. Each service is tested in isolated Docker containers to
ensure production-grade reliability and prevent development workflow interruptions.

Features:
- Isolated Docker testing environments for each service
- Parallel test execution across services
- Comprehensive test reporting and metrics
- Integration with existing test suites
- Automatic container lifecycle management
- Health checks and service validation
- Performance benchmarking in containerized environments
"""

import asyncio
import docker
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import requests
import structlog

# Configure structured logging
logger = structlog.get_logger(__name__)


@dataclass
class DockerTestResult:
    """Container for Docker test execution results"""
    service_name: str
    test_type: str
    status: str  # 'PASS', 'FAIL', 'ERROR', 'SKIP'
    duration: float
    output: str
    error_message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ServiceTestConfig:
    """Configuration for service-specific Docker testing"""
    service_name: str
    image_name: str
    container_name: str
    test_command: List[str]
    health_check_url: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)
    ports: Dict[str, str] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    timeout: int = 300  # 5 minutes default timeout


class DockerTestRunner:
    """Docker-based automated testing framework"""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.test_results: List[DockerTestResult] = []
        self.active_containers: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.project_name = "agentic-testing"

    def get_service_configs(self) -> Dict[str, ServiceTestConfig]:
        """Get test configurations for all services"""

        base_env = {
            "POSTGRES_HOST": "postgres_test",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "agentic_test",
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_password",
            "REDIS_HOST": "redis_test",
            "REDIS_PORT": "6379",
            "RABBITMQ_HOST": "rabbitmq_test",
            "RABBITMQ_PORT": "5672",
            "RABBITMQ_USER": "test_user",
            "RABBITMQ_PASSWORD": "test_password"
        }

        return {
            "ingestion-coordinator": ServiceTestConfig(
                service_name="ingestion-coordinator",
                image_name="agentic/ingestion-coordinator:test",
                container_name="test_ingestion_coordinator",
                test_command=["python", "-m", "pytest", "/app/tests/", "-v"],
                health_check_url="http://localhost:8080/health",
                environment_vars={**base_env, "INGESTION_COORDINATOR_PORT": "8080"},
                ports={"8080": "8080"},
                depends_on=["postgres_test", "redis_test", "rabbitmq_test"]
            ),

            "output-coordinator": ServiceTestConfig(
                service_name="output-coordinator",
                image_name="agentic/output-coordinator:test",
                container_name="test_output_coordinator",
                test_command=["python", "-m", "pytest", "/app/tests/", "-v"],
                health_check_url="http://localhost:8081/health",
                environment_vars={**base_env, "OUTPUT_COORDINATOR_PORT": "8081"},
                ports={"8081": "8081"},
                depends_on=["postgres_test", "redis_test"]
            ),

            "agent-orchestrator": ServiceTestConfig(
                service_name="agent-orchestrator",
                image_name="agentic/agent-orchestrator:test",
                container_name="test_agent_orchestrator",
                test_command=["python", "-m", "pytest", "/app/tests/", "-v"],
                health_check_url="http://localhost:8200/health",
                environment_vars={**base_env, "AGENT_ORCHESTRATOR_PORT": "8200"},
                ports={"8200": "8200"},
                depends_on=["postgres_test", "redis_test", "rabbitmq_test"]
            ),

            "plugin-registry": ServiceTestConfig(
                service_name="plugin-registry",
                image_name="agentic/plugin-registry:test",
                container_name="test_plugin_registry",
                test_command=["python", "-m", "pytest", "/app/tests/", "-v"],
                health_check_url="http://localhost:8201/health",
                environment_vars={**base_env, "PLUGIN_REGISTRY_PORT": "8201"},
                ports={"8201": "8201"},
                depends_on=["postgres_test", "redis_test"]
            ),

            "workflow-engine": ServiceTestConfig(
                service_name="workflow-engine",
                image_name="agentic/workflow-engine:test",
                container_name="test_workflow_engine",
                test_command=["python", "-m", "pytest", "/app/tests/", "-v"],
                health_check_url="http://localhost:8202/health",
                environment_vars={**base_env, "WORKFLOW_ENGINE_PORT": "8202"},
                ports={"8202": "8202"},
                depends_on=["postgres_test", "redis_test", "rabbitmq_test"]
            ),

            "vector-ui": ServiceTestConfig(
                service_name="vector-ui",
                image_name="agentic/vector-ui:test",
                container_name="test_vector_ui",
                test_command=["python", "-m", "pytest", "/app/tests/", "-v", "--tb=short"],
                health_check_url="http://localhost:8082/health",
                environment_vars={**base_env, "VECTOR_UI_PORT": "8082"},
                ports={"8082": "8082"},
                depends_on=["qdrant_test", "redis_test"]
            ),

            "brain-factory": ServiceTestConfig(
                service_name="brain-factory",
                image_name="agentic/brain-factory:test",
                container_name="test_brain_factory",
                test_command=["python", "-m", "pytest", "/app/tests/", "-v"],
                health_check_url="http://localhost:8301/health",
                environment_vars={**base_env, "BRAIN_FACTORY_PORT": "8301"},
                ports={"8301": "8301"},
                depends_on=["postgres_test", "redis_test"]
            ),

            "data-encryption": ServiceTestConfig(
                service_name="data-encryption",
                image_name="agentic/data-encryption:test",
                container_name="test_data_encryption",
                test_command=["python", "-m", "pytest", "/app/tests/", "-v"],
                health_check_url="http://localhost:8094/health",
                environment_vars={**base_env, "DATA_ENCRYPTION_PORT": "8094"},
                ports={"8094": "8094"},
                depends_on=[]
            ),

            "port-manager": ServiceTestConfig(
                service_name="port-manager",
                image_name="agentic/port-manager:test",
                container_name="test_port_manager",
                test_command=["python", "-m", "pytest", "/app/tests/", "-v"],
                health_check_url="http://localhost:8098/health",
                environment_vars={**base_env, "PORT_MANAGER_PORT": "8098"},
                ports={"8098": "8098"},
                depends_on=[]
            )
        }

    def create_test_infrastructure(self) -> Dict[str, Any]:
        """Create test infrastructure containers (databases, message queues, etc.)"""

        logger.info("Creating test infrastructure containers")

        infrastructure = {}

        # PostgreSQL test database
        postgres_config = {
            "image": "postgres:15-alpine",
            "name": "postgres_test",
            "environment": {
                "POSTGRES_DB": "agentic_test",
                "POSTGRES_USER": "test_user",
                "POSTGRES_PASSWORD": "test_password"
            },
            "ports": {"5432": "5432"},
            "healthcheck": {
                "test": ["CMD-SHELL", "pg_isready -U test_user -d agentic_test"],
                "interval": 10000000000,  # 10 seconds
                "timeout": 5000000000,   # 5 seconds
                "retries": 5
            }
        }

        # Redis test cache
        redis_config = {
            "image": "redis:7-alpine",
            "name": "redis_test",
            "ports": {"6379": "6379"},
            "healthcheck": {
                "test": ["CMD", "redis-cli", "ping"],
                "interval": 10000000000,
                "timeout": 3000000000,
                "retries": 5
            }
        }

        # RabbitMQ test message queue
        rabbitmq_config = {
            "image": "rabbitmq:3-management-alpine",
            "name": "rabbitmq_test",
            "environment": {
                "RABBITMQ_DEFAULT_USER": "test_user",
                "RABBITMQ_DEFAULT_PASS": "test_password"
            },
            "ports": {"5672": "5672", "15672": "15672"},
            "healthcheck": {
                "test": ["CMD", "rabbitmqctl", "status"],
                "interval": 10000000000,
                "timeout": 10000000000,
                "retries": 5
            }
        }

        # Qdrant test vector database
        qdrant_config = {
            "image": "qdrant/qdrant:v1.7.4",
            "name": "qdrant_test",
            "ports": {"6333": "6333", "6334": "6334"},
            "volumes": {
                "/tmp/qdrant_test_data": {"bind": "/qdrant/storage", "mode": "rw"}
            },
            "healthcheck": {
                "test": ["CMD", "curl", "-f", "http://localhost:6333/health"],
                "interval": 10000000000,
                "timeout": 5000000000,
                "retries": 5
            }
        }

        infrastructure_configs = {
            "postgres": postgres_config,
            "redis": redis_config,
            "rabbitmq": rabbitmq_config,
            "qdrant": qdrant_config
        }

        # Start infrastructure containers
        for name, config in infrastructure_configs.items():
            try:
                logger.info(f"Starting {name} test infrastructure")

                container = self.docker_client.containers.run(
                    config["image"],
                    name=config["name"],
                    environment=config.get("environment", {}),
                    ports=config.get("ports", {}),
                    volumes=config.get("volumes", {}),
                    healthcheck=config.get("healthcheck"),
                    detach=True,
                    remove=True
                )

                infrastructure[name] = container
                self.active_containers[config["name"]] = container

                # Wait for health check
                self._wait_for_container_health(container, timeout=60)

            except Exception as e:
                logger.error(f"Failed to start {name} infrastructure", error=str(e))
                raise

        return infrastructure

    def _wait_for_container_health(self, container, timeout: int = 60):
        """Wait for container to become healthy"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                container.reload()
                if container.status == "running":
                    # Check health status if healthcheck is configured
                    if hasattr(container, 'health') and container.health:
                        if container.health == "healthy":
                            return True
                    else:
                        # No healthcheck configured, assume healthy after short delay
                        time.sleep(2)
                        return True

                time.sleep(1)
            except Exception:
                time.sleep(1)

        raise TimeoutError(f"Container {container.name} failed to become healthy within {timeout}s")

    async def run_service_test(self, config: ServiceTestConfig) -> DockerTestResult:
        """Run Docker-based test for a specific service"""

        start_time = time.time()

        try:
            logger.info(f"Starting Docker test for {config.service_name}")

            # Build test image if needed
            self._build_test_image(config)

            # Run test container
            container = self.docker_client.containers.run(
                config.image_name,
                name=config.container_name,
                command=config.test_command,
                environment=config.environment_vars,
                ports=config.ports,
                volumes=config.volumes,
                network="agentic-testing",
                detach=False,  # Run synchronously for test output
                remove=True,
                timeout=config.timeout
            )

            # Wait for completion and get logs
            result = container.wait(timeout=config.timeout)
            logs = container.logs().decode('utf-8')

            duration = time.time() - start_time

            if result["StatusCode"] == 0:
                status = "PASS"
                error_message = ""
            else:
                status = "FAIL"
                error_message = f"Test failed with exit code {result['StatusCode']}"

            return DockerTestResult(
                service_name=config.service_name,
                test_type="docker_integration",
                status=status,
                duration=duration,
                output=logs,
                error_message=error_message,
                metrics={
                    "exit_code": result["StatusCode"],
                    "container_id": container.id[:12] if container.id else None
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Docker test failed for {config.service_name}", error=str(e))

            return DockerTestResult(
                service_name=config.service_name,
                test_type="docker_integration",
                status="ERROR",
                duration=duration,
                output="",
                error_message=str(e)
            )

    def _build_test_image(self, config: ServiceTestConfig):
        """Build Docker test image for service if it doesn't exist"""

        try:
            self.docker_client.images.get(config.image_name)
            logger.debug(f"Test image {config.image_name} already exists")
            return
        except docker.errors.ImageNotFound:
            logger.info(f"Building test image {config.image_name}")

            # Find service directory
            service_dir = Path(__file__).parent.parent / "services" / config.service_name.replace("-", "_")

            if not service_dir.exists():
                raise FileNotFoundError(f"Service directory not found: {service_dir}")

            # Build test Dockerfile if it doesn't exist
            dockerfile_path = service_dir / "Dockerfile.test"
            if not dockerfile_path.exists():
                self._create_test_dockerfile(service_dir, dockerfile_path)

            # Build image
            self.docker_client.images.build(
                path=str(service_dir),
                dockerfile="Dockerfile.test",
                tag=config.image_name,
                rm=True
            )

            logger.info(f"Successfully built test image {config.image_name}")

    def _create_test_dockerfile(self, service_dir: Path, dockerfile_path: Path):
        """Create a test Dockerfile for the service"""

        dockerfile_content = f"""FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy service files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install test dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov

# Create test directory
RUN mkdir -p /app/tests

# Copy test files if they exist
COPY tests/ /app/tests/ 2>/dev/null || true

# Expose service port
EXPOSE {self._get_service_port(service_dir.name)}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{self._get_service_port(service_dir.name)}/health || exit 1

# Default command
CMD ["python", "main.py"]
"""

        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        logger.info(f"Created test Dockerfile: {dockerfile_path}")

    def _get_service_port(self, service_name: str) -> int:
        """Get the default port for a service"""
        port_map = {
            "ingestion_coordinator": 8080,
            "output_coordinator": 8081,
            "vector_ui": 8082,
            "agent_orchestrator": 8200,
            "plugin_registry": 8201,
            "workflow_engine": 8202,
            "brain_factory": 8301,
            "data_encryption": 8094,
            "port_manager": 8098
        }
        return port_map.get(service_name, 8080)

    async def run_parallel_tests(self, services: List[str] = None) -> List[DockerTestResult]:
        """Run Docker tests for multiple services in parallel"""

        if services is None:
            services = list(self.get_service_configs().keys())

        logger.info(f"Running Docker tests for services: {', '.join(services)}")

        # Create test infrastructure
        try:
            infrastructure = self.create_test_infrastructure()
            logger.info("Test infrastructure created successfully")
        except Exception as e:
            logger.error("Failed to create test infrastructure", error=str(e))
            raise

        # Run service tests in parallel
        configs = self.get_service_configs()
        test_tasks = []

        for service_name in services:
            if service_name in configs:
                config = configs[service_name]
                task = asyncio.create_task(self.run_service_test(config))
                test_tasks.append(task)

        # Wait for all tests to complete
        results = await asyncio.gather(*test_tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                # Handle exceptions from test tasks
                error_result = DockerTestResult(
                    service_name="unknown",
                    test_type="docker_integration",
                    status="ERROR",
                    duration=0,
                    output="",
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        # Cleanup infrastructure
        self._cleanup_containers()

        return processed_results

    def _cleanup_containers(self):
        """Clean up test containers"""
        logger.info("Cleaning up test containers")

        for container_name, container in self.active_containers.items():
            try:
                container.stop(timeout=10)
                logger.debug(f"Stopped container: {container_name}")
            except Exception as e:
                logger.warning(f"Failed to stop container {container_name}", error=str(e))

        self.active_containers.clear()

    def generate_test_report(self, results: List[DockerTestResult]) -> Dict[str, Any]:
        """Generate comprehensive test report"""

        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == "PASS"])
        failed_tests = len([r for r in results if r.status == "FAIL"])
        error_tests = len([r for r in results if r.status == "ERROR"])
        skipped_tests = len([r for r in results if r.status == "SKIP"])

        total_duration = sum(r.duration for r in results)

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "skipped": skipped_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_duration": round(total_duration, 2),
                "average_duration": round(total_duration / total_tests, 2) if total_tests > 0 else 0
            },
            "results": [
                {
                    "service_name": r.service_name,
                    "test_type": r.test_type,
                    "status": r.status,
                    "duration": round(r.duration, 2),
                    "error_message": r.error_message,
                    "timestamp": r.timestamp,
                    "metrics": r.metrics
                }
                for r in results
            ],
            "failed_tests": [
                {
                    "service_name": r.service_name,
                    "error_message": r.error_message,
                    "output": r.output[-500:] if r.output else "",  # Last 500 chars
                    "duration": round(r.duration, 2)
                }
                for r in results if r.status in ["FAIL", "ERROR"]
            ],
            "timestamp": time.time()
        }

        return report

    def save_test_report(self, report: Dict[str, Any], filename: str = None):
        """Save test report to JSON file"""

        if filename is None:
            timestamp = int(time.time())
            filename = f"docker_test_report_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Docker test report saved to: {filename}")

        return filename


async def run_docker_tests(services: List[str] = None, report_file: str = None) -> int:
    """Main function to run Docker-based tests"""

    logger.info("🚀 Starting Docker-Based Automated Testing Suite")
    logger.info("=" * 60)

    runner = DockerTestRunner()

    try:
        # Run tests
        results = await runner.run_parallel_tests(services)

        # Generate report
        report = runner.generate_test_report(results)

        # Save report
        saved_file = runner.save_test_report(report, report_file)

        # Print summary
        summary = report["summary"]
        print("\n" + "=" * 60)
        print("📊 DOCKER TESTING RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Tests:     {summary['total_tests']}")
        print(f"Passed:          {summary['passed']} ✅")
        print(f"Failed:          {summary['failed']} ❌")
        print(f"Errors:          {summary['errors']} 💥")
        print(f"Skipped:         {summary['skipped']} ⚠️")
        print(f"Duration:        {summary['total_duration']:.2f}s")
        print(f"Success Rate:    {summary['success_rate']:.1f}%")
        print(f"Report Saved:    {saved_file}")

        # Print failed tests
        if summary['failed'] + summary['errors'] > 0:
            print("\n❌ FAILED TESTS:")
            for failed in report["failed_tests"]:
                print(f"  • {failed['service_name']}: {failed['error_message']}")

        # Determine exit code
        if summary['failed'] == 0 and summary['errors'] == 0:
            print("\n🎉 ALL DOCKER TESTS PASSED!")
            return 0
        elif summary['success_rate'] >= 70:
            print("\n⚠️ MOST TESTS PASSED. Review failed tests before deployment.")
            return 1
        else:
            print("\n❌ CRITICAL ISSUES DETECTED. Docker testing failed.")
            return 2

    except KeyboardInterrupt:
        logger.warning("Docker testing interrupted by user")
        return 130
    except Exception as e:
        logger.error("Docker testing suite crashed", error=str(e))
        return 1
    finally:
        runner._cleanup_containers()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Docker-Based Automated Testing")
    parser.add_argument("--services", nargs="*", help="Specific services to test")
    parser.add_argument("--report", help="Output report filename")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")

    args = parser.parse_args()

    # Run async tests
    exit_code = asyncio.run(run_docker_tests(args.services, args.report))
    sys.exit(exit_code)
