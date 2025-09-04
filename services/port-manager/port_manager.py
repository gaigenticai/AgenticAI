#!/usr/bin/env python3
"""
Port Manager Service for Agentic Platform

Automatic port conflict resolution service that:
- Monitors Docker container port usage
- Detects port conflicts before they occur
- Suggests alternative ports
- Updates configuration files automatically
"""

import asyncio
import json
import os
import socket
import subprocess
import time
from typing import Dict, List, Optional, Set

import docker
import structlog
from fastapi import FastAPI, HTTPException

# Configure logging
logger = structlog.get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="Port Manager Service",
    description="Automatic port conflict resolution for Docker services",
    version="1.0.0"
)

# Docker client
docker_client = docker.from_env()

# Port monitoring state
monitored_ports: Set[int] = set()
last_check_time = 0

def check_port_in_use(port: int, host: str = '0.0.0.0') -> bool:
    """Check if a port is currently in use"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0  # Port is in use if connection succeeds
    except:
        return False

def get_docker_container_ports() -> Dict[str, List[int]]:
    """Get ports currently used by Docker containers"""
    container_ports = {}

    try:
        for container in docker_client.containers.list():
            container_info = docker_client.api.inspect_container(container.id)
            ports = container_info.get('NetworkSettings', {}).get('Ports', {})

            container_ports[container.name] = []
            for port_mapping in ports.values():
                if port_mapping:
                    for mapping in port_mapping:
                        if 'HostPort' in mapping:
                            try:
                                port = int(mapping['HostPort'])
                                container_ports[container.name].append(port)
                            except ValueError:
                                continue

    except Exception as e:
        logger.error("Failed to get Docker container ports", error=str(e))

    return container_ports

def find_available_port(start_port: int, exclude_ports: Set[int] = None) -> Optional[int]:
    """Find an available port starting from the given port"""
    if exclude_ports is None:
        exclude_ports = set()

    max_attempts = 100
    for port in range(start_port, start_port + max_attempts):
        if port not in exclude_ports and not check_port_in_use(port):
            return port
    return None

def resolve_port_conflicts() -> Dict[str, Dict[str, int]]:
    """Resolve port conflicts for services"""
    conflicts_resolved = {}
    container_ports = get_docker_container_ports()

    # Flatten all used ports
    used_ports = set()
    for ports in container_ports.values():
        used_ports.update(ports)

    # Check common service ports from docker-compose.yml
    common_ports = {
        'INGESTION_COORDINATOR_PORT': 8080,
        'OUTPUT_COORDINATOR_PORT': 8081,
        'VECTOR_UI_PORT': 8082,
        'CSV_INGESTION_PORT': 8082,
        'PDF_INGESTION_PORT': 8083,
        'EXCEL_INGESTION_PORT': 8084,
        'JSON_INGESTION_PORT': 8085,
        'API_INGESTION_PORT': 8086,
        'UI_SCRAPER_PORT': 8087,
        'RABBITMQ_PORT': 5672,
        'POSTGRES_INGESTION_PORT': 5432,
        'POSTGRES_OUTPUT_PORT': 5433,
        'REDIS_INGESTION_PORT': 6379,
        'MINIO_BRONZE_PORT': 9000,
        'MINIO_SILVER_PORT': 9010,
        'MINIO_GOLD_PORT': 9020,
        'GRAFANA_PORT': 3000,
        'PROMETHEUS_PORT': 9090,
        'QDRANT_HTTP_PORT': 6333,
        'QDRANT_GRPC_PORT': 6334,
        'JAEGER_UI_PORT': 16686
    }

    for service_name, default_port in common_ports.items():
        if default_port in used_ports:
            # Port conflict detected
            alternative_port = find_available_port(default_port + 1, used_ports)
            if alternative_port:
                conflicts_resolved[service_name] = {
                    'original_port': default_port,
                    'suggested_port': alternative_port,
                    'conflict_with': get_conflicting_service(default_port, container_ports)
                }
                logger.info(f"Port conflict resolved for {service_name}",
                          original=default_port,
                          suggested=alternative_port)

    return conflicts_resolved

def get_conflicting_service(port: int, container_ports: Dict[str, List[int]]) -> Optional[str]:
    """Find which service is using the conflicting port"""
    for service_name, ports in container_ports.items():
        if port in ports:
            return service_name
    return None

def update_environment_file(conflicts: Dict[str, Dict[str, int]]):
    """Update .env file with resolved ports"""
    if not conflicts:
        return

    env_file = '.env'
    env_lines = []

    # Read existing .env file or .env.example
    source_file = env_file if os.path.exists(env_file) else '.env.example'

    try:
        with open(source_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    if key in conflicts:
                        new_value = str(conflicts[key]['suggested_port'])
                        env_lines.append(f"{key}={new_value}")
                        logger.info(f"Updated {key}: {value} -> {new_value}")
                    else:
                        env_lines.append(line)
                else:
                    env_lines.append(line)

        # Write updated .env file
        with open(env_file, 'w') as f:
            f.write('\n'.join(env_lines) + '\n')

        logger.info("Environment file updated successfully")

    except Exception as e:
        logger.error("Failed to update environment file", error=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "port-manager",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.get("/check-conflicts")
async def check_conflicts():
    """Check for port conflicts"""
    conflicts = resolve_port_conflicts()
    return {
        "conflicts_found": len(conflicts),
        "conflicts": conflicts,
        "timestamp": time.time()
    }

@app.post("/resolve-conflicts")
async def resolve_conflicts():
    """Resolve detected port conflicts"""
    conflicts = resolve_port_conflicts()

    if not conflicts:
        return {
            "status": "success",
            "message": "No port conflicts detected",
            "conflicts_resolved": 0
        }

    # Update environment file
    update_environment_file(conflicts)

    return {
        "status": "success",
        "message": f"Resolved {len(conflicts)} port conflicts",
        "conflicts": conflicts
    }

@app.get("/used-ports")
async def get_used_ports():
    """Get currently used ports"""
    container_ports = get_docker_container_ports()
    used_ports = set()

    for ports in container_ports.values():
        used_ports.update(ports)

    return {
        "used_ports": sorted(list(used_ports)),
        "container_ports": container_ports,
        "total_used": len(used_ports)
    }

@app.get("/available-port/{start_port}")
async def find_available_port_endpoint(start_port: int):
    """Find an available port starting from the given port"""
    container_ports = get_docker_container_ports()
    used_ports = set()

    for ports in container_ports.values():
        used_ports.update(ports)

    available_port = find_available_port(start_port, used_ports)

    if available_port:
        return {
            "available_port": available_port,
            "searched_from": start_port
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No available ports found starting from {start_port}"
        )

if __name__ == "__main__":
    import uvicorn

    logger.info("Port Manager Service starting up...")

    uvicorn.run(
        "port_manager:app",
        host="0.0.0.0",
        port=8088,  # Default port for port manager
        reload=False,
        log_level="info"
    )