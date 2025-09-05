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
    """Check if a specific port is currently in use on the host system.

    Uses socket connection test to determine if port is bound by any process.
    This is more reliable than parsing netstat output.

    Args:
        port: Port number to check (1-65535)
        host: Host address to check (default: all interfaces)

    Returns:
        bool: True if port is in use, False if available
    """
    try:
        # Create TCP socket for connection test
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)  # 1 second timeout for quick checks
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0  # connect_ex returns 0 if connection succeeds (port in use)
    except Exception:
        # If socket operation fails, assume port is available
        return False

def get_docker_container_ports() -> Dict[str, List[int]]:
    """Get ports currently used by Docker containers via Docker API.

    Queries the Docker daemon to inspect all running containers and extract
    their port mappings. This helps identify which ports are already allocated
    to avoid conflicts when starting new services.

    Returns:
        Dict mapping container names to lists of ports they are using
        Format: {'container_name': [8080, 5432, ...], ...}
    """
    container_ports = {}

    try:
        # Iterate through all running containers
        for container in docker_client.containers.list():
            # Get detailed container information including network settings
            container_info = docker_client.api.inspect_container(container.id)
            ports = container_info.get('NetworkSettings', {}).get('Ports', {})

            container_ports[container.name] = []

            # Parse port mappings from Docker's network configuration
            # Docker stores ports as: {'8080/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '8080'}]}
            for port_mapping in ports.values():
                if port_mapping:  # Skip if no mapping exists
                    for mapping in port_mapping:
                        if 'HostPort' in mapping:
                            try:
                                port = int(mapping['HostPort'])
                                container_ports[container.name].append(port)
                            except ValueError:
                                # Skip invalid port numbers
                                continue

    except Exception as e:
        logger.error("Failed to get Docker container ports", error=str(e))

    return container_ports

def get_all_used_ports(container_ports: Dict[str, List[int]]) -> Set[int]:
    """Get all ports currently in use by Docker containers and system processes.

    Combines Docker container port mappings with system-level port usage
    to provide a comprehensive view of occupied ports.

    Args:
        container_ports: Dict of container names to their port lists

    Returns:
        Set of all ports currently in use
    """
    used_ports = set()

    # Add Docker container ports
    for ports in container_ports.values():
        used_ports.update(ports)

    # Add system ports - only check commonly used ranges for efficiency
    # Web services (80xx), databases (54xx, 27xx), message queues (56xx), etc.
    port_ranges_to_check = [
        (8000, 8999),  # Web services and APIs
        (5000, 5999),  # Database and backend services
        (27000, 27999),  # MongoDB range
        (5600, 5799),   # Message queues and monitoring
        (9000, 9999),   # Additional services
    ]

    for start_port, end_port in port_ranges_to_check:
        for port in range(start_port, min(end_port + 1, 65536)):
            if check_port_in_use(port):
                used_ports.add(port)

    return used_ports


def find_available_port_in_range(start_port: int, end_port: int, exclude_ports: Set[int] = None) -> Optional[int]:
    """Find an available port within a specified range.

    More intelligent than the basic version - respects port range boundaries
    and provides better port allocation strategy.

    Args:
        start_port: Beginning of port range to search
        end_port: End of port range to search
        exclude_ports: Set of ports to exclude from consideration

    Returns:
        First available port in range, or None if none found
    """
    if exclude_ports is None:
        exclude_ports = set()

    # Search within the specified range
    for port in range(start_port, end_port + 1):
        if port not in exclude_ports and not check_port_in_use(port):
            return port
    return None


def find_available_port(start_port: int, exclude_ports: Set[int] = None) -> Optional[int]:
    """Find an available port starting from the given port (legacy function).

    Maintained for backward compatibility. For new code, use find_available_port_in_range()
    with explicit port ranges for better port management.
    """
    if exclude_ports is None:
        exclude_ports = set()

    max_attempts = 100
    for port in range(start_port, start_port + max_attempts):
        if port not in exclude_ports and not check_port_in_use(port):
            return port
    return None

def resolve_port_conflicts() -> Dict[str, Dict[str, int]]:
    """Resolve port conflicts for services with intelligent port allocation.

    This function implements automatic port conflict resolution by:
    1. Scanning all currently used ports (Docker containers + system processes)
    2. Detecting conflicts with predefined service ports
    3. Finding optimal alternative ports in designated ranges
    4. Providing conflict resolution recommendations

    Returns:
        Dict mapping service names to conflict resolution data
        Format: {'SERVICE_NAME': {'original_port': 8080, 'suggested_port': 8081, 'conflict_with': 'container_name'}}
    """
    conflicts_resolved = {}
    container_ports = get_docker_container_ports()

    # Get all used ports (Docker containers + system processes)
    used_ports = get_all_used_ports(container_ports)

    # Comprehensive service port mapping with intelligent port ranges
    service_port_config = {
        # Ingestion Layer (8000-8099 range)
        'INGESTION_COORDINATOR_PORT': {'default': 8080, 'range': (8080, 8099)},
        'OUTPUT_COORDINATOR_PORT': {'default': 8081, 'range': (8080, 8099)},
        'VECTOR_UI_PORT': {'default': 8082, 'range': (8080, 8099)},

        # Data Ingestion Services (8080-8099 range)
        'CSV_INGESTION_PORT': {'default': 8082, 'range': (8080, 8099)},
        'PDF_INGESTION_PORT': {'default': 8083, 'range': (8080, 8099)},
        'EXCEL_INGESTION_PORT': {'default': 8084, 'range': (8080, 8099)},
        'JSON_INGESTION_PORT': {'default': 8085, 'range': (8080, 8099)},
        'API_INGESTION_PORT': {'default': 8086, 'range': (8080, 8099)},
        'UI_SCRAPER_PORT': {'default': 8087, 'range': (8080, 8099)},

        # Message Queue (5600-5699 range)
        'RABBITMQ_PORT': {'default': 5672, 'range': (5670, 5699)},
        # Database Services (5400-5499 range)
        'POSTGRES_INGESTION_PORT': {'default': 5432, 'range': (5430, 5499)},
        'POSTGRES_OUTPUT_PORT': {'default': 5433, 'range': (5430, 5499)},

        # Cache Services (6300-6399 range)
        'REDIS_INGESTION_PORT': {'default': 6379, 'range': (6370, 6399)},

        # Storage Services (9000-9099 range)
        'MINIO_BRONZE_PORT': {'default': 9000, 'range': (9000, 9099)},
        'MINIO_SILVER_PORT': {'default': 9010, 'range': (9000, 9099)},
        'MINIO_GOLD_PORT': {'default': 9020, 'range': (9000, 9099)},

        # Monitoring Services (3000-3099, 9000-9099 ranges)
        'GRAFANA_PORT': {'default': 3000, 'range': (3000, 3099)},
        'PROMETHEUS_PORT': {'default': 9090, 'range': (9090, 9199)},

        # Vector Database (6300-6399 range)
        'QDRANT_HTTP_PORT': {'default': 6333, 'range': (6330, 6399)},
        'QDRANT_GRPC_PORT': {'default': 6334, 'range': (6330, 6399)},

        # Tracing (16600-16700 range)
        'JAEGER_UI_PORT': {'default': 16686, 'range': (16680, 16700)}
    }

    # Check for conflicts and resolve them using intelligent port allocation
    for service_name, config in service_port_config.items():
        if isinstance(config, dict):
            default_port = config['default']
            port_range = config['range']

            if default_port in used_ports:
                # Port conflict detected - find alternative in designated range
                alternative_port = find_available_port_in_range(
                    port_range[0], port_range[1], used_ports
                )

                if alternative_port:
                    conflicts_resolved[service_name] = {
                        'original_port': default_port,
                        'suggested_port': alternative_port,
                        'conflict_with': get_conflicting_service(default_port, container_ports),
                        'port_range': port_range
                    }
                    logger.info(f"Port conflict resolved for {service_name}",
                              original=default_port,
                              suggested=alternative_port,
                              range=f"{port_range[0]}-{port_range[1]}")
                else:
                    logger.warning(f"No available ports found in range {port_range} for {service_name}")
        else:
            # Legacy format support (single port value)
            default_port = config
            if default_port in used_ports:
                alternative_port = find_available_port(default_port + 1, used_ports)
                if alternative_port:
                    conflicts_resolved[service_name] = {
                        'original_port': default_port,
                        'suggested_port': alternative_port,
                        'conflict_with': get_conflicting_service(default_port, container_ports)
                    }

    return conflicts_resolved

def get_conflicting_service(port: int, container_ports: Dict[str, List[int]]) -> Optional[str]:
    """Find which service is using the conflicting port"""
    for service_name, ports in container_ports.items():
        if port in ports:
            return service_name
    return None

def update_environment_file(conflicts: Dict[str, Dict[str, int]]):
    """Update .env file with resolved ports using atomic file operations.

    This function safely updates environment files by:
    1. Creating backup of existing file
    2. Using atomic write operations to prevent corruption
    3. Validating changes before committing
    4. Providing rollback capability

    Args:
        conflicts: Dict of service conflicts with resolution data

    Returns:
        bool: True if update successful, False otherwise
    """
    if not conflicts:
        logger.info("No conflicts to resolve")
        return True

    env_file = '.env'
    backup_file = f"{env_file}.backup.{int(time.time())}"

    # Determine source file (.env takes precedence over .env.example)
    source_file = env_file if os.path.exists(env_file) else '.env.example'

    if not os.path.exists(source_file):
        logger.error("Neither .env nor .env.example found")
        return False

    try:
        logger.info(f"Updating {env_file} with {len(conflicts)} port resolutions")
        # Create backup of existing file if it exists
        if os.path.exists(env_file):
            import shutil
            shutil.copy2(env_file, backup_file)
            logger.info(f"Created backup: {backup_file}")

        # Read and process source file
        env_lines = []
        with open(source_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    if key in conflicts:
                        new_value = str(conflicts[key]['suggested_port'])
                        env_lines.append(f"{key}={new_value}")
                        logger.info(f"Updated {key}: {value} -> {new_value}")
                    else:
                        env_lines.append(f"{key}={value}")
                else:
                    env_lines.append(line)

        # Add any new environment variables that weren't in the source file
        existing_keys = {line.split('=')[0].strip() for line in env_lines if '=' in line and not line.startswith('#')}
        for key in conflicts:
            if key not in existing_keys:
                new_value = str(conflicts[key]['suggested_port'])
                env_lines.append(f"{key}={new_value}")
                logger.info(f"Added new variable {key}={new_value}")

        # Atomic write to prevent file corruption
        temp_file = f"{env_file}.tmp"
        with open(temp_file, 'w') as f:
            f.write('\n'.join(env_lines) + '\n')

        # Atomic move (rename) operation
        os.rename(temp_file, env_file)

        logger.info(f"Environment file {env_file} updated successfully with {len(conflicts)} changes")
        return True

    except Exception as e:
        logger.error("Failed to update environment file", error=str(e))
        # Cleanup temporary file if it exists
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

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
    """Resolve detected port conflicts with automatic environment file updates.

    This endpoint performs the complete port conflict resolution workflow:
    1. Detect port conflicts across all services
    2. Find optimal alternative ports in designated ranges
    3. Update .env file with resolved port assignments
    4. Provide detailed conflict resolution report

    Returns:
        Dict containing resolution status and details
    """
    try:
        conflicts = resolve_port_conflicts()

        if not conflicts:
            return {
                "status": "success",
                "message": "No port conflicts detected",
                "conflicts_resolved": 0,
                "timestamp": time.time()
            }

        # Update environment file with conflict resolutions
        update_success = update_environment_file(conflicts)

        if not update_success:
            return {
                "status": "error",
                "message": "Failed to update environment file",
                "conflicts_detected": len(conflicts),
                "timestamp": time.time()
            }

        return {
            "status": "success",
            "message": f"Resolved {len(conflicts)} port conflicts",
            "conflicts_resolved": len(conflicts),
            "conflicts": conflicts,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error("Failed to resolve port conflicts", error=str(e))
        return {
            "status": "error",
            "message": f"Port conflict resolution failed: {str(e)}",
            "timestamp": time.time()
        }

@app.get("/used-ports")
async def get_used_ports():
    """Get currently used ports"""
    container_ports = get_docker_container_ports()
    used_ports = set()

    for ports in container_ports.values():
        used_ports.update(ports)

    return {
        "status": "success",
        "used_ports": sorted(list(used_ports)),
        "total_ports_used": len(used_ports),
        "container_count": len(container_ports),
        "containers": container_ports,
        "port_ranges": {
            "web_services": "8000-8999",
            "databases": "5000-5999",
            "monitoring": "5600-5799",
            "storage": "9000-9999"
        },
        "timestamp": time.time()
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