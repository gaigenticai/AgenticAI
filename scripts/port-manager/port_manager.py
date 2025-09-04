#!/usr/bin/env python3
"""
Port Manager Service for Agentic Platform
Automatically resolves port conflicts and manages dynamic port allocation

This service provides automatic port conflict resolution to ensure
development never stops due to port conflicts (Rule 8 compliance).
"""

import asyncio
import json
import logging
import os
import socket
import subprocess
import time
from typing import Dict, List, Optional, Set

import docker
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic Port Manager",
    description="Automatic port conflict resolution service",
    version="1.0.0"
)

# Global state
port_assignments: Dict[str, int] = {}
reserved_ports: Set[int] = set()
docker_client = docker.from_env()

# Port ranges for different services
PORT_RANGES = {
    'ingestion': (8000, 8099),
    'output': (8100, 8199),
    'database': (8200, 8299),
    'monitoring': (8300, 8399),
    'streaming': (8400, 8499),
    'storage': (8500, 8599),
    'security': (8600, 8699)
}


class PortManager:
    """Manages dynamic port allocation and conflict resolution"""

    def __init__(self):
        self.used_ports = self._scan_used_ports()
        self.load_existing_assignments()

    def _scan_used_ports(self) -> Set[int]:
        """Scan for currently used ports"""
        used_ports = set()

        # Check listening ports
        try:
            result = subprocess.run(
                ['netstat', '-tln'],
                capture_output=True,
                text=True,
                timeout=10
            )
            for line in result.stdout.split('\n'):
                if 'LISTEN' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        port = parts[3].split(':')[-1]
                        try:
                            used_ports.add(int(port))
                        except ValueError:
                            continue
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Could not scan listening ports")

        # Check Docker containers
        try:
            containers = docker_client.containers.list()
            for container in containers:
                if container.status == 'running':
                    ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                    for port_info in ports.values():
                        if port_info:
                            for binding in port_info:
                                host_port = binding.get('HostPort')
                                if host_port:
                                    try:
                                        used_ports.add(int(host_port))
                                    except ValueError:
                                        continue
        except Exception as e:
            logger.warning(f"Could not scan Docker ports: {e}")

        return used_ports

    def load_existing_assignments(self):
        """Load existing port assignments from file"""
        try:
            if os.path.exists('/app/port_assignments.json'):
                with open('/app/port_assignments.json', 'r') as f:
                    data = json.load(f)
                    port_assignments.update(data.get('assignments', {}))
                    reserved_ports.update(data.get('reserved', []))
        except Exception as e:
            logger.warning(f"Could not load port assignments: {e}")

    def save_assignments(self):
        """Save port assignments to file"""
        try:
            data = {
                'assignments': port_assignments,
                'reserved': list(reserved_ports),
                'last_updated': time.time()
            }
            with open('/app/port_assignments.json', 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save port assignments: {e}")

    def find_available_port(self, service_type: str, preferred_port: Optional[int] = None) -> int:
        """Find an available port for the given service type"""
        # Check if preferred port is available
        if preferred_port and self._is_port_available(preferred_port):
            return preferred_port

        # Find port in service-specific range
        if service_type in PORT_RANGES:
            min_port, max_port = PORT_RANGES[service_type]
            for port in range(min_port, max_port + 1):
                if self._is_port_available(port):
                    return port

        # Fallback to general range
        for port in range(9000, 9999):
            if self._is_port_available(port):
                return port

        raise RuntimeError(f"No available ports found for service type: {service_type}")

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        if port in self.used_ports or port in reserved_ports:
            return False

        # Test if port can be bound
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            return True
        except OSError:
            return False

    def assign_port(self, service_name: str, service_type: str, preferred_port: Optional[int] = None) -> int:
        """Assign a port to a service"""
        if service_name in port_assignments:
            current_port = port_assignments[service_name]
            if self._is_port_available(current_port):
                return current_port
            else:
                logger.info(f"Port {current_port} for {service_name} is no longer available")

        # Find new port
        new_port = self.find_available_port(service_type, preferred_port)
        port_assignments[service_name] = new_port
        reserved_ports.add(new_port)
        self.save_assignments()

        logger.info(f"Assigned port {new_port} to {service_name}")
        return new_port

    def release_port(self, service_name: str):
        """Release a port assignment"""
        if service_name in port_assignments:
            port = port_assignments[service_name]
            reserved_ports.discard(port)
            del port_assignments[service_name]
            self.save_assignments()
            logger.info(f"Released port {port} from {service_name}")

    def get_service_port(self, service_name: str) -> Optional[int]:
        """Get the assigned port for a service"""
        return port_assignments.get(service_name)

    def list_assignments(self) -> Dict:
        """List all port assignments"""
        return {
            'assignments': port_assignments,
            'reserved_ports': list(reserved_ports),
            'used_ports_count': len(self.used_ports)
        }


# Global port manager instance
port_manager = PortManager()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "port-manager"}


@app.post("/assign")
async def assign_port(
    service_name: str,
    service_type: str,
    preferred_port: Optional[int] = None
):
    """Assign a port to a service"""
    try:
        port = port_manager.assign_port(service_name, service_type, preferred_port)
        return {
            "service_name": service_name,
            "assigned_port": port,
            "service_type": service_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/release/{service_name}")
async def release_port(service_name: str):
    """Release a port assignment"""
    port_manager.release_port(service_name)
    return {"message": f"Released port for {service_name}"}


@app.get("/port/{service_name}")
async def get_service_port(service_name: str):
    """Get assigned port for a service"""
    port = port_manager.get_service_port(service_name)
    if port is None:
        raise HTTPException(status_code=404, detail=f"No port assigned for {service_name}")
    return {"service_name": service_name, "port": port}


@app.get("/assignments")
async def list_assignments():
    """List all port assignments"""
    return port_manager.list_assignments()


@app.post("/scan")
async def scan_ports():
    """Rescan for used ports"""
    port_manager.used_ports = port_manager._scan_used_ports()
    return {
        "message": "Port scan completed",
        "used_ports_count": len(port_manager.used_ports)
    }


@app.on_event("startup")
async def startup_event():
    """Initialize port manager on startup"""
    logger.info("Port Manager starting up...")
    logger.info(f"Found {len(port_manager.used_ports)} used ports")
    logger.info(f"Loaded {len(port_assignments)} port assignments")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Port Manager shutting down...")
    port_manager.save_assignments()


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "port_manager:app",
        host="0.0.0.0",
        port=8082,
        reload=False,
        log_level="info"
    )
