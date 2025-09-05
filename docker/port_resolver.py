#!/usr/bin/env python3
"""
Automatic Port Conflict Resolution for Docker Compose

This script integrates with docker-compose to automatically resolve port conflicts
before starting services. It works by:

1. Checking for port conflicts using the port-manager service
2. Updating .env file with resolved port assignments
3. Ensuring docker-compose starts without port conflicts

Usage:
    python port_resolver.py [check|resolve|auto]

Arguments:
    check   - Check for port conflicts without resolving
    resolve - Resolve conflicts and update .env file
    auto    - Automatically resolve conflicts (default)

Environment Variables:
    PORT_MANAGER_URL - URL of port-manager service (default: http://localhost:8000)
    ENV_FILE         - Path to environment file (default: .env)
    BACKUP_ENV       - Create backup of .env file (default: true)
"""

import requests
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional


class PortResolver:
    """Automatic port conflict resolver for Docker Compose"""

    def __init__(self):
        self.port_manager_url = os.getenv('PORT_MANAGER_URL', 'http://localhost:8000')
        self.env_file = os.getenv('ENV_FILE', '.env')
        self.backup_env = os.getenv('BACKUP_ENV', 'true').lower() == 'true'
        self.session = requests.Session()
        self.session.timeout = 30

    def check_service_health(self) -> bool:
        """Check if port-manager service is available"""
        try:
            response = self.session.get(f"{self.port_manager_url}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_for_service(self, timeout: int = 60) -> bool:
        """Wait for port-manager service to become available"""
        print("Waiting for port-manager service to be available...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_service_health():
                print("Port-manager service is available")
                return True
            time.sleep(2)

        print("Timeout waiting for port-manager service")
        return False

    def check_conflicts(self) -> Dict[str, Any]:
        """Check for port conflicts"""
        try:
            response = self.session.get(f"{self.port_manager_url}/check-conflicts")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to check conflicts: {e}")
            return {"status": "error", "message": str(e)}

    def resolve_conflicts(self) -> Dict[str, Any]:
        """Resolve port conflicts"""
        try:
            response = self.session.post(f"{self.port_manager_url}/resolve-conflicts")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to resolve conflicts: {e}")
            return {"status": "error", "message": str(e)}

    def create_backup(self) -> Optional[str]:
        """Create backup of .env file"""
        if not self.backup_env or not Path(self.env_file).exists():
            return None

        backup_file = f"{self.env_file}.backup.{int(time.time())}"
        try:
            with open(self.env_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())
            print(f"Created backup: {backup_file}")
            return backup_file
        except Exception as e:
            print(f"Failed to create backup: {e}")
            return None

    def run_check(self) -> int:
        """Check for port conflicts"""
        print("Checking for port conflicts...")

        if not self.check_service_health():
            if not self.wait_for_service():
                print("Port-manager service is not available")
                return 1

        result = self.check_conflicts()

        if result.get("status") == "error":
            print(f"Error checking conflicts: {result.get('message')}")
            return 1

        conflicts = result.get("conflicts", {})
        if not conflicts:
            print("✅ No port conflicts detected")
            return 0

        print(f"⚠️  Found {len(conflicts)} port conflicts:")
        for service, conflict in conflicts.items():
            print(f"   {service}: {conflict['original_port']} -> {conflict['suggested_port']}")

        return len(conflicts)

    def run_resolve(self) -> int:
        """Resolve port conflicts"""
        print("Resolving port conflicts...")

        if not self.check_service_health():
            if not self.wait_for_service():
                print("Port-manager service is not available")
                return 1

        # Create backup before making changes
        self.create_backup()

        result = self.resolve_conflicts()

        if result.get("status") == "error":
            print(f"Error resolving conflicts: {result.get('message')}")
            return 1

        conflicts_resolved = result.get("conflicts_resolved", 0)
        if conflicts_resolved == 0:
            print("✅ No port conflicts to resolve")
            return 0

        print(f"✅ Successfully resolved {conflicts_resolved} port conflicts")
        conflicts = result.get("conflicts", {})
        for service, conflict in conflicts.items():
            print(f"   {service}: {conflict['original_port']} -> {conflict['suggested_port']}")

        return 0

    def run_auto(self) -> int:
        """Automatically check and resolve conflicts"""
        conflicts_found = self.run_check()

        if conflicts_found > 0:
            print("\nAuto-resolving conflicts...")
            return self.run_resolve()

        return 0


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        command = 'auto'

    resolver = PortResolver()

    if command == 'check':
        return resolver.run_check()
    elif command == 'resolve':
        return resolver.run_resolve()
    elif command == 'auto':
        return resolver.run_auto()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python port_resolver.py [check|resolve|auto]")
        return 1


if __name__ == "__main__":
    sys.exit(main())