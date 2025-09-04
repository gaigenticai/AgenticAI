#!/usr/bin/env python3
"""
Port Conflict Resolution Utility

This utility helps resolve Docker port conflicts by:
- Checking if specified ports are available
- Suggesting alternative ports if conflicts exist
- Updating docker-compose.yml with resolved ports
"""

import json
import os
import socket
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

def check_port_available(port: int, host: str = 'localhost') -> bool:
    """Check if a port is available on the specified host"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0  # Port is available if connection fails
    except:
        return False

def find_available_port(start_port: int, max_attempts: int = 100) -> Optional[int]:
    """Find an available port starting from the given port"""
    for port in range(start_port, start_port + max_attempts):
        if check_port_available(port):
            return port
    return None

def get_docker_compose_ports() -> Dict[str, int]:
    """Extract port mappings from docker-compose.yml"""
    ports = {}

    try:
        with open('docker-compose.yml', 'r') as f:
            content = f.read()

        # Extract environment variables that define ports
        import re
        env_vars = re.findall(r'\$\{([^}]+)\}', content)

        for var in env_vars:
            if 'PORT' in var:
                # Get the default value from .env.example if it exists
                default_value = get_env_default(var)
                if default_value and default_value.isdigit():
                    ports[var] = int(default_value)

    except Exception as e:
        print(f"Error reading docker-compose.yml: {e}")

    return ports

def get_env_default(var_name: str) -> Optional[str]:
    """Get default value for environment variable from .env.example"""
    try:
        with open('.env.example', 'r') as f:
            for line in f:
                if line.startswith(f'{var_name}='):
                    return line.split('=', 1)[1].strip()
    except:
        pass
    return None

def resolve_port_conflicts() -> Dict[str, Tuple[int, int]]:
    """Resolve port conflicts and return mapping of old->new ports"""
    resolved_ports = {}
    current_ports = get_docker_compose_ports()

    print("🔍 Checking for port conflicts...")

    for var_name, port in current_ports.items():
        if not check_port_available(port):
            print(f"⚠️  Port {port} ({var_name}) is already in use")

            # Find alternative port
            alternative = find_available_port(port + 1)
            if alternative:
                resolved_ports[var_name] = (port, alternative)
                print(f"✅ Found alternative port: {alternative}")
            else:
                print(f"❌ No available ports found near {port}")
        else:
            print(f"✅ Port {port} ({var_name}) is available")

    return resolved_ports

def update_docker_compose(resolved_ports: Dict[str, Tuple[int, int]]):
    """Update docker-compose.yml with resolved ports"""
    if not resolved_ports:
        print("🎉 No port conflicts found!")
        return

    print("\n📝 Updating docker-compose.yml...")

    try:
        with open('docker-compose.yml', 'r') as f:
            content = f.read()

        for var_name, (old_port, new_port) in resolved_ports.items():
            # Update port mappings
            old_mapping = f'"${{{var_name}}}'
            new_mapping = f'"{new_port}'
            content = content.replace(old_mapping, new_mapping)

            print(f"🔄 Updated {var_name}: {old_port} → {new_port}")

        with open('docker-compose.yml', 'w') as f:
            f.write(content)

        print("✅ docker-compose.yml updated successfully!")

    except Exception as e:
        print(f"❌ Error updating docker-compose.yml: {e}")

def create_env_file(resolved_ports: Dict[str, Tuple[int, int]]):
    """Create .env file with resolved ports"""
    if not resolved_ports:
        return

    print("\n📄 Creating .env file with resolved ports...")

    env_content = "# Environment configuration with resolved ports\n"

    # Read existing .env.example
    try:
        with open('.env.example', 'r') as f:
            for line in f:
                var_name = line.split('=')[0] if '=' in line else None

                if var_name and var_name in resolved_ports:
                    old_port, new_port = resolved_ports[var_name]
                    env_content += f"{var_name}={new_port}\n"
                else:
                    env_content += line
    except:
        # Fallback: create basic .env file
        for var_name, (old_port, new_port) in resolved_ports.items():
            env_content += f"{var_name}={new_port}\n"

    with open('.env', 'w') as f:
        f.write(env_content)

    print("✅ .env file created!")

def main():
    """Main port resolution function"""
    print("🚀 Port Conflict Resolution Utility")
    print("=" * 40)

    resolved_ports = resolve_port_conflicts()

    if resolved_ports:
        print(f"\n📋 Summary of changes:")
        for var_name, (old_port, new_port) in resolved_ports.items():
            print(f"   {var_name}: {old_port} → {new_port}")

        # Ask for confirmation
        response = input("\n🔧 Apply these changes? (y/N): ").lower().strip()
        if response == 'y':
            update_docker_compose(resolved_ports)
            create_env_file(resolved_ports)
            print("\n🎉 Port conflicts resolved! You can now run: docker-compose up")
        else:
            print("\n❌ Changes not applied.")
    else:
        print("\n🎉 All ports are available!")

if __name__ == "__main__":
    main()
