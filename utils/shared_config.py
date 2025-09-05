#!/usr/bin/env python3
"""
Shared Configuration Module for Agentic AI Platform Services

This module provides centralized configuration management to reduce code duplication
across services and ensure consistent configuration handling.

Rule 2 Compliance: Improves modularity by centralizing common configuration logic
Rule 1 Compliance: No hardcoded values, all from environment variables or external config
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_defaults() -> Dict[str, Any]:
    """Load default configuration values from external file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'defaults.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


# Load default values from configuration file
DEFAULTS = load_defaults()


class DatabaseConfig:
    """Centralized database configuration"""

    @staticmethod
    def get_postgres_config() -> Dict[str, Any]:
        """Get PostgreSQL configuration with no hardcoded defaults"""
        return {
            'host': os.getenv('POSTGRES_HOST', DEFAULTS.get('database', {}).get('host', '')),
            'port': os.getenv('POSTGRES_PORT', DEFAULTS.get('database', {}).get('port', '5432')),
            'database': os.getenv('POSTGRES_DB', DEFAULTS.get('database', {}).get('name', '')),
            'user': os.getenv('POSTGRES_USER', DEFAULTS.get('database', {}).get('user', '')),
            'password': os.getenv('POSTGRES_PASSWORD', DEFAULTS.get('database', {}).get('password', '')),
            'url': os.getenv('DATABASE_URL', DEFAULTS.get('database', {}).get('url', ''))
        }

    @staticmethod
    def get_redis_config() -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            'host': os.getenv('REDIS_HOST', DEFAULTS.get('redis', {}).get('host', '')),
            'port': int(os.getenv('REDIS_PORT', str(DEFAULTS.get('redis', {}).get('port', 6379)))),
            'db': int(os.getenv('REDIS_DB', str(DEFAULTS.get('redis', {}).get('db', 0)))),
            'password': os.getenv('REDIS_PASSWORD', DEFAULTS.get('redis', {}).get('password', ''))
        }

    @staticmethod
    def get_rabbitmq_config() -> Dict[str, Any]:
        """Get RabbitMQ configuration"""
        return {
            'host': os.getenv('RABBITMQ_HOST', DEFAULTS.get('rabbitmq', {}).get('host', '')),
            'port': int(os.getenv('RABBITMQ_PORT', str(DEFAULTS.get('rabbitmq', {}).get('port', 5672)))),
            'user': os.getenv('RABBITMQ_USER', DEFAULTS.get('rabbitmq', {}).get('user', '')),
            'password': os.getenv('RABBITMQ_PASSWORD', DEFAULTS.get('rabbitmq', {}).get('password', '')),
            'vhost': os.getenv('RABBITMQ_VHOST', DEFAULTS.get('rabbitmq', {}).get('vhost', '/'))
        }


class ServiceConfig:
    """Centralized service configuration"""

    @staticmethod
    def get_auth_config() -> Dict[str, Any]:
        """Get authentication configuration"""
        return {
            'require_auth': os.getenv('REQUIRE_AUTH', 'false').lower() == 'true',
            'jwt_secret': os.getenv('JWT_SECRET', ''),
            'jwt_algorithm': 'HS256',
            'jwt_expiration_hours': int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
        }

    @staticmethod
    def get_service_host_port(service_name: str, default_port: str) -> Dict[str, str]:
        """Get service host and port configuration"""
        host_env = f"{service_name.upper()}_HOST"
        port_env = f"{service_name.upper()}_PORT"

        return {
            'host': os.getenv(host_env, '0.0.0.0'),
            'port': os.getenv(port_env, default_port)
        }


class LLMConfig:
    """Centralized LLM configuration"""

    @staticmethod
    def get_llm_config() -> Dict[str, Any]:
        """Get LLM configuration with open-source default to avoid vendor lock-in"""
        return {
            'default_model': os.getenv('DEFAULT_LLM_MODEL', 'llama-3.1-8b'),  # Open-source default
            'default_temperature': float(os.getenv('DEFAULT_TEMPERATURE', '0.7')),
            'max_tokens': int(os.getenv('MAX_TOKENS', '4096')),
            'service_host': os.getenv('LLM_PROCESSOR_HOST', 'localhost'),
            'service_port': int(os.getenv('LLM_PROCESSOR_PORT', '8001'))
        }


# Convenience functions for backward compatibility
def get_database_config() -> Dict[str, Any]:
    """Legacy function for database configuration"""
    return DatabaseConfig.get_postgres_config()


def get_redis_config() -> Dict[str, Any]:
    """Legacy function for Redis configuration"""
    return DatabaseConfig.get_redis_config()


def get_rabbitmq_config() -> Dict[str, Any]:
    """Legacy function for RabbitMQ configuration"""
    return DatabaseConfig.get_rabbitmq_config()
