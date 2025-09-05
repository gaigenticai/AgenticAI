#!/usr/bin/env python3
"""Prometheus helper (moved/created to avoid hardcoded defaults)"""
import os
import logging
logger = logging.getLogger(__name__)

def require_postgres_password():
    if not os.getenv('POSTGRES_PASSWORD'):
        logger.error('POSTGRES_PASSWORD not configured for Prometheus integration')
        raise RuntimeError('POSTGRES_PASSWORD not configured for Prometheus integration')


