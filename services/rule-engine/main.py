#!/usr/bin/env python3
"""
Rule Engine Service for Agentic Brain Platform

This service provides powerful business rule processing and decision logic execution
capabilities for the Agentic Brain platform. It supports complex rule evaluation,
conditional logic, and automated decision making based on configurable rule sets.

Features:
- Rule definition and management with versioning
- Complex rule evaluation with multiple conditions
- Rule execution orchestration and performance monitoring
- Support for various rule types (validation, decision, transformation)
- Rule dependency management and conflict resolution
- Real-time rule evaluation and batch processing
- Rule performance analytics and optimization
- RESTful API for rule operations
- Comprehensive monitoring and metrics
- Authentication and authorization support
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import operator
import re

import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON, Float, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func
import httpx
import uvicorn

# JWT support for authentication
try:
    import jwt
except ImportError:
    jwt = None

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for Rule Engine Service"""

    # Database Configuration
    DB_HOST = os.getenv('POSTGRES_HOST', 'postgresql_ingestion')
    DB_PORT = os.getenv('POSTGRES_PORT', '5432')
    DB_NAME = os.getenv('POSTGRES_DB', 'agentic_ingestion')
    DB_USER = os.getenv('POSTGRES_USER', 'agentic_user')
    DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'agentic123')
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Redis Configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis_ingestion')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB = 4  # Use DB 4 for rule engine

    # Service Configuration
    SERVICE_HOST = os.getenv('RULE_ENGINE_HOST', '0.0.0.0')
    SERVICE_PORT = int(os.getenv('RULE_ENGINE_PORT', '8204'))

    # Authentication Configuration
    REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-jwt-key-change-in-production')
    JWT_ALGORITHM = 'HS256'

    # Rule Configuration
    RULE_CACHE_ENABLED = os.getenv('RULE_CACHE_ENABLED', 'true').lower() == 'true'
    RULE_CACHE_TTL_SECONDS = int(os.getenv('RULE_CACHE_TTL_SECONDS', '3600'))
    MAX_RULES_PER_SET = int(os.getenv('MAX_RULES_PER_SET', '100'))
    RULE_EXECUTION_TIMEOUT_SECONDS = int(os.getenv('RULE_EXECUTION_TIMEOUT_SECONDS', '300'))

    # Performance Configuration
    BATCH_PROCESSING_ENABLED = os.getenv('BATCH_PROCESSING_ENABLED', 'true').lower() == 'true'
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '1000'))

    # Monitoring Configuration
    METRICS_ENABLED = os.getenv('AGENT_METRICS_ENABLED', 'true').lower() == 'true'

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class RuleSet(Base):
    """Database model for rule sets"""
    __tablename__ = 'rule_sets'

    id = Column(Integer, primary_key=True)
    rule_set_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Text
    domain = Column(String(100))
    version = Column(String(50), default='1.0.0')
    is_active = Column(Boolean, default=True)
    rule_count = Column(Integer, default=0)
    created_by = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Rule(Base):
    """Database model for individual rules"""
    __tablename__ = 'rules'

    id = Column(Integer, primary_key=True)
    rule_id = Column(String(100), unique=True, nullable=False)
    rule_set_id = Column(String(100), nullable=False)
    name = Column(String(255), nullable=False)
    description = Text
    rule_type = Column(String(50), nullable=False)  # validation, decision, transformation, scoring
    priority = Column(Integer, default=1)
    conditions = Column(JSON, nullable=False)
    actions = Column(JSON, nullable=False)
    metadata = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    execution_count = Column(BigInteger, default=0)
    success_count = Column(BigInteger, default=0)
    average_execution_time_ms = Column(Float)
    last_executed = Column(DateTime)
    created_by = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class RuleExecution(Base):
    """Database model for rule execution tracking"""
    __tablename__ = 'rule_executions'

    id = Column(Integer, primary_key=True)
    execution_id = Column(String(100), unique=True, nullable=False)
    rule_id = Column(String(100), nullable=False)
    rule_set_id = Column(String(100), nullable=False)
    input_data = Column(JSON)
    output_data = Column(JSON)
    execution_time_ms = Column(Integer)
    result = Column(String(50))  # pass, fail, error
    error_message = Text
    matched_conditions = Column(JSON, default=list)
    executed_actions = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)

class RulePerformance(Base):
    """Database model for rule performance metrics"""
    __tablename__ = 'rule_performance'

    id = Column(Integer, primary_key=True)
    rule_id = Column(String(100), nullable=False)
    rule_set_id = Column(String(100), nullable=False)
    execution_count = Column(BigInteger, default=0)
    success_count = Column(BigInteger, default=0)
    failure_count = Column(BigInteger, default=0)
    average_execution_time_ms = Column(Float)
    max_execution_time_ms = Column(Integer)
    min_execution_time_ms = Column(Integer)
    last_updated = Column(DateTime, default=datetime.utcnow)

# =============================================================================
# RULE CONDITION OPERATORS
# =============================================================================

class ConditionOperator(Enum):
    """Enumeration of supported condition operators"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex_match"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"
    NOT_BETWEEN = "not_between"

class LogicalOperator(Enum):
    """Enumeration of logical operators for combining conditions"""
    AND = "and"
    OR = "or"

# =============================================================================
# RULE EVALUATION ENGINE
# =============================================================================

class RuleEvaluator:
    """Evaluates rule conditions against input data"""

    def __init__(self):
        self.operators = {
            ConditionOperator.EQUALS: self._equals,
            ConditionOperator.NOT_EQUALS: self._not_equals,
            ConditionOperator.GREATER_THAN: self._greater_than,
            ConditionOperator.LESS_THAN: self._less_than,
            ConditionOperator.GREATER_EQUAL: self._greater_equal,
            ConditionOperator.LESS_EQUAL: self._less_equal,
            ConditionOperator.CONTAINS: self._contains,
            ConditionOperator.NOT_CONTAINS: self._not_contains,
            ConditionOperator.STARTS_WITH: self._starts_with,
            ConditionOperator.ENDS_WITH: self._ends_with,
            ConditionOperator.REGEX_MATCH: self._regex_match,
            ConditionOperator.IN_LIST: self._in_list,
            ConditionOperator.NOT_IN_LIST: self._not_in_list,
            ConditionOperator.IS_NULL: self._is_null,
            ConditionOperator.IS_NOT_NULL: self._is_not_null,
            ConditionOperator.BETWEEN: self._between,
            ConditionOperator.NOT_BETWEEN: self._not_between,
        }

    def evaluate_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        try:
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')

            if not field or not operator:
                return False

            # Get field value from data
            field_value = self._get_nested_value(data, field)
            if field_value is None:
                # Handle null checks
                if operator in [ConditionOperator.IS_NULL.value]:
                    return True
                elif operator in [ConditionOperator.IS_NOT_NULL.value]:
                    return False
                else:
                    return False

            # Get operator function
            operator_func = self.operators.get(ConditionOperator(operator))
            if not operator_func:
                logger.warning(f"Unknown operator: {operator}")
                return False

            return operator_func(field_value, value)

        except Exception as e:
            logger.error(f"Error evaluating condition: {str(e)}")
            return False

    def evaluate_conditions(self, conditions: List[Dict[str, Any]], data: Dict[str, Any],
                          logical_operator: str = "and") -> Dict[str, Any]:
        """Evaluate multiple conditions with logical operators"""
        if not conditions:
            return {"result": True, "matched_conditions": [], "failed_conditions": []}

        matched_conditions = []
        failed_conditions = []
        results = []

        for condition in conditions:
            result = self.evaluate_condition(condition, data)
            results.append(result)

            if result:
                matched_conditions.append(condition)
            else:
                failed_conditions.append(condition)

        if logical_operator == LogicalOperator.AND.value:
            final_result = all(results)
        elif logical_operator == LogicalOperator.OR.value:
            final_result = any(results)
        else:
            final_result = False

        return {
            "result": final_result,
            "matched_conditions": matched_conditions,
            "failed_conditions": failed_conditions,
            "logical_operator": logical_operator
        }

    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested value from data using dot notation"""
        keys = field_path.split('.')
        current = data

        try:
            for key in keys:
                if isinstance(current, dict):
                    current = current[key]
                elif isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                else:
                    return None
            return current
        except (KeyError, IndexError, TypeError):
            return None

    # Condition operator implementations
    def _equals(self, field_value: Any, expected_value: Any) -> bool:
        return field_value == expected_value

    def _not_equals(self, field_value: Any, expected_value: Any) -> bool:
        return field_value != expected_value

    def _greater_than(self, field_value: Any, expected_value: Any) -> bool:
        try:
            return float(field_value) > float(expected_value)
        except (ValueError, TypeError):
            return False

    def _less_than(self, field_value: Any, expected_value: Any) -> bool:
        try:
            return float(field_value) < float(expected_value)
        except (ValueError, TypeError):
            return False

    def _greater_equal(self, field_value: Any, expected_value: Any) -> bool:
        try:
            return float(field_value) >= float(expected_value)
        except (ValueError, TypeError):
            return False

    def _less_equal(self, field_value: Any, expected_value: Any) -> bool:
        try:
            return float(field_value) <= float(expected_value)
        except (ValueError, TypeError):
            return False

    def _contains(self, field_value: Any, expected_value: Any) -> bool:
        if isinstance(field_value, str) and isinstance(expected_value, str):
            return expected_value in field_value
        elif isinstance(field_value, list):
            return expected_value in field_value
        return False

    def _not_contains(self, field_value: Any, expected_value: Any) -> bool:
        return not self._contains(field_value, expected_value)

    def _starts_with(self, field_value: Any, expected_value: Any) -> bool:
        if isinstance(field_value, str) and isinstance(expected_value, str):
            return field_value.startswith(expected_value)
        return False

    def _ends_with(self, field_value: Any, expected_value: Any) -> bool:
        if isinstance(field_value, str) and isinstance(expected_value, str):
            return field_value.endswith(expected_value)
        return False

    def _regex_match(self, field_value: Any, pattern: str) -> bool:
        if not isinstance(field_value, str):
            return False
        try:
            return bool(re.match(pattern, field_value))
        except re.error:
            return False

    def _in_list(self, field_value: Any, expected_list: List[Any]) -> bool:
        if not isinstance(expected_list, list):
            return False
        return field_value in expected_list

    def _not_in_list(self, field_value: Any, expected_list: List[Any]) -> bool:
        return not self._in_list(field_value, expected_list)

    def _is_null(self, field_value: Any, expected_value: Any) -> bool:
        return field_value is None

    def _is_not_null(self, field_value: Any, expected_value: Any) -> bool:
        return field_value is not None

    def _between(self, field_value: Any, range_values: List[Any]) -> bool:
        if not isinstance(range_values, list) or len(range_values) != 2:
            return False
        try:
            min_val, max_val = float(range_values[0]), float(range_values[1])
            field_val = float(field_value)
            return min_val <= field_val <= max_val
        except (ValueError, TypeError):
            return False

    def _not_between(self, field_value: Any, range_values: List[Any]) -> bool:
        return not self._between(field_value, range_values)

# =============================================================================
# RULE ACTION EXECUTOR
# =============================================================================

class RuleActionExecutor:
    """Executes rule actions based on evaluation results"""

    def __init__(self):
        self.actions = {
            "set_field": self._set_field,
            "add_to_list": self._add_to_list,
            "remove_from_list": self._remove_from_list,
            "calculate_score": self._calculate_score,
            "flag_record": self._flag_record,
            "transform_data": self._transform_data,
            "send_notification": self._send_notification,
            "update_status": self._update_status,
            "log_event": self._log_event,
        }

    def execute_actions(self, actions: List[Dict[str, Any]], data: Dict[str, Any],
                       evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a list of actions"""
        executed_actions = []
        results = {}

        for action in actions:
            try:
                action_type = action.get('type')
                action_func = self.actions.get(action_type)

                if action_func:
                    result = action_func(action, data, evaluation_result)
                    executed_actions.append(action)
                    results[action_type] = result
                else:
                    logger.warning(f"Unknown action type: {action_type}")

            except Exception as e:
                logger.error(f"Error executing action {action}: {str(e)}")
                results[action.get('type', 'unknown')] = {"error": str(e)}

        return {
            "executed_actions": executed_actions,
            "results": results
        }

    def _set_field(self, action: Dict[str, Any], data: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Set a field value"""
        field = action.get('field')
        value = action.get('value')

        if field:
            self._set_nested_value(data, field, value)
            return {"field": field, "value": value, "status": "set"}

        return {"error": "Missing field parameter"}

    def _add_to_list(self, action: Dict[str, Any], data: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Add value to a list field"""
        field = action.get('field')
        value = action.get('value')

        if field:
            current_value = self._get_nested_value(data, field) or []
            if not isinstance(current_value, list):
                current_value = [current_value]
            current_value.append(value)
            self._set_nested_value(data, field, current_value)
            return {"field": field, "value": value, "status": "added"}

        return {"error": "Missing field parameter"}

    def _remove_from_list(self, action: Dict[str, Any], data: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove value from a list field"""
        field = action.get('field')
        value = action.get('value')

        if field:
            current_value = self._get_nested_value(data, field) or []
            if isinstance(current_value, list) and value in current_value:
                current_value.remove(value)
                self._set_nested_value(data, field, current_value)
                return {"field": field, "value": value, "status": "removed"}

        return {"field": field, "value": value, "status": "not_found"}

    def _calculate_score(self, action: Dict[str, Any], data: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate a score based on conditions"""
        score_field = action.get('score_field', 'score')
        base_score = action.get('base_score', 0)
        multipliers = action.get('multipliers', {})

        score = base_score

        # Apply multipliers based on matched conditions
        matched_conditions = evaluation_result.get('matched_conditions', [])
        for condition in matched_conditions:
            condition_name = condition.get('name', '')
            if condition_name in multipliers:
                score *= multipliers[condition_name]

        self._set_nested_value(data, score_field, score)
        return {"score_field": score_field, "score": score, "status": "calculated"}

    def _flag_record(self, action: Dict[str, Any], data: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Flag a record with a specific status"""
        flag_field = action.get('flag_field', 'flags')
        flag_value = action.get('flag_value', 'flagged')
        reason = action.get('reason', 'Rule triggered')

        flags = self._get_nested_value(data, flag_field) or []
        if not isinstance(flags, list):
            flags = [flags]

        flag_entry = {
            "flag": flag_value,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "rule_id": evaluation_result.get('rule_id')
        }

        flags.append(flag_entry)
        self._set_nested_value(data, flag_field, flags)

        return {"flag_field": flag_field, "flag_value": flag_value, "status": "flagged"}

    def _transform_data(self, action: Dict[str, Any], data: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data using mapping rules"""
        transformations = action.get('transformations', {})

        transformed_fields = {}
        for target_field, source_field in transformations.items():
            value = self._get_nested_value(data, source_field)
            if value is not None:
                self._set_nested_value(data, target_field, value)
                transformed_fields[target_field] = value

        return {"transformed_fields": transformed_fields, "status": "transformed"}

    def _send_notification(self, action: Dict[str, Any], data: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification (placeholder for actual implementation)"""
        notification_type = action.get('notification_type', 'email')
        recipient = action.get('recipient', 'admin@example.com')
        message = action.get('message', 'Rule triggered')

        # In a real implementation, this would integrate with email/SMS services
        logger.info(f"Sending {notification_type} notification to {recipient}: {message}")

        return {
            "notification_type": notification_type,
            "recipient": recipient,
            "message": message,
            "status": "sent"
        }

    def _update_status(self, action: Dict[str, Any], data: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update record status"""
        status_field = action.get('status_field', 'status')
        new_status = action.get('new_status', 'processed')

        self._set_nested_value(data, status_field, new_status)

        return {"status_field": status_field, "new_status": new_status, "status": "updated"}

    def _log_event(self, action: Dict[str, Any], data: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Log an event"""
        event_type = action.get('event_type', 'rule_triggered')
        message = action.get('message', 'Rule evaluation completed')
        level = action.get('level', 'info')

        log_data = {
            "event_type": event_type,
            "message": message,
            "level": level,
            "rule_id": evaluation_result.get('rule_id'),
            "timestamp": datetime.utcnow().isoformat(),
            "data_summary": {k: v for k, v in data.items() if k in ['id', 'status', 'score']}
        }

        if level == 'error':
            logger.error(message, extra=log_data)
        elif level == 'warning':
            logger.warning(message, extra=log_data)
        else:
            logger.info(message, extra=log_data)

        return {"event_type": event_type, "level": level, "status": "logged"}

    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested value from data using dot notation"""
        keys = field_path.split('.')
        current = data

        try:
            for key in keys:
                if isinstance(current, dict):
                    current = current[key]
                elif isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                else:
                    return None
            return current
        except (KeyError, IndexError, TypeError):
            return None

    def _set_nested_value(self, data: Dict[str, Any], field_path: str, value: Any):
        """Set nested value in data using dot notation"""
        keys = field_path.split('.')
        current = data

        for i, key in enumerate(keys[:-1]):
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

# =============================================================================
# BUSINESS LOGIC CLASSES
# =============================================================================

class RuleEngine:
    """Main rule engine for processing rules and evaluating conditions"""

    def __init__(self, db_session: Session, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.evaluator = RuleEvaluator()
        self.action_executor = RuleActionExecutor()

        # Load built-in rule sets
        self._load_builtin_rule_sets()

    def _load_builtin_rule_sets(self):
        """Load built-in rule sets"""
        builtin_sets = [
            self._create_fraud_detection_rules(),
            self._create_compliance_rules(),
            self._create_risk_assessment_rules(),
            self._create_data_validation_rules()
        ]

        for rule_set in builtin_sets:
            try:
                self.create_rule_set(rule_set, "system")
            except Exception as e:
                logger.error(f"Failed to load built-in rule set {rule_set['rule_set_id']}: {str(e)}")

    def _create_fraud_detection_rules(self) -> Dict[str, Any]:
        """Create fraud detection rule set"""
        return {
            "rule_set_id": "fraud_detection",
            "name": "Fraud Detection Rules",
            "description": "Rules for detecting potential fraudulent activities",
            "domain": "fraud",
            "rules": [
                {
                    "rule_id": "high_amount_transaction",
                    "name": "High Amount Transaction",
                    "description": "Flag transactions above threshold",
                    "rule_type": "decision",
                    "priority": 1,
                    "conditions": [
                        {
                            "field": "amount",
                            "operator": "greater_than",
                            "value": 50000
                        }
                    ],
                    "actions": [
                        {
                            "type": "flag_record",
                            "flag_value": "high_amount",
                            "reason": "Transaction amount exceeds threshold"
                        },
                        {
                            "type": "update_status",
                            "new_status": "requires_review"
                        }
                    ]
                },
                {
                    "rule_id": "rapid_transactions",
                    "name": "Rapid Successive Transactions",
                    "description": "Flag rapid transaction patterns",
                    "rule_type": "decision",
                    "priority": 2,
                    "conditions": [
                        {
                            "field": "transaction_frequency",
                            "operator": "greater_than",
                            "value": 5
                        }
                    ],
                    "actions": [
                        {
                            "type": "flag_record",
                            "flag_value": "rapid_transactions",
                            "reason": "High frequency of transactions detected"
                        }
                    ]
                }
            ]
        }

    def _create_compliance_rules(self) -> Dict[str, Any]:
        """Create compliance rule set"""
        return {
            "rule_set_id": "compliance_rules",
            "name": "Compliance Rules",
            "description": "Rules for regulatory compliance validation",
            "domain": "compliance",
            "rules": [
                {
                    "rule_id": "data_retention_check",
                    "name": "Data Retention Compliance",
                    "description": "Ensure data retention policies are followed",
                    "rule_type": "validation",
                    "priority": 1,
                    "conditions": [
                        {
                            "field": "data_age_days",
                            "operator": "greater_than",
                            "value": 2555  # 7 years
                        }
                    ],
                    "actions": [
                        {
                            "type": "flag_record",
                            "flag_value": "retention_violation",
                            "reason": "Data exceeds retention period"
                        }
                    ]
                }
            ]
        }

    def _create_risk_assessment_rules(self) -> Dict[str, Any]:
        """Create risk assessment rule set"""
        return {
            "rule_set_id": "risk_assessment",
            "name": "Risk Assessment Rules",
            "description": "Rules for assessing various types of risk",
            "domain": "risk",
            "rules": [
                {
                    "rule_id": "credit_score_low",
                    "name": "Low Credit Score",
                    "description": "Flag applications with low credit scores",
                    "rule_type": "scoring",
                    "priority": 1,
                    "conditions": [
                        {
                            "field": "credit_score",
                            "operator": "less_than",
                            "value": 620
                        }
                    ],
                    "actions": [
                        {
                            "type": "calculate_score",
                            "score_field": "risk_score",
                            "base_score": 40,
                            "multipliers": {}
                        }
                    ]
                }
            ]
        }

    def _create_data_validation_rules(self) -> Dict[str, Any]:
        """Create data validation rule set"""
        return {
            "rule_set_id": "data_validation",
            "name": "Data Validation Rules",
            "description": "Rules for validating data integrity and quality",
            "domain": "data_quality",
            "rules": [
                {
                    "rule_id": "required_fields_check",
                    "name": "Required Fields Validation",
                    "description": "Ensure required fields are present",
                    "rule_type": "validation",
                    "priority": 1,
                    "conditions": [
                        {
                            "field": "required_field",
                            "operator": "is_null"
                        }
                    ],
                    "actions": [
                        {
                            "type": "flag_record",
                            "flag_value": "missing_data",
                            "reason": "Required field is missing"
                        }
                    ]
                }
            ]
        }

    def create_rule_set(self, rule_set_data: Dict[str, Any], created_by: str) -> str:
        """Create a new rule set"""
        rule_set_id = rule_set_data['rule_set_id']

        # Check if rule set already exists
        existing = self.db.query(RuleSet).filter_by(rule_set_id=rule_set_id).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Rule set {rule_set_id} already exists")

        # Create rule set
        rule_set = RuleSet(
            rule_set_id=rule_set_id,
            name=rule_set_data['name'],
            description=rule_set_data.get('description'),
            domain=rule_set_data.get('domain'),
            created_by=created_by
        )
        self.db.add(rule_set)

        # Create rules
        rules = rule_set_data.get('rules', [])
        for rule_data in rules:
            rule = Rule(
                rule_id=rule_data['rule_id'],
                rule_set_id=rule_set_id,
                name=rule_data['name'],
                description=rule_data.get('description'),
                rule_type=rule_data['rule_type'],
                priority=rule_data.get('priority', 1),
                conditions=rule_data['conditions'],
                actions=rule_data['actions'],
                metadata=rule_data.get('metadata', {}),
                created_by=created_by
            )
            self.db.add(rule)

        # Update rule count
        rule_set.rule_count = len(rules)

        self.db.commit()
        return rule_set_id

    def get_rule_set(self, rule_set_id: str) -> Optional[Dict[str, Any]]:
        """Get a rule set by ID"""
        rule_set = self.db.query(RuleSet).filter_by(rule_set_id=rule_set_id, is_active=True).first()
        if not rule_set:
            return None

        # Get rules
        rules = self.db.query(Rule).filter_by(rule_set_id=rule_set_id, is_active=True).all()

        return {
            'rule_set_id': rule_set.rule_set_id,
            'name': rule_set.name,
            'description': rule_set.description,
            'domain': rule_set.domain,
            'version': rule_set.version,
            'rule_count': rule_set.rule_count,
            'rules': [{
                'rule_id': rule.rule_id,
                'name': rule.name,
                'description': rule.description,
                'rule_type': rule.rule_type,
                'priority': rule.priority,
                'conditions': rule.conditions,
                'actions': rule.actions,
                'execution_count': rule.execution_count,
                'success_count': rule.success_count
            } for rule in rules],
            'created_by': rule_set.created_by,
            'created_at': rule_set.created_at.isoformat(),
            'updated_at': rule_set.updated_at.isoformat()
        }

    def evaluate_rule_set(self, rule_set_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a rule set against input data"""
        rule_set = self.get_rule_set(rule_set_id)
        if not rule_set:
            raise HTTPException(status_code=404, detail=f"Rule set {rule_set_id} not found")

        execution_id = str(uuid.uuid4())
        start_time = time.time()

        results = []
        matched_rules = []
        executed_actions = []

        # Sort rules by priority
        rules = sorted(rule_set['rules'], key=lambda r: r['priority'])

        for rule in rules:
            try:
                # Evaluate conditions
                conditions_result = self.evaluator.evaluate_conditions(
                    rule['conditions'],
                    input_data,
                    rule.get('logical_operator', 'and')
                )

                rule_result = {
                    'rule_id': rule['rule_id'],
                    'rule_name': rule['name'],
                    'conditions_matched': conditions_result['result'],
                    'matched_conditions': conditions_result['matched_conditions'],
                    'failed_conditions': conditions_result['failed_conditions']
                }

                # Execute actions if conditions matched
                if conditions_result['result']:
                    matched_rules.append(rule['rule_id'])
                    actions_result = self.action_executor.execute_actions(
                        rule['actions'],
                        input_data,
                        {
                            'rule_id': rule['rule_id'],
                            'matched_conditions': conditions_result['matched_conditions']
                        }
                    )
                    rule_result['executed_actions'] = actions_result['executed_actions']
                    executed_actions.extend(actions_result['executed_actions'])

                results.append(rule_result)

                # Update rule statistics
                self._update_rule_stats(rule['rule_id'], conditions_result['result'], time.time() - start_time)

            except Exception as e:
                logger.error(f"Error evaluating rule {rule['rule_id']}: {str(e)}")
                results.append({
                    'rule_id': rule['rule_id'],
                    'rule_name': rule['name'],
                    'error': str(e)
                })

        execution_time = int((time.time() - start_time) * 1000)

        # Create execution record
        execution = RuleExecution(
            execution_id=execution_id,
            rule_set_id=rule_set_id,
            input_data=input_data,
            output_data={
                'results': results,
                'matched_rules': matched_rules,
                'executed_actions': executed_actions,
                'execution_time_ms': execution_time
            },
            result='success' if matched_rules else 'no_match',
            matched_conditions=matched_rules,
            executed_actions=executed_actions
        )
        self.db.add(execution)
        self.db.commit()

        return {
            'execution_id': execution_id,
            'rule_set_id': rule_set_id,
            'input_data': input_data,
            'output_data': input_data,  # Modified by actions
            'results': results,
            'matched_rules': matched_rules,
            'executed_actions': executed_actions,
            'execution_time_ms': execution_time,
            'timestamp': datetime.utcnow().isoformat()
        }

    def _update_rule_stats(self, rule_id: str, success: bool, execution_time: float):
        """Update rule execution statistics"""
        rule = self.db.query(Rule).filter_by(rule_id=rule_id).first()
        if rule:
            rule.execution_count += 1
            if success:
                rule.success_count += 1
            rule.average_execution_time_ms = (
                (rule.average_execution_time_ms * (rule.execution_count - 1)) + (execution_time * 1000)
            ) / rule.execution_count
            rule.last_executed = datetime.utcnow()
            self.db.commit()

    def list_rule_sets(self, domain: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """List available rule sets"""
        query = self.db.query(RuleSet)

        if domain:
            query = query.filter_by(domain=domain)
        if active_only:
            query = query.filter_by(is_active=True)

        rule_sets = query.all()

        return [{
            'rule_set_id': rs.rule_set_id,
            'name': rs.name,
            'description': rs.description,
            'domain': rs.domain,
            'version': rs.version,
            'rule_count': rs.rule_count,
            'created_by': rs.created_by,
            'created_at': rs.created_at.isoformat()
        } for rs in rule_sets]

    def get_rule_performance(self, rule_id: Optional[str] = None, rule_set_id: Optional[str] = None) -> Dict[str, Any]:
        """Get rule performance statistics"""
        query = self.db.query(Rule)

        if rule_id:
            query = query.filter_by(rule_id=rule_id)
        if rule_set_id:
            query = query.filter_by(rule_set_id=rule_set_id)

        rules = query.all()

        performance_data = []
        for rule in rules:
            success_rate = (rule.success_count / rule.execution_count * 100) if rule.execution_count > 0 else 0

            performance_data.append({
                'rule_id': rule.rule_id,
                'rule_set_id': rule.rule_set_id,
                'execution_count': rule.execution_count,
                'success_count': rule.success_count,
                'success_rate': success_rate,
                'average_execution_time_ms': rule.average_execution_time_ms,
                'last_executed': rule.last_executed.isoformat() if rule.last_executed else None
            })

        return {
            'performance_data': performance_data,
            'total_rules': len(performance_data),
            'generated_at': datetime.utcnow().isoformat()
        }

# =============================================================================
# API MODELS
# =============================================================================

class RuleCondition(BaseModel):
    """Model for rule conditions"""
    field: str
    operator: str
    value: Optional[Any] = None

class RuleAction(BaseModel):
    """Model for rule actions"""
    type: str
    field: Optional[str] = None
    value: Optional[Any] = None
    message: Optional[str] = None

class RuleDefinition(BaseModel):
    """Model for rule definition"""
    rule_id: str
    name: str
    description: Optional[str] = None
    rule_type: str
    priority: int = 1
    conditions: List[RuleCondition]
    actions: List[RuleAction]
    logical_operator: str = "and"
    metadata: Optional[Dict[str, Any]] = None

class RuleSetDefinition(BaseModel):
    """Model for rule set definition"""
    rule_set_id: str
    name: str
    description: Optional[str] = None
    domain: Optional[str] = None
    rules: List[RuleDefinition]

class RuleEvaluationRequest(BaseModel):
    """Model for rule evaluation request"""
    rule_set_id: str
    input_data: Dict[str, Any]
    evaluation_options: Optional[Dict[str, Any]] = None

class RuleSetSearchRequest(BaseModel):
    """Model for rule set search request"""
    domain: Optional[str] = None
    active_only: bool = True
    limit: int = 20
    offset: int = 0

# =============================================================================
# MONITORING & METRICS
# =============================================================================

class MetricsCollector:
    """Collects and exposes Prometheus metrics"""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Rule metrics
        self.rule_sets_active = Gauge('rule_engine_rule_sets_active', 'Number of active rule sets', registry=self.registry)
        self.rule_evaluations = Counter('rule_engine_rule_evaluations_total', 'Total rule evaluations', ['rule_set_id', 'result'], registry=self.registry)
        self.rule_evaluation_time = Histogram('rule_engine_evaluation_duration_seconds', 'Rule evaluation duration', ['rule_set_id'], registry=self.registry)
        self.rule_execution_count = Counter('rule_engine_rule_executions_total', 'Total rule executions', ['rule_id', 'result'], registry=self.registry)

        # Performance metrics
        self.request_count = Counter('rule_engine_requests_total', 'Total number of requests', ['method', 'endpoint'], registry=self.registry)
        self.request_duration = Histogram('rule_engine_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'], registry=self.registry)
        self.error_count = Counter('rule_engine_errors_total', 'Total number of errors', ['type'], registry=self.registry)

    def update_rule_metrics(self, rule_engine: RuleEngine):
        """Update rule-related metrics"""
        try:
            rule_sets = rule_engine.list_rule_sets()
            self.rule_sets_active.set(len([rs for rs in rule_sets]))

        except Exception as e:
            logger.error(f"Failed to update rule metrics: {str(e)}")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Rule Engine Service",
    description="Business rule processing and decision logic execution",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db_engine = create_engine(Config.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=Config.REDIS_DB, decode_responses=True)
metrics_collector = MetricsCollector()

# Dependency injection
def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_rule_engine(db: Session = Depends(get_db)):
    """Rule engine dependency"""
    return RuleEngine(db, redis_client)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "rule-engine",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    if not Config.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    return generate_latest(metrics_collector.registry)

@app.get("/rules/sets")
async def list_rule_sets(
    domain: Optional[str] = None,
    active_only: bool = True,
    limit: int = 20,
    offset: int = 0,
    rule_engine: RuleEngine = Depends(get_rule_engine)
):
    """List available rule sets"""
    metrics_collector.request_count.labels(method='GET', endpoint='/rules/sets').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/rules/sets').time():
        rule_sets = rule_engine.list_rule_sets(domain=domain, active_only=active_only)

        # Apply pagination
        paginated_sets = rule_sets[offset:offset + limit]

    return {
        "rule_sets": paginated_sets,
        "total_count": len(rule_sets),
        "limit": limit,
        "offset": offset
    }

@app.get("/rules/sets/{rule_set_id}")
async def get_rule_set(
    rule_set_id: str,
    rule_engine: RuleEngine = Depends(get_rule_engine)
):
    """Get detailed information about a rule set"""
    metrics_collector.request_count.labels(method='GET', endpoint='/rules/sets/{rule_set_id}').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/rules/sets/{rule_set_id}').time():
        rule_set = rule_engine.get_rule_set(rule_set_id)

        if not rule_set:
            raise HTTPException(status_code=404, detail=f"Rule set {rule_set_id} not found")

    return rule_set

@app.post("/rules/evaluate")
async def evaluate_rules(
    request: RuleEvaluationRequest,
    rule_engine: RuleEngine = Depends(get_rule_engine)
):
    """Evaluate rules against input data"""
    metrics_collector.request_count.labels(method='POST', endpoint='/rules/evaluate').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/rules/evaluate').time():
        result = rule_engine.evaluate_rule_set(request.rule_set_id, request.input_data)

    return result

@app.get("/rules/performance")
async def get_rule_performance(
    rule_id: Optional[str] = None,
    rule_set_id: Optional[str] = None,
    rule_engine: RuleEngine = Depends(get_rule_engine)
):
    """Get rule performance statistics"""
    metrics_collector.request_count.labels(method='GET', endpoint='/rules/performance').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/rules/performance').time():
        performance = rule_engine.get_rule_performance(rule_id=rule_id, rule_set_id=rule_set_id)

    return performance

@app.get("/rules/operators")
async def get_condition_operators():
    """Get available condition operators"""
    operators = [
        {
            "operator": "equals",
            "name": "Equals",
            "description": "Check if field equals value",
            "value_required": True
        },
        {
            "operator": "not_equals",
            "name": "Not Equals",
            "description": "Check if field does not equal value",
            "value_required": True
        },
        {
            "operator": "greater_than",
            "name": "Greater Than",
            "description": "Check if field is greater than value",
            "value_required": True
        },
        {
            "operator": "less_than",
            "name": "Less Than",
            "description": "Check if field is less than value",
            "value_required": True
        },
        {
            "operator": "contains",
            "name": "Contains",
            "description": "Check if field contains value",
            "value_required": True
        },
        {
            "operator": "is_null",
            "name": "Is Null",
            "description": "Check if field is null",
            "value_required": False
        },
        {
            "operator": "in_list",
            "name": "In List",
            "description": "Check if field value is in a list",
            "value_required": True
        },
        {
            "operator": "between",
            "name": "Between",
            "description": "Check if field value is between two values",
            "value_required": True
        }
    ]

    return {"operators": operators}

@app.get("/rules/actions")
async def get_action_types():
    """Get available action types"""
    actions = [
        {
            "type": "set_field",
            "name": "Set Field",
            "description": "Set a field to a specific value",
            "parameters": ["field", "value"]
        },
        {
            "type": "flag_record",
            "name": "Flag Record",
            "description": "Add a flag to the record",
            "parameters": ["flag_value", "reason"]
        },
        {
            "type": "update_status",
            "name": "Update Status",
            "description": "Update the record status",
            "parameters": ["new_status"]
        },
        {
            "type": "calculate_score",
            "name": "Calculate Score",
            "description": "Calculate a score based on conditions",
            "parameters": ["score_field", "base_score"]
        },
        {
            "type": "send_notification",
            "name": "Send Notification",
            "description": "Send a notification",
            "parameters": ["notification_type", "recipient", "message"]
        },
        {
            "type": "log_event",
            "name": "Log Event",
            "description": "Log an event",
            "parameters": ["event_type", "message", "level"]
        }
    ]

    return {"actions": actions}

@app.get("/rules/domains")
async def get_domains():
    """Get available business domains"""
    domains = [
        {"id": "fraud", "name": "Fraud Detection", "description": "Rules for fraud detection and prevention"},
        {"id": "compliance", "name": "Compliance", "description": "Regulatory compliance rules"},
        {"id": "risk", "name": "Risk Assessment", "description": "Risk assessment and scoring rules"},
        {"id": "underwriting", "name": "Underwriting", "description": "Insurance underwriting rules"},
        {"id": "claims", "name": "Claims", "description": "Insurance claims processing rules"},
        {"id": "data_quality", "name": "Data Quality", "description": "Data validation and quality rules"}
    ]

    return {"domains": domains}

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    metrics_collector.error_count.labels(type='validation').inc()

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "message": "Invalid request data"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    metrics_collector.error_count.labels(type='http').inc()

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    metrics_collector.error_count.labels(type='general').inc()

    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": str(uuid.uuid4())
        }
    )

# =============================================================================
# LIFECYCLE MANAGEMENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Starting Rule Engine Service...")

    # Create database tables
    try:
        Base.metadata.create_all(bind=db_engine)
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise

    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        raise

    logger.info(f"Rule Engine Service started on {Config.SERVICE_HOST}:{Config.SERVICE_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down Rule Engine Service...")

    # Close Redis connection
    try:
        redis_client.close()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {str(e)}")

    logger.info("Rule Engine Service shutdown complete")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.SERVICE_HOST,
        port=Config.SERVICE_PORT,
        reload=False,
        log_level="info",
        access_log=True
    )
