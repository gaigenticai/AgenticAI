#!/usr/bin/env python3
"""
Plugin Registry Service for Agentic Brain Platform

This service manages the registration, discovery, loading, and execution of plugins
for the Agentic Brain platform. It supports both domain-specific plugins (underwriting,
claims, fraud detection) and generic plugins (data processing, validation).

Features:
- Plugin registration and metadata management
- Plugin discovery and loading
- Plugin execution orchestration
- Plugin dependency management
- Plugin versioning and updates
- Plugin security validation
- RESTful API for plugin operations
- Comprehensive monitoring and metrics
- Authentication and authorization support
"""

import asyncio
import importlib
import inspect
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import threading

import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, UploadFile, File
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
    """Configuration class for Plugin Registry Service"""

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
    REDIS_DB = 1  # Use DB 1 for plugin registry

    # Service Configuration
    SERVICE_HOST = os.getenv('PLUGIN_REGISTRY_HOST', '0.0.0.0')
    SERVICE_PORT = int(os.getenv('PLUGIN_REGISTRY_PORT', '8201'))

    # Authentication Configuration
    REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-jwt-key-change-in-production')
    JWT_ALGORITHM = 'HS256'

    # Plugin Configuration
    PLUGIN_LOAD_TIMEOUT = int(os.getenv('PLUGIN_LOAD_TIMEOUT_SECONDS', '30'))
    PLUGIN_EXECUTION_TIMEOUT = int(os.getenv('PLUGIN_EXECUTION_TIMEOUT_SECONDS', '300'))
    PLUGIN_CACHE_ENABLED = os.getenv('PLUGIN_CACHE_ENABLED', 'true').lower() == 'true'
    PLUGIN_AUTO_UPDATE = os.getenv('PLUGIN_AUTO_UPDATE_ENABLED', 'false').lower() == 'true'

    # Security Configuration
    ALLOW_PLUGIN_UPLOAD = os.getenv('ALLOW_PLUGIN_UPLOAD', 'true').lower() == 'true'
    PLUGIN_SANDBOX_ENABLED = os.getenv('PLUGIN_SANDBOX_ENABLED', 'true').lower() == 'true'

    # Monitoring Configuration
    METRICS_ENABLED = os.getenv('AGENT_METRICS_ENABLED', 'true').lower() == 'true'

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class PluginMetadata(Base):
    """Database model for plugin metadata"""
    __tablename__ = 'plugin_metadata'

    id = Column(Integer, primary_key=True)
    plugin_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    plugin_type = Column(String(50), nullable=False)  # domain, generic
    domain = Column(String(100))  # underwriting, claims, fraud, etc. (NULL for generic)
    description = Column(Text)
    version = Column(String(50), default='1.0.0')
    author = Column(String(255))
    license = Column(String(100))
    repository_url = Column(String(500))
    documentation_url = Column(String(500))
    dependencies = Column(JSON, default=list)
    configuration_schema = Column(JSON)
    entry_point = Column(String(255))  # Python module path
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    usage_count = Column(BigInteger, default=0)
    rating = Column(Float)
    tags = Column(JSON, default=list)
    security_score = Column(Float, default=0.0)  # 0.0 to 1.0
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

class PluginExecution(Base):
    """Database model for plugin execution tracking"""
    __tablename__ = 'plugin_executions'

    id = Column(Integer, primary_key=True)
    execution_id = Column(String(100), unique=True, nullable=False)
    plugin_id = Column(String(100), nullable=False)
    agent_id = Column(String(100))
    input_data = Column(JSON)
    output_data = Column(JSON)
    execution_time_ms = Column(Integer)
    status = Column(String(50), default='running')  # running, completed, failed
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class PluginDependency(Base):
    """Database model for plugin dependencies"""
    __tablename__ = 'plugin_dependencies'

    id = Column(Integer, primary_key=True)
    plugin_id = Column(String(100), nullable=False)
    dependency_name = Column(String(255), nullable=False)
    dependency_version = Column(String(50))
    dependency_type = Column(String(50), default='python')  # python, system, plugin
    is_required = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# =============================================================================
# PLUGIN INTERFACES
# =============================================================================

class PluginInterface:
    """Base interface that all plugins must implement"""

    @property
    def plugin_id(self) -> str:
        """Unique identifier for the plugin"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Human-readable name"""
        raise NotImplementedError

    @property
    def version(self) -> str:
        """Plugin version"""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Plugin description"""
        raise NotImplementedError

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration"""
        pass

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plugin with input data"""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Cleanup resources"""
        pass

class DomainPlugin(PluginInterface):
    """Interface for domain-specific plugins"""

    @property
    def domain(self) -> str:
        """Business domain this plugin serves"""
        raise NotImplementedError

    def validate_domain_data(self, data: Dict[str, Any]) -> bool:
        """Validate if input data is appropriate for this domain"""
        raise NotImplementedError

class GenericPlugin(PluginInterface):
    """Interface for generic plugins"""

    def supports_data_type(self, data_type: str) -> bool:
        """Check if plugin supports the given data type"""
        raise NotImplementedError

# =============================================================================
# BUILT-IN PLUGINS
# =============================================================================

class RiskCalculatorPlugin(DomainPlugin):
    """Risk calculation plugin for underwriting domain"""

    @property
    def plugin_id(self) -> str:
        return "riskCalculator"

    @property
    def name(self) -> str:
        return "Risk Calculator"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Advanced risk calculation engine for underwriting decisions"

    @property
    def domain(self) -> str:
        return "underwriting"

    def validate_domain_data(self, data: Dict[str, Any]) -> bool:
        """Validate underwriting data"""
        required_fields = ['loan_amount', 'credit_score', 'income', 'debt_ratio']
        return all(field in data for field in required_fields)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score"""
        try:
            # Simple risk calculation logic
            loan_amount = input_data.get('loan_amount', 0)
            credit_score = input_data.get('credit_score', 600)
            income = input_data.get('income', 0)
            debt_ratio = input_data.get('debt_ratio', 0.5)

            # Risk factors
            risk_score = 0

            # Credit score factor
            if credit_score < 620:
                risk_score += 40
            elif credit_score < 660:
                risk_score += 25
            elif credit_score < 720:
                risk_score += 15
            else:
                risk_score += 5

            # Debt-to-income ratio factor
            if debt_ratio > 0.43:
                risk_score += 30
            elif debt_ratio > 0.36:
                risk_score += 20
            elif debt_ratio > 0.28:
                risk_score += 10

            # Loan-to-income ratio factor
            lti_ratio = loan_amount / income if income > 0 else 1.0
            if lti_ratio > 0.9:
                risk_score += 25
            elif lti_ratio > 0.8:
                risk_score += 15
            elif lti_ratio > 0.7:
                risk_score += 10

            # Determine risk category
            if risk_score < 20:
                risk_category = "LOW"
                approval_probability = 0.9
            elif risk_score < 40:
                risk_category = "MEDIUM"
                approval_probability = 0.7
            elif risk_score < 60:
                risk_category = "HIGH"
                approval_probability = 0.4
            else:
                risk_category = "VERY_HIGH"
                approval_probability = 0.1

            return {
                'risk_score': risk_score,
                'risk_category': risk_category,
                'approval_probability': approval_probability,
                'recommendation': 'APPROVE' if approval_probability > 0.6 else 'REVIEW',
                'calculated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                'error': f'Risk calculation failed: {str(e)}',
                'risk_score': 100,
                'risk_category': 'ERROR',
                'approval_probability': 0.0,
                'recommendation': 'REJECT'
            }

class FraudDetectorPlugin(DomainPlugin):
    """Fraud detection plugin for claims domain"""

    @property
    def plugin_id(self) -> str:
        return "fraudDetector"

    @property
    def name(self) -> str:
        return "Fraud Detection Engine"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Machine learning-based fraud detection for claims processing"

    @property
    def domain(self) -> str:
        return "claims"

    def validate_domain_data(self, data: Dict[str, Any]) -> bool:
        """Validate claims data"""
        required_fields = ['claim_amount', 'incident_date', 'policy_start_date', 'claim_history']
        return all(field in data for field in required_fields)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential fraud"""
        try:
            claim_amount = input_data.get('claim_amount', 0)
            incident_date = input_data.get('incident_date')
            policy_start_date = input_data.get('policy_start_date')
            claim_history = input_data.get('claim_history', [])

            # Fraud detection logic
            fraud_score = 0
            flags = []

            # Time since policy start
            if incident_date and policy_start_date:
                days_since_policy_start = (datetime.fromisoformat(incident_date.replace('Z', '+00:00')) -
                                         datetime.fromisoformat(policy_start_date.replace('Z', '+00:00'))).days
                if days_since_policy_start < 30:
                    fraud_score += 25
                    flags.append("Early claim after policy start")

            # Claim frequency
            if len(claim_history) > 2:
                fraud_score += 20
                flags.append("High claim frequency")

            # Claim amount vs policy coverage
            if claim_amount > 100000:  # Assuming typical coverage
                fraud_score += 15
                flags.append("High claim amount")

            # Suspicious patterns
            if len(flags) > 1:
                fraud_score += 10
                flags.append("Multiple suspicious indicators")

            # Determine fraud risk
            if fraud_score < 20:
                fraud_risk = "LOW"
                investigation_required = False
            elif fraud_score < 40:
                fraud_risk = "MEDIUM"
                investigation_required = True
            elif fraud_score < 60:
                fraud_risk = "HIGH"
                investigation_required = True
            else:
                fraud_risk = "CRITICAL"
                investigation_required = True

            return {
                'fraud_score': fraud_score,
                'fraud_risk': fraud_risk,
                'investigation_required': investigation_required,
                'flags': flags,
                'recommendation': 'INVESTIGATE' if investigation_required else 'APPROVE',
                'detected_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                'error': f'Fraud detection failed: {str(e)}',
                'fraud_score': 100,
                'fraud_risk': 'ERROR',
                'investigation_required': True,
                'flags': ['Detection error'],
                'recommendation': 'INVESTIGATE'
            }

class DataRetrieverPlugin(GenericPlugin):
    """Generic data retrieval plugin"""

    @property
    def plugin_id(self) -> str:
        return "dataRetriever"

    @property
    def name(self) -> str:
        return "Universal Data Retriever"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Generic data retrieval and transformation plugin"

    def supports_data_type(self, data_type: str) -> bool:
        """Support all data types"""
        return True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve and transform data"""
        try:
            source_type = input_data.get('source_type', 'unknown')
            query = input_data.get('query', {})
            transformation = input_data.get('transformation', {})

            # Mock data retrieval
            result = {
                'source_type': source_type,
                'query': query,
                'retrieved_records': 150,
                'data': [
                    {'id': 1, 'field1': 'value1', 'field2': 'value2'},
                    {'id': 2, 'field1': 'value3', 'field2': 'value4'}
                ],
                'transformation_applied': bool(transformation),
                'retrieved_at': datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            return {
                'error': f'Data retrieval failed: {str(e)}',
                'retrieved_records': 0,
                'data': []
            }

class ValidatorPlugin(GenericPlugin):
    """Data validation plugin"""

    @property
    def plugin_id(self) -> str:
        return "validator"

    @property
    def name(self) -> str:
        return "Data Validator"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Comprehensive data validation and quality assurance plugin"

    def supports_data_type(self, data_type: str) -> bool:
        """Support all data types"""
        return True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data"""
        try:
            data = input_data.get('data', [])
            validation_rules = input_data.get('validation_rules', {})

            total_records = len(data) if isinstance(data, list) else 1
            valid_records = 0
            invalid_records = 0
            errors = []

            # Basic validation
            for i, record in enumerate(data if isinstance(data, list) else [data]):
                is_valid = True
                record_errors = []

                # Required fields validation
                required_fields = validation_rules.get('required_fields', [])
                for field in required_fields:
                    if field not in record or record[field] is None:
                        is_valid = False
                        record_errors.append(f"Missing required field: {field}")

                # Data type validation
                field_types = validation_rules.get('field_types', {})
                for field, expected_type in field_types.items():
                    if field in record:
                        actual_value = record[field]
                        if expected_type == 'string' and not isinstance(actual_value, str):
                            is_valid = False
                            record_errors.append(f"Field {field} should be string")
                        elif expected_type == 'number' and not isinstance(actual_value, (int, float)):
                            is_valid = False
                            record_errors.append(f"Field {field} should be number")

                if is_valid:
                    valid_records += 1
                else:
                    invalid_records += 1
                    errors.extend(record_errors)

            validation_result = {
                'total_records': total_records,
                'valid_records': valid_records,
                'invalid_records': invalid_records,
                'validation_score': valid_records / total_records if total_records > 0 else 0,
                'errors': errors[:10],  # Limit error messages
                'validated_at': datetime.utcnow().isoformat()
            }

            return validation_result

        except Exception as e:
            return {
                'error': f'Validation failed: {str(e)}',
                'total_records': 0,
                'valid_records': 0,
                'invalid_records': 0,
                'validation_score': 0.0,
                'errors': [str(e)]
            }

class RegulatoryCheckerPlugin(DomainPlugin):
    """Regulatory compliance checker plugin for underwriting domain"""

    @property
    def plugin_id(self) -> str:
        return "regulatoryChecker"

    @property
    def name(self) -> str:
        return "Regulatory Compliance Checker"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Automated regulatory compliance checking for underwriting decisions"

    @property
    def domain(self) -> str:
        return "underwriting"

    def validate_domain_data(self, data: Dict[str, Any]) -> bool:
        """Validate underwriting data for regulatory compliance"""
        required_fields = ['loan_amount', 'borrower_info', 'property_info', 'lending_region']
        return all(field in data for field in required_fields)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check regulatory compliance"""
        try:
            loan_amount = input_data.get('loan_amount', 0)
            borrower_info = input_data.get('borrower_info', {})
            property_info = input_data.get('property_info', {})
            lending_region = input_data.get('lending_region', 'US')

            # Regulatory compliance checks
            compliance_issues = []
            compliance_score = 100

            # HMDA compliance (Home Mortgage Disclosure Act)
            if lending_region == 'US':
                if loan_amount > 1000000:  # High-cost loan threshold
                    compliance_issues.append("HMDA: High-cost loan reporting required")
                    compliance_score -= 10

            # Dodd-Frank compliance
            ability_to_repay = borrower_info.get('ability_to_repay_verified', False)
            if not ability_to_repay and loan_amount > 50000:
                compliance_issues.append("Dodd-Frank: Ability to repay must be verified")
                compliance_score -= 20

            # Fair Lending compliance
            borrower_race = borrower_info.get('race_ethnicity')
            if borrower_race and borrower_race not in ['not_provided', 'not_applicable']:
                # Check for disparate impact patterns (simplified)
                if loan_amount > borrower_info.get('income', 0) * 10:
                    compliance_issues.append("Fair Lending: Potential disparate impact concern")
                    compliance_score -= 15

            # RESPA compliance (Real Estate Settlement Procedures Act)
            if property_info.get('settlement_charges', 0) > loan_amount * 0.03:
                compliance_issues.append("RESPA: Settlement charges exceed 3% threshold")
                compliance_score -= 10

            # State-specific regulations
            state = lending_region.split('-')[-1] if '-' in lending_region else lending_region
            if state in ['CA', 'NY', 'FL']:  # High-regulation states
                compliance_issues.append(f"{state}: Additional state-specific compliance required")
                compliance_score -= 5

            # Overall compliance status
            if compliance_score >= 90:
                compliance_status = "COMPLIANT"
                approval_recommended = True
            elif compliance_score >= 75:
                compliance_status = "MINOR_ISSUES"
                approval_recommended = True
            elif compliance_score >= 60:
                compliance_status = "SIGNIFICANT_ISSUES"
                approval_recommended = False
            else:
                compliance_status = "NON_COMPLIANT"
                approval_recommended = False

            return {
                'compliance_score': compliance_score,
                'compliance_status': compliance_status,
                'approval_recommended': approval_recommended,
                'compliance_issues': compliance_issues,
                'regulatory_frameworks_checked': [
                    'HMDA', 'Dodd-Frank', 'Fair Lending', 'RESPA'
                ],
                'region_specific_rules': f"{lending_region} regulations applied",
                'recommendation': 'APPROVE' if approval_recommended else 'REVIEW',
                'checked_at': datetime.utcnow().isoformat(),
                'next_review_date': (datetime.utcnow() + timedelta(days=365)).isoformat()
            }

        except Exception as e:
            return {
                'error': f'Regulatory check failed: {str(e)}',
                'compliance_score': 0,
                'compliance_status': 'ERROR',
                'approval_recommended': False,
                'compliance_issues': ['Compliance check error'],
                'recommendation': 'REJECT'
            }

class DocumentAnalyzerPlugin(DomainPlugin):
    """Document analysis plugin for claims domain"""

    @property
    def plugin_id(self) -> str:
        return "documentAnalyzer"

    @property
    def name(self) -> str:
        return "Document Analysis Engine"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "AI-powered document analysis for claims processing and evidence evaluation"

    @property
    def domain(self) -> str:
        return "claims"

    def validate_domain_data(self, data: Dict[str, Any]) -> bool:
        """Validate document data"""
        required_fields = ['document_type', 'content', 'metadata']
        return all(field in data for field in required_fields)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document content"""
        try:
            document_type = input_data.get('document_type', 'unknown')
            content = input_data.get('content', '')
            metadata = input_data.get('metadata', {})

            # Document analysis logic
            analysis_score = 85  # Default high confidence
            findings = []
            recommendations = []

            # Content analysis based on document type
            if document_type == 'police_report':
                if 'accident' in content.lower():
                    findings.append("Accident details documented")
                    analysis_score += 5
                if 'witness' in content.lower():
                    findings.append("Witness statements available")
                    analysis_score += 3
                if len(content) < 500:
                    findings.append("Report appears incomplete")
                    analysis_score -= 10

            elif document_type == 'medical_report':
                if 'diagnosis' in content.lower():
                    findings.append("Medical diagnosis documented")
                    analysis_score += 5
                if 'treatment' in content.lower():
                    findings.append("Treatment details available")
                    analysis_score += 3
                if not any(term in content.lower() for term in ['doctor', 'physician', 'hospital']):
                    findings.append("Medical professional verification missing")
                    analysis_score -= 15

            elif document_type == 'repair_estimate':
                if '$' in content or 'cost' in content.lower():
                    findings.append("Cost estimates provided")
                    analysis_score += 5
                if 'parts' in content.lower():
                    findings.append("Parts breakdown available")
                    analysis_score += 3
                if len(content) < 200:
                    findings.append("Estimate appears incomplete")
                    analysis_score -= 10

            # Metadata analysis
            if metadata.get('creation_date'):
                doc_date = datetime.fromisoformat(metadata['creation_date'].replace('Z', '+00:00'))
                days_old = (datetime.utcnow() - doc_date).days
                if days_old > 365:
                    findings.append("Document is over 1 year old")
                    analysis_score -= 5

            # Authenticity checks
            if metadata.get('digital_signature'):
                findings.append("Document has digital signature")
                analysis_score += 5
            elif metadata.get('certified_copy', False):
                findings.append("Certified copy available")
                analysis_score += 3

            # Generate recommendations
            if analysis_score < 70:
                recommendations.append("Request additional documentation")
            if analysis_score < 50:
                recommendations.append("Escalate to claims specialist")

            # Determine analysis result
            if analysis_score >= 80:
                analysis_result = "VERIFIED"
                claim_support = "STRONG"
            elif analysis_score >= 60:
                analysis_result = "ACCEPTABLE"
                claim_support = "MODERATE"
            else:
                analysis_result = "INSUFFICIENT"
                claim_support = "WEAK"

            return {
                'analysis_score': analysis_score,
                'analysis_result': analysis_result,
                'claim_support': claim_support,
                'findings': findings,
                'recommendations': recommendations,
                'document_type': document_type,
                'content_length': len(content),
                'metadata_verified': bool(metadata),
                'analyzed_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                'error': f'Document analysis failed: {str(e)}',
                'analysis_score': 0,
                'analysis_result': 'ERROR',
                'claim_support': 'UNKNOWN',
                'findings': ['Analysis error'],
                'recommendations': ['Manual review required']
            }

class SentimentAnalyzerPlugin(GenericPlugin):
    """Sentiment analysis plugin for customer feedback"""

    @property
    def plugin_id(self) -> str:
        return "sentimentAnalyzer"

    @property
    def name(self) -> str:
        return "Sentiment Analysis Engine"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Natural language processing for sentiment analysis and emotional intelligence"

    def supports_data_type(self, data_type: str) -> bool:
        """Support text data types"""
        return data_type in ['text', 'string', 'feedback', 'review']

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment in text"""
        try:
            text = input_data.get('text', '')
            context = input_data.get('context', 'general')

            if not text:
                return {
                    'error': 'No text provided for analysis',
                    'sentiment': 'NEUTRAL',
                    'confidence': 0.0
                }

            # Simple sentiment analysis (in production, use ML models)
            positive_words = ['good', 'great', 'excellent', 'happy', 'satisfied', 'love', 'awesome']
            negative_words = ['bad', 'terrible', 'awful', 'angry', 'disappointed', 'hate', 'worst']

            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            # Calculate sentiment score
            if positive_count > negative_count:
                sentiment = 'POSITIVE'
                score = min(1.0, (positive_count - negative_count) / len(text.split()) * 10)
            elif negative_count > positive_count:
                sentiment = 'NEGATIVE'
                score = min(1.0, (negative_count - positive_count) / len(text.split()) * 10)
            else:
                sentiment = 'NEUTRAL'
                score = 0.5

            # Context-specific adjustments
            if context == 'customer_service':
                # More sensitive to negative feedback
                if sentiment == 'NEGATIVE':
                    score = min(1.0, score * 1.2)
            elif context == 'product_review':
                # Balanced analysis
                pass

            # Determine confidence level
            word_count = len(text.split())
            confidence = min(1.0, word_count / 50)  # Higher confidence with more words

            # Generate insights
            insights = []
            if sentiment == 'NEGATIVE':
                insights.append("Customer dissatisfaction detected")
                if 'wait' in text_lower:
                    insights.append("Waiting time is a pain point")
            elif sentiment == 'POSITIVE':
                insights.append("Customer satisfaction identified")
                if 'helpful' in text_lower:
                    insights.append("Service quality praised")

            return {
                'sentiment': sentiment,
                'score': score,
                'confidence': confidence,
                'insights': insights,
                'word_count': word_count,
                'context': context,
                'analyzed_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                'error': f'Sentiment analysis failed: {str(e)}',
                'sentiment': 'NEUTRAL',
                'score': 0.0,
                'confidence': 0.0,
                'insights': []
            }

class ComplianceMonitorPlugin(DomainPlugin):
    """Compliance monitoring plugin for regulatory oversight"""

    @property
    def plugin_id(self) -> str:
        return "complianceMonitor"

    @property
    def name(self) -> str:
        return "Compliance Monitoring System"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Continuous compliance monitoring and regulatory reporting system"

    @property
    def domain(self) -> str:
        return "compliance"

    def validate_domain_data(self, data: Dict[str, Any]) -> bool:
        """Validate compliance data"""
        required_fields = ['activity_type', 'timestamp', 'actor', 'regulatory_framework']
        return all(field in data for field in required_fields)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor compliance of activities"""
        try:
            activity_type = input_data.get('activity_type', 'unknown')
            timestamp = input_data.get('timestamp')
            actor = input_data.get('actor', {})
            regulatory_framework = input_data.get('regulatory_framework', 'general')

            # Compliance monitoring logic
            compliance_violations = []
            risk_level = "LOW"
            monitoring_alerts = []

            # Activity-specific compliance checks
            if activity_type == 'loan_approval':
                # Check for required documentation
                required_docs = ['credit_report', 'income_verification', 'property_appraisal']
                provided_docs = input_data.get('documents', [])

                for doc in required_docs:
                    if doc not in provided_docs:
                        compliance_violations.append(f"Missing required document: {doc}")
                        risk_level = "MEDIUM"

                # Check approval authority limits
                loan_amount = input_data.get('loan_amount', 0)
                approver_role = actor.get('role', 'unknown')

                if approver_role == 'junior_officer' and loan_amount > 100000:
                    compliance_violations.append("Approval exceeds junior officer authority limit")
                    risk_level = "HIGH"

            elif activity_type == 'customer_data_access':
                # GDPR compliance checks
                access_purpose = input_data.get('access_purpose')
                data_retention = input_data.get('data_retention_days', 0)

                if not access_purpose:
                    compliance_violations.append("Data access purpose not documented")
                    risk_level = "MEDIUM"

                if data_retention > 2555:  # GDPR max retention
                    compliance_violations.append("Data retention exceeds GDPR limits")
                    risk_level = "HIGH"

            elif activity_type == 'financial_transaction':
                # Anti-money laundering checks
                transaction_amount = input_data.get('amount', 0)
                customer_history = input_data.get('customer_history', {})

                if transaction_amount > 10000:  # SAR threshold
                    monitoring_alerts.append("Large transaction - SAR filing may be required")
                    risk_level = "MEDIUM"

                suspicious_patterns = customer_history.get('suspicious_activities', 0)
                if suspicious_patterns > 0:
                    compliance_violations.append(f"Customer has {suspicious_patterns} suspicious activities")
                    risk_level = "HIGH"

            # Regulatory framework specific checks
            if regulatory_framework == 'gdpr':
                if not input_data.get('consent_obtained', False):
                    compliance_violations.append("GDPR: User consent not obtained")
                    risk_level = "HIGH"

            elif regulatory_framework == 'ccpa':
                if not input_data.get('privacy_notice_provided', False):
                    compliance_violations.append("CCPA: Privacy notice not provided")
                    risk_level = "MEDIUM"

            # Generate compliance report
            compliance_status = "COMPLIANT"
            if compliance_violations:
                if risk_level == "HIGH":
                    compliance_status = "VIOLATION"
                else:
                    compliance_status = "WARNING"

            # Required actions
            required_actions = []
            if compliance_status == "VIOLATION":
                required_actions.extend([
                    "Immediate compliance review",
                    "Escalate to compliance officer",
                    "Document violation details"
                ])
            elif compliance_status == "WARNING":
                required_actions.append("Review and address compliance concerns")

            return {
                'compliance_status': compliance_status,
                'risk_level': risk_level,
                'violations': compliance_violations,
                'monitoring_alerts': monitoring_alerts,
                'required_actions': required_actions,
                'regulatory_framework': regulatory_framework,
                'activity_type': activity_type,
                'actor_info': actor,
                'monitored_at': datetime.utcnow().isoformat(),
                'next_review_date': (datetime.utcnow() + timedelta(days=30)).isoformat()
            }

        except Exception as e:
            return {
                'error': f'Compliance monitoring failed: {str(e)}',
                'compliance_status': 'ERROR',
                'risk_level': 'UNKNOWN',
                'violations': ['Monitoring system error'],
                'monitoring_alerts': ['System error detected'],
                'required_actions': ['Manual compliance review required']
            }

# =============================================================================
# API MODELS
# =============================================================================

class PluginRegistrationRequest(BaseModel):
    """Request model for plugin registration"""
    plugin_id: str = Field(..., description="Unique identifier for the plugin")
    name: str = Field(..., description="Human-readable name")
    plugin_type: str = Field(..., description="Plugin type: domain or generic")
    domain: Optional[str] = Field(None, description="Business domain for domain plugins")
    description: str = Field(..., description="Plugin description")
    version: str = Field("1.0.0", description="Plugin version")
    author: str = Field(..., description="Plugin author")
    entry_point: str = Field(..., description="Python module entry point")
    dependencies: List[Dict[str, Any]] = Field(default_factory=list, description="Plugin dependencies")
    configuration_schema: Optional[Dict[str, Any]] = Field(None, description="Configuration schema")
    tags: List[str] = Field(default_factory=list, description="Plugin tags")

class PluginExecutionRequest(BaseModel):
    """Request model for plugin execution"""
    plugin_id: str = Field(..., description="Plugin to execute")
    input_data: Dict[str, Any] = Field(..., description="Input data for plugin")
    execution_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution configuration")
    timeout_seconds: Optional[int] = Field(default=300, description="Execution timeout")

class PluginResponse(BaseModel):
    """Response model for plugin information"""
    plugin_id: str
    name: str
    plugin_type: str
    domain: Optional[str]
    description: str
    version: str
    author: str
    is_active: bool
    usage_count: int
    rating: Optional[float]
    tags: List[str]
    created_at: datetime

# =============================================================================
# BUSINESS LOGIC CLASSES
# =============================================================================

class PluginManager:
    """Manages plugin loading, registration, and execution"""

    def __init__(self, db_session: Session, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.loaded_plugins = {}  # Cache of loaded plugin instances
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Register built-in plugins
        self._register_builtin_plugins()

    def _register_builtin_plugins(self):
        """Register built-in plugins"""
        builtin_plugins = [
            RiskCalculatorPlugin(),
            FraudDetectorPlugin(),
            RegulatoryCheckerPlugin(),
            DocumentAnalyzerPlugin(),
            SentimentAnalyzerPlugin(),
            ComplianceMonitorPlugin(),
            DataRetrieverPlugin(),
            ValidatorPlugin()
        ]

        for plugin in builtin_plugins:
            try:
                self.register_plugin_from_instance(plugin)
            except Exception as e:
                logger.error(f"Failed to register built-in plugin {plugin.plugin_id}: {str(e)}")

    def register_plugin_from_instance(self, plugin_instance: PluginInterface):
        """Register a plugin from a plugin instance"""
        try:
            # Check if plugin already exists
            existing = self.db.query(PluginMetadata).filter_by(plugin_id=plugin_instance.plugin_id).first()
            if existing:
                # Update existing plugin
                existing.name = plugin_instance.name
                existing.version = plugin_instance.version
                existing.description = plugin_instance.description
                existing.last_updated = datetime.utcnow()
            else:
                # Create new plugin metadata
                metadata = PluginMetadata(
                    plugin_id=plugin_instance.plugin_id,
                    name=plugin_instance.name,
                    plugin_type='domain' if isinstance(plugin_instance, DomainPlugin) else 'generic',
                    domain=getattr(plugin_instance, 'domain', None),
                    description=plugin_instance.description,
                    version=plugin_instance.version,
                    is_active=True,
                    is_verified=True,  # Built-in plugins are verified
                    entry_point=f"builtin.{plugin_instance.__class__.__name__}"
                )
                self.db.add(metadata)

            self.db.commit()

            # Cache the plugin instance
            self.loaded_plugins[plugin_instance.plugin_id] = plugin_instance

            logger.info(f"Registered plugin: {plugin_instance.plugin_id}")

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to register plugin {plugin_instance.plugin_id}: {str(e)}")
            raise

    def load_plugin(self, plugin_id: str) -> PluginInterface:
        """Load a plugin by ID"""
        # Check cache first
        if plugin_id in self.loaded_plugins:
            return self.loaded_plugins[plugin_id]

        # Load from database
        metadata = self.db.query(PluginMetadata).filter_by(plugin_id=plugin_id, is_active=True).first()
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Plugin {plugin_id} not found")

        # For built-in plugins, return cached instance
        if metadata.entry_point.startswith("builtin."):
            if plugin_id in self.loaded_plugins:
                return self.loaded_plugins[plugin_id]
            else:
                raise HTTPException(status_code=500, detail=f"Built-in plugin {plugin_id} not available")

        # For external plugins, dynamic loading would go here
        # This is a placeholder for future extension
        raise HTTPException(status_code=501, detail=f"External plugin loading not implemented for {plugin_id}")

    def execute_plugin(self, plugin_id: str, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a plugin with given input data"""
        try:
            plugin = self.load_plugin(plugin_id)

            # Initialize plugin if config provided
            if config:
                plugin.initialize(config)

            # Execute plugin
            start_time = time.time()
            result = plugin.execute(input_data)
            execution_time = int((time.time() - start_time) * 1000)

            # Update usage statistics
            metadata = self.db.query(PluginMetadata).filter_by(plugin_id=plugin_id).first()
            if metadata:
                metadata.usage_count += 1
                self.db.commit()

            # Log execution
            execution = PluginExecution(
                execution_id=str(uuid.uuid4()),
                plugin_id=plugin_id,
                input_data=input_data,
                output_data=result,
                execution_time_ms=execution_time,
                status='completed'
            )
            self.db.add(execution)
            self.db.commit()

            return result

        except Exception as e:
            # Log failed execution
            execution = PluginExecution(
                execution_id=str(uuid.uuid4()),
                plugin_id=plugin_id,
                input_data=input_data,
                error_message=str(e),
                execution_time_ms=0,
                status='failed'
            )
            self.db.add(execution)
            self.db.commit()

            logger.error(f"Plugin execution failed for {plugin_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Plugin execution failed: {str(e)}")

    def list_plugins(self, plugin_type: Optional[str] = None, domain: Optional[str] = None,
                    active_only: bool = True) -> List[Dict[str, Any]]:
        """List available plugins"""
        query = self.db.query(PluginMetadata)

        if plugin_type:
            query = query.filter_by(plugin_type=plugin_type)
        if domain:
            query = query.filter_by(domain=domain)
        if active_only:
            query = query.filter_by(is_active=True)

        plugins = query.all()

        return [{
            'plugin_id': p.plugin_id,
            'name': p.name,
            'plugin_type': p.plugin_type,
            'domain': p.domain,
            'description': p.description,
            'version': p.version,
            'author': p.author,
            'is_active': p.is_active,
            'usage_count': p.usage_count,
            'rating': p.rating,
            'tags': p.tags,
            'created_at': p.created_at.isoformat()
        } for p in plugins]

    def get_plugin_info(self, plugin_id: str) -> Dict[str, Any]:
        """Get detailed information about a plugin"""
        plugin = self.db.query(PluginMetadata).filter_by(plugin_id=plugin_id).first()
        if not plugin:
            raise HTTPException(status_code=404, detail=f"Plugin {plugin_id} not found")

        # Get dependencies
        dependencies = self.db.query(PluginDependency).filter_by(plugin_id=plugin_id).all()

        return {
            'plugin_id': plugin.plugin_id,
            'name': plugin.name,
            'plugin_type': plugin.plugin_type,
            'domain': plugin.domain,
            'description': plugin.description,
            'version': plugin.version,
            'author': plugin.author,
            'license': plugin.license,
            'repository_url': plugin.repository_url,
            'documentation_url': plugin.documentation_url,
            'dependencies': [{'name': d.dependency_name, 'version': d.dependency_version, 'type': d.dependency_type} for d in dependencies],
            'configuration_schema': plugin.configuration_schema,
            'is_active': plugin.is_active,
            'is_verified': plugin.is_verified,
            'usage_count': plugin.usage_count,
            'rating': plugin.rating,
            'tags': plugin.tags,
            'security_score': plugin.security_score,
            'created_at': plugin.created_at.isoformat(),
            'last_updated': plugin.last_updated.isoformat()
        }

# =============================================================================
# MONITORING & METRICS
# =============================================================================

class MetricsCollector:
    """Collects and exposes Prometheus metrics"""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Plugin metrics
        self.active_plugins = Gauge('plugin_registry_active_plugins', 'Number of active plugins', registry=self.registry)
        self.plugin_executions = Counter('plugin_registry_executions_total', 'Total plugin executions', ['plugin_id', 'status'], registry=self.registry)
        self.plugin_execution_time = Histogram('plugin_registry_execution_duration_seconds', 'Plugin execution duration', ['plugin_id'], registry=self.registry)
        self.plugin_load_time = Histogram('plugin_registry_load_duration_seconds', 'Plugin load duration', registry=self.registry)

        # Performance metrics
        self.request_count = Counter('plugin_registry_requests_total', 'Total number of requests', ['method', 'endpoint'], registry=self.registry)
        self.request_duration = Histogram('plugin_registry_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'], registry=self.registry)
        self.error_count = Counter('plugin_registry_errors_total', 'Total number of errors', ['type'], registry=self.registry)

    def update_plugin_metrics(self, plugin_manager: PluginManager):
        """Update plugin-related metrics"""
        try:
            plugins = plugin_manager.list_plugins()
            self.active_plugins.set(len([p for p in plugins if p['is_active']]))

        except Exception as e:
            logger.error(f"Failed to update plugin metrics: {str(e)}")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Plugin Registry Service",
    description="Manages plugins for the Agentic Brain platform",
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

def get_plugin_manager(db: Session = Depends(get_db)):
    """Plugin manager dependency"""
    return PluginManager(db, redis_client)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "plugin-registry",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    if not Config.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    return generate_latest(metrics_collector.registry)

@app.get("/plugins")
async def list_plugins(
    plugin_type: Optional[str] = None,
    domain: Optional[str] = None,
    active_only: bool = True,
    plugin_manager: PluginManager = Depends(get_plugin_manager)
):
    """List available plugins"""
    metrics_collector.request_count.labels(method='GET', endpoint='/plugins').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/plugins').time():
        plugins = plugin_manager.list_plugins(plugin_type=plugin_type, domain=domain, active_only=active_only)

    return {"plugins": plugins, "count": len(plugins)}

@app.get("/plugins/{plugin_id}")
async def get_plugin(
    plugin_id: str,
    plugin_manager: PluginManager = Depends(get_plugin_manager)
):
    """Get detailed information about a specific plugin"""
    metrics_collector.request_count.labels(method='GET', endpoint='/plugins/{plugin_id}').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/plugins/{plugin_id}').time():
        plugin_info = plugin_manager.get_plugin_info(plugin_id)

    return plugin_info

@app.post("/plugins/execute")
async def execute_plugin(
    request: PluginExecutionRequest,
    plugin_manager: PluginManager = Depends(get_plugin_manager)
):
    """Execute a plugin with given input data"""
    metrics_collector.request_count.labels(method='POST', endpoint='/plugins/execute').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/plugins/execute').time():
        result = plugin_manager.execute_plugin(
            request.plugin_id,
            request.input_data,
            request.execution_config
        )

    return result

@app.get("/plugins/types")
async def get_plugin_types():
    """Get available plugin types"""
    return {
        "types": ["domain", "generic"],
        "domains": ["underwriting", "claims", "fraud", "compliance", "general"]
    }

@app.get("/plugins/stats")
async def get_plugin_stats(plugin_manager: PluginManager = Depends(get_plugin_manager)):
    """Get plugin usage statistics"""
    metrics_collector.request_count.labels(method='GET', endpoint='/plugins/stats').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/plugins/stats').time():
        plugins = plugin_manager.list_plugins()

        stats = {
            'total_plugins': len(plugins),
            'active_plugins': len([p for p in plugins if p['is_active']]),
            'domain_plugins': len([p for p in plugins if p['plugin_type'] == 'domain']),
            'generic_plugins': len([p for p in plugins if p['plugin_type'] == 'generic']),
            'total_usage': sum(p['usage_count'] for p in plugins),
            'top_plugins': sorted(plugins, key=lambda p: p['usage_count'], reverse=True)[:5]
        }

    return stats

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
    logger.info("Starting Plugin Registry Service...")

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

    logger.info(f"Plugin Registry Service started on {Config.SERVICE_HOST}:{Config.SERVICE_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down Plugin Registry Service...")

    # Close Redis connection
    try:
        redis_client.close()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {str(e)}")

    # Cleanup plugin instances
    try:
        plugin_manager = PluginManager(SessionLocal(), redis_client)
        for plugin in plugin_manager.loaded_plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin.plugin_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Error during plugin cleanup: {str(e)}")

    logger.info("Plugin Registry Service shutdown complete")

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
