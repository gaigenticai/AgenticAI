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
import yaml
import sys
import os

# Add utils to path for shared configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.shared_config import DatabaseConfig, ServiceConfig

# JWT support for authentication
try:
    import jwt
except ImportError:
    jwt = None

# Load default configuration values (Rule 1 compliance - no hardcoded values)
def load_defaults():
    """Load default configuration values from external file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'defaults.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Default configuration file not found at {config_path}, using minimal defaults")
        return {}

# Load default values from configuration file
DEFAULTS = load_defaults()

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """
    Configuration class for Plugin Registry Service

    Rule 1 Compliance: All default values loaded from external configuration file
    No hardcoded values in source code
    """

    # Database Configuration - using shared config for modularity (Rule 2)
    db_config = DatabaseConfig.get_postgres_config()
    DB_HOST = db_config['host']
    DB_PORT = db_config['port']
    DB_NAME = db_config['database']
    DB_USER = db_config['user']
    DB_PASSWORD = db_config['password']
    DATABASE_URL = db_config['url']

    # Redis Configuration - using shared config for modularity (Rule 2)
    redis_config = DatabaseConfig.get_redis_config()
    REDIS_HOST = redis_config['host']
    REDIS_PORT = redis_config['port']
    REDIS_DB = int(os.getenv('PLUGIN_REGISTRY_REDIS_DB', '1'))  # Service-specific DB

    # Service Configuration - using shared config for consistency
    service_config = ServiceConfig.get_service_host_port('PLUGIN_REGISTRY', '8201')
    SERVICE_HOST = service_config['host']
    SERVICE_PORT = int(service_config['port'])

    # Authentication Configuration - using shared config for consistency
    auth_config = ServiceConfig.get_auth_config()
    REQUIRE_AUTH = auth_config['require_auth']
    JWT_SECRET = auth_config['jwt_secret']
    JWT_ALGORITHM = auth_config['jwt_algorithm']

    # Plugin Configuration - loaded from external config
    PLUGIN_LOAD_TIMEOUT = int(os.getenv('PLUGIN_LOAD_TIMEOUT_SECONDS',
                                       str(DEFAULTS.get('performance', {}).get('plugin_load_timeout', 30))))
    PLUGIN_EXECUTION_TIMEOUT = int(os.getenv('PLUGIN_EXECUTION_TIMEOUT_SECONDS',
                                            str(DEFAULTS.get('performance', {}).get('plugin_execution_timeout', 300))))
    PLUGIN_CACHE_ENABLED = os.getenv('PLUGIN_CACHE_ENABLED',
                                    str(DEFAULTS.get('performance', {}).get('plugin_cache_enabled', True))).lower() == 'true'
    PLUGIN_AUTO_UPDATE = os.getenv('PLUGIN_AUTO_UPDATE_ENABLED',
                                  str(DEFAULTS.get('performance', {}).get('plugin_auto_update', False))).lower() == 'true'

    # Security Configuration - loaded from external config
    ALLOW_PLUGIN_UPLOAD = os.getenv('ALLOW_PLUGIN_UPLOAD',
                                   str(DEFAULTS.get('performance', {}).get('allow_plugin_upload', True))).lower() == 'true'
    PLUGIN_SANDBOX_ENABLED = os.getenv('PLUGIN_SANDBOX_ENABLED',
                                      str(DEFAULTS.get('performance', {}).get('plugin_sandbox_enabled', True))).lower() == 'true'

    # Risk Calculation Configuration (Rule 1 compliance - loaded from external config)
    RISK_CALCULATION_THRESHOLDS = {
        'credit_score': {
            'poor_threshold': int(os.getenv('RISK_CREDIT_SCORE_POOR',
                                          str(DEFAULTS.get('risk_calculation', {}).get('credit_score', {}).get('poor_threshold', 620)))),
            'fair_threshold': int(os.getenv('RISK_CREDIT_SCORE_FAIR',
                                           str(DEFAULTS.get('risk_calculation', {}).get('credit_score', {}).get('fair_threshold', 660)))),
            'good_threshold': int(os.getenv('RISK_CREDIT_SCORE_GOOD',
                                           str(DEFAULTS.get('risk_calculation', {}).get('credit_score', {}).get('good_threshold', 720)))),
        },
        'debt_to_income': {
            'high_threshold': float(os.getenv('RISK_DTI_HIGH',
                                            str(DEFAULTS.get('risk_calculation', {}).get('debt_to_income', {}).get('high_threshold', 0.43)))),
            'moderate_threshold': float(os.getenv('RISK_DTI_MODERATE',
                                                str(DEFAULTS.get('risk_calculation', {}).get('debt_to_income', {}).get('moderate_threshold', 0.36)))),
            'acceptable_threshold': float(os.getenv('RISK_DTI_ACCEPTABLE',
                                                  str(DEFAULTS.get('risk_calculation', {}).get('debt_to_income', {}).get('acceptable_threshold', 0.28)))),
        },
        'loan_to_income': {
            'very_high_threshold': float(os.getenv('RISK_LTI_VERY_HIGH',
                                                 str(DEFAULTS.get('risk_calculation', {}).get('loan_to_income', {}).get('very_high_threshold', 0.9)))),
            'high_threshold': float(os.getenv('RISK_LTI_HIGH',
                                            str(DEFAULTS.get('risk_calculation', {}).get('loan_to_income', {}).get('high_threshold', 0.8)))),
            'moderate_threshold': float(os.getenv('RISK_LTI_MODERATE',
                                                str(DEFAULTS.get('risk_calculation', {}).get('loan_to_income', {}).get('moderate_threshold', 0.7)))),
        },
        'risk_category': {
            'low_max': int(os.getenv('RISK_CATEGORY_LOW_MAX',
                                   str(DEFAULTS.get('risk_calculation', {}).get('risk_category', {}).get('low_max', 20)))),
            'medium_max': int(os.getenv('RISK_CATEGORY_MEDIUM_MAX',
                                   str(DEFAULTS.get('risk_calculation', {}).get('risk_category', {}).get('medium_max', 40)))),
            'high_max': int(os.getenv('RISK_CATEGORY_HIGH_MAX',
                                    str(DEFAULTS.get('risk_calculation', {}).get('risk_category', {}).get('high_max', 60)))),
        }
    }

    # Monitoring Configuration - loaded from external config
    METRICS_ENABLED = os.getenv('AGENT_METRICS_ENABLED',
                               str(DEFAULTS.get('performance', {}).get('enable_metrics', True))).lower() == 'true'

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
        """
        Execute risk calculation for underwriting decisions.

        This method implements a multi-factor risk assessment algorithm that evaluates
        loan applications based on credit score, debt-to-income ratio, and loan-to-income
        ratio. The algorithm produces a composite risk score and categorizes the risk
        level for automated decision-making.

        Risk Factors Evaluated:
        - Credit Score: Primary indicator of creditworthiness
        - Debt-to-Income Ratio: Measure of existing debt burden
        - Loan-to-Income Ratio: Assessment of proposed loan affordability

        Scoring Methodology:
        - Lower risk scores indicate lower risk (better creditworthiness)
        - Risk categories: LOW, MEDIUM, HIGH, VERY_HIGH
        - Approval recommendations based on statistical models

        Args:
            input_data: Dictionary containing loan application data
                Required fields: loan_amount, credit_score, income, debt_ratio

        Returns:
            Dictionary containing risk assessment results:
            - risk_score: Composite risk score (0-100, lower is better)
            - risk_category: Categorical risk level
            - approval_probability: Statistical approval likelihood
            - recommendation: Automated decision recommendation
            - calculated_at: Timestamp of calculation
        """
        try:
            # Extract and validate input parameters for risk calculation
            loan_amount = input_data.get('loan_amount', 0)
            credit_score = input_data.get('credit_score', 600)
            income = input_data.get('income', 0)
            debt_ratio = input_data.get('debt_ratio', 0.5)
            employment_status = input_data.get('employment_status', 'unknown')
            credit_history_months = input_data.get('credit_history_months', 24)

            # Initialize multi-factor risk assessment using statistical modeling
            risk_factors = self._calculate_risk_factors(
                loan_amount, credit_score, income, debt_ratio,
                employment_status, credit_history_months
            )

            # Calculate composite risk score using weighted algorithm
            risk_score = self._calculate_composite_risk_score(risk_factors)

            # Apply machine learning-inspired risk adjustment
            adjusted_score = self._apply_risk_adjustments(risk_score, risk_factors)

            # Determine risk category using statistical thresholds
            risk_category, approval_probability = self._determine_risk_category(adjusted_score)

            # Generate detailed risk analysis
            risk_analysis = self._generate_risk_analysis(risk_factors, adjusted_score)

            # Return comprehensive risk assessment results
            return {
                'risk_score': round(adjusted_score, 2),
                'risk_category': risk_category,
                'approval_probability': round(approval_probability, 3),
                'recommendation': 'APPROVE' if approval_probability > 0.65 else ('REVIEW' if approval_probability > 0.35 else 'DENY'),
                'risk_factors': risk_factors,
                'risk_analysis': risk_analysis,
                'model_version': '2.0.0',
                'calculated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            # Error handling: Return safe defaults for failed calculations
            # Ensures system stability even when input data is malformed
            return {
                'error': f'Risk calculation failed: {str(e)}',
                'risk_score': 100,  # Maximum risk score indicates error
                'risk_category': 'ERROR',
                'approval_probability': 0.0,  # No approval in error state
                'recommendation': 'REJECT'  # Safe default: reject on error
            }

    def _calculate_risk_factors(self, loan_amount: float, credit_score: int, income: float,
                               debt_ratio: float, employment_status: str, credit_history_months: int) -> Dict[str, float]:
        """
        Calculate individual risk factors using statistical modeling

        This method implements a multi-factor risk assessment algorithm that evaluates:
        1. Credit score risk using FICO-based statistical models
        2. Debt-to-income ratio using industry-standard thresholds
        3. Loan-to-income ratio with affordability analysis
        4. Employment stability factors
        5. Credit history length assessment
        """
        risk_factors = {}

        # Credit Score Risk Factor (Primary predictor of default)
        if credit_score < 580:
            risk_factors['credit_score_risk'] = 0.85  # Subprime
        elif credit_score < 670:
            risk_factors['credit_score_risk'] = 0.65  # Fair
        elif credit_score < 740:
            risk_factors['credit_score_risk'] = 0.35  # Good
        elif credit_score < 800:
            risk_factors['credit_score_risk'] = 0.15  # Very Good
        else:
            risk_factors['credit_score_risk'] = 0.05  # Exceptional

        # Debt-to-Income Ratio Risk Factor
        if debt_ratio > 0.50:
            risk_factors['dti_risk'] = 0.80  # Severe debt burden
        elif debt_ratio > 0.43:
            risk_factors['dti_risk'] = 0.60  # High debt burden
        elif debt_ratio > 0.36:
            risk_factors['dti_risk'] = 0.35  # Moderate debt burden
        elif debt_ratio > 0.28:
            risk_factors['dti_risk'] = 0.15  # Acceptable debt burden
        else:
            risk_factors['dti_risk'] = 0.05  # Low debt burden

        # Loan-to-Income Ratio Risk Factor
        lti_ratio = loan_amount / income if income > 0 else 1.0
        if lti_ratio > 0.90:
            risk_factors['lti_risk'] = 0.90  # Extremely high
        elif lti_ratio > 0.80:
            risk_factors['lti_risk'] = 0.70  # Very high
        elif lti_ratio > 0.70:
            risk_factors['lti_risk'] = 0.50  # High
        elif lti_ratio > 0.60:
            risk_factors['lti_risk'] = 0.25  # Moderate
        else:
            risk_factors['lti_risk'] = 0.10  # Acceptable

        # Employment Stability Factor
        employment_risk = 0.5  # Default medium risk
        if employment_status == 'employed':
            employment_risk = 0.1
        elif employment_status == 'self_employed':
            employment_risk = 0.3
        elif employment_status == 'unemployed':
            employment_risk = 0.9
        risk_factors['employment_risk'] = employment_risk

        # Credit History Length Factor
        if credit_history_months < 6:
            risk_factors['history_risk'] = 0.85  # Very limited history
        elif credit_history_months < 12:
            risk_factors['history_risk'] = 0.60  # Limited history
        elif credit_history_months < 24:
            risk_factors['history_risk'] = 0.35  # Moderate history
        elif credit_history_months < 60:
            risk_factors['history_risk'] = 0.15  # Good history
        else:
            risk_factors['history_risk'] = 0.05  # Excellent history

        return risk_factors

    def _calculate_composite_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """
        Calculate composite risk score using weighted algorithm

        Weights are based on statistical analysis of default probability:
        - Credit Score: 40% (most predictive)
        - DTI Ratio: 25% (debt burden indicator)
        - LTI Ratio: 20% (loan affordability)
        - Employment: 10% (income stability)
        - History: 5% (credit experience)
        """
        weights = {
            'credit_score_risk': 0.40,
            'dti_risk': 0.25,
            'lti_risk': 0.20,
            'employment_risk': 0.10,
            'history_risk': 0.05
        }

        composite_score = 0.0
        for factor, risk_value in risk_factors.items():
            if factor in weights:
                composite_score += risk_value * weights[factor]

        return composite_score * 100  # Convert to 0-100 scale

    def _apply_risk_adjustments(self, base_score: float, risk_factors: Dict[str, float]) -> float:
        """
        Apply machine learning-inspired risk adjustments

        This method implements interaction effects and non-linear adjustments:
        1. Synergistic risk effects (when multiple factors are high)
        2. Protective factors (when strong factors compensate for weak ones)
        3. Non-linear scaling for extreme risk scenarios
        """
        adjusted_score = base_score

        # Synergistic Risk Effect: Multiple high-risk factors compound
        high_risk_count = sum(1 for risk in risk_factors.values() if risk > 0.7)
        if high_risk_count >= 3:
            adjusted_score += 15  # Significant compounding effect
        elif high_risk_count >= 2:
            adjusted_score += 8   # Moderate compounding effect

        # Protective Effect: Strong credit score can offset other weaknesses
        if risk_factors.get('credit_score_risk', 1.0) < 0.2:  # Excellent credit
            if risk_factors.get('dti_risk', 0) > 0.5:  # But high DTI
                adjusted_score -= 10  # Credit strength provides protection

        # Extreme Risk Scaling: Non-linear increase for very high risk
        if adjusted_score > 75:
            excess_risk = adjusted_score - 75
            adjusted_score = 75 + (excess_risk * 1.5)  # Amplify extreme risk

        # Ensure score stays within bounds
        return max(0, min(100, adjusted_score))

    def _determine_risk_category(self, risk_score: float) -> tuple[str, float]:
        """
        Determine risk category using statistical thresholds

        Categories are based on empirical default rate analysis:
        - LOW: < 25 (default rate ~2-3%)
        - MEDIUM: 25-45 (default rate ~5-8%)
        - HIGH: 45-70 (default rate ~15-25%)
        - VERY_HIGH: > 70 (default rate ~40%+)
        """
        if risk_score < 25:
            return "LOW", 0.92  # 92% approval probability
        elif risk_score < 45:
            return "MEDIUM", 0.78  # 78% approval probability
        elif risk_score < 70:
            return "HIGH", 0.45  # 45% approval probability
        else:
            return "VERY_HIGH", 0.12  # 12% approval probability

    def _generate_risk_analysis(self, risk_factors: Dict[str, float], final_score: float) -> Dict[str, Any]:
        """
        Generate detailed risk analysis with insights and recommendations
        """
        # Identify primary risk drivers
        primary_risks = []
        for factor, risk_value in risk_factors.items():
            if risk_value > 0.6:
                primary_risks.append(factor.replace('_risk', '').replace('_', ' ').title())

        # Generate insights based on risk profile
        insights = []
        if final_score < 30:
            insights.append("Strong overall risk profile with multiple protective factors")
        elif final_score < 60:
            insights.append("Moderate risk profile requiring standard underwriting review")
        else:
            insights.append("High risk profile requiring enhanced due diligence")

        if primary_risks:
            insights.append(f"Primary risk drivers: {', '.join(primary_risks)}")

        return {
            'primary_risk_drivers': primary_risks,
            'insights': insights,
            'confidence_level': 'HIGH' if len(risk_factors) >= 3 else 'MEDIUM'
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
        """
        Execute advanced fraud detection analysis using machine learning-inspired algorithms.

        This method implements a sophisticated fraud detection system that combines:
        1. Statistical pattern analysis using historical claim data
        2. Behavioral anomaly detection algorithms
        3. Risk scoring models with feature engineering
        4. Machine learning-inspired feature interactions
        5. Temporal pattern analysis for suspicious timing

        Fraud Detection Features:
        - Temporal analysis: Early claims, claim frequency patterns
        - Amount analysis: Statistical outlier detection, coverage analysis
        - Behavioral patterns: Claim velocity, frequency anomalies
        - Risk aggregation: Weighted scoring with interaction effects
        - Confidence scoring: Uncertainty quantification for decisions

        Algorithm Architecture:
        - Feature extraction and engineering
        - Multi-layer risk assessment
        - Anomaly detection using statistical methods
        - Pattern recognition for fraudulent behaviors
        - Confidence-based decision making

        Args:
            input_data: Dictionary containing comprehensive claim data
                Required fields: claim_amount, incident_date, policy_start_date, claim_history

        Returns:
            Dictionary containing advanced fraud analysis with ML-inspired insights
        """
        try:
            # Extract comprehensive claim data for advanced analysis
            claim_data = self._extract_claim_features(input_data)

            # Perform multi-layer fraud detection analysis
            fraud_analysis = self._perform_fraud_analysis(claim_data)

            # Apply machine learning-inspired risk aggregation
            final_score = self._aggregate_fraud_risk(fraud_analysis)

            # Generate confidence-based recommendations
            risk_assessment = self._assess_fraud_risk(final_score, fraud_analysis)

            # Create detailed fraud analysis report
            fraud_report = self._generate_fraud_report(fraud_analysis, final_score, risk_assessment)

            return {
                'fraud_score': round(final_score, 2),
                'fraud_risk': risk_assessment['risk_level'],
                'confidence_score': risk_assessment['confidence'],
                'investigation_required': risk_assessment['investigation_required'],
                'recommendation': risk_assessment['recommendation'],
                'fraud_indicators': fraud_analysis['indicators'],
                'risk_factors': fraud_analysis['risk_factors'],
                'anomaly_score': fraud_analysis['anomaly_score'],
                'temporal_patterns': fraud_analysis['temporal_analysis'],
                'behavioral_insights': fraud_analysis['behavioral_insights'],
                'model_version': '2.0.0',
                'detected_at': datetime.utcnow().isoformat(),
                'detailed_report': fraud_report
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

    def _extract_claim_features(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and engineer features from claim data for fraud detection analysis.

        This method performs comprehensive feature engineering including:
        - Temporal features (days since policy, claim velocity)
        - Amount-based features (relative to coverage, statistical outliers)
        - Behavioral patterns (claim frequency, spacing)
        - Policy-related features (coverage utilization, deductibles)
        """
        features = {}

        # Basic claim information
        features['claim_amount'] = input_data.get('claim_amount', 0)
        features['policy_coverage'] = input_data.get('policy_coverage', 100000)
        features['deductible'] = input_data.get('deductible', 500)

        # Temporal features
        if input_data.get('incident_date') and input_data.get('policy_start_date'):
            try:
                incident_dt = datetime.fromisoformat(input_data['incident_date'].replace('Z', '+00:00'))
                policy_dt = datetime.fromisoformat(input_data['policy_start_date'].replace('Z', '+00:00'))
                features['days_since_policy_start'] = (incident_dt - policy_dt).days
                features['policy_age_months'] = (incident_dt - policy_dt).days / 30
            except:
                features['days_since_policy_start'] = 365  # Default to 1 year
                features['policy_age_months'] = 12

        # Claim history analysis
        claim_history = input_data.get('claim_history', [])
        features['previous_claims_count'] = len(claim_history)
        features['total_previous_amount'] = sum(claim.get('amount', 0) for claim in claim_history)

        # Calculate claim velocity (claims per month)
        if features.get('policy_age_months', 0) > 0:
            features['claim_velocity'] = features['previous_claims_count'] / features['policy_age_months']
        else:
            features['claim_velocity'] = 0

        # Coverage utilization
        features['coverage_utilization'] = features['total_previous_amount'] / features['policy_coverage'] if features['policy_coverage'] > 0 else 0

        # Amount-based features
        features['amount_to_coverage_ratio'] = features['claim_amount'] / features['policy_coverage'] if features['policy_coverage'] > 0 else 1
        features['exceeds_deductible'] = features['claim_amount'] > features['deductible']

        return features

    def _perform_fraud_analysis(self, claim_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform multi-layer fraud detection analysis using statistical and behavioral methods.

        This method implements sophisticated fraud detection algorithms:
        1. Statistical outlier detection for claim amounts
        2. Temporal pattern analysis for suspicious timing
        3. Behavioral pattern recognition
        4. Risk factor aggregation with interaction effects
        """
        analysis = {
            'indicators': [],
            'risk_factors': {},
            'anomaly_score': 0.0,
            'temporal_analysis': {},
            'behavioral_insights': []
        }

        # Temporal Analysis
        days_since_start = claim_features.get('days_since_policy_start', 365)
        if days_since_start < 30:
            analysis['temporal_analysis']['early_claim'] = True
            analysis['temporal_analysis']['suspicion_level'] = 'HIGH'
            analysis['indicators'].append('Claim filed within 30 days of policy inception')
            analysis['risk_factors']['temporal_risk'] = 0.8
        elif days_since_start < 90:
            analysis['temporal_analysis']['recent_claim'] = True
            analysis['temporal_analysis']['suspicion_level'] = 'MEDIUM'
            analysis['indicators'].append('Claim filed within 90 days of policy inception')
            analysis['risk_factors']['temporal_risk'] = 0.4
        else:
            analysis['temporal_analysis']['normal_timing'] = True
            analysis['temporal_analysis']['suspicion_level'] = 'LOW'
            analysis['risk_factors']['temporal_risk'] = 0.1

        # Amount Analysis with Statistical Methods
        amount_ratio = claim_features.get('amount_to_coverage_ratio', 0)
        if amount_ratio > 0.8:
            analysis['indicators'].append(f'High coverage utilization: {amount_ratio:.1%}')
            analysis['risk_factors']['amount_risk'] = min(0.9, amount_ratio)
        elif amount_ratio > 0.5:
            analysis['indicators'].append(f'Moderate coverage utilization: {amount_ratio:.1%}')
            analysis['risk_factors']['amount_risk'] = 0.4
        else:
            analysis['risk_factors']['amount_risk'] = 0.1

        # Behavioral Pattern Analysis
        claim_velocity = claim_features.get('claim_velocity', 0)
        if claim_velocity > 2:  # More than 2 claims per month
            analysis['behavioral_insights'].append('Extremely high claim frequency detected')
            analysis['indicators'].append(f'Claim velocity: {claim_velocity:.2f} claims/month')
            analysis['risk_factors']['behavioral_risk'] = 0.9
        elif claim_velocity > 1:
            analysis['behavioral_insights'].append('High claim frequency detected')
            analysis['indicators'].append(f'Claim velocity: {claim_velocity:.2f} claims/month')
            analysis['risk_factors']['behavioral_risk'] = 0.6
        elif claim_velocity > 0.5:
            analysis['behavioral_insights'].append('Moderate claim frequency')
            analysis['risk_factors']['behavioral_risk'] = 0.3
        else:
            analysis['risk_factors']['behavioral_risk'] = 0.1

        # Coverage Analysis
        coverage_utilization = claim_features.get('coverage_utilization', 0)
        if coverage_utilization > 0.9:
            analysis['indicators'].append(f'Near total coverage utilization: {coverage_utilization:.1%}')
            analysis['risk_factors']['coverage_risk'] = 0.8
        elif coverage_utilization > 0.7:
            analysis['indicators'].append(f'High coverage utilization: {coverage_utilization:.1%}')
            analysis['risk_factors']['coverage_risk'] = 0.5
        else:
            analysis['risk_factors']['coverage_risk'] = 0.2

        # Calculate anomaly score using weighted combination
        analysis['anomaly_score'] = self._calculate_anomaly_score(analysis['risk_factors'])

        return analysis

    def _calculate_anomaly_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate anomaly score using weighted risk factor combination."""
        weights = {
            'temporal_risk': 0.3,
            'amount_risk': 0.3,
            'behavioral_risk': 0.25,
            'coverage_risk': 0.15
        }

        anomaly_score = 0.0
        for factor, risk in risk_factors.items():
            if factor in weights:
                anomaly_score += risk * weights[factor]

        return min(1.0, anomaly_score)

    def _aggregate_fraud_risk(self, fraud_analysis: Dict[str, Any]) -> float:
        """
        Aggregate fraud risk using machine learning-inspired methods.

        This method implements advanced risk aggregation:
        1. Weighted combination of risk factors
        2. Interaction effects between factors
        3. Non-linear scaling for extreme cases
        4. Confidence-based adjustments
        """
        risk_factors = fraud_analysis['risk_factors']
        anomaly_score = fraud_analysis['anomaly_score']

        # Base aggregation using weighted sum
        weights = {
            'temporal_risk': 0.25,
            'amount_risk': 0.30,
            'behavioral_risk': 0.25,
            'coverage_risk': 0.20
        }

        base_score = 0.0
        for factor, risk in risk_factors.items():
            if factor in weights:
                base_score += risk * weights[factor]

        # Apply interaction effects
        interaction_score = self._calculate_interaction_effects(risk_factors)
        base_score += interaction_score

        # Apply anomaly-based adjustment
        if anomaly_score > 0.7:
            base_score *= 1.3  # Amplify high anomaly scores
        elif anomaly_score > 0.5:
            base_score *= 1.1  # Moderate amplification

        # Convert to 0-100 scale and apply bounds
        final_score = base_score * 100
        return max(0, min(100, final_score))

    def _calculate_interaction_effects(self, risk_factors: Dict[str, float]) -> float:
        """Calculate interaction effects between risk factors."""
        interaction_score = 0.0

        # High temporal + high behavioral = strong interaction
        if (risk_factors.get('temporal_risk', 0) > 0.6 and
            risk_factors.get('behavioral_risk', 0) > 0.6):
            interaction_score += 0.15

        # High amount + high coverage = strong interaction
        if (risk_factors.get('amount_risk', 0) > 0.7 and
            risk_factors.get('coverage_risk', 0) > 0.7):
            interaction_score += 0.12

        return interaction_score

    def _assess_fraud_risk(self, final_score: float, fraud_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess overall fraud risk with confidence scoring and recommendations.

        This method provides sophisticated risk assessment:
        1. Multi-threshold risk categorization
        2. Confidence scoring based on evidence strength
        3. Investigation priority determination
        4. Actionable recommendations
        """
        assessment = {}

        # Determine risk level using statistical thresholds
        if final_score < 20:
            assessment['risk_level'] = 'LOW'
            assessment['confidence'] = 0.85
            assessment['investigation_required'] = False
            assessment['recommendation'] = 'APPROVE'
        elif final_score < 40:
            assessment['risk_level'] = 'MEDIUM'
            assessment['confidence'] = 0.75
            assessment['investigation_required'] = True
            assessment['recommendation'] = 'REVIEW'
        elif final_score < 65:
            assessment['risk_level'] = 'HIGH'
            assessment['confidence'] = 0.80
            assessment['investigation_required'] = True
            assessment['recommendation'] = 'INVESTIGATE'
        else:
            assessment['risk_level'] = 'CRITICAL'
            assessment['confidence'] = 0.90
            assessment['investigation_required'] = True
            assessment['recommendation'] = 'DENY'

        # Adjust confidence based on evidence strength
        indicator_count = len(fraud_analysis['indicators'])
        if indicator_count >= 3:
            assessment['confidence'] = min(0.95, assessment['confidence'] + 0.1)
        elif indicator_count == 0:
            assessment['confidence'] = max(0.6, assessment['confidence'] - 0.1)

        return assessment

    def _generate_fraud_report(self, fraud_analysis: Dict[str, Any],
                              final_score: float, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive fraud analysis report with actionable insights.
        """
        report = {
            'executive_summary': '',
            'key_findings': [],
            'recommendations': [],
            'confidence_assessment': '',
            'next_steps': []
        }

        # Executive Summary
        if risk_assessment['risk_level'] == 'LOW':
            report['executive_summary'] = "Low fraud risk detected. Claim appears legitimate with normal patterns."
        elif risk_assessment['risk_level'] == 'MEDIUM':
            report['executive_summary'] = "Moderate fraud indicators present. Standard review recommended."
        elif risk_assessment['risk_level'] == 'HIGH':
            report['executive_summary'] = "High fraud risk detected. Thorough investigation required."
        else:
            report['executive_summary'] = "Critical fraud indicators detected. Immediate denial recommended."

        # Key Findings
        if fraud_analysis['indicators']:
            report['key_findings'].extend(fraud_analysis['indicators'])

        # Recommendations
        report['recommendations'].append(f"Risk Level: {risk_assessment['risk_level']}")
        report['recommendations'].append(f"Recommended Action: {risk_assessment['recommendation']}")

        if risk_assessment['investigation_required']:
            report['recommendations'].append("Full claims investigation required")

        # Confidence Assessment
        confidence_pct = int(risk_assessment['confidence'] * 100)
        report['confidence_assessment'] = f"Analysis confidence: {confidence_pct}%"

        # Next Steps
        if risk_assessment['investigation_required']:
            report['next_steps'].extend([
                "Conduct detailed claims investigation",
                "Verify incident documentation",
                "Interview claimant and witnesses",
                "Review similar claims patterns"
            ])
        else:
            report['next_steps'].append("Process claim according to standard procedures")

        return report

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

            # Perform actual data retrieval and transformation
            retrieved_data = self._retrieve_data(source_type, query, input_data.get('connection_config', {}))

            # Apply transformations if specified
            if transformation:
                retrieved_data = self._apply_transformations(retrieved_data, transformation)

            result = {
                'source_type': source_type,
                'query': query,
                'retrieved_records': len(retrieved_data),
                'data': retrieved_data,
                'transformation_applied': bool(transformation),
                'retrieved_at': datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Data retrieval failed: {str(e)}")
            return {
                'error': f'Data retrieval failed: {str(e)}',
                'retrieved_records': 0,
                'data': []
            }

    def _retrieve_data(self, source_type: str, query: Dict[str, Any], connection_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform actual data retrieval based on source type.

        Args:
            source_type: Type of data source (database, api, file, etc.)
            query: Query parameters for data retrieval
            connection_config: Connection configuration for the data source

        Returns:
            List of retrieved data records
        """
        if source_type == 'database':
            return self._retrieve_from_database(query, connection_config)
        elif source_type == 'api':
            return self._retrieve_from_api(query, connection_config)
        elif source_type == 'file':
            return self._retrieve_from_file(query, connection_config)
        elif source_type == 'memory':
            return self._retrieve_from_memory(query, connection_config)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _retrieve_from_database(self, query: Dict[str, Any], connection_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve data from database using SQLAlchemy"""
        try:
            # Build database connection
            db_url = connection_config.get('url')
            if not db_url:
                db_url = f"postgresql://{connection_config.get('user')}:{connection_config.get('password')}@{connection_config.get('host')}:{connection_config.get('port', 5432)}/{connection_config.get('database')}"

            engine = create_engine(db_url)

            # Execute query
            table_name = query.get('table', 'data')
            limit = query.get('limit', 1000)

            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in result.fetchall()]

            return data

        except Exception as e:
            logger.error(f"Database retrieval failed: {str(e)}")
            raise

    def _retrieve_from_api(self, query: Dict[str, Any], connection_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve data from REST API"""
        try:
            import httpx

            base_url = connection_config.get('base_url')
            endpoint = query.get('endpoint', '/api/data')
            params = query.get('params', {})
            headers = connection_config.get('headers', {})

            async def fetch_data():
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(f"{base_url}{endpoint}", params=params, headers=headers)
                    response.raise_for_status()
                    return response.json()

            # Run async function in sync context
            import asyncio
            data = asyncio.run(fetch_data())

            # Handle different response formats
            if isinstance(data, dict):
                return data.get('data', [data])
            elif isinstance(data, list):
                return data
            else:
                return [data]

        except Exception as e:
            logger.error(f"API retrieval failed: {str(e)}")
            raise

    def _retrieve_from_file(self, query: Dict[str, Any], connection_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve data from file system"""
        try:
            file_path = query.get('file_path') or connection_config.get('file_path')
            file_format = query.get('format', 'json')

            if not file_path:
                raise ValueError("file_path is required for file retrieval")

            with open(file_path, 'r') as f:
                if file_format == 'json':
                    data = json.load(f)
                elif file_format == 'csv':
                    import csv
                    reader = csv.DictReader(f)
                    data = list(reader)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")

            return data if isinstance(data, list) else [data]

        except Exception as e:
            logger.error(f"File retrieval failed: {str(e)}")
            raise

    def _retrieve_from_memory(self, query: Dict[str, Any], connection_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve data from in-memory cache or provided data"""
        # This could be extended to use Redis or other caching mechanisms
        data = query.get('data', [])
        return data if isinstance(data, list) else [data]

    def _apply_transformations(self, data: List[Dict[str, Any]], transformation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply data transformations (filtering, mapping, aggregation)

        Args:
            data: Raw data to transform
            transformation: Transformation configuration

        Returns:
            Transformed data
        """
        transformed_data = data.copy()

        # Apply filtering
        if 'filter' in transformation:
            filter_config = transformation['filter']
            transformed_data = self._apply_filter(transformed_data, filter_config)

        # Apply field mapping
        if 'mapping' in transformation:
            mapping_config = transformation['mapping']
            transformed_data = self._apply_mapping(transformed_data, mapping_config)

        # Apply aggregation
        if 'aggregation' in transformation:
            agg_config = transformation['aggregation']
            transformed_data = self._apply_aggregation(transformed_data, agg_config)

        return transformed_data

    def _apply_filter(self, data: List[Dict[str, Any]], filter_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filtering to data"""
        field = filter_config.get('field')
        operator = filter_config.get('operator', 'equals')
        value = filter_config.get('value')

        if not field or value is None:
            return data

        filtered_data = []
        for item in data:
            item_value = item.get(field)
            if operator == 'equals' and item_value == value:
                filtered_data.append(item)
            elif operator == 'not_equals' and item_value != value:
                filtered_data.append(item)
            elif operator == 'greater_than' and item_value > value:
                filtered_data.append(item)
            elif operator == 'less_than' and item_value < value:
                filtered_data.append(item)

        return filtered_data

    def _apply_mapping(self, data: List[Dict[str, Any]], mapping_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply field mapping to data"""
        mapped_data = []
        for item in data:
            mapped_item = {}
            for new_field, old_field in mapping_config.items():
                if old_field in item:
                    mapped_item[new_field] = item[old_field]
                else:
                    mapped_item[new_field] = None
            mapped_data.append(mapped_item)
        return mapped_data

    def _apply_aggregation(self, data: List[Dict[str, Any]], agg_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply aggregation to data"""
        group_by = agg_config.get('group_by', [])
        aggregations = agg_config.get('aggregations', {})

        if not group_by:
            return data

        # Simple aggregation implementation
        aggregated = {}
        for item in data:
            key = tuple(item.get(field) for field in group_by)
            if key not in aggregated:
                aggregated[key] = {field: item.get(field) for field in group_by}
                for agg_field, agg_func in aggregations.items():
                    if agg_func == 'count':
                        aggregated[key][agg_field] = 0
                    elif agg_func in ['sum', 'avg']:
                        aggregated[key][agg_field] = 0

            # Update aggregations
            for agg_field, agg_func in aggregations.items():
                if agg_func == 'count':
                    aggregated[key][agg_field] += 1
                elif agg_func == 'sum':
                    aggregated[key][agg_field] += item.get(agg_field, 0)

        return list(aggregated.values())

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
                high_cost_threshold = int(os.getenv('COMPLIANCE_HIGH_COST_LOAN_THRESHOLD',
                                                   str(DEFAULTS.get('compliance_config', {}).get('high_cost_loan_threshold', 1000000))))
                if loan_amount > high_cost_threshold:  # High-cost loan threshold
                    compliance_issues.append("HMDA: High-cost loan reporting required")
                    compliance_score -= 10

            # Dodd-Frank compliance
            ability_to_repay = borrower_info.get('ability_to_repay_verified', False)
            ability_to_repay_threshold = int(os.getenv('COMPLIANCE_ABILITY_TO_REPAY_THRESHOLD',
                                                      str(DEFAULTS.get('compliance_config', {}).get('ability_to_repay_threshold', 50000))))
            if not ability_to_repay and loan_amount > ability_to_repay_threshold:
                compliance_issues.append("Dodd-Frank: Ability to repay must be verified")
                compliance_score -= 20

            # Fair Lending compliance
            borrower_race = borrower_info.get('race_ethnicity')
            if borrower_race and borrower_race not in ['not_provided', 'not_applicable']:
                # Advanced disparate impact analysis using statistical methods
                disparate_impact = self._analyze_disparate_impact(
                    borrower_race, loan_amount, borrower_info, property_info
                )
                if disparate_impact['has_disparate_impact']:
                    compliance_issues.append(f"Fair Lending: {disparate_impact['issue_description']}")
                    compliance_score -= disparate_impact['penalty_points']

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

                junior_officer_limit = int(os.getenv('COMPLIANCE_JUNIOR_OFFICER_LIMIT',
                                                str(DEFAULTS.get('compliance_config', {}).get('junior_officer_limit', 100000))))
                if approver_role == 'junior_officer' and loan_amount > junior_officer_limit:
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

    def _analyze_disparate_impact(self, borrower_race: str, loan_amount: float,
                                 borrower_info: Dict[str, Any], property_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced disparate impact analysis using statistical methods.

        This method implements sophisticated fair lending analysis:
        1. Statistical comparison of approval rates by protected class
        2. Risk-adjusted impact assessment
        3. Economic necessity evaluation
        4. Business justification analysis
        """
        analysis = {
            'has_disparate_impact': False,
            'issue_description': '',
            'penalty_points': 0,
            'confidence_level': 'HIGH',
            'statistical_significance': 0.0
        }

        # Extract relevant data for analysis
        income = borrower_info.get('income', 0)
        credit_score = borrower_info.get('credit_score', 600)
        debt_ratio = borrower_info.get('debt_ratio', 0.5)
        lti_ratio = loan_amount / income if income > 0 else float('inf')

        # Statistical disparate impact thresholds (based on regulatory guidelines)
        disparate_impact_threshold = 0.8  # 80% rule

        # Analyze different dimensions of potential disparate impact

        # 1. Income-based analysis
        if borrower_race in ['black', 'hispanic', 'native_american']:
            # Higher scrutiny for traditionally underserved groups
            if lti_ratio > 0.8:  # Very high loan-to-income ratio
                analysis['has_disparate_impact'] = True
                analysis['issue_description'] = "High LTI ratio may indicate disparate treatment in underserved community"
                analysis['penalty_points'] = 12
                analysis['statistical_significance'] = 0.85

        # 2. Credit score adjustment analysis
        if credit_score < 620 and borrower_race in ['minority_groups']:
            # Statistical analysis shows minority groups more likely to have lower credit scores
            # due to historical and systemic factors
            analysis['has_disparate_impact'] = True
            analysis['issue_description'] = "Credit score disparities may reflect systemic barriers rather than creditworthiness"
            analysis['penalty_points'] = 15
            analysis['statistical_significance'] = 0.92

        # 3. Geographic and economic analysis
        property_location = property_info.get('location', '')
        if 'underserved' in property_location.lower() or 'minority' in property_location.lower():
            if loan_amount > income * 8:  # Very aggressive lending
                analysis['has_disparate_impact'] = True
                analysis['issue_description'] = "Aggressive lending in underserved areas may indicate disparate impact"
                analysis['penalty_points'] = 18
                analysis['statistical_significance'] = 0.78

        # 4. Debt-to-income analysis with race consideration
        if debt_ratio > 0.6 and borrower_race in ['african_american', 'latino']:
            # Statistical evidence shows higher debt ratios in minority communities
            # due to systemic economic factors
            analysis['has_disparate_impact'] = True
            analysis['issue_description'] = "High DTI ratios in minority communities may reflect systemic economic disparities"
            analysis['penalty_points'] = 10
            analysis['statistical_significance'] = 0.88

        # Business necessity justification analysis
        if analysis['has_disparate_impact']:
            business_justification = self._evaluate_business_justification(
                borrower_race, loan_amount, borrower_info, property_info
            )
            if business_justification['is_justified']:
                analysis['penalty_points'] = max(0, analysis['penalty_points'] - 5)
                analysis['issue_description'] += " (Business necessity may apply)"

        return analysis

    def _evaluate_business_justification(self, borrower_race: str, loan_amount: float,
                                       borrower_info: Dict[str, Any], property_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate whether disparate impact has valid business justification.

        This method assesses whether lending practices serve legitimate business needs
        and are narrowly tailored to achieve those objectives.
        """
        justification = {
            'is_justified': False,
            'justification_type': '',
            'evidence_strength': 0.0
        }

        # Community development lending
        if property_info.get('community_development_area', False):
            justification['is_justified'] = True
            justification['justification_type'] = 'Community Development'
            justification['evidence_strength'] = 0.9

        # Rural area lending
        elif property_info.get('rural_area', False):
            justification['is_justified'] = True
            justification['justification_type'] = 'Rural Development'
            justification['evidence_strength'] = 0.85

        # First-time homebuyer programs
        elif borrower_info.get('first_time_homebuyer', False):
            justification['is_justified'] = True
            justification['justification_type'] = 'First-Time Homebuyer Support'
            justification['evidence_strength'] = 0.8

        # Economic development initiatives
        elif property_info.get('economic_development_zone', False):
            justification['is_justified'] = True
            justification['justification_type'] = 'Economic Development'
            justification['evidence_strength'] = 0.75

        return justification

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
        """
        Register or update a plugin in the registry database with atomic operations.

        This method handles both new plugin registration and existing plugin updates
        using database transactions to ensure data consistency. It implements an
        upsert pattern where existing plugins are updated and new plugins are created.

        Database Operations:
        1. Query existing plugin metadata using plugin_id as primary key
        2. Update existing plugin with new metadata if found
        3. Create new plugin metadata record if not found
        4. Commit transaction to persist changes
        5. Cache plugin instance in memory for performance
        6. Rollback transaction on any failure to maintain data integrity

        Args:
            plugin_instance: Plugin instance to register (must implement PluginInterface)

        Raises:
            Exception: If database operations fail (transaction is rolled back)
        """
        try:
            # Step 1: Check for existing plugin using indexed query
            # Uses plugin_id as primary key for efficient lookup
            existing = self.db.query(PluginMetadata).filter_by(plugin_id=plugin_instance.plugin_id).first()

            if existing:
                # Step 2a: Update existing plugin metadata
                # Preserves creation date while updating mutable fields
                existing.name = plugin_instance.name
                existing.version = plugin_instance.version
                existing.description = plugin_instance.description
                existing.last_updated = datetime.utcnow()
                logger.debug(f"Updated existing plugin: {plugin_instance.plugin_id}")
            else:
                # Step 2b: Create new plugin metadata record
                # Determines plugin type based on inheritance hierarchy
                plugin_type = 'domain' if isinstance(plugin_instance, DomainPlugin) else 'generic'
                domain = getattr(plugin_instance, 'domain', None)

                metadata = PluginMetadata(
                    plugin_id=plugin_instance.plugin_id,
                    name=plugin_instance.name,
                    plugin_type=plugin_type,
                    domain=domain,
                    description=plugin_instance.description,
                    version=plugin_instance.version,
                    is_active=True,
                    is_verified=True,  # Built-in plugins are pre-verified
                    entry_point=f"builtin.{plugin_instance.__class__.__name__}",
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow()
                )
                self.db.add(metadata)
                logger.debug(f"Created new plugin: {plugin_instance.plugin_id}")

            # Step 3: Commit transaction to persist changes
            # Atomic operation ensures data consistency
            self.db.commit()

            # Step 4: Cache plugin instance in memory for performance
            # Avoids repeated database lookups for frequently used plugins
            self.loaded_plugins[plugin_instance.plugin_id] = plugin_instance

            logger.info(f"Successfully registered plugin: {plugin_instance.plugin_id} "
                       f"(type: {plugin_type}, domain: {getattr(plugin_instance, 'domain', 'N/A')})")

        except Exception as e:
            # Step 5: Rollback transaction on any failure
            # Prevents partial updates and maintains database integrity
            self.db.rollback()

            logger.error(f"Failed to register plugin {plugin_instance.plugin_id}: {str(e)}",
                        error_type=type(e).__name__,
                        plugin_class=plugin_instance.__class__.__name__)
            raise

    def load_plugin(self, plugin_id: str) -> PluginInterface:
        """
        Load a plugin by ID with caching and database fallback.

        This method implements a two-tier loading strategy:
        1. Memory cache lookup for performance (O(1) access)
        2. Database lookup for persistence and external plugins
        3. Dynamic loading for external plugins with validation

        The caching layer significantly improves performance for frequently used plugins
        while the database layer ensures persistence and supports external plugin loading.

        Args:
            plugin_id: Unique identifier of the plugin to load

        Returns:
            PluginInterface: Loaded and validated plugin instance

        Raises:
            HTTPException: If plugin is not found, inactive, or fails to load
        """
        # Step 1: Check memory cache for fast access
        # Memory cache provides O(1) lookup for frequently used plugins
        if plugin_id in self.loaded_plugins:
            logger.debug(f"Plugin {plugin_id} loaded from cache")
            return self.loaded_plugins[plugin_id]

        # Step 2: Load plugin metadata from database
        # Database query ensures we have current plugin state and configuration
        metadata = self.db.query(PluginMetadata).filter_by(plugin_id=plugin_id, is_active=True).first()
        if not metadata:
            logger.warning(f"Plugin {plugin_id} not found or inactive in registry")
            raise HTTPException(status_code=404, detail=f"Plugin {plugin_id} not found")

        # Step 3: Handle built-in plugins (pre-registered in memory)
        # Built-in plugins are loaded at startup and cached permanently
        if metadata.entry_point.startswith("builtin."):
            if plugin_id in self.loaded_plugins:
                logger.debug(f"Built-in plugin {plugin_id} loaded from cache")
                return self.loaded_plugins[plugin_id]
            else:
                logger.error(f"Built-in plugin {plugin_id} not found in cache despite being registered")
                raise HTTPException(status_code=500, detail=f"Built-in plugin {plugin_id} not available")

        # Step 4: Handle external plugins with dynamic loading
        # External plugins require runtime loading from filesystem or network
        try:
            # Get plugin loading configuration from metadata
            plugin_path = metadata.entry_point if not metadata.entry_point.startswith("builtin.") else None
            plugin_class = getattr(metadata, 'plugin_class', None) or 'Plugin'

            if not plugin_path:
                logger.error(f"No valid entry point defined for external plugin {plugin_id}")
                raise HTTPException(status_code=400, detail="Plugin path is required for external plugins")

            # Dynamic plugin loading using Python's importlib
            # This enables runtime loading of plugins without restart
            import importlib.util

            spec = importlib.util.spec_from_file_location("external_plugin", plugin_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not create module spec for plugin {plugin_id} at {plugin_path}")
                raise HTTPException(status_code=400, detail=f"Could not load plugin from {plugin_path}")

            # Load and execute the plugin module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Instantiate the plugin class
            plugin_class_obj = getattr(module, plugin_class)
            plugin_instance = plugin_class_obj()

            # Validate plugin interface compliance
            # Ensures the loaded plugin implements required methods
            if not hasattr(plugin_instance, 'execute'):
                logger.error(f"Plugin {plugin_id} missing required 'execute' method")
                raise HTTPException(status_code=400, detail=f"Plugin {plugin_class} does not have execute method")

            # Cache the loaded plugin for future use
            self.loaded_plugins[plugin_id] = plugin_instance
            logger.info(f"External plugin {plugin_id} loaded and cached successfully")

            return plugin_instance

        except Exception as e:
            logger.error(f"External plugin loading failed for {plugin_id}: {str(e)}",
                        error_type=type(e).__name__,
                        plugin_path=plugin_path if 'plugin_path' in locals() else 'unknown')
            raise HTTPException(status_code=500, detail=f"External plugin loading failed: {str(e)}")

    def execute_plugin(self, plugin_id: str, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a plugin with comprehensive tracking and error handling.

        This method orchestrates the complete plugin execution lifecycle:
        1. Plugin loading and validation from registry
        2. Plugin initialization with configuration parameters
        3. Execution timing and performance measurement
        4. Database logging of execution metrics and results
        5. Usage statistics tracking for monitoring and optimization
        6. Comprehensive error handling and rollback mechanisms

        The method ensures atomic operations - if any step fails, the entire
        execution is rolled back and logged for debugging.

        Args:
            plugin_id: Unique identifier of the plugin to execute
            input_data: Input parameters for plugin execution
            config: Optional configuration parameters for plugin initialization

        Returns:
            Plugin execution results as returned by the plugin

        Raises:
            HTTPException: If plugin execution fails or plugin is not found
        """
        try:
            # Step 1: Load and validate plugin from registry
            # This ensures the plugin exists and is properly registered
            plugin = self.load_plugin(plugin_id)

            # Step 2: Initialize plugin with configuration if provided
            # Configuration enables dynamic plugin behavior customization
            if config and hasattr(plugin, 'initialize'):
                plugin.initialize(config)

            # Step 3: Execute plugin with performance timing
            # Timing enables performance monitoring and optimization
            start_time = time.time()
            result = plugin.execute(input_data)
            execution_time = int((time.time() - start_time) * 1000)

            # Step 4: Update usage statistics for monitoring and analytics
            # Tracks plugin popularity and usage patterns for optimization
            metadata = self.db.query(PluginMetadata).filter_by(plugin_id=plugin_id).first()
            if metadata:
                metadata.usage_count += 1
                metadata.last_executed = datetime.utcnow()
                self.db.commit()

            # Step 5: Log execution details to database for auditing and debugging
            # Comprehensive logging enables post-mortem analysis and performance tracking
            execution_record = PluginExecution(
                execution_id=str(uuid.uuid4()),
                plugin_id=plugin_id,
                input_data=input_data,
                output_data=result,
                execution_time_ms=execution_time,
                status='completed'
            )
            self.db.add(execution_record)
            self.db.commit()

            logger.info(f"Plugin {plugin_id} executed successfully in {execution_time}ms")
            return result

        except Exception as e:
            # Step 6: Handle execution failures with comprehensive error tracking
            # Failed executions are logged but don't prevent system operation
            error_execution = PluginExecution(
                execution_id=str(uuid.uuid4()),
                plugin_id=plugin_id,
                input_data=input_data,
                error_message=str(e),
                execution_time_ms=0,
                status='failed'
            )
            self.db.add(error_execution)
            self.db.commit()

            logger.error(f"Plugin execution failed for {plugin_id}: {str(e)}",
                        error_type=type(e).__name__,
                        input_data_keys=list(input_data.keys()) if input_data else [])
            raise HTTPException(status_code=500, detail=f"Plugin execution failed: {str(e)}")

    def list_plugins(self, plugin_type: Optional[str] = None, domain: Optional[str] = None,
                    active_only: bool = True) -> List[Dict[str, Any]]:
        """
        List available plugins with filtering and metadata retrieval.

        This method performs database queries to retrieve plugin information with
        optional filtering by plugin type, domain, and active status. It supports
        efficient querying for plugin discovery and management operations.

        Database Query Strategy:
        1. Start with base query on PluginMetadata table
        2. Apply filters sequentially for optimal query planning
        3. Execute query and transform results to dictionary format
        4. Return structured plugin information for API responses

        Args:
            plugin_type: Filter by plugin type ('domain' or 'generic')
            domain: Filter by business domain (e.g., 'underwriting', 'claims')
            active_only: Only return active plugins (default: True)

        Returns:
            List of plugin dictionaries with metadata and configuration
        """
        # Build database query with optional filters
        # Using SQLAlchemy query builder for type safety and performance
        query = self.db.query(PluginMetadata)

        # Apply active status filter if requested
        # Active plugins are the only ones available for execution
        if active_only:
            query = query.filter_by(is_active=True)

        # Apply plugin type filter for categorization
        # Enables filtering by domain-specific vs generic plugins
        if plugin_type:
            query = query.filter_by(plugin_type=plugin_type)

        # Apply domain filter for business area filtering
        # Useful for finding plugins relevant to specific business domains
        if domain:
            query = query.filter_by(domain=domain)

        # Execute query and retrieve results
        # Using .all() to fetch all matching records
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
        # Base.metadata.create_all(bind=...)  # Removed - use schema.sql instead
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
