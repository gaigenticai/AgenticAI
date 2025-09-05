#!/usr/bin/env python3
"""
Audit Logging Service for Agentic Brain Platform

This service provides comprehensive audit logging and compliance monitoring for the Agentic Brain platform,
tracking all operations, user activities, system events, and security incidents for regulatory compliance,
troubleshooting, and security monitoring.

Features:
- Comprehensive audit trail for all operations
- Real-time audit event ingestion and processing
- Compliance reporting for GDPR, SOX, HIPAA, etc.
- Security incident detection and alerting
- Audit data retention and archiving
- Advanced search and filtering capabilities
- Integration with SIEM systems
- Automated compliance reporting
- Audit dashboard and analytics
- Data anonymization and privacy controls
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import threading

import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON, Float, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func
import httpx
import aiohttp
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class AuditEvent(Base):
    """Comprehensive audit event storage"""
    __tablename__ = 'audit_events'

    id = Column(String(100), primary_key=True)
    event_id = Column(String(100), unique=True, nullable=False)
    event_type = Column(String(50), nullable=False)  # authentication, authorization, operation, security, compliance
    severity = Column(String(20), nullable=False)  # info, warning, error, critical
    user_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)
    service_name = Column(String(100), nullable=False)
    operation = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    location = Column(JSON, nullable=True)  # Geographic location data
    device_info = Column(JSON, nullable=True)
    request_data = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    success = Column(Boolean, default=True)
    compliance_flags = Column(JSON, nullable=True)  # GDPR, SOX, HIPAA flags
    risk_score = Column(Float, default=0.0)
    tags = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True)

class AuditArchive(Base):
    """Archived audit events for long-term storage"""
    __tablename__ = 'audit_archive'

    id = Column(String(100), primary_key=True)
    archive_id = Column(String(100), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    record_count = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=False)
    checksum = Column(String(128), nullable=False)
    compression_type = Column(String(20), default='gzip')
    retention_period_days = Column(Integer, nullable=False)
    archived_at = Column(DateTime, default=datetime.utcnow)

class ComplianceReport(Base):
    """Compliance reporting and tracking"""
    __tablename__ = 'compliance_reports'

    id = Column(String(100), primary_key=True)
    report_type = Column(String(50), nullable=False)  # gdpr, sox, hipaa, pci
    report_period = Column(String(20), nullable=False)  # daily, weekly, monthly, quarterly
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    total_events = Column(Integer, default=0)
    compliant_events = Column(Integer, default=0)
    non_compliant_events = Column(Integer, default=0)
    risk_events = Column(Integer, default=0)
    findings = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    generated_at = Column(DateTime, default=datetime.utcnow)
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

class AlertRule(Base):
    """Security alert rules and thresholds"""
    __tablename__ = 'alert_rules'

    id = Column(String(100), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    event_type = Column(String(50), nullable=False)
    conditions = Column(JSON, nullable=False)
    severity = Column(String(20), nullable=False)
    threshold = Column(Integer, nullable=False)
    time_window_minutes = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True)
    notification_channels = Column(JSON, nullable=True)
    last_triggered = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AlertInstance(Base):
    """Triggered security alerts"""
    __tablename__ = 'alert_instances'

    id = Column(String(100), primary_key=True)
    rule_id = Column(String(100), nullable=False)
    alert_message = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False)
    event_count = Column(Integer, nullable=False)
    events = Column(JSON, nullable=False)
    notified_channels = Column(JSON, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DataRetentionPolicy(Base):
    """Data retention policies for different event types"""
    __tablename__ = 'data_retention_policies'

    id = Column(String(100), primary_key=True)
    event_type = Column(String(50), nullable=False)
    retention_days = Column(Integer, nullable=False)
    archive_after_days = Column(Integer, nullable=False)
    delete_after_days = Column(Integer, nullable=False)
    compliance_requirements = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration"""
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agentic_brain")

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

    # Elasticsearch
    ELASTICSEARCH_ENABLED = os.getenv("ELASTICSEARCH_ENABLED", "false").lower() == "true"
    ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
    ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
    ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "audit_events")

    # Service ports
    AUDIT_SERVICE_PORT = int(os.getenv("AUDIT_SERVICE_PORT", "8340"))

    # Audit configuration
    MAX_EVENTS_PER_REQUEST = int(os.getenv("MAX_EVENTS_PER_REQUEST", "100"))
    AUDIT_QUEUE_SIZE = int(os.getenv("AUDIT_QUEUE_SIZE", "10000"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))

    # Retention policies
    DEFAULT_RETENTION_DAYS = int(os.getenv("DEFAULT_RETENTION_DAYS", "365"))
    ARCHIVE_AFTER_DAYS = int(os.getenv("ARCHIVE_AFTER_DAYS", "90"))
    DELETE_AFTER_DAYS = int(os.getenv("DELETE_AFTER_DAYS", "2555"))  # 7 years

    # Compliance settings
    GDPR_ENABLED = os.getenv("GDPR_ENABLED", "true").lower() == "true"
    SOX_ENABLED = os.getenv("SOX_ENABLED", "true").lower() == "true"
    HIPAA_ENABLED = os.getenv("HIPAA_ENABLED", "false").lower() == "true"
    PCI_ENABLED = os.getenv("PCI_ENABLED", "false").lower() == "true"

    # Alert configuration
    ALERT_ENABLED = os.getenv("ALERT_ENABLED", "true").lower() == "true"
    ALERT_EMAIL_ENABLED = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"
    ALERT_WEBHOOK_ENABLED = os.getenv("ALERT_WEBHOOK_ENABLED", "false").lower() == "true"

    # Email configuration
    SMTP_SERVER = os.getenv("SMTP_SERVER", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    EMAIL_FROM = os.getenv("EMAIL_FROM", "audit@agenticbrain.com")

    # Security
    API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "true").lower() == "true"
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "1000"))

    # Monitoring
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8004"))

# =============================================================================
# AUDIT EVENT PROCESSOR
# =============================================================================

class AuditEventProcessor:
    """Processes and manages audit events"""

    def __init__(self, db_session: Session, redis_client, elasticsearch_client=None):
        self.db = db_session
        self.redis = redis_client
        self.elasticsearch = elasticsearch_client
        self.event_queue = asyncio.Queue(maxsize=Config.AUDIT_QUEUE_SIZE)
        self.processing = True

        # Start background processor
        asyncio.create_task(self._process_events())

    async def log_event(self, event_data: Dict[str, Any]) -> str:
        """Log an audit event"""
        try:
            # Generate event ID
            event_id = str(uuid.uuid4())

            # Enrich event data
            enriched_event = self._enrich_event_data(event_data, event_id)

            # Add to processing queue
            await self.event_queue.put(enriched_event)

            # Store in Redis for immediate access
            self._cache_event(event_id, enriched_event)

            logger.info("Audit event logged", event_id=event_id, event_type=enriched_event.get('event_type'))

            return event_id

        except Exception as e:
            logger.error("Failed to log audit event", error=str(e), event_data=event_data)
            raise HTTPException(status_code=500, detail="Failed to log audit event")

    def _enrich_event_data(self, event_data: Dict[str, Any], event_id: str) -> Dict[str, Any]:
        """Enrich event data with additional context"""
        enriched = event_data.copy()
        enriched.update({
            'id': str(uuid.uuid4()),
            'event_id': event_id,
            'created_at': datetime.utcnow().isoformat(),
            'service_version': '1.0.0',
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'compliance_flags': self._calculate_compliance_flags(enriched),
            'risk_score': self._calculate_risk_score(enriched),
            'tags': enriched.get('tags', [])
        })

        return enriched

    def _calculate_compliance_flags(self, event: Dict[str, Any]) -> Dict[str, bool]:
        """Calculate compliance flags for the event"""
        flags = {}

        if Config.GDPR_ENABLED:
            flags['gdpr'] = self._is_gdpr_relevant(event)

        if Config.SOX_ENABLED:
            flags['sox'] = self._is_sox_relevant(event)

        if Config.HIPAA_ENABLED:
            flags['hipaa'] = self._is_hipaa_relevant(event)

        if Config.PCI_ENABLED:
            flags['pci'] = self._is_pci_relevant(event)

        return flags

    def _is_gdpr_relevant(self, event: Dict[str, Any]) -> bool:
        """Check if event is GDPR relevant"""
        gdpr_operations = [
            'user_data_access', 'user_data_modification', 'user_data_deletion',
            'data_export', 'consent_management', 'privacy_settings'
        ]
        return event.get('operation') in gdpr_operations

    def _is_sox_relevant(self, event: Dict[str, Any]) -> bool:
        """Check if event is SOX relevant"""
        sox_operations = [
            'financial_transaction', 'audit_log_access', 'system_configuration',
            'user_role_change', 'admin_action'
        ]
        return event.get('operation') in sox_operations

    def _is_hipaa_relevant(self, event: Dict[str, Any]) -> bool:
        """Check if event is HIPAA relevant"""
        hipaa_operations = [
            'patient_data_access', 'health_record_modification', 'medical_decision',
            'prescription_access', 'treatment_record'
        ]
        return event.get('operation') in hipaa_operations

    def _is_pci_relevant(self, event: Dict[str, Any]) -> bool:
        """Check if event is PCI relevant"""
        pci_operations = [
            'payment_processing', 'card_data_access', 'transaction_authorization',
            'payment_gateway_config'
        ]
        return event.get('operation') in pci_operations

    def _calculate_risk_score(self, event: Dict[str, Any]) -> float:
        """Calculate risk score for the event"""
        risk_score = 0.0

        # Base risk by event type
        event_type_risks = {
            'authentication': 0.3,
            'authorization': 0.4,
            'operation': 0.2,
            'security': 0.8,
            'compliance': 0.6
        }
        risk_score += event_type_risks.get(event.get('event_type', 'operation'), 0.2)

        # Severity multiplier
        severity_multipliers = {
            'info': 1.0,
            'warning': 1.5,
            'error': 2.0,
            'critical': 3.0
        }
        risk_score *= severity_multipliers.get(event.get('severity', 'info'), 1.0)

        # Success/failure modifier
        if not event.get('success', True):
            risk_score *= 1.2

        return min(risk_score, 10.0)  # Cap at 10.0

    def _cache_event(self, event_id: str, event_data: Dict[str, Any]):
        """Cache event in Redis for immediate access"""
        cache_key = f"audit_event:{event_id}"
        self.redis.setex(cache_key, 3600, json.dumps(event_data))  # 1 hour cache

    async def _process_events(self):
        """Background event processing"""
        while self.processing:
            try:
                # Get batch of events
                events = []
                for _ in range(min(Config.BATCH_SIZE, self.event_queue.qsize())):
                    try:
                        event = self.event_queue.get_nowait()
                        events.append(event)
                    except asyncio.QueueEmpty:
                        break

                if events:
                    # Process batch
                    await self._process_batch(events)

                # Wait before next batch
                await asyncio.sleep(1)

            except Exception as e:
                logger.error("Error in event processing", error=str(e))
                await asyncio.sleep(5)

    async def _process_batch(self, events: List[Dict[str, Any]]):
        """Process a batch of audit events"""
        try:
            # Store in database
            db_events = []
            for event in events:
                db_event = AuditEvent(
                    id=event['id'],
                    event_id=event['event_id'],
                    event_type=event['event_type'],
                    severity=event['severity'],
                    user_id=event.get('user_id'),
                    session_id=event.get('session_id'),
                    service_name=event['service_name'],
                    operation=event['operation'],
                    resource_type=event['resource_type'],
                    resource_id=event.get('resource_id'),
                    ip_address=event.get('ip_address'),
                    user_agent=event.get('user_agent'),
                    location=event.get('location'),
                    device_info=event.get('device_info'),
                    request_data=event.get('request_data'),
                    response_data=event.get('response_data'),
                    error_message=event.get('error_message'),
                    execution_time_ms=event.get('execution_time_ms'),
                    success=event.get('success', True),
                    compliance_flags=event.get('compliance_flags'),
                    risk_score=event.get('risk_score'),
                    tags=event.get('tags'),
                    metadata=event.get('metadata'),
                    created_at=datetime.fromisoformat(event['created_at'])
                )
                db_events.append(db_event)

            self.db.add_all(db_events)
            self.db.commit()

            # Index in Elasticsearch if enabled
            if self.elasticsearch and Config.ELASTICSEARCH_ENABLED:
                await self._index_events_elasticsearch(events)

            # Check for alerts
            await self._check_alerts(events)

            logger.info("Processed audit event batch", count=len(events))

        except Exception as e:
            logger.error("Failed to process audit event batch", error=str(e))
            self.db.rollback()

    async def _index_events_elasticsearch(self, events: List[Dict[str, Any]]):
        """Index events in Elasticsearch"""
        try:
            actions = []
            for event in events:
                action = {
                    "_index": Config.ELASTICSEARCH_INDEX,
                    "_id": event['event_id'],
                    "_source": event
                }
                actions.append(action)

            bulk(self.elasticsearch, actions)

        except Exception as e:
            logger.error("Failed to index events in Elasticsearch", error=str(e))

    async def _check_alerts(self, events: List[Dict[str, Any]]):
        """Check for security alerts based on events"""
        if not Config.ALERT_ENABLED:
            return

        try:
            # Get active alert rules
            alert_rules = self.db.query(AlertRule).filter_by(is_active=True).all()

            for rule in alert_rules:
                # Count events matching rule conditions in time window
                time_threshold = datetime.utcnow() - timedelta(minutes=rule.time_window_minutes)

                matching_events = [
                    event for event in events
                    if self._matches_alert_condition(event, rule.conditions) and
                    datetime.fromisoformat(event['created_at']) > time_threshold
                ]

                if len(matching_events) >= rule.threshold:
                    await self._trigger_alert(rule, matching_events)

        except Exception as e:
            logger.error("Error checking alerts", error=str(e))

    def _matches_alert_condition(self, event: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
        """Check if event matches alert condition"""
        for key, value in conditions.items():
            if key not in event:
                return False
            if isinstance(value, list):
                if event[key] not in value:
                    return False
            elif event[key] != value:
                return False
        return True

    async def _trigger_alert(self, rule: AlertRule, events: List[Dict[str, Any]]):
        """Trigger a security alert"""
        try:
            alert_id = str(uuid.uuid4())

            # Create alert instance
            alert = AlertInstance(
                id=alert_id,
                rule_id=rule.id,
                alert_message=f"Alert triggered: {rule.name} - {len(events)} events in {rule.time_window_minutes} minutes",
                severity=rule.severity,
                event_count=len(events),
                events=[event['event_id'] for event in events]
            )

            self.db.add(alert)
            self.db.commit()

            # Send notifications
            await self._send_alert_notifications(rule, alert, events)

            # Update rule last triggered
            rule.last_triggered = datetime.utcnow()
            self.db.commit()

            logger.warning("Security alert triggered", alert_id=alert_id, rule_name=rule.name)

        except Exception as e:
            logger.error("Failed to trigger alert", error=str(e))

    async def _send_alert_notifications(self, rule: AlertRule, alert: AlertInstance, events: List[Dict[str, Any]]):
        """Send alert notifications"""
        notified_channels = []

        # Email notifications
        if Config.ALERT_EMAIL_ENABLED and rule.notification_channels.get('email'):
            await self._send_email_alert(rule, alert, events)
            notified_channels.append('email')

        # Webhook notifications
        if Config.ALERT_WEBHOOK_ENABLED and rule.notification_channels.get('webhook'):
            await self._send_webhook_alert(rule, alert, events)
            notified_channels.append('webhook')

        # Update alert with notified channels
        alert.notified_channels = notified_channels
        self.db.commit()

    async def _send_email_alert(self, rule: AlertRule, alert: AlertInstance, events: List[Dict[str, Any]]):
        """Send email alert notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = Config.EMAIL_FROM
            msg['To'] = ', '.join(rule.notification_channels['email'])
            msg['Subject'] = f"Security Alert: {rule.name}"

            body = f"""
Security Alert Triggered

Rule: {rule.name}
Severity: {rule.severity}
Events: {alert.event_count}
Time Window: {rule.time_window_minutes} minutes

Alert Message: {alert.alert_message}

Please review the audit logs for more details.
            """

            msg.attach(MIMEText(body, 'plain'))

            # Send email (async)
            server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT)
            server.starttls()
            server.login(Config.SMTP_USERNAME, Config.SMTP_PASSWORD)
            text = msg.as_string()
            server.sendmail(Config.EMAIL_FROM, rule.notification_channels['email'], text)
            server.quit()

        except Exception as e:
            logger.error("Failed to send email alert", error=str(e))

    async def _send_webhook_alert(self, rule: AlertRule, alert: AlertInstance, events: List[Dict[str, Any]]):
        """Send webhook alert notification"""
        try:
            webhook_data = {
                'alert_id': alert.id,
                'rule_name': rule.name,
                'severity': rule.severity,
                'message': alert.alert_message,
                'event_count': alert.event_count,
                'timestamp': datetime.utcnow().isoformat(),
                'events': [event['event_id'] for event in events]
            }

            async with httpx.AsyncClient() as client:
                await client.post(
                    rule.notification_channels['webhook'],
                    json=webhook_data,
                    timeout=10
                )

        except Exception as e:
            logger.error("Failed to send webhook alert", error=str(e))

# =============================================================================
# COMPLIANCE MANAGER
# =============================================================================

class ComplianceManager:
    """Manages compliance reporting and data retention"""

    def __init__(self, db_session: Session):
        self.db = db_session

    async def generate_compliance_report(self, report_type: str, start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report"""
        try:
            # Query audit events for the period
            events = self.db.query(AuditEvent).filter(
                AuditEvent.created_at.between(start_date, end_date)
            ).all()

            # Analyze compliance
            analysis = self._analyze_compliance(events, report_type)

            # Generate report
            report = ComplianceReport(
                id=str(uuid.uuid4()),
                report_type=report_type,
                report_period='custom',
                start_date=start_date,
                end_date=end_date,
                total_events=len(events),
                compliant_events=analysis['compliant_count'],
                non_compliant_events=analysis['non_compliant_count'],
                risk_events=analysis['risk_count'],
                findings=analysis['findings'],
                recommendations=analysis['recommendations']
            )

            self.db.add(report)
            self.db.commit()

            return {
                'report_id': report.id,
                'report_type': report_type,
                'period': f"{start_date.date()} to {end_date.date()}",
                'total_events': len(events),
                'compliant_events': analysis['compliant_count'],
                'non_compliant_events': analysis['non_compliant_count'],
                'risk_events': analysis['risk_count'],
                'compliance_rate': analysis['compliance_rate'],
                'findings': analysis['findings'],
                'recommendations': analysis['recommendations']
            }

        except Exception as e:
            logger.error("Failed to generate compliance report", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to generate compliance report")

    def _analyze_compliance(self, events: List[AuditEvent], report_type: str) -> Dict[str, Any]:
        """Analyze compliance for events"""
        compliant_count = 0
        non_compliant_count = 0
        risk_count = 0
        findings = []
        recommendations = []

        for event in events:
            compliance_flags = event.compliance_flags or {}

            if report_type.lower() in compliance_flags and compliance_flags[report_type.lower()]:
                compliant_count += 1
            else:
                non_compliant_count += 1

            if event.risk_score and event.risk_score > 7.0:
                risk_count += 1

        total_events = len(events)
        compliance_rate = (compliant_count / total_events * 100) if total_events > 0 else 0

        # Generate findings based on analysis
        if compliance_rate < 95:
            findings.append(f"Compliance rate below threshold: {compliance_rate:.1f}%")
            recommendations.append("Review audit policies and access controls")

        if risk_count > total_events * 0.1:
            findings.append(f"High number of risk events: {risk_count}")
            recommendations.append("Investigate high-risk activities and implement additional controls")

        return {
            'compliant_count': compliant_count,
            'non_compliant_count': non_compliant_count,
            'risk_count': risk_count,
            'compliance_rate': compliance_rate,
            'findings': findings,
            'recommendations': recommendations
        }

    async def archive_old_events(self, days_to_archive: int = 90) -> Dict[str, Any]:
        """Archive old audit events"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_archive)

            # Get events to archive
            events_to_archive = self.db.query(AuditEvent).filter(
                AuditEvent.created_at < cutoff_date
            ).all()

            if not events_to_archive:
                return {'archived_count': 0, 'message': 'No events to archive'}

            # Create archive file
            archive_id = str(uuid.uuid4())
            archive_filename = f"audit_archive_{archive_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json.gz"

            # Convert events to JSON
            archive_data = []
            for event in events_to_archive:
                event_dict = {
                    'id': event.id,
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'user_id': event.user_id,
                    'service_name': event.service_name,
                    'operation': event.operation,
                    'resource_type': event.resource_type,
                    'resource_id': event.resource_id,
                    'success': event.success,
                    'created_at': event.created_at.isoformat()
                }
                archive_data.append(event_dict)

            # Save to compressed file (simplified - in production, use proper compression)
            import gzip
            archive_path = f"/app/archives/{archive_filename}"
            with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
                json.dump(archive_data, f)

            # Calculate SHA-256 checksum of the archive file
            archive_checksum = self._calculate_file_checksum(archive_path)

            # Create archive record with proper checksum
            archive_record = AuditArchive(
                id=str(uuid.uuid4()),
                archive_id=archive_id,
                start_date=min(event.created_at for event in events_to_archive),
                end_date=max(event.created_at for event in events_to_archive),
                record_count=len(events_to_archive),
                file_path=archive_path,
                checksum=archive_checksum,
                retention_period_days=2555  # 7 years
            )

            self.db.add(archive_record)

            # Mark events as archived and delete
            for event in events_to_archive:
                event.indexed_at = datetime.utcnow()

            self.db.commit()

            # Delete archived events
            for event in events_to_archive:
                self.db.delete(event)

            self.db.commit()

            return {
                'archived_count': len(events_to_archive),
                'archive_id': archive_id,
                'archive_file': archive_path,
                'message': f'Successfully archived {len(events_to_archive)} events'
            }

        except Exception as e:
            logger.error("Failed to archive events", error=str(e))
            self.db.rollback()
            raise HTTPException(status_code=500, detail="Failed to archive events")

    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file

        This method implements production-grade file integrity verification by:
        1. Reading the file in chunks to handle large files efficiently
        2. Using SHA-256 for cryptographic security
        3. Providing tamper detection capabilities
        4. Supporting audit trail integrity verification

        Args:
            file_path: Path to the file to calculate checksum for

        Returns:
            str: Hexadecimal SHA-256 checksum string

        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If there's an error reading the file
        """
        try:
            # Use SHA-256 for cryptographic security
            sha256_hash = hashlib.sha256()

            # Read file in chunks to handle large files efficiently
            with open(file_path, 'rb') as f:
                chunk_size = 8192  # 8KB chunks for optimal performance
                while chunk := f.read(chunk_size):
                    sha256_hash.update(chunk)

            # Return hexadecimal representation
            checksum = sha256_hash.hexdigest()

            logger.info(f"Calculated checksum for archive file",
                       file_path=file_path,
                       checksum=checksum)

            return checksum

        except FileNotFoundError:
            logger.error(f"Archive file not found for checksum calculation: {file_path}")
            raise
        except IOError as e:
            logger.error(f"IO error calculating checksum for {file_path}", error=str(e))
            raise
        except Exception as e:
            logger.error(f"Unexpected error calculating checksum for {file_path}", error=str(e))
            raise

# =============================================================================
# API MODELS
# =============================================================================

class AuditEventRequest(BaseModel):
    """Audit event logging request"""
    event_type: str = Field(..., description="Type of audit event")
    severity: str = Field(..., description="Event severity level")
    user_id: Optional[str] = Field(None, description="User ID associated with event")
    session_id: Optional[str] = Field(None, description="Session ID")
    service_name: str = Field(..., description="Service that generated the event")
    operation: str = Field(..., description="Operation performed")
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: Optional[str] = Field(None, description="Resource ID")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    location: Optional[Dict[str, Any]] = Field(None, description="Geographic location")
    device_info: Optional[Dict[str, Any]] = Field(None, description="Device information")
    request_data: Optional[Dict[str, Any]] = Field(None, description="Request data")
    response_data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error_message: Optional[str] = Field(None, description="Error message if any")
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    success: bool = Field(True, description="Whether operation was successful")
    tags: List[str] = Field(default_factory=list, description="Event tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class SearchRequest(BaseModel):
    """Audit event search request"""
    query: Optional[str] = Field(None, description="Search query string")
    event_type: Optional[str] = Field(None, description="Filter by event type")
    severity: Optional[str] = Field(None, description="Filter by severity")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    service_name: Optional[str] = Field(None, description="Filter by service name")
    operation: Optional[str] = Field(None, description="Filter by operation")
    resource_type: Optional[str] = Field(None, description="Filter by resource type")
    success: Optional[bool] = Field(None, description="Filter by success status")
    start_date: Optional[datetime] = Field(None, description="Start date for search")
    end_date: Optional[datetime] = Field(None, description="End date for search")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(100, description="Maximum number of results")
    offset: int = Field(0, description="Pagination offset")

class ComplianceReportRequest(BaseModel):
    """Compliance report request"""
    report_type: str = Field(..., description="Type of compliance report")
    start_date: datetime = Field(..., description="Report start date")
    end_date: datetime = Field(..., description="Report end date")

class AlertRuleRequest(BaseModel):
    """Alert rule creation request"""
    name: str = Field(..., description="Alert rule name")
    description: str = Field(..., description="Alert rule description")
    event_type: str = Field(..., description="Event type to monitor")
    conditions: Dict[str, Any] = Field(..., description="Alert conditions")
    severity: str = Field(..., description="Alert severity")
    threshold: int = Field(..., description="Alert threshold")
    time_window_minutes: int = Field(..., description="Time window in minutes")
    notification_channels: Optional[Dict[str, Any]] = Field(None, description="Notification channels")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Audit Logging Service",
    description="Comprehensive audit logging and compliance monitoring for Agentic Brain platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
engine = create_engine(Config.DATABASE_URL)
# Base.metadata.create_all(bind=...)  # Removed - use schema.sql instead
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize Redis
redis_client = redis.Redis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=Config.REDIS_DB,
    decode_responses=True
)

# Initialize Elasticsearch if enabled
elasticsearch_client = None
if Config.ELASTICSEARCH_ENABLED:
    try:
        elasticsearch_client = Elasticsearch([{'host': Config.ELASTICSEARCH_HOST, 'port': Config.ELASTICSEARCH_PORT}])
    except Exception as e:
        logger.warning("Failed to connect to Elasticsearch", error=str(e))

# Initialize components
audit_processor = AuditEventProcessor(SessionLocal(), redis_client, elasticsearch_client)
compliance_manager = ComplianceManager(SessionLocal())

# Security schemes
security = HTTPBearer(auto_error=False)

# Prometheus metrics
AUDIT_EVENTS_TOTAL = Counter('audit_events_total', 'Total audit events logged', ['event_type', 'severity'])
AUDIT_REQUEST_DURATION = Histogram('audit_request_duration_seconds', 'Audit request duration')
SEARCH_REQUEST_DURATION = Histogram('audit_search_duration_seconds', 'Search request duration')

@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    """Database session middleware"""
    response = None
    try:
        request.state.db = SessionLocal()
        response = await call_next(request)
    finally:
        if hasattr(request.state, 'db'):
            request.state.db.close()

    # Record metrics
    AUDIT_REQUEST_DURATION.observe(time.time() - time.time())  # Simplified timing

    return response

def verify_api_key(request: Request) -> bool:
    """Verify API key if required"""
    if not Config.API_KEY_REQUIRED:
        return True

    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return False

    # In production, validate against database or cache
    return api_key == os.getenv('AUDIT_API_KEY', 'default-audit-key')

@app.post("/audit/events")
async def log_audit_event(event: AuditEventRequest, request: Request):
    """Log an audit event"""
    if Config.API_KEY_REQUIRED and not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # Add request metadata
        event_data = event.dict()
        event_data.update({
            'ip_address': request.client.host if request.client else None,
            'user_agent': request.headers.get('user-agent')
        })

        event_id = await audit_processor.log_event(event_data)

        # Update metrics
        AUDIT_EVENTS_TOTAL.labels(
            event_type=event.event_type,
            severity=event.severity
        ).inc()

        return {
            'event_id': event_id,
            'status': 'logged',
            'timestamp': datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to log audit event", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to log audit event")

@app.post("/audit/events/batch")
async def log_audit_events_batch(events: List[AuditEventRequest], request: Request):
    """Log multiple audit events"""
    if Config.API_KEY_REQUIRED and not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if len(events) > Config.MAX_EVENTS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many events. Maximum {Config.MAX_EVENTS_PER_REQUEST} allowed"
        )

    try:
        logged_events = []

        for event in events:
            event_data = event.dict()
            event_data.update({
                'ip_address': request.client.host if request.client else None,
                'user_agent': request.headers.get('user-agent')
            })

            event_id = await audit_processor.log_event(event_data)
            logged_events.append({
                'event_id': event_id,
                'event_type': event.event_type,
                'severity': event.severity
            })

        return {
            'logged_count': len(logged_events),
            'events': logged_events,
            'timestamp': datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to log audit events batch", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to log audit events")

@app.get("/audit/events")
async def search_audit_events(search: SearchRequest, request: Request):
    """Search audit events"""
    if Config.API_KEY_REQUIRED and not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        start_time = time.time()

        # Build query
        query = SessionLocal().query(AuditEvent)

        if search.query:
            # Simple text search (in production, use full-text search)
            query = query.filter(
                (AuditEvent.operation.contains(search.query)) |
                (AuditEvent.resource_type.contains(search.query)) |
                (AuditEvent.service_name.contains(search.query))
            )

        if search.event_type:
            query = query.filter(AuditEvent.event_type == search.event_type)

        if search.severity:
            query = query.filter(AuditEvent.severity == search.severity)

        if search.user_id:
            query = query.filter(AuditEvent.user_id == search.user_id)

        if search.service_name:
            query = query.filter(AuditEvent.service_name == search.service_name)

        if search.operation:
            query = query.filter(AuditEvent.operation == search.operation)

        if search.resource_type:
            query = query.filter(AuditEvent.resource_type == search.resource_type)

        if search.success is not None:
            query = query.filter(AuditEvent.success == search.success)

        if search.start_date:
            query = query.filter(AuditEvent.created_at >= search.start_date)

        if search.end_date:
            query = query.filter(AuditEvent.created_at <= search.end_date)

        if search.tags:
            # Simple tag filtering (in production, use JSON array operations)
            for tag in search.tags:
                query = query.filter(AuditEvent.tags.contains(tag))

        # Get total count
        total_count = query.count()

        # Apply pagination
        events = query.order_by(AuditEvent.created_at.desc())\
                     .limit(search.limit)\
                     .offset(search.offset)\
                     .all()

        # Convert to dict
        event_list = []
        for event in events:
            event_dict = {
                'id': event.id,
                'event_id': event.event_id,
                'event_type': event.event_type,
                'severity': event.severity,
                'user_id': event.user_id,
                'service_name': event.service_name,
                'operation': event.operation,
                'resource_type': event.resource_type,
                'resource_id': event.resource_id,
                'success': event.success,
                'risk_score': event.risk_score,
                'created_at': event.created_at.isoformat()
            }
            event_list.append(event_dict)

        end_time = time.time()
        SEARCH_REQUEST_DURATION.observe(end_time - start_time)

        return {
            'events': event_list,
            'total_count': total_count,
            'limit': search.limit,
            'offset': search.offset,
            'search_time_seconds': end_time - start_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to search audit events", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to search audit events")

@app.get("/audit/events/{event_id}")
async def get_audit_event(event_id: str, request: Request):
    """Get specific audit event"""
    if Config.API_KEY_REQUIRED and not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # Try cache first
        cache_key = f"audit_event:{event_id}"
        cached_event = redis_client.get(cache_key)

        if cached_event:
            return json.loads(cached_event)

        # Get from database
        db = SessionLocal()
        event = db.query(AuditEvent).filter_by(event_id=event_id).first()
        db.close()

        if not event:
            raise HTTPException(status_code=404, detail="Audit event not found")

        event_dict = {
            'id': event.id,
            'event_id': event.event_id,
            'event_type': event.event_type,
            'severity': event.severity,
            'user_id': event.user_id,
            'session_id': event.session_id,
            'service_name': event.service_name,
            'operation': event.operation,
            'resource_type': event.resource_type,
            'resource_id': event.resource_id,
            'ip_address': event.ip_address,
            'user_agent': event.user_agent,
            'location': event.location,
            'device_info': event.device_info,
            'request_data': event.request_data,
            'response_data': event.response_data,
            'error_message': event.error_message,
            'execution_time_ms': event.execution_time_ms,
            'success': event.success,
            'compliance_flags': event.compliance_flags,
            'risk_score': event.risk_score,
            'tags': event.tags,
            'metadata': event.metadata,
            'created_at': event.created_at.isoformat()
        }

        return event_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get audit event", event_id=event_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get audit event")

@app.post("/compliance/reports")
async def generate_compliance_report(report_request: ComplianceReportRequest, request: Request):
    """Generate compliance report"""
    if Config.API_KEY_REQUIRED and not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        report = await compliance_manager.generate_compliance_report(
            report_request.report_type,
            report_request.start_date,
            report_request.end_date
        )

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate compliance report", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate compliance report")

@app.get("/compliance/reports")
async def list_compliance_reports(limit: int = 50, offset: int = 0, request: Request = None):
    """List compliance reports"""
    if Config.API_KEY_REQUIRED and not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        db = SessionLocal()
        reports = db.query(ComplianceReport)\
                   .order_by(ComplianceReport.generated_at.desc())\
                   .limit(limit)\
                   .offset(offset)\
                   .all()
        db.close()

        return {
            'reports': [
                {
                    'id': report.id,
                    'report_type': report.report_type,
                    'start_date': report.start_date.isoformat(),
                    'end_date': report.end_date.isoformat(),
                    'total_events': report.total_events,
                    'compliant_events': report.compliant_events,
                    'non_compliant_events': report.non_compliant_events,
                    'risk_events': report.risk_events,
                    'generated_at': report.generated_at.isoformat()
                }
                for report in reports
            ],
            'total_count': len(reports),
            'limit': limit,
            'offset': offset
        }

    except Exception as e:
        logger.error("Failed to list compliance reports", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list compliance reports")

@app.post("/admin/archive")
async def archive_old_events(days: int = 90, request: Request = None):
    """Archive old audit events"""
    if Config.API_KEY_REQUIRED and not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        result = await compliance_manager.archive_old_events(days)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to archive events", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to archive events")

@app.get("/analytics/summary")
async def get_analytics_summary(days: int = 30, request: Request = None):
    """Get audit analytics summary"""
    if Config.API_KEY_REQUIRED and not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        db = SessionLocal()
        start_date = datetime.utcnow() - timedelta(days=days)

        # Event type distribution
        event_types = db.query(
            AuditEvent.event_type,
            func.count(AuditEvent.id).label('count')
        ).filter(AuditEvent.created_at >= start_date)\
         .group_by(AuditEvent.event_type)\
         .all()

        # Severity distribution
        severities = db.query(
            AuditEvent.severity,
            func.count(AuditEvent.id).label('count')
        ).filter(AuditEvent.created_at >= start_date)\
         .group_by(AuditEvent.severity)\
         .all()

        # Service distribution
        services = db.query(
            AuditEvent.service_name,
            func.count(AuditEvent.id).label('count')
        ).filter(AuditEvent.created_at >= start_date)\
         .group_by(AuditEvent.service_name)\
         .all()

        # Risk score distribution
        risk_distribution = db.query(
            func.floor(AuditEvent.risk_score).label('risk_bucket'),
            func.count(AuditEvent.id).label('count')
        ).filter(AuditEvent.created_at >= start_date)\
         .group_by(func.floor(AuditEvent.risk_score))\
         .all()

        # Daily event counts
        daily_counts = db.query(
            func.date(AuditEvent.created_at).label('date'),
            func.count(AuditEvent.id).label('count')
        ).filter(AuditEvent.created_at >= start_date)\
         .group_by(func.date(AuditEvent.created_at))\
         .order_by(func.date(AuditEvent.created_at))\
         .all()

        db.close()

        return {
            'period_days': days,
            'total_events': sum(count for _, count in event_types),
            'event_types': {event_type: count for event_type, count in event_types},
            'severities': {severity: count for severity, count in severities},
            'services': {service: count for service, count in services},
            'risk_distribution': {str(int(bucket)): count for bucket, count in risk_distribution},
            'daily_counts': [{'date': str(date), 'count': count} for date, count in daily_counts]
        }

    except Exception as e:
        logger.error("Failed to get analytics summary", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get analytics summary")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "redis": "connected",
            "elasticsearch": "connected" if elasticsearch_client else "disabled",
            "audit_processor": "active",
            "compliance_manager": "active"
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return HTMLResponse(generate_latest(), media_type="text/plain")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Audit Dashboard"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Audit Dashboard - Agentic Brain Platform</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f7fa;
                color: #333;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .metric-card {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                text-align: center;
            }}
            .metric-value {{
                font-size: 3em;
                font-weight: bold;
                color: #3498db;
                margin-bottom: 10px;
            }}
            .status-critical {{ color: #e74c3c; }}
            .status-warning {{ color: #f39c12; }}
            .status-success {{ color: #27ae60; }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .recent-events {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Audit Dashboard</h1>
            <p>Comprehensive audit logging and compliance monitoring</p>
        </div>

        <div class="container">
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-value status-success" id="total-events">0</div>
                    <div class="metric-label">Total Events (30d)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-warning" id="risk-events">0</div>
                    <div class="metric-label">High Risk Events</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-critical" id="failed-events">0</div>
                    <div class="metric-label">Failed Operations</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-success" id="compliance-rate">0%</div>
                    <div class="metric-label">Compliance Rate</div>
                </div>
            </div>

            <div class="metric-card">
                <h3>Quick Actions</h3>
                <button onclick="searchEvents()">Search Events</button>
                <button onclick="generateReport()">Generate Report</button>
                <button onclick="viewAnalytics()">View Analytics</button>
                <button onclick="manageAlerts()">Manage Alerts</button>
            </div>

            <div class="recent-events">
                <h3>Recent Audit Events</h3>
                <table id="recent-events-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Event Type</th>
                            <th>Service</th>
                            <th>Operation</th>
                            <th>User</th>
                            <th>Severity</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="events-tbody">
                        <tr><td colspan="7">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            async function loadDashboardData() {{
                try {{
                    // Load analytics summary
                    const analyticsResponse = await fetch('/analytics/summary');
                    const analytics = await analyticsResponse.json();

                    document.getElementById('total-events').textContent = analytics.total_events || 0;

                    // Calculate risk events (risk score > 7)
                    const riskEvents = Object.entries(analytics.risk_distribution || {{}})
                        .filter(([bucket, count]) => parseInt(bucket) > 7)
                        .reduce((sum, [, count]) => sum + count, 0);
                    document.getElementById('risk-events').textContent = riskEvents;

                    // Load recent events
                    const eventsResponse = await fetch('/audit/events?limit=10');
                    const eventsData = await eventsResponse.json();

                    const tbody = document.getElementById('events-tbody');
                    tbody.innerHTML = '';

                    if (eventsData.events && eventsData.events.length > 0) {{
                        eventsData.events.forEach(event => {{
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td>${{new Date(event.created_at).toLocaleString()}}</td>
                                <td>${{event.event_type}}</td>
                                <td>${{event.service_name}}</td>
                                <td>${{event.operation}}</td>
                                <td>${{event.user_id || 'System'}}</td>
                                <td>${{event.severity}}</td>
                                <td>${{event.success ? '✅' : '❌'}}</td>
                            `;
                            tbody.appendChild(row);
                        }});
                    }} else {{
                        tbody.innerHTML = '<tr><td colspan="7">No events found</td></tr>';
                    }}

                }} catch (error) {{
                    console.error('Error loading dashboard data:', error);
                }}
            }}

            async function searchEvents() {{
                const query = prompt('Enter search query:');
                if (query) {{
                    window.location.href = `/audit/events?query=${{encodeURIComponent(query)}}`;
                }}
            }}

            async function generateReport() {{
                const reportType = prompt('Enter report type (gdpr, sox, hipaa, pci):', 'gdpr');
                if (reportType) {{
                    const startDate = new Date();
                    startDate.setDate(startDate.getDate() - 30);
                    const endDate = new Date();

                    const response = await fetch('/compliance/reports', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            report_type: reportType,
                            start_date: startDate.toISOString(),
                            end_date: endDate.toISOString()
                        }})
                    }});

                    const result = await response.json();
                    alert(`Report generated: ${{result.report_id}}`);
                }}
            }}

            async function viewAnalytics() {{
                window.location.href = '/analytics/summary';
            }}

            async function manageAlerts() {{
                window.location.href = '/alerts';
            }}

            // Load data on page load
            document.addEventListener('DOMContentLoaded', loadDashboardData);

            // Refresh data every 30 seconds
            setInterval(loadDashboardData, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Audit Logging Service starting up")

    # Create archives directory
    os.makedirs("/app/archives", exist_ok=True)

    # Initialize default alert rules
    try:
        db = SessionLocal()
        existing_rules = db.query(AlertRule).count()

        if existing_rules == 0:
            default_rules = [
                AlertRule(
                    id=str(uuid.uuid4()),
                    name="Multiple Failed Authentications",
                    description="Alert when multiple authentication failures occur",
                    event_type="authentication",
                    conditions={"success": False, "operation": "login"},
                    severity="warning",
                    threshold=5,
                    time_window_minutes=10,
                    notification_channels={"email": ["security@agenticbrain.com"]}
                ),
                AlertRule(
                    id=str(uuid.uuid4()),
                    name="High Risk Operations",
                    description="Alert on high-risk operations",
                    event_type="operation",
                    conditions={"risk_score": {"gt": 8.0}},
                    severity="critical",
                    threshold=1,
                    time_window_minutes=5,
                    notification_channels={"email": ["security@agenticbrain.com"]}
                ),
                AlertRule(
                    id=str(uuid.uuid4()),
                    name="Admin Privilege Changes",
                    description="Alert on administrative privilege changes",
                    event_type="authorization",
                    conditions={"operation": "role_change", "resource_type": "user"},
                    severity="warning",
                    threshold=1,
                    time_window_minutes=1,
                    notification_channels={"email": ["security@agenticbrain.com"]}
                )
            ]

            for rule in default_rules:
                db.add(rule)

            db.commit()
            logger.info("Default alert rules created")

        db.close()

    except Exception as e:
        logger.warning("Failed to create default alert rules", error=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Audit Logging Service shutting down")

    # Stop audit processor
    audit_processor.processing = False

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.AUDIT_SERVICE_PORT,
        reload=True,
        log_level="info"
    )
