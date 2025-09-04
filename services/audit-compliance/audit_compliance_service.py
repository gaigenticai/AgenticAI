#!/usr/bin/env python3
"""
Audit Compliance Service for Agentic Platform

This service provides comprehensive audit logging and compliance framework with:
- Real-time audit logging for all system activities
- Compliance reporting for GDPR, HIPAA, SOX, PCI-DSS
- Data retention policy enforcement
- Security event monitoring and alerting
- Audit trail integrity verification
- Automated compliance assessments
- Incident response and forensic analysis
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import structlog
import uvicorn

# Configure structured logging
logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.StdLibLoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# FastAPI app
app = FastAPI(
    title="Audit Compliance Service",
    description="Comprehensive audit logging and compliance framework",
    version="1.0.0"
)

# Prometheus metrics
AUDIT_EVENTS_LOGGED = Counter('audit_events_logged_total', 'Total audit events logged', ['event_category', 'severity'])
COMPLIANCE_REPORTS_GENERATED = Counter('compliance_reports_generated_total', 'Total compliance reports generated', ['compliance_standard'])
RETENTION_POLICY_EXECUTIONS = Counter('retention_policy_executions_total', 'Total retention policy executions', ['policy_type'])
SECURITY_ALERTS_TRIGGERED = Counter('security_alerts_triggered_total', 'Total security alerts triggered', ['alert_type', 'severity'])
AUDIT_INTEGRITY_CHECKS = Counter('audit_integrity_checks_total', 'Total audit integrity checks', ['check_type'])

# Global variables
database_connection = None

# Pydantic models
class AuditEvent(BaseModel):
    """Audit event model"""
    event_type: str = Field(..., description="Type of audit event")
    event_category: str = Field(..., description="Category: SECURITY, DATA_ACCESS, USER_ACTION, SYSTEM, COMPLIANCE")
    severity: str = Field("info", description="Severity: info, warning, error, critical")
    user_id: Optional[str] = Field(None, description="User ID associated with event")
    session_id: Optional[str] = Field(None, description="Session ID")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    resource_type: Optional[str] = Field(None, description="Resource type: TABLE, FILE, API, SERVICE")
    resource_id: Optional[str] = Field(None, description="Resource identifier")
    action: str = Field(..., description="Action performed")
    old_values: Optional[Dict[str, Any]] = Field(None, description="Old values (for updates)")
    new_values: Optional[Dict[str, Any]] = Field(None, description="New values (for updates)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    compliance_flags: List[str] = Field([], description="Compliance flags: GDPR, HIPAA, SOX, PCI_DSS")
    success: bool = Field(True, description="Whether the action was successful")
    error_message: Optional[str] = Field(None, description="Error message if action failed")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")

class ComplianceReportRequest(BaseModel):
    """Compliance report request model"""
    compliance_standard: str = Field(..., description="Compliance standard: GDPR, HIPAA, SOX, PCI_DSS")
    report_period_days: int = Field(30, description="Report period in days")
    include_details: bool = Field(True, description="Include detailed findings")
    output_format: str = Field("json", description="Output format: json, csv, pdf")

class RetentionPolicyExecution(BaseModel):
    """Retention policy execution model"""
    policy_name: str = Field(..., description="Name of the retention policy")
    dry_run: bool = Field(True, description="Whether to perform dry run")
    force_delete: bool = Field(False, description="Force deletion regardless of compliance")

class SecurityAlert(BaseModel):
    """Security alert model"""
    alert_type: str = Field(..., description="Type of security alert")
    severity: str = Field(..., description="Severity level")
    description: str = Field(..., description="Alert description")
    affected_resources: List[str] = Field([], description="Affected resources")
    recommended_actions: List[str] = Field([], description="Recommended actions")
    incident_response_required: bool = Field(False, description="Whether incident response is required")

# Audit Manager Class
class AuditComplianceManager:
    """Comprehensive audit compliance manager"""

    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.compliance_standards = {
            "GDPR": self._check_gdpr_compliance,
            "HIPAA": self._check_hipaa_compliance,
            "SOX": self._check_sox_compliance,
            "PCI_DSS": self._check_pci_dss_compliance
        }

    def log_audit_event(self, event: AuditEvent) -> Dict[str, Any]:
        """Log an audit event to the database"""
        try:
            audit_id = str(uuid.uuid4())

            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO audit_log (
                        audit_id, event_type, event_category, severity, user_id, session_id,
                        ip_address, user_agent, resource_type, resource_id, action,
                        old_values, new_values, metadata, compliance_flags,
                        success, error_message, processing_time_ms, event_timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    audit_id,
                    event.event_type,
                    event.event_category,
                    event.severity,
                    event.user_id,
                    event.session_id,
                    event.ip_address,
                    event.user_agent,
                    event.resource_type,
                    event.resource_id,
                    event.action,
                    json.dumps(event.old_values) if event.old_values else None,
                    json.dumps(event.new_values) if event.new_values else None,
                    json.dumps(event.metadata) if event.metadata else None,
                    event.compliance_flags,
                    event.success,
                    event.error_message,
                    event.processing_time_ms,
                    datetime.utcnow()
                ))

                audit_log_id = cursor.fetchone()[0]
                self.db_connection.commit()

                # Log specialized audit events
                if event.event_category == "DATA_ACCESS":
                    self._log_data_access_event(audit_log_id, event)
                elif event.event_category == "USER_ACTION":
                    self._log_user_activity_event(audit_log_id, event)
                elif event.event_category == "SECURITY":
                    self._log_security_event(audit_log_id, event)

                # Update metrics
                AUDIT_EVENTS_LOGGED.labels(
                    event_category=event.event_category,
                    severity=event.severity
                ).inc()

                # Check for security alerts
                self._check_security_alerts(event)

                logger.info("Audit event logged", audit_id=audit_id, event_type=event.event_type)

                return {
                    "audit_id": audit_id,
                    "audit_log_id": audit_log_id,
                    "status": "logged",
                    "timestamp": datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error("Failed to log audit event", error=str(e), event_type=event.event_type)
            self.db_connection.rollback()
            raise

    def _log_data_access_event(self, audit_log_id: int, event: AuditEvent):
        """Log data access specific audit information"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO data_access_audit (
                        audit_log_id, data_source, table_name, column_names,
                        record_ids, query_text, rows_affected, data_classification,
                        encryption_used, masking_applied, access_purpose
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    audit_log_id,
                    event.metadata.get("data_source") if event.metadata else None,
                    event.resource_id,
                    event.metadata.get("column_names") if event.metadata else None,
                    event.metadata.get("record_ids") if event.metadata else None,
                    event.metadata.get("query_text") if event.metadata else None,
                    event.metadata.get("rows_affected") if event.metadata else None,
                    event.metadata.get("data_classification") if event.metadata else "INTERNAL",
                    event.metadata.get("encryption_used", False) if event.metadata else False,
                    event.metadata.get("masking_applied", False) if event.metadata else False,
                    event.metadata.get("access_purpose") if event.metadata else None
                ))
                self.db_connection.commit()

        except Exception as e:
            logger.error("Failed to log data access event", error=str(e))

    def _log_user_activity_event(self, audit_log_id: int, event: AuditEvent):
        """Log user activity specific audit information"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO user_activity_audit (
                        audit_log_id, user_id, activity_type, device_info,
                        location_info, risk_score, suspicious_activity,
                        mfa_used, session_duration
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    audit_log_id,
                    event.user_id,
                    event.event_type,
                    json.dumps(event.metadata.get("device_info")) if event.metadata and event.metadata.get("device_info") else None,
                    json.dumps(event.metadata.get("location_info")) if event.metadata and event.metadata.get("location_info") else None,
                    event.metadata.get("risk_score") if event.metadata else None,
                    event.metadata.get("suspicious_activity", False) if event.metadata else False,
                    event.metadata.get("mfa_used", False) if event.metadata else False,
                    event.metadata.get("session_duration") if event.metadata else None
                ))
                self.db_connection.commit()

        except Exception as e:
            logger.error("Failed to log user activity event", error=str(e))

    def _log_security_event(self, audit_log_id: int, event: AuditEvent):
        """Log security event specific audit information"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO security_events_audit (
                        audit_log_id, event_type, threat_level, source_ip,
                        target_resource, attack_vector, mitigation_action,
                        alert_generated, incident_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    audit_log_id,
                    event.event_type,
                    event.severity,
                    event.ip_address,
                    event.resource_id,
                    event.metadata.get("attack_vector") if event.metadata else None,
                    event.metadata.get("mitigation_action") if event.metadata else None,
                    True,  # Alert is generated by default for security events
                    event.metadata.get("incident_id") if event.metadata else None
                ))
                self.db_connection.commit()

        except Exception as e:
            logger.error("Failed to log security event", error=str(e))

    def _check_security_alerts(self, event: AuditEvent):
        """Check for security alerts based on audit events"""
        alerts_triggered = []

        # Failed login attempts
        if event.event_type == "FAILED_LOGIN" and event.severity in ["warning", "error"]:
            alerts_triggered.append({
                "type": "failed_login_spike",
                "severity": "medium",
                "description": f"Failed login attempt from {event.ip_address}",
                "affected_resources": [event.user_id or "unknown_user"]
            })

        # Suspicious data access
        if event.event_category == "DATA_ACCESS" and event.metadata and event.metadata.get("suspicious_activity"):
            alerts_triggered.append({
                "type": "suspicious_data_access",
                "severity": "high",
                "description": f"Suspicious data access detected: {event.action} on {event.resource_id}",
                "affected_resources": [event.resource_id]
            })

        # Policy violations
        if event.event_type == "POLICY_VIOLATION":
            alerts_triggered.append({
                "type": "policy_violation",
                "severity": "high",
                "description": f"Policy violation: {event.metadata.get('policy_name') if event.metadata else 'Unknown'}",
                "affected_resources": event.metadata.get("affected_resources", []) if event.metadata else []
            })

        # Trigger alerts
        for alert in alerts_triggered:
            SECURITY_ALERTS_TRIGGERED.labels(
                alert_type=alert["type"],
                severity=alert["severity"]
            ).inc()

            logger.warning("Security alert triggered",
                         alert_type=alert["type"],
                         severity=alert["severity"],
                         description=alert["description"])

    def generate_compliance_report(self, request: ComplianceReportRequest) -> Dict[str, Any]:
        """Generate compliance report for specified standard"""
        try:
            COMPLIANCE_REPORTS_GENERATED.labels(compliance_standard=request.compliance_standard).inc()

            if request.compliance_standard not in self.compliance_standards:
                raise ValueError(f"Unsupported compliance standard: {request.compliance_standard}")

            compliance_checker = self.compliance_standards[request.compliance_standard]
            report_data = compliance_checker(request.report_period_days, request.include_details)

            return {
                "compliance_standard": request.compliance_standard,
                "report_period_days": request.report_period_days,
                "generated_at": datetime.utcnow().isoformat(),
                "assessment_result": report_data["result"],
                "findings": report_data["findings"] if request.include_details else [],
                "recommendations": report_data["recommendations"],
                "compliance_score": report_data["score"]
            }

        except Exception as e:
            logger.error("Failed to generate compliance report", error=str(e), standard=request.compliance_standard)
            raise

    def _check_gdpr_compliance(self, period_days: int, include_details: bool) -> Dict[str, Any]:
        """Check GDPR compliance"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)

            with self.db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Check data retention compliance
                cursor.execute("""
                    SELECT COUNT(*) as violations
                    FROM audit_log al
                    JOIN data_access_audit daa ON al.id = daa.audit_log_id
                    WHERE al.event_timestamp < %s
                    AND al.event_category = 'DATA_ACCESS'
                    AND 'GDPR' = ANY(al.compliance_flags)
                """, (cutoff_date,))

                retention_violations = cursor.fetchone()["violations"]

                # Check data subject access requests
                cursor.execute("""
                    SELECT COUNT(*) as access_requests
                    FROM audit_log
                    WHERE event_type = 'DATA_SUBJECT_ACCESS_REQUEST'
                    AND event_timestamp >= %s
                    AND 'GDPR' = ANY(compliance_flags)
                """, (cutoff_date,))

                access_requests = cursor.fetchone()["access_requests"]

                # Check data deletion requests
                cursor.execute("""
                    SELECT COUNT(*) as deletion_requests
                    FROM audit_log
                    WHERE event_type = 'DATA_SUBJECT_DELETION_REQUEST'
                    AND event_timestamp >= %s
                    AND 'GDPR' = ANY(compliance_flags)
                """, (cutoff_date,))

                deletion_requests = cursor.fetchone()["deletion_requests"]

            findings = []
            score = 100

            if retention_violations > 0:
                findings.append(f"Data retention violations: {retention_violations} records exceed retention period")
                score -= min(retention_violations * 5, 40)

            if access_requests == 0:
                findings.append("No data subject access requests processed in reporting period")
                score -= 10

            if deletion_requests == 0:
                findings.append("No data subject deletion requests processed in reporting period")
                score -= 10

            return {
                "result": "PASS" if score >= 70 else "FAIL",
                "findings": findings if include_details else [],
                "recommendations": [
                    "Implement automated data retention policies",
                    "Establish data subject request processing workflow",
                    "Conduct regular GDPR compliance audits",
                    "Implement data encryption for personal data"
                ],
                "score": max(score, 0)
            }

        except Exception as e:
            logger.error("GDPR compliance check failed", error=str(e))
            return {
                "result": "ERROR",
                "findings": [f"Compliance check failed: {str(e)}"],
                "recommendations": ["Review system logs and fix compliance check"],
                "score": 0
            }

    def _check_hipaa_compliance(self, period_days: int, include_details: bool) -> Dict[str, Any]:
        """Check HIPAA compliance"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)

            with self.db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Check for unauthorized PHI access
                cursor.execute("""
                    SELECT COUNT(*) as unauthorized_access
                    FROM audit_log
                    WHERE event_category = 'DATA_ACCESS'
                    AND event_type = 'UNAUTHORIZED_ACCESS'
                    AND event_timestamp >= %s
                    AND 'HIPAA' = ANY(compliance_flags)
                """, (cutoff_date,))

                unauthorized_access = cursor.fetchone()["unauthorized_access"]

                # Check encryption usage
                cursor.execute("""
                    SELECT COUNT(*) as total_phi_access,
                           COUNT(CASE WHEN metadata->>'encryption_used' = 'true' THEN 1 END) as encrypted_access
                    FROM audit_log al
                    JOIN data_access_audit daa ON al.id = daa.audit_log_id
                    WHERE al.event_timestamp >= %s
                    AND daa.data_classification = 'RESTRICTED'
                    AND 'HIPAA' = ANY(al.compliance_flags)
                """, (cutoff_date,))

                phi_stats = cursor.fetchone()

            findings = []
            score = 100

            if unauthorized_access > 0:
                findings.append(f"Unauthorized PHI access incidents: {unauthorized_access}")
                score -= min(unauthorized_access * 20, 60)

            if phi_stats["total_phi_access"] > 0:
                encryption_rate = (phi_stats["encrypted_access"] / phi_stats["total_phi_access"]) * 100
                if encryption_rate < 100:
                    findings.append(f"PHI encryption rate: {encryption_rate:.1f}% (should be 100%)")
                    score -= (100 - encryption_rate)

            return {
                "result": "PASS" if score >= 80 else "FAIL",
                "findings": findings if include_details else [],
                "recommendations": [
                    "Implement strict access controls for PHI",
                    "Ensure all PHI data is encrypted at rest and in transit",
                    "Conduct regular security risk assessments",
                    "Maintain detailed audit logs for all PHI access"
                ],
                "score": max(score, 0)
            }

        except Exception as e:
            logger.error("HIPAA compliance check failed", error=str(e))
            return {
                "result": "ERROR",
                "findings": [f"Compliance check failed: {str(e)}"],
                "recommendations": ["Review system logs and fix compliance check"],
                "score": 0
            }

    def _check_sox_compliance(self, period_days: int, include_details: bool) -> Dict[str, Any]:
        """Check SOX compliance"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)

            with self.db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Check for segregation of duties violations
                cursor.execute("""
                    SELECT COUNT(*) as sod_violations
                    FROM audit_log
                    WHERE event_type = 'SEGREGATION_OF_DUTIES_VIOLATION'
                    AND event_timestamp >= %s
                    AND 'SOX' = ANY(compliance_flags)
                """, (cutoff_date,))

                sod_violations = cursor.fetchone()["sod_violations"]

                # Check change management compliance
                cursor.execute("""
                    SELECT COUNT(*) as total_changes,
                           COUNT(CASE WHEN metadata->>'approved_change' = 'true' THEN 1 END) as approved_changes
                    FROM audit_log
                    WHERE event_category = 'SYSTEM'
                    AND event_type IN ('CONFIGURATION_CHANGE', 'SOFTWARE_UPDATE')
                    AND event_timestamp >= %s
                    AND 'SOX' = ANY(compliance_flags)
                """, (cutoff_date,))

                change_stats = cursor.fetchone()

            findings = []
            score = 100

            if sod_violations > 0:
                findings.append(f"Segregation of duties violations: {sod_violations}")
                score -= min(sod_violations * 15, 50)

            if change_stats["total_changes"] > 0:
                approval_rate = (change_stats["approved_changes"] / change_stats["total_changes"]) * 100
                if approval_rate < 95:
                    findings.append(f"Change approval rate: {approval_rate:.1f}% (should be ≥95%)")
                    score -= (95 - approval_rate)

            return {
                "result": "PASS" if score >= 85 else "FAIL",
                "findings": findings if include_details else [],
                "recommendations": [
                    "Implement and enforce segregation of duties policies",
                    "Establish formal change management procedures",
                    "Conduct regular internal control assessments",
                    "Maintain detailed audit trails for financial systems"
                ],
                "score": max(score, 0)
            }

        except Exception as e:
            logger.error("SOX compliance check failed", error=str(e))
            return {
                "result": "ERROR",
                "findings": [f"Compliance check failed: {str(e)}"],
                "recommendations": ["Review system logs and fix compliance check"],
                "score": 0
            }

    def _check_pci_dss_compliance(self, period_days: int, include_details: bool) -> Dict[str, Any]:
        """Check PCI DSS compliance"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)

            with self.db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Check cardholder data access
                cursor.execute("""
                    SELECT COUNT(*) as chd_access_events
                    FROM audit_log al
                    JOIN data_access_audit daa ON al.id = daa.audit_log_id
                    WHERE al.event_timestamp >= %s
                    AND daa.data_classification = 'RESTRICTED'
                    AND 'PCI_DSS' = ANY(al.compliance_flags)
                """, (cutoff_date,))

                chd_access = cursor.fetchone()["chd_access_events"]

                # Check encryption compliance
                cursor.execute("""
                    SELECT COUNT(*) as total_card_access,
                           COUNT(CASE WHEN daa.encryption_used = true THEN 1 END) as encrypted_access,
                           COUNT(CASE WHEN daa.masking_applied = true THEN 1 END) as masked_access
                    FROM audit_log al
                    JOIN data_access_audit daa ON al.id = daa.audit_log_id
                    WHERE al.event_timestamp >= %s
                    AND daa.data_classification = 'RESTRICTED'
                    AND 'PCI_DSS' = ANY(al.compliance_flags)
                """, (cutoff_date,))

                encryption_stats = cursor.fetchone()

            findings = []
            score = 100

            if chd_access == 0:
                findings.append("No cardholder data access events found - verify CHD handling")
                score -= 20

            if encryption_stats["total_card_access"] > 0:
                encrypted_rate = (encryption_stats["encrypted_access"] / encryption_stats["total_card_access"]) * 100
                if encrypted_rate < 100:
                    findings.append(f"CHD encryption rate: {encrypted_rate:.1f}% (must be 100%)")
                    score -= (100 - encrypted_rate)

            return {
                "result": "PASS" if score >= 90 else "FAIL",
                "findings": findings if include_details else [],
                "recommendations": [
                    "Implement strict controls for cardholder data",
                    "Ensure all CHD is encrypted and tokenized",
                    "Regular security scans and penetration testing",
                    "Maintain detailed access logs for CHD"
                ],
                "score": max(score, 0)
            }

        except Exception as e:
            logger.error("PCI DSS compliance check failed", error=str(e))
            return {
                "result": "ERROR",
                "findings": [f"Compliance check failed: {str(e)}"],
                "recommendations": ["Review system logs and fix compliance check"],
                "score": 0
            }

    def execute_retention_policy(self, execution: RetentionPolicyExecution) -> Dict[str, Any]:
        """Execute retention policy for audit data"""
        try:
            RETENTION_POLICY_EXECUTIONS.labels(policy_type=execution.policy_name).inc()

            with self.db_connection.cursor() as cursor:
                # Get retention policy details
                cursor.execute("""
                    SELECT * FROM audit_retention_policies
                    WHERE policy_name = %s AND is_active = true
                """, (execution.policy_name,))

                policy = cursor.fetchone()

                if not policy:
                    raise ValueError(f"Retention policy '{execution.policy_name}' not found or inactive")

                # Calculate cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=policy[3])  # retention_period_days

                if execution.dry_run:
                    # Count records that would be affected
                    cursor.execute("""
                        SELECT COUNT(*) as records_to_delete
                        FROM audit_log
                        WHERE event_category = %s
                        AND event_timestamp < %s
                    """, (policy[2], cutoff_date))  # event_category

                    count = cursor.fetchone()[0]

                    return {
                        "policy_name": execution.policy_name,
                        "dry_run": True,
                        "records_to_delete": count,
                        "cutoff_date": cutoff_date.isoformat(),
                        "status": "preview"
                    }
                else:
                    # Execute deletion
                    cursor.execute("""
                        DELETE FROM audit_log
                        WHERE event_category = %s
                        AND event_timestamp < %s
                    """, (policy[2], cutoff_date))

                    deleted_count = cursor.rowcount
                    self.db_connection.commit()

                    # Log retention execution
                    audit_event = AuditEvent(
                        event_type="RETENTION_POLICY_EXECUTION",
                        event_category="COMPLIANCE",
                        severity="info",
                        action="DELETE",
                        resource_type="AUDIT_LOG",
                        metadata={
                            "policy_name": execution.policy_name,
                            "records_deleted": deleted_count,
                            "cutoff_date": cutoff_date.isoformat()
                        },
                        compliance_flags=["GDPR", "SOX"]
                    )
                    self.log_audit_event(audit_event)

                    return {
                        "policy_name": execution.policy_name,
                        "dry_run": False,
                        "records_deleted": deleted_count,
                        "cutoff_date": cutoff_date.isoformat(),
                        "status": "executed"
                    }

        except Exception as e:
            logger.error("Retention policy execution failed", error=str(e), policy=execution.policy_name)
            self.db_connection.rollback()
            raise

    def get_audit_trail_integrity(self) -> Dict[str, Any]:
        """Check audit trail integrity"""
        try:
            AUDIT_INTEGRITY_CHECKS.labels(check_type="full").inc()

            with self.db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Check for gaps in audit log sequence
                cursor.execute("""
                    SELECT COUNT(*) as total_records,
                           COUNT(DISTINCT id) as unique_ids,
                           MAX(id) - MIN(id) + 1 as expected_count
                    FROM audit_log
                    WHERE event_timestamp >= CURRENT_TIMESTAMP - INTERVAL '30 days'
                """)

                integrity_stats = cursor.fetchone()

                # Check for tampered records (basic check)
                cursor.execute("""
                    SELECT COUNT(*) as records_with_future_timestamps
                    FROM audit_log
                    WHERE event_timestamp > CURRENT_TIMESTAMP + INTERVAL '1 minute'
                """)

                future_timestamps = cursor.fetchone()["records_with_future_timestamps"]

            issues = []

            if integrity_stats["total_records"] != integrity_stats["expected_count"]:
                issues.append("Gaps detected in audit log sequence")

            if integrity_stats["total_records"] != integrity_stats["unique_ids"]:
                issues.append("Duplicate audit record IDs found")

            if future_timestamps > 0:
                issues.append(f"Records with future timestamps: {future_timestamps}")

            return {
                "integrity_status": "COMPROMISED" if issues else "INTACT",
                "issues": issues,
                "stats": dict(integrity_stats),
                "last_check": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Audit integrity check failed", error=str(e))
            return {
                "integrity_status": "ERROR",
                "issues": [f"Integrity check failed: {str(e)}"],
                "last_check": datetime.utcnow().isoformat()
            }

# Global instance
audit_manager = None

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "audit-compliance-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/audit/events")
async def log_audit_event(event: AuditEvent):
    """Log an audit event"""
    try:
        result = audit_manager.log_audit_event(event)
        return result

    except Exception as e:
        logger.error("Audit event logging failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Audit logging failed: {str(e)}")

@app.post("/compliance/reports")
async def generate_compliance_report(request: ComplianceReportRequest):
    """Generate compliance report"""
    try:
        report = audit_manager.generate_compliance_report(request)
        return report

    except Exception as e:
        logger.error("Compliance report generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.post("/retention/execute")
async def execute_retention_policy(execution: RetentionPolicyExecution):
    """Execute retention policy"""
    try:
        result = audit_manager.execute_retention_policy(execution)
        return result

    except Exception as e:
        logger.error("Retention policy execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Retention execution failed: {str(e)}")

@app.get("/integrity/check")
async def check_audit_integrity():
    """Check audit trail integrity"""
    try:
        result = audit_manager.get_audit_trail_integrity()
        return result

    except Exception as e:
        logger.error("Integrity check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Integrity check failed: {str(e)}")

@app.get("/audit/events")
async def get_audit_events(
    event_category: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Retrieve audit events"""
    try:
        with database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            query = """
                SELECT * FROM audit_log
                WHERE 1=1
            """
            params = []

            if event_category:
                query += " AND event_category = %s"
                params.append(event_category)

            if user_id:
                query += " AND user_id = %s"
                params.append(user_id)

            query += " ORDER BY event_timestamp DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cursor.execute(query, params)
            events = cursor.fetchall()

            return {
                "events": [dict(event) for event in events],
                "total": len(events),
                "limit": limit,
                "offset": offset
            }

    except Exception as e:
        logger.error("Audit events retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Events retrieval failed: {str(e)}")

@app.get("/stats")
async def get_audit_stats():
    """Get audit service statistics"""
    return {
        "service": "audit-compliance-service",
        "metrics": {
            "audit_events_logged_total": AUDIT_EVENTS_LOGGED._value.get(),
            "compliance_reports_generated_total": COMPLIANCE_REPORTS_GENERATED._value.get(),
            "retention_policy_executions_total": RETENTION_POLICY_EXECUTIONS._value.get(),
            "security_alerts_triggered_total": SECURITY_ALERTS_TRIGGERED._value.get(),
            "audit_integrity_checks_total": AUDIT_INTEGRITY_CHECKS._value.get()
        }
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global audit_manager, database_connection

    logger.info("Audit Compliance Service starting up...")

    # Setup database connection
    try:
        db_config = {
            "host": os.getenv("POSTGRES_HOST", "postgresql_ingestion"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agentic_ingestion"),
            "user": os.getenv("POSTGRES_USER", "agentic_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "agentic123")
        }

        database_connection = psycopg2.connect(**db_config)
        audit_manager = AuditComplianceManager(database_connection)

        logger.info("Database connection established")

    except Exception as e:
        logger.warning("Database connection failed", error=str(e))

    logger.info("Audit Compliance Service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global database_connection

    logger.info("Audit Compliance Service shutting down...")

    # Close database connection
    if database_connection:
        database_connection.close()

    logger.info("Audit Compliance Service shutdown complete")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "audit_compliance_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8096)),
        reload=False,
        log_level="info"
    )
