-- Agentic Platform Database Schema
-- Comprehensive schema for modular agentic data platform
-- Supports both ingestion and output layers with full audit trail

-- ===========================================
-- CORE SYSTEM TABLES
-- ===========================================

-- System configuration and settings
CREATE TABLE IF NOT EXISTS system_config (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value TEXT,
    config_type VARCHAR(50) NOT NULL DEFAULT 'string',
    is_encrypted BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

-- Service registry for microservices
CREATE TABLE IF NOT EXISTS services (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(255) UNIQUE NOT NULL,
    service_type VARCHAR(100) NOT NULL,
    service_version VARCHAR(50),
    host VARCHAR(255),
    port INTEGER,
    health_endpoint VARCHAR(500),
    status VARCHAR(50) DEFAULT 'stopped',
    last_health_check TIMESTAMP WITH TIME ZONE,
    startup_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- INGESTION LAYER TABLES
-- ===========================================

-- Data sources configuration
CREATE TABLE IF NOT EXISTS data_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(255) UNIQUE NOT NULL,
    source_type VARCHAR(100) NOT NULL, -- csv, excel, pdf, json, api, ui
    connection_string TEXT,
    authentication_type VARCHAR(50),
    authentication_config JSONB,
    schema_definition JSONB,
    metadata JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

-- Ingestion jobs tracking
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id SERIAL PRIMARY KEY,
    job_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    source_id INTEGER REFERENCES data_sources(id),
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    priority INTEGER DEFAULT 1,
    batch_size INTEGER DEFAULT 1000,
    total_records INTEGER,
    processed_records INTEGER DEFAULT 0,
    failed_records INTEGER DEFAULT 0,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    duration INTERVAL,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Data validation results
CREATE TABLE IF NOT EXISTS data_validation_results (
    id SERIAL PRIMARY KEY,
    job_id INTEGER REFERENCES ingestion_jobs(id),
    record_id VARCHAR(255),
    validation_type VARCHAR(100) NOT NULL,
    validation_rule VARCHAR(500),
    is_valid BOOLEAN DEFAULT true,
    error_message TEXT,
    error_severity VARCHAR(50) DEFAULT 'error',
    field_name VARCHAR(255),
    field_value TEXT,
    expected_value TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Data quality metrics
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES data_sources(id),
    job_id INTEGER REFERENCES ingestion_jobs(id),
    metric_name VARCHAR(255) NOT NULL,
    metric_value NUMERIC,
    metric_unit VARCHAR(50),
    threshold_min NUMERIC,
    threshold_max NUMERIC,
    is_within_threshold BOOLEAN,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Metadata catalog
CREATE TABLE IF NOT EXISTS metadata_catalog (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES data_sources(id),
    table_name VARCHAR(255) NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    data_type VARCHAR(100),
    is_nullable BOOLEAN DEFAULT true,
    is_primary_key BOOLEAN DEFAULT false,
    is_foreign_key BOOLEAN DEFAULT false,
    references_table VARCHAR(255),
    references_column VARCHAR(255),
    description TEXT,
    tags TEXT[],
    sensitivity_level VARCHAR(50) DEFAULT 'public',
    pii_detection BOOLEAN DEFAULT false,
    encryption_required BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, table_name, column_name)
);

-- ===========================================
-- OUTPUT LAYER TABLES
-- ===========================================

-- Output targets configuration
CREATE TABLE IF NOT EXISTS output_targets (
    id SERIAL PRIMARY KEY,
    target_name VARCHAR(255) UNIQUE NOT NULL,
    target_type VARCHAR(100) NOT NULL, -- postgresql, mongodb, qdrant, elasticsearch, etc.
    connection_string TEXT,
    authentication_config JSONB,
    storage_config JSONB,
    retention_policy JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

-- Data routing rules
CREATE TABLE IF NOT EXISTS data_routing_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(255) UNIQUE NOT NULL,
    source_pattern VARCHAR(500),
    target_id INTEGER REFERENCES output_targets(id),
    routing_conditions JSONB,
    transformation_rules JSONB,
    priority INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

-- Output jobs tracking
CREATE TABLE IF NOT EXISTS output_jobs (
    id SERIAL PRIMARY KEY,
    job_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    target_id INTEGER REFERENCES output_targets(id),
    source_job_id INTEGER REFERENCES ingestion_jobs(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    records_processed INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    duration INTERVAL,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- USER MANAGEMENT & SECURITY (for REQUIRE_AUTH=true)
-- ===========================================

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(500),
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    last_login TIMESTAMP WITH TIME ZONE,
    password_changed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id),
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Permissions and roles
CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    role_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB,
    is_system_role BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User role assignments
CREATE TABLE IF NOT EXISTS user_roles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    role_id INTEGER REFERENCES roles(id),
    assigned_by INTEGER REFERENCES users(id),
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(user_id, role_id)
);

-- ===========================================
-- OAUTH2/OIDC TABLES
-- ===========================================

-- OAuth2 clients
CREATE TABLE IF NOT EXISTS oauth_clients (
    id SERIAL PRIMARY KEY,
    client_id VARCHAR(100) UNIQUE NOT NULL,
    client_secret VARCHAR(200) NOT NULL,
    client_name VARCHAR(200) NOT NULL,
    client_uri VARCHAR(500),
    redirect_uris TEXT[], -- Array of allowed redirect URIs
    grant_types TEXT[], -- Array of supported grant types
    response_types TEXT[], -- Array of supported response types
    scope VARCHAR(500) DEFAULT 'openid profile email',
    token_endpoint_auth_method VARCHAR(50) DEFAULT 'client_secret_basic',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Authorization codes
CREATE TABLE IF NOT EXISTS authorization_codes (
    id SERIAL PRIMARY KEY,
    code VARCHAR(200) UNIQUE NOT NULL,
    client_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    redirect_uri VARCHAR(500) NOT NULL,
    scope VARCHAR(500),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_used BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    used_at TIMESTAMP WITH TIME ZONE
);

-- Refresh tokens
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id SERIAL PRIMARY KEY,
    token_id VARCHAR(100) UNIQUE NOT NULL,
    client_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    scope VARCHAR(500),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_revoked BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP WITH TIME ZONE,
    revoked_reason VARCHAR(200)
);

-- Access tokens (for introspection)
CREATE TABLE IF NOT EXISTS access_tokens (
    id SERIAL PRIMARY KEY,
    token_id VARCHAR(100) UNIQUE NOT NULL,
    client_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    scope VARCHAR(500),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_revoked BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP WITH TIME ZONE,
    revoked_reason VARCHAR(200)
);

-- Sessions for web-based authentication
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    client_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit log for OAuth2/OIDC operations
CREATE TABLE IF NOT EXISTS oauth_audit_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    client_id VARCHAR(100),
    user_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    event_details JSONB,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- ENCRYPTION & SECURITY TABLES
-- ===========================================

-- Encryption keys management
CREATE TABLE IF NOT EXISTS encryption_keys (
    id SERIAL PRIMARY KEY,
    key_id VARCHAR(100) UNIQUE NOT NULL,
    key_type VARCHAR(50) NOT NULL, -- AES, RSA_PRIVATE, RSA_PUBLIC
    key_size INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    rotation_count INTEGER DEFAULT 0,
    last_rotated TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- Encrypted data tracking
CREATE TABLE IF NOT EXISTS encrypted_data (
    id SERIAL PRIMARY KEY,
    data_id VARCHAR(100) UNIQUE NOT NULL,
    encryption_key_id VARCHAR(100) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    original_size_bytes INTEGER,
    encrypted_size_bytes INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    data_classification VARCHAR(50), -- PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- TLS certificates tracking
CREATE TABLE IF NOT EXISTS tls_certificates (
    id SERIAL PRIMARY KEY,
    certificate_id VARCHAR(100) UNIQUE NOT NULL,
    subject_name VARCHAR(500) NOT NULL,
    issuer_name VARCHAR(500) NOT NULL,
    serial_number VARCHAR(100) NOT NULL,
    valid_from TIMESTAMP WITH TIME ZONE NOT NULL,
    valid_until TIMESTAMP WITH TIME ZONE NOT NULL,
    certificate_data TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Data masking rules
CREATE TABLE IF NOT EXISTS masking_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) UNIQUE NOT NULL,
    rule_type VARCHAR(50) NOT NULL, -- FIELD_MASK, PATTERN_MASK, CUSTOM
    field_name VARCHAR(100),
    masking_pattern VARCHAR(200),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- ===========================================
-- AUDIT & COMPLIANCE TABLES
-- ===========================================

-- Main audit log for all system activities
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    audit_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    event_category VARCHAR(50) NOT NULL, -- SECURITY, DATA_ACCESS, USER_ACTION, SYSTEM, COMPLIANCE
    severity VARCHAR(20) DEFAULT 'info', -- info, warning, error, critical
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    resource_type VARCHAR(50), -- TABLE, FILE, API, SERVICE
    resource_id VARCHAR(200),
    action VARCHAR(50) NOT NULL, -- CREATE, READ, UPDATE, DELETE, LOGIN, LOGOUT, etc.
    old_values JSONB,
    new_values JSONB,
    metadata JSONB,
    compliance_flags TEXT[], -- GDPR, HIPAA, SOX, PCI_DSS, etc.
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    processing_time_ms INTEGER,
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Data access audit log (subset of audit_log for data operations)
CREATE TABLE IF NOT EXISTS data_access_audit (
    id SERIAL PRIMARY KEY,
    audit_log_id INTEGER REFERENCES audit_log(id),
    data_source VARCHAR(100) NOT NULL,
    table_name VARCHAR(100),
    column_names TEXT[],
    record_ids TEXT[],
    query_text TEXT,
    rows_affected INTEGER,
    data_classification VARCHAR(50), -- PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED
    encryption_used BOOLEAN DEFAULT false,
    masking_applied BOOLEAN DEFAULT false,
    access_purpose VARCHAR(100),
    retention_period_days INTEGER DEFAULT 2555, -- 7 years
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User activity audit log
CREATE TABLE IF NOT EXISTS user_activity_audit (
    id SERIAL PRIMARY KEY,
    audit_log_id INTEGER REFERENCES audit_log(id),
    user_id VARCHAR(100) NOT NULL,
    activity_type VARCHAR(50) NOT NULL, -- LOGIN, LOGOUT, PASSWORD_CHANGE, PROFILE_UPDATE, etc.
    device_info JSONB,
    location_info JSONB,
    risk_score DECIMAL(3,2), -- 0.00 to 1.00
    suspicious_activity BOOLEAN DEFAULT false,
    mfa_used BOOLEAN DEFAULT false,
    session_duration INTEGER, -- seconds
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Security events audit log
CREATE TABLE IF NOT EXISTS security_events_audit (
    id SERIAL PRIMARY KEY,
    audit_log_id INTEGER REFERENCES audit_log(id),
    event_type VARCHAR(100) NOT NULL, -- FAILED_LOGIN, SUSPICIOUS_ACTIVITY, POLICY_VIOLATION, etc.
    threat_level VARCHAR(20) DEFAULT 'low', -- low, medium, high, critical
    source_ip INET,
    target_resource VARCHAR(200),
    attack_vector VARCHAR(100),
    mitigation_action VARCHAR(100),
    alert_generated BOOLEAN DEFAULT false,
    incident_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Compliance audit log
CREATE TABLE IF NOT EXISTS compliance_audit (
    id SERIAL PRIMARY KEY,
    audit_log_id INTEGER REFERENCES audit_log(id),
    compliance_standard VARCHAR(50) NOT NULL, -- GDPR, HIPAA, SOX, PCI_DSS, etc.
    requirement_id VARCHAR(100),
    assessment_result VARCHAR(20), -- PASS, FAIL, NA
    remediation_required BOOLEAN DEFAULT false,
    remediation_deadline TIMESTAMP WITH TIME ZONE,
    auditor_notes TEXT,
    evidence_documentation TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit retention policies
CREATE TABLE IF NOT EXISTS audit_retention_policies (
    id SERIAL PRIMARY KEY,
    policy_name VARCHAR(100) UNIQUE NOT NULL,
    event_category VARCHAR(50) NOT NULL,
    retention_period_days INTEGER NOT NULL,
    archive_after_days INTEGER,
    delete_after_days INTEGER,
    compliance_requirements TEXT[],
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit configuration
CREATE TABLE IF NOT EXISTS audit_configuration (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB,
    description TEXT,
    is_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert default audit retention policies
INSERT INTO audit_retention_policies (policy_name, event_category, retention_period_days, archive_after_days, delete_after_days, compliance_requirements) VALUES
('security_events', 'SECURITY', 2555, 365, 2555, ARRAY['GDPR', 'HIPAA', 'SOX']), -- 7 years
('data_access', 'DATA_ACCESS', 2555, 365, 2555, ARRAY['GDPR', 'HIPAA', 'PCI_DSS']),
('user_activity', 'USER_ACTION', 1825, 180, 1825, ARRAY['GDPR', 'SOX']), -- 5 years
('system_events', 'SYSTEM', 1095, 90, 1095, ARRAY['GDPR', 'SOX']), -- 3 years
('compliance_audit', 'COMPLIANCE', 3650, 365, 3650, ARRAY['GDPR', 'HIPAA', 'SOX', 'PCI_DSS']); -- 10 years

-- Insert default audit configuration
INSERT INTO audit_configuration (config_key, config_value, description) VALUES
('audit_enabled', 'true', 'Master switch for audit logging'),
('real_time_alerts', 'true', 'Enable real-time security alerts'),
('compliance_reporting', 'true', 'Enable automated compliance reporting'),
('data_masking_audit', 'true', 'Audit data masking operations'),
('encryption_audit', 'true', 'Audit encryption operations'),
('retention_enforcement', 'true', 'Enforce audit retention policies'),
('chain_of_custody', 'true', 'Maintain audit log integrity');

-- ===========================================
-- BACKUP & DISASTER RECOVERY TABLES
-- ===========================================

-- Backup metadata tracking
CREATE TABLE IF NOT EXISTS backup_metadata (
    id SERIAL PRIMARY KEY,
    backup_id VARCHAR(100) UNIQUE NOT NULL,
    backup_type VARCHAR(50) NOT NULL, -- database, object_store, cache, config
    target_system VARCHAR(100) NOT NULL,
    storage_location VARCHAR(500) NOT NULL,
    retention_days INTEGER DEFAULT 30,
    size_bytes BIGINT DEFAULT 0,
    status VARCHAR(50) DEFAULT 'in_progress', -- in_progress, completed, failed
    compression_used BOOLEAN DEFAULT false,
    encryption_used BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB
);

-- Backup schedules
CREATE TABLE IF NOT EXISTS backup_schedules (
    id SERIAL PRIMARY KEY,
    schedule_name VARCHAR(100) UNIQUE NOT NULL,
    backup_type VARCHAR(50) NOT NULL,
    cron_expression VARCHAR(100) NOT NULL,
    target_systems TEXT[], -- Array of target systems
    storage_location VARCHAR(500),
    retention_policy VARCHAR(100) DEFAULT 'standard',
    is_active BOOLEAN DEFAULT true,
    last_run TIMESTAMP WITH TIME ZONE,
    next_run TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Disaster recovery plans
CREATE TABLE IF NOT EXISTS disaster_recovery_plans (
    id SERIAL PRIMARY KEY,
    plan_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    trigger_conditions TEXT[], -- Array of conditions that trigger recovery
    recovery_steps JSONB, -- Detailed recovery procedure
    estimated_rto_minutes INTEGER, -- Recovery Time Objective
    estimated_rpo_seconds INTEGER, -- Recovery Point Objective
    test_frequency_days INTEGER DEFAULT 30,
    last_tested TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Replication configuration
CREATE TABLE IF NOT EXISTS replication_config (
    id SERIAL PRIMARY KEY,
    source_cluster VARCHAR(100) NOT NULL,
    target_cluster VARCHAR(100) NOT NULL,
    replication_type VARCHAR(50) DEFAULT 'async', -- async, sync, semi-sync
    data_types TEXT[], -- Types of data to replicate
    lag_tolerance_seconds INTEGER DEFAULT 300,
    is_active BOOLEAN DEFAULT true,
    last_sync TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Backup verification results
CREATE TABLE IF NOT EXISTS backup_verification (
    id SERIAL PRIMARY KEY,
    backup_id VARCHAR(100) NOT NULL,
    verification_type VARCHAR(50) NOT NULL, -- integrity, consistency, corruption
    result VARCHAR(20) NOT NULL, -- pass, fail, warning
    details JSONB,
    verified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    verified_by VARCHAR(100),
    FOREIGN KEY (backup_id) REFERENCES backup_metadata(backup_id)
);

-- Insert default backup schedules
INSERT INTO backup_schedules (schedule_name, backup_type, cron_expression, target_systems, storage_location, retention_policy) VALUES
('daily_database_backup', 'database', '0 2 * * *', ARRAY['postgresql_ingestion'], '/backups/database', 'standard'),
('hourly_cache_backup', 'cache', '0 * * * *', ARRAY['redis_ingestion'], '/backups/cache', 'short'),
('daily_config_backup', 'config', '0 3 * * *', ARRAY['platform_configs'], '/backups/config', 'long'),
('weekly_object_store_backup', 'object_store', '0 4 * * 0', ARRAY['minio_bronze', 'minio_silver', 'minio_gold'], '/backups/object_store', 'long');

-- Insert default disaster recovery plan
INSERT INTO disaster_recovery_plans (plan_name, description, trigger_conditions, recovery_steps, estimated_rto_minutes, estimated_rpo_seconds) VALUES
('primary_datacenter_failure', 'Recovery plan for primary datacenter failure',
 ARRAY['datacenter_unreachable', 'service_degradation'],
 '[
   {"step": 1, "action": "Activate secondary datacenter", "duration_minutes": 10},
   {"step": 2, "action": "Failover database to secondary", "duration_minutes": 15},
   {"step": 3, "action": "Update DNS to secondary datacenter", "duration_minutes": 5},
   {"step": 4, "action": "Verify service availability", "duration_minutes": 5}
 ]',
 35, 300);

-- ===========================================
-- PORT MANAGEMENT TABLES
-- ===========================================

-- Port assignments tracking
CREATE TABLE IF NOT EXISTS port_assignments (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    assigned_port INTEGER NOT NULL,
    protocol VARCHAR(10) DEFAULT 'tcp',
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reassigned_at TIMESTAMP WITH TIME ZONE,
    released_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active', -- active, released, conflict
    conflict_count INTEGER DEFAULT 0,
    metadata JSONB,
    UNIQUE(service_name, assigned_port, protocol)
);

-- Port conflict history
CREATE TABLE IF NOT EXISTS port_conflicts (
    id SERIAL PRIMARY KEY,
    port INTEGER NOT NULL,
    conflicting_services TEXT[],
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_action VARCHAR(50), -- reassign, release, manual
    resolution_details TEXT,
    status VARCHAR(20) DEFAULT 'detected' -- detected, resolved, escalated
);

-- Port usage analytics
CREATE TABLE IF NOT EXISTS port_usage_stats (
    id SERIAL PRIMARY KEY,
    port INTEGER NOT NULL,
    service_name VARCHAR(100),
    usage_start TIMESTAMP WITH TIME ZONE,
    usage_end TIMESTAMP WITH TIME ZONE,
    total_connections INTEGER DEFAULT 0,
    peak_connections INTEGER DEFAULT 0,
    data_transferred_bytes BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- (Duplicate audit_log table removed - using comprehensive definition above)

-- Data lineage tracking
CREATE TABLE IF NOT EXISTS data_lineage (
    id SERIAL PRIMARY KEY,
    lineage_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    source_record_id VARCHAR(255),
    source_table VARCHAR(255),
    source_column VARCHAR(255),
    target_record_id VARCHAR(255),
    target_table VARCHAR(255),
    target_column VARCHAR(255),
    transformation_type VARCHAR(100),
    transformation_details JSONB,
    data_quality_score NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Compliance events
CREATE TABLE IF NOT EXISTS compliance_events (
    id SERIAL PRIMARY KEY,
    event_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    compliance_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) DEFAULT 'info',
    description TEXT,
    affected_resources JSONB,
    remediation_steps TEXT,
    is_resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- MONITORING & METRICS TABLES
-- ===========================================

-- Service health checks
CREATE TABLE IF NOT EXISTS service_health_checks (
    id SERIAL PRIMARY KEY,
    service_id INTEGER REFERENCES services(id),
    check_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    response_time_ms INTEGER,
    error_message TEXT,
    metadata JSONB,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value NUMERIC,
    metric_unit VARCHAR(50),
    labels JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Error tracking
CREATE TABLE IF NOT EXISTS error_logs (
    id SERIAL PRIMARY KEY,
    error_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    service_name VARCHAR(255),
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT,
    stack_trace TEXT,
    context_data JSONB,
    severity VARCHAR(50) DEFAULT 'error',
    is_resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- BACKUP & RECOVERY TABLES
-- ===========================================

-- Backup metadata
CREATE TABLE IF NOT EXISTS backups (
    id SERIAL PRIMARY KEY,
    backup_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    backup_type VARCHAR(100) NOT NULL,
    target_name VARCHAR(255) NOT NULL,
    backup_path VARCHAR(1000),
    compression_type VARCHAR(50),
    encryption_enabled BOOLEAN DEFAULT false,
    size_bytes BIGINT,
    status VARCHAR(50) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    duration INTERVAL,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Recovery points
CREATE TABLE IF NOT EXISTS recovery_points (
    id SERIAL PRIMARY KEY,
    recovery_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    backup_id INTEGER REFERENCES backups(id),
    recovery_type VARCHAR(100) NOT NULL,
    recovery_path VARCHAR(1000),
    status VARCHAR(50) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    success BOOLEAN DEFAULT false,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- DATA LAKE & STORAGE TABLES
-- ===========================================

-- Data lake objects
CREATE TABLE IF NOT EXISTS data_lake_objects (
    id SERIAL PRIMARY KEY,
    object_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    bucket VARCHAR(255) NOT NULL,
    object_key VARCHAR(1000) NOT NULL,
    layer VARCHAR(50) NOT NULL, -- bronze, silver, gold
    source_id INTEGER REFERENCES data_sources(id),
    file_format VARCHAR(50),
    compression VARCHAR(50),
    size_bytes BIGINT,
    row_count BIGINT,
    schema_version VARCHAR(50),
    partition_info JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Storage usage tracking
CREATE TABLE IF NOT EXISTS storage_usage (
    id SERIAL PRIMARY KEY,
    target_name VARCHAR(255) NOT NULL,
    target_type VARCHAR(100) NOT NULL,
    total_size_bytes BIGINT DEFAULT 0,
    used_size_bytes BIGINT DEFAULT 0,
    available_size_bytes BIGINT DEFAULT 0,
    compression_ratio NUMERIC,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- INDEXES FOR PERFORMANCE
-- ===========================================

-- Core indexes
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_source_id ON ingestion_jobs(source_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_created_at ON ingestion_jobs(created_at);

CREATE INDEX IF NOT EXISTS idx_output_jobs_status ON output_jobs(status);
CREATE INDEX IF NOT EXISTS idx_output_jobs_target_id ON output_jobs(target_id);
CREATE INDEX IF NOT EXISTS idx_output_jobs_created_at ON output_jobs(created_at);

CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_data_validation_job_id ON data_validation_results(job_id);
CREATE INDEX IF NOT EXISTS idx_data_quality_source_id ON data_quality_metrics(source_id);
CREATE INDEX IF NOT EXISTS idx_metadata_source_id ON metadata_catalog(source_id);
CREATE INDEX IF NOT EXISTS idx_data_lineage_source ON data_lineage(source_table, source_column);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_service ON performance_metrics(service_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_error_logs_service ON error_logs(service_name, created_at);

-- ===========================================
-- DEFAULT DATA INSERTION
-- ===========================================

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, config_type) VALUES
('require_auth', 'false', 'boolean'),
('log_level', 'INFO', 'string'),
('metrics_enabled', 'true', 'boolean'),
('tracing_enabled', 'true', 'boolean'),
('data_validation_enabled', 'true', 'boolean'),
('audit_enabled', 'true', 'boolean')
ON CONFLICT (config_key) DO NOTHING;

-- Insert default roles
INSERT INTO roles (role_name, description, permissions, is_system_role) VALUES
('admin', 'System administrator with full access', '{"*": true}', true),
('data_engineer', 'Data engineer with ingestion and processing access', '{"ingestion": true, "processing": true, "monitoring": true}', true),
('data_analyst', 'Data analyst with read access', '{"read": true, "query": true}', true),
('auditor', 'Compliance auditor with audit access', '{"audit": true, "read": true}', true)
ON CONFLICT (role_name) DO NOTHING;

-- Insert default service registry entries
INSERT INTO services (service_name, service_type, service_version) VALUES
('ingestion-coordinator', 'coordinator', '1.0.0'),
('output-coordinator', 'coordinator', '1.0.0'),
('csv-ingestion-service', 'ingestion', '1.0.0'),
('pdf-ingestion-service', 'ingestion', '1.0.0'),
('excel-ingestion-service', 'ingestion', '1.0.0'),
('json-ingestion-service', 'ingestion', '1.0.0'),
('api-ingestion-service', 'ingestion', '1.0.0'),
('ui-scraper-service', 'ingestion', '1.0.0')
ON CONFLICT (service_name) DO NOTHING;

-- ===========================================
-- FUNCTIONS AND TRIGGERS
-- ===========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_data_sources_updated_at BEFORE UPDATE ON data_sources FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_output_targets_updated_at BEFORE UPDATE ON output_targets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_metadata_catalog_updated_at BEFORE UPDATE ON metadata_catalog FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_data_lake_objects_updated_at BEFORE UPDATE ON data_lake_objects FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function for audit logging
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
DECLARE
    action_type TEXT;
    old_data JSONB;
    new_data JSONB;
BEGIN
    -- Determine action type
    IF TG_OP = 'INSERT' THEN
        action_type := 'INSERT';
        old_data := NULL;
        new_data := row_to_json(NEW)::JSONB;
    ELSIF TG_OP = 'UPDATE' THEN
        action_type := 'UPDATE';
        old_data := row_to_json(OLD)::JSONB;
        new_data := row_to_json(NEW)::JSONB;
    ELSIF TG_OP = 'DELETE' THEN
        action_type := 'DELETE';
        old_data := row_to_json(OLD)::JSONB;
        new_data := NULL;
    END IF;

    -- Insert audit record
    INSERT INTO audit_log (
        action,
        resource_type,
        resource_id,
        action_details,
        success
    ) VALUES (
        action_type,
        TG_TABLE_NAME,
        CASE
            WHEN TG_OP = 'DELETE' THEN (old_data->>'id')::TEXT
            ELSE (new_data->>'id')::TEXT
        END,
        jsonb_build_object(
            'table_name', TG_TABLE_NAME,
            'operation', TG_OP,
            'old_data', old_data,
            'new_data', new_data
        ),
        true
    );

    RETURN CASE
        WHEN TG_OP = 'DELETE' THEN OLD
        ELSE NEW
    END;
END;
$$ LANGUAGE plpgsql;

-- Add audit triggers to sensitive tables (uncomment as needed)
-- CREATE TRIGGER audit_users AFTER INSERT OR UPDATE OR DELETE ON users FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
-- CREATE TRIGGER audit_data_sources AFTER INSERT OR UPDATE OR DELETE ON data_sources FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- ===========================================
-- VIEWS FOR COMMON QUERIES
-- ===========================================

-- View for active data sources with latest metrics
CREATE OR REPLACE VIEW active_data_sources_view AS
SELECT
    ds.*,
    COUNT(ij.id) as total_jobs,
    COUNT(CASE WHEN ij.status = 'completed' THEN 1 END) as completed_jobs,
    COUNT(CASE WHEN ij.status = 'failed' THEN 1 END) as failed_jobs,
    AVG(EXTRACT(EPOCH FROM ij.duration)) as avg_job_duration_seconds,
    MAX(ij.created_at) as last_job_time
FROM data_sources ds
LEFT JOIN ingestion_jobs ij ON ds.id = ij.source_id
WHERE ds.is_active = true
GROUP BY ds.id;

-- View for system health overview
CREATE OR REPLACE VIEW system_health_view AS
SELECT
    s.service_name,
    s.service_type,
    s.status,
    s.last_health_check,
    COUNT(shc.id) as health_check_count,
    AVG(shc.response_time_ms) as avg_response_time,
    COUNT(CASE WHEN shc.status = 'healthy' THEN 1 END) as healthy_checks,
    COUNT(CASE WHEN shc.status = 'unhealthy' THEN 1 END) as unhealthy_checks
FROM services s
LEFT JOIN service_health_checks shc ON s.id = shc.service_id
    AND shc.checked_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY s.id, s.service_name, s.service_type, s.status, s.last_health_check;

-- View for data quality dashboard
CREATE OR REPLACE VIEW data_quality_dashboard AS
SELECT
    ds.source_name,
    ds.source_type,
    COUNT(DISTINCT ij.id) as total_jobs,
    AVG(dqm.metric_value) as avg_quality_score,
    COUNT(CASE WHEN dvr.is_valid = false THEN 1 END) as validation_errors,
    MAX(ij.created_at) as last_ingestion_time
FROM data_sources ds
LEFT JOIN ingestion_jobs ij ON ds.id = ij.source_id
LEFT JOIN data_quality_metrics dqm ON ds.id = dqm.source_id
LEFT JOIN data_validation_results dvr ON ij.id = dvr.job_id
WHERE ds.is_active = true
GROUP BY ds.id, ds.source_name, ds.source_type;

-- ========================================
-- VECTOR OPERATIONS TABLES
-- ========================================

-- Vector collections table
CREATE TABLE IF NOT EXISTS vector_collections (
    id SERIAL PRIMARY KEY,
    collection_name VARCHAR(255) UNIQUE NOT NULL,
    collection_type VARCHAR(50) DEFAULT 'documents',
    vector_dimension INTEGER NOT NULL DEFAULT 384,
    distance_metric VARCHAR(20) DEFAULT 'cosine',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Vector documents table
CREATE TABLE IF NOT EXISTS vector_documents (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) UNIQUE NOT NULL,
    collection_name VARCHAR(255) NOT NULL,
    text_content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    vector_id BIGINT, -- Qdrant internal ID
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (collection_name) REFERENCES vector_collections(collection_name) ON DELETE CASCADE
);

-- Vector search history table
CREATE TABLE IF NOT EXISTS vector_search_history (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    collection_name VARCHAR(255) NOT NULL,
    search_results JSONB,
    execution_time_ms INTEGER,
    result_count INTEGER,
    user_id VARCHAR(255), -- For REQUIRE_AUTH integration
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Embedding cache table for performance optimization
CREATE TABLE IF NOT EXISTS embedding_cache (
    id SERIAL PRIMARY KEY,
    text_hash VARCHAR(64) UNIQUE NOT NULL,
    text_content TEXT NOT NULL,
    embedding_vector JSONB NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for vector operations
CREATE INDEX IF NOT EXISTS idx_vector_documents_collection ON vector_documents(collection_name);
CREATE INDEX IF NOT EXISTS idx_vector_documents_doc_id ON vector_documents(document_id);
CREATE INDEX IF NOT EXISTS idx_vector_search_history_collection ON vector_search_history(collection_name);
CREATE INDEX IF NOT EXISTS idx_vector_search_history_user ON vector_search_history(user_id);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_hash ON embedding_cache(text_hash);

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_vector_collections_updated_at
    BEFORE UPDATE ON vector_collections
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vector_documents_updated_at
    BEFORE UPDATE ON vector_documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- DASHBOARD & MONITORING TABLES
-- ========================================

-- Dashboard metrics table for storing real-time metrics
CREATE TABLE IF NOT EXISTS dashboard_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,6),
    metric_type VARCHAR(50) DEFAULT 'gauge', -- gauge, counter, histogram
    labels JSONB DEFAULT '{}', -- Additional labels/tags
    service_name VARCHAR(255),
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE -- For time-based cleanup
);

-- Service health monitoring table
CREATE TABLE IF NOT EXISTS service_health (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    service_type VARCHAR(100),
    status VARCHAR(50) DEFAULT 'unknown', -- healthy, degraded, unhealthy, unknown
    response_time_ms INTEGER,
    error_message TEXT,
    last_check TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    next_check TIMESTAMP WITH TIME ZONE,
    consecutive_failures INTEGER DEFAULT 0,
    uptime_percentage DECIMAL(5,2) DEFAULT 100.00,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User activity tracking for dashboard analytics
CREATE TABLE IF NOT EXISTS user_activity (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255), -- NULL for anonymous users
    session_id VARCHAR(255),
    activity_type VARCHAR(100) NOT NULL, -- login, api_call, page_view, etc.
    activity_details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    service_name VARCHAR(255),
    endpoint VARCHAR(500),
    response_time_ms INTEGER,
    status_code INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alert and notification system
CREATE TABLE IF NOT EXISTS system_alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(100) NOT NULL, -- system, service, performance, security
    severity VARCHAR(20) DEFAULT 'info', -- info, warning, error, critical
    title VARCHAR(500) NOT NULL,
    description TEXT,
    affected_service VARCHAR(255),
    alert_data JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    acknowledged BOOLEAN DEFAULT false,
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance monitoring table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    metric_type VARCHAR(100) NOT NULL, -- cpu, memory, disk, network, response_time
    metric_value DECIMAL(15,6),
    unit VARCHAR(20), -- percentage, bytes, milliseconds, etc.
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    retention_period INTERVAL DEFAULT '30 days'
);

-- Dashboard configuration table
CREATE TABLE IF NOT EXISTS dashboard_config (
    id SERIAL PRIMARY KEY,
    dashboard_name VARCHAR(255) UNIQUE NOT NULL,
    config_data JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API usage tracking table
CREATE TABLE IF NOT EXISTS api_usage (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    endpoint VARCHAR(500) NOT NULL,
    method VARCHAR(10) NOT NULL,
    user_id VARCHAR(255),
    request_count INTEGER DEFAULT 1,
    total_response_time_ms INTEGER,
    avg_response_time_ms DECIMAL(10,2),
    error_count INTEGER DEFAULT 0,
    last_request_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    first_request_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(service_name, endpoint, method, user_id)
);

-- Session management table
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    login_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    logout_time TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    session_data JSONB DEFAULT '{}'
);

-- ========================================
-- INDEXES FOR DASHBOARD TABLES
-- ========================================

-- Dashboard metrics indexes
CREATE INDEX IF NOT EXISTS idx_dashboard_metrics_name_time ON dashboard_metrics(metric_name, collected_at);
CREATE INDEX IF NOT EXISTS idx_dashboard_metrics_service ON dashboard_metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_dashboard_metrics_expires ON dashboard_metrics(expires_at) WHERE expires_at IS NOT NULL;

-- Service health indexes
CREATE INDEX IF NOT EXISTS idx_service_health_name ON service_health(service_name);
CREATE INDEX IF NOT EXISTS idx_service_health_status ON service_health(status);
CREATE INDEX IF NOT EXISTS idx_service_health_last_check ON service_health(last_check);

-- User activity indexes
CREATE INDEX IF NOT EXISTS idx_user_activity_user ON user_activity(user_id);
CREATE INDEX IF NOT EXISTS idx_user_activity_session ON user_activity(session_id);
CREATE INDEX IF NOT EXISTS idx_user_activity_type ON user_activity(activity_type);
CREATE INDEX IF NOT EXISTS idx_user_activity_created ON user_activity(created_at);

-- System alerts indexes
CREATE INDEX IF NOT EXISTS idx_system_alerts_type ON system_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_system_alerts_severity ON system_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_system_alerts_active ON system_alerts(is_active);
CREATE INDEX IF NOT EXISTS idx_system_alerts_created ON system_alerts(created_at);

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_service ON performance_metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_collected ON performance_metrics(collected_at);

-- API usage indexes
CREATE INDEX IF NOT EXISTS idx_api_usage_service ON api_usage(service_name);
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_usage(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_usage_user ON api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_last_request ON api_usage(last_request_at);

-- Session management indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_user_sessions_last_activity ON user_sessions(last_activity);

-- ========================================
-- TRIGGERS FOR DASHBOARD TABLES
-- ========================================

-- Triggers for updated_at on dashboard tables
CREATE TRIGGER update_service_health_updated_at
    BEFORE UPDATE ON service_health
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_alerts_updated_at
    BEFORE UPDATE ON system_alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_dashboard_config_updated_at
    BEFORE UPDATE ON dashboard_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- CLEANUP FUNCTIONS
-- ========================================

-- Function to clean up old dashboard metrics
CREATE OR REPLACE FUNCTION cleanup_expired_metrics()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM dashboard_metrics
    WHERE expires_at IS NOT NULL
    AND expires_at < CURRENT_TIMESTAMP;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old performance metrics
CREATE OR REPLACE FUNCTION cleanup_old_performance_metrics()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM performance_metrics
    WHERE collected_at < (CURRENT_TIMESTAMP - retention_period);

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update user session last activity
CREATE OR REPLACE FUNCTION update_session_activity(session_id_param VARCHAR(255))
RETURNS VOID AS $$
BEGIN
    UPDATE user_sessions
    SET last_activity = CURRENT_TIMESTAMP
    WHERE session_id = session_id_param
    AND is_active = true;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- VIEWS FOR DASHBOARD ANALYTICS
-- ========================================

-- View for active service health summary
CREATE OR REPLACE VIEW active_service_health AS
SELECT
    service_name,
    service_type,
    status,
    response_time_ms,
    consecutive_failures,
    uptime_percentage,
    last_check,
    CASE
        WHEN last_check > CURRENT_TIMESTAMP - INTERVAL '5 minutes' THEN 'current'
        WHEN last_check > CURRENT_TIMESTAMP - INTERVAL '15 minutes' THEN 'stale'
        ELSE 'outdated'
    END as health_status
FROM service_health
WHERE status IN ('healthy', 'degraded')
ORDER BY service_name;

-- View for recent user activity summary
CREATE OR REPLACE VIEW recent_user_activity AS
SELECT
    DATE_TRUNC('hour', created_at) as hour,
    activity_type,
    COUNT(*) as activity_count,
    COUNT(DISTINCT COALESCE(user_id, session_id)) as unique_users
FROM user_activity
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at), activity_type
ORDER BY hour DESC, activity_count DESC;

-- View for API usage summary
CREATE OR REPLACE VIEW api_usage_summary AS
SELECT
    service_name,
    endpoint,
    method,
    SUM(request_count) as total_requests,
    AVG(avg_response_time_ms) as avg_response_time,
    SUM(error_count) as total_errors,
    MAX(last_request_at) as last_request
FROM api_usage
GROUP BY service_name, endpoint, method
ORDER BY total_requests DESC;

-- ========================================
-- DASHBOARD DEFAULT CONFIGURATION
-- ========================================

-- Insert default dashboard configuration
INSERT INTO dashboard_config (dashboard_name, config_data, created_by) VALUES
('main-dashboard', '{
  "title": "Agentic Platform Dashboard",
  "description": "Real-time monitoring and control center",
  "refresh_interval": 30,
  "widgets": [
    {
      "type": "metric",
      "title": "Active Microservices",
      "metric": "service_health_count",
      "position": {"x": 0, "y": 0, "width": 3, "height": 2}
    },
    {
      "type": "metric",
      "title": "API Endpoints",
      "metric": "api_endpoints_count",
      "position": {"x": 3, "y": 0, "width": 3, "height": 2}
    },
    {
      "type": "metric",
      "title": "Data Formats",
      "metric": "data_formats_count",
      "position": {"x": 6, "y": 0, "width": 3, "height": 2}
    },
    {
      "type": "metric",
      "title": "Platform Uptime",
      "metric": "platform_uptime",
      "position": {"x": 9, "y": 0, "width": 3, "height": 2}
    }
  ],
  "theme": "professional",
  "auto_refresh": true
}', 'system')
ON CONFLICT (dashboard_name) DO NOTHING;
