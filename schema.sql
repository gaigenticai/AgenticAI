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

-- ===========================================
-- AGENTIC BRAIN TABLES
-- ===========================================

-- Agent registry - main table for all deployed agents
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(100) NOT NULL, -- underwriting, claims, fraud_detection, etc.
    description TEXT,
    persona JSONB NOT NULL, -- {"role": "string", "expertise": ["array"], "personality": "string"}
    reasoning_pattern VARCHAR(50) DEFAULT 'react', -- react, reflection, planning, multi_agent
    status VARCHAR(50) DEFAULT 'inactive', -- active, inactive, error, deploying
    version VARCHAR(50) DEFAULT '1.0.0',
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP WITH TIME ZONE,
    deployment_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,2) DEFAULT 0.00
);

-- Agent configurations - stores the complete agent configuration JSON
CREATE TABLE IF NOT EXISTS agent_configs (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    config_version VARCHAR(50) NOT NULL,
    config_data JSONB NOT NULL, -- Complete AgentConfig JSON from AgentBrain.md
    is_active BOOLEAN DEFAULT false,
    checksum VARCHAR(64), -- For integrity verification
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, config_version)
);

-- Agent templates - prebuilt templates for different domains
CREATE TABLE IF NOT EXISTS agent_templates (
    id SERIAL PRIMARY KEY,
    template_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(100) NOT NULL,
    description TEXT,
    template_data JSONB NOT NULL, -- Template YAML/JSON structure
    thumbnail_url VARCHAR(500),
    category VARCHAR(100) DEFAULT 'general',
    tags TEXT[],
    is_public BOOLEAN DEFAULT true,
    usage_count INTEGER DEFAULT 0,
    rating DECIMAL(3,2),
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent deployments - tracks deployment history and status
CREATE TABLE IF NOT EXISTS agent_deployments (
    id SERIAL PRIMARY KEY,
    deployment_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, deploying, deployed, failed, stopped
    deployment_type VARCHAR(50) DEFAULT 'manual', -- manual, auto, rollback
    source_template_id VARCHAR(100), -- Reference to template if deployed from template
    config_checksum VARCHAR(64),
    deployed_by VARCHAR(255),
    deployed_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    rollback_deployment_id UUID, -- Reference to previous deployment for rollback
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent sessions - tracks active agent execution sessions
CREATE TABLE IF NOT EXISTS agent_sessions (
    id SERIAL PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    task_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'running', -- running, completed, failed, paused
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration INTERVAL,
    task_input JSONB,
    task_output JSONB,
    error_message TEXT,
    memory_used_bytes BIGINT DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    cost_estimate DECIMAL(10,4) DEFAULT 0.0000,
    metadata JSONB DEFAULT '{}'
);

-- Agent plugins - registry of available plugins
CREATE TABLE IF NOT EXISTS agent_plugins (
    id SERIAL PRIMARY KEY,
    plugin_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    plugin_type VARCHAR(50) NOT NULL, -- domain, generic
    domain VARCHAR(100), -- underwriting, claims, fraud, etc. (NULL for generic)
    description TEXT,
    version VARCHAR(50) DEFAULT '1.0.0',
    author VARCHAR(255),
    license VARCHAR(100),
    repository_url VARCHAR(500),
    documentation_url VARCHAR(500),
    dependencies JSONB DEFAULT '[]', -- Array of required dependencies
    configuration_schema JSONB, -- JSON Schema for plugin configuration
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    usage_count INTEGER DEFAULT 0,
    rating DECIMAL(3,2),
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent workflows - defines workflow components and connections
CREATE TABLE IF NOT EXISTS agent_workflows (
    id SERIAL PRIMARY KEY,
    workflow_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    workflow_version VARCHAR(50) DEFAULT '1.0.0',
    components JSONB NOT NULL, -- Array of workflow components
    connections JSONB NOT NULL, -- Array of connections between components
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent memory management - handles different types of memory
CREATE TABLE IF NOT EXISTS agent_memory (
    id SERIAL PRIMARY KEY,
    memory_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    memory_type VARCHAR(50) NOT NULL, -- working, episodic, semantic, long_term
    content_key VARCHAR(255) NOT NULL,
    content_value JSONB NOT NULL,
    importance_score DECIMAL(3,2) DEFAULT 0.50, -- 0.00 to 1.00
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE, -- For TTL-based memory
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, memory_type, content_key)
);

-- Agent metrics - performance and usage metrics
CREATE TABLE IF NOT EXISTS agent_metrics (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6),
    metric_unit VARCHAR(50),
    metric_type VARCHAR(50) DEFAULT 'gauge', -- gauge, counter, histogram
    labels JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    retention_period INTERVAL DEFAULT '30 days'
);

-- Agent audit log - specific audit trail for agent operations
CREATE TABLE IF NOT EXISTS agent_audit_log (
    id SERIAL PRIMARY KEY,
    audit_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    agent_id INTEGER REFERENCES agents(id),
    operation VARCHAR(100) NOT NULL, -- create, update, deploy, execute, delete
    resource_type VARCHAR(50) NOT NULL, -- agent, workflow, plugin, memory, session
    resource_id VARCHAR(255),
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    operation_details JSONB,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    processing_time_ms INTEGER,
    compliance_flags TEXT[], -- GDPR, HIPAA, SOX, etc.
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Template usage tracking
CREATE TABLE IF NOT EXISTS template_usage (
    id SERIAL PRIMARY KEY,
    template_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    usage_type VARCHAR(50) NOT NULL, -- view, instantiate, deploy
    usage_context JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Plugin usage tracking
CREATE TABLE IF NOT EXISTS plugin_usage (
    id SERIAL PRIMARY KEY,
    plugin_id VARCHAR(100) NOT NULL,
    agent_id INTEGER REFERENCES agents(id),
    usage_type VARCHAR(50) NOT NULL, -- load, execute, configure
    usage_context JSONB DEFAULT '{}',
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- AGENTIC BRAIN INDEXES
-- ===========================================

-- Core agent indexes
CREATE INDEX IF NOT EXISTS idx_agents_agent_id ON agents(agent_id);
CREATE INDEX IF NOT EXISTS idx_agents_domain ON agents(domain);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_created_at ON agents(created_at);

-- Agent config indexes
CREATE INDEX IF NOT EXISTS idx_agent_configs_agent_id ON agent_configs(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_configs_active ON agent_configs(is_active);

-- Agent deployment indexes
CREATE INDEX IF NOT EXISTS idx_agent_deployments_agent_id ON agent_deployments(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_deployments_status ON agent_deployments(status);
CREATE INDEX IF NOT EXISTS idx_agent_deployments_created_at ON agent_deployments(created_at);

-- Agent session indexes
CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent_id ON agent_sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_status ON agent_sessions(status);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_started_at ON agent_sessions(started_at);

-- Agent plugin indexes
CREATE INDEX IF NOT EXISTS idx_agent_plugins_plugin_type ON agent_plugins(plugin_type);
CREATE INDEX IF NOT EXISTS idx_agent_plugins_domain ON agent_plugins(domain);
CREATE INDEX IF NOT EXISTS idx_agent_plugins_active ON agent_plugins(is_active);

-- Agent workflow indexes
CREATE INDEX IF NOT EXISTS idx_agent_workflows_agent_id ON agent_workflows(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_workflows_active ON agent_workflows(is_active);

-- Agent memory indexes
CREATE INDEX IF NOT EXISTS idx_agent_memory_agent_id ON agent_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_type ON agent_memory(memory_type);
CREATE INDEX IF NOT EXISTS idx_agent_memory_expires ON agent_memory(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_agent_memory_last_accessed ON agent_memory(last_accessed);

-- Agent metrics indexes
CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent_id ON agent_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_name_time ON agent_metrics(metric_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_recorded_at ON agent_metrics(recorded_at);

-- Agent audit indexes
CREATE INDEX IF NOT EXISTS idx_agent_audit_log_agent_id ON agent_audit_log(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_audit_log_operation ON agent_audit_log(operation);
CREATE INDEX IF NOT EXISTS idx_agent_audit_log_timestamp ON agent_audit_log(event_timestamp);

-- Template usage indexes
CREATE INDEX IF NOT EXISTS idx_template_usage_template_id ON template_usage(template_id);
CREATE INDEX IF NOT EXISTS idx_template_usage_created_at ON template_usage(created_at);

-- Plugin usage indexes
CREATE INDEX IF NOT EXISTS idx_plugin_usage_plugin_id ON plugin_usage(plugin_id);
CREATE INDEX IF NOT EXISTS idx_plugin_usage_agent_id ON plugin_usage(agent_id);
CREATE INDEX IF NOT EXISTS idx_plugin_usage_created_at ON plugin_usage(created_at);

-- ===========================================
-- AGENTIC BRAIN VIEWS
-- ===========================================

-- View for active agents with latest metrics
CREATE OR REPLACE VIEW active_agents_view AS
SELECT
    a.*,
    ac.config_version as current_config_version,
    COUNT(DISTINCT s.id) as active_sessions,
    AVG(m.metric_value) FILTER (WHERE m.metric_name = 'response_time_ms') as avg_response_time,
    MAX(m.metric_value) FILTER (WHERE m.metric_name = 'success_rate') as success_rate,
    COUNT(DISTINCT d.id) FILTER (WHERE d.status = 'deployed') as deployment_count
FROM agents a
LEFT JOIN agent_configs ac ON a.id = ac.agent_id AND ac.is_active = true
LEFT JOIN agent_sessions s ON a.id = s.agent_id AND s.status = 'running'
LEFT JOIN agent_metrics m ON a.id = m.agent_id AND m.recorded_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
LEFT JOIN agent_deployments d ON a.id = d.agent_id AND d.status = 'deployed'
WHERE a.status = 'active'
GROUP BY a.id, ac.config_version;

-- View for agent performance summary
CREATE OR REPLACE VIEW agent_performance_view AS
SELECT
    a.agent_id,
    a.name,
    a.domain,
    COUNT(DISTINCT s.id) as total_sessions,
    COUNT(DISTINCT CASE WHEN s.status = 'completed' THEN s.id END) as completed_sessions,
    COUNT(DISTINCT CASE WHEN s.status = 'failed' THEN s.id END) as failed_sessions,
    AVG(s.duration) FILTER (WHERE s.status = 'completed') as avg_completion_time,
    AVG(s.tokens_used) as avg_tokens_used,
    SUM(s.cost_estimate) as total_cost,
    MAX(s.completed_at) as last_execution_time
FROM agents a
LEFT JOIN agent_sessions s ON a.id = s.agent_id
    AND s.created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY a.id, a.agent_id, a.name, a.domain;

-- View for plugin usage analytics
CREATE OR REPLACE VIEW plugin_usage_analytics AS
SELECT
    p.plugin_id,
    p.name,
    p.plugin_type,
    p.domain,
    COUNT(DISTINCT pu.agent_id) as unique_agents_using,
    COUNT(pu.id) as total_usages,
    AVG(pu.execution_time_ms) FILTER (WHERE pu.success = true) as avg_execution_time,
    COUNT(CASE WHEN pu.success = false THEN 1 END) as failure_count,
    MAX(pu.created_at) as last_used
FROM agent_plugins p
LEFT JOIN plugin_usage pu ON p.plugin_id = pu.plugin_id
    AND pu.created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY p.plugin_id, p.name, p.plugin_type, p.domain;

-- ===========================================
-- AGENTIC BRAIN DEFAULT DATA
-- ===========================================

-- Insert default agent templates
INSERT INTO agent_templates (template_id, name, domain, description, template_data, category, tags) VALUES
('underwriting_template', 'Underwriting Agent', 'underwriting',
 'Comprehensive underwriting agent with risk assessment and decision making',
 '{
   "components": [
     {"id": "data_input", "type": "ingestion", "service": "csv-ingestion-service"},
     {"id": "risk_assess", "type": "llm-processor", "config": {"model": "gpt-4", "temperature": 0.6}},
     {"id": "decision", "type": "decision-node", "config": {"approveThreshold": 0.3, "declineThreshold": 0.7}},
     {"id": "policy_output", "type": "database-output", "service": "postgresql-output"}
   ],
   "connections": [
     {"from": "data_input", "to": "risk_assess"},
     {"from": "risk_assess", "to": "decision"},
     {"from": "decision", "to": "policy_output"}
   ],
   "persona": {"role": "Underwriting Analyst", "expertise": ["risk assessment", "compliance"], "personality": "balanced"},
   "reasoningPattern": "react"
 }', 'business_process', ARRAY['underwriting', 'risk', 'insurance', 'decision-making']),
('claims_template', 'Claims Processing Agent', 'claims',
 'Intelligent claims processing with fraud detection and settlement calculation',
 '{
   "components": [
     {"id": "claim_input", "type": "ingestion", "service": "api-ingestion-service"},
     {"id": "fraud_detect", "type": "rule-engine", "config": {"ruleSet": "fraud_rules"}},
     {"id": "adjust_calc", "type": "llm-processor", "config": {"model": "gpt-4", "temperature": 0.7}},
     {"id": "email_notify", "type": "email-output", "config": {"template": "claim_notification"}}
   ],
   "connections": [
     {"from": "claim_input", "to": "fraud_detect"},
     {"from": "fraud_detect", "to": "adjust_calc"},
     {"from": "adjust_calc", "to": "email_notify"}
   ],
   "persona": {"role": "Claims Specialist", "expertise": ["fraud detection", "settlement"], "personality": "efficient"},
   "reasoningPattern": "reflection"
 }', 'business_process', ARRAY['claims', 'fraud', 'insurance', 'settlement'])
ON CONFLICT (template_id) DO NOTHING;

-- Insert default agent plugins
INSERT INTO agent_plugins (plugin_id, name, plugin_type, domain, description, version, author, tags) VALUES
('riskCalculator', 'Risk Calculator', 'domain', 'underwriting',
 'Advanced risk calculation engine for underwriting decisions', '1.0.0', 'system',
 ARRAY['risk', 'underwriting', 'calculation', 'insurance']),
('fraudDetector', 'Fraud Detection Engine', 'domain', 'claims',
 'Machine learning-based fraud detection for claims processing', '1.0.0', 'system',
 ARRAY['fraud', 'claims', 'ml', 'detection']),
('regulatoryChecker', 'Regulatory Compliance Checker', 'domain', 'compliance',
 'Automated regulatory compliance verification and reporting', '1.0.0', 'system',
 ARRAY['compliance', 'regulation', 'audit', 'reporting']),
('dataRetriever', 'Universal Data Retriever', 'generic', NULL,
 'Generic data retrieval and transformation plugin', '1.0.0', 'system',
 ARRAY['data', 'retrieval', 'transformation', 'generic']),
('validator', 'Data Validator', 'generic', NULL,
 'Comprehensive data validation and quality assurance plugin', '1.0.0', 'system',
 ARRAY['validation', 'quality', 'data', 'assurance'])
ON CONFLICT (plugin_id) DO NOTHING;

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

-- =============================================================================
-- AUTHENTICATION AND AUTHORIZATION TABLES
-- =============================================================================

-- User accounts
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(100) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    full_name VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    email_verified_at TIMESTAMP,
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    oauth_provider VARCHAR(50),
    oauth_id VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',
    permissions JSON,
    profile_data JSON,
    last_login_at TIMESTAMP,
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for users table
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_oauth_provider ON users(oauth_provider);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- User sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address VARCHAR(45),
    user_agent TEXT,
    device_info JSON,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for user_sessions table
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_refresh_token ON user_sessions(refresh_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active, expires_at);

-- API keys
CREATE TABLE IF NOT EXISTS api_keys (
    id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    permissions JSON,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for api_keys table
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active, expires_at);

-- OAuth providers
CREATE TABLE IF NOT EXISTS oauth_providers (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    client_id VARCHAR(255) NOT NULL,
    client_secret VARCHAR(255) NOT NULL,
    authorization_url VARCHAR(500) NOT NULL,
    token_url VARCHAR(500) NOT NULL,
    user_info_url VARCHAR(500) NOT NULL,
    scope VARCHAR(255) DEFAULT 'openid email profile',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for oauth_providers table
CREATE INDEX IF NOT EXISTS idx_oauth_providers_name ON oauth_providers(name);
CREATE INDEX IF NOT EXISTS idx_oauth_providers_active ON oauth_providers(is_active);

-- Audit logging
CREATE TABLE IF NOT EXISTS audit_logs (
    id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    ip_address VARCHAR(45),
    user_agent TEXT,
    details JSON,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for audit_logs table
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

-- Roles and permissions
CREATE TABLE IF NOT EXISTS roles (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    permissions JSON NOT NULL,
    is_system_role BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for roles table
CREATE INDEX IF NOT EXISTS idx_roles_name ON roles(name);

-- Insert default roles
INSERT INTO roles (id, name, description, permissions, is_system_role) VALUES
('role_admin', 'admin', 'System administrator with full access', '["read", "write", "delete", "admin", "manage_users", "manage_system", "view_audit_logs"]', true),
('role_user', 'user', 'Standard user with basic access', '["read", "write"]', true),
('role_viewer', 'viewer', 'Read-only user', '["read"]', true)
ON CONFLICT (name) DO NOTHING;

-- Insert default admin user (password: admin123!)
INSERT INTO users (id, email, username, password_hash, full_name, role, is_active, is_verified, permissions) VALUES
('user_admin', 'admin@agenticbrain.com', 'admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LEsBpIw2LJ8XcKweW', 'System Administrator', 'admin', true, true, '["read", "write", "delete", "admin", "manage_users", "manage_system", "view_audit_logs"]')
ON CONFLICT (username) DO NOTHING;

-- Insert default OAuth providers (disabled by default)
INSERT INTO oauth_providers (id, name, client_id, client_secret, authorization_url, token_url, user_info_url, scope, is_active) VALUES
('oauth_google', 'google', '', '', 'https://accounts.google.com/o/oauth2/auth', 'https://oauth2.googleapis.com/token', 'https://openidconnect.googleapis.com/v1/userinfo', 'openid email profile', false),
('oauth_github', 'github', '', '', 'https://github.com/login/oauth/authorize', 'https://github.com/login/oauth/access_token', 'https://api.github.com/user', 'user:email', false)
ON CONFLICT (name) DO NOTHING;

-- Create view for active user sessions
CREATE OR REPLACE VIEW active_user_sessions AS
SELECT
    us.id,
    us.user_id,
    u.username,
    u.email,
    us.ip_address,
    us.user_agent,
    us.created_at,
    us.last_activity_at,
    us.expires_at,
    EXTRACT(EPOCH FROM (us.expires_at - CURRENT_TIMESTAMP)) as seconds_until_expiry
FROM user_sessions us
JOIN users u ON us.user_id = u.id
WHERE us.is_active = true
AND us.expires_at > CURRENT_TIMESTAMP
ORDER BY us.last_activity_at DESC;

-- Create view for user activity summary
CREATE OR REPLACE VIEW user_activity_summary AS
SELECT
    u.id,
    u.username,
    u.email,
    u.role,
    u.last_login_at,
    u.login_attempts,
    COUNT(us.id) as active_sessions,
    COUNT(ak.id) as active_api_keys,
    MAX(us.last_activity_at) as last_activity,
    CASE
        WHEN u.locked_until IS NOT NULL AND u.locked_until > CURRENT_TIMESTAMP THEN true
        ELSE false
    END as is_locked
FROM users u
LEFT JOIN user_sessions us ON u.id = us.user_id AND us.is_active = true AND us.expires_at > CURRENT_TIMESTAMP
LEFT JOIN api_keys ak ON u.id = ak.user_id AND ak.is_active = true AND (ak.expires_at IS NULL OR ak.expires_at > CURRENT_TIMESTAMP)
GROUP BY u.id, u.username, u.email, u.role, u.last_login_at, u.login_attempts, u.locked_until;

-- Create view for security audit summary
CREATE OR REPLACE VIEW security_audit_summary AS
SELECT
    DATE(created_at) as date,
    action,
    resource_type,
    COUNT(*) as total_events,
    COUNT(CASE WHEN success = true THEN 1 END) as successful_events,
    COUNT(CASE WHEN success = false THEN 1 END) as failed_events,
    COUNT(DISTINCT user_id) as unique_users_affected
FROM audit_logs
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY DATE(created_at), action, resource_type
ORDER BY date DESC, total_events DESC;

-- =============================================================================
-- AUDIT LOGGING TABLES
-- =============================================================================

-- Comprehensive audit events
CREATE TABLE IF NOT EXISTS audit_events (
    id VARCHAR(100) PRIMARY KEY,
    event_id VARCHAR(100) UNIQUE NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    service_name VARCHAR(100) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    ip_address VARCHAR(45),
    user_agent TEXT,
    location JSON,
    device_info JSON,
    request_data JSON,
    response_data JSON,
    error_message TEXT,
    execution_time_ms FLOAT,
    success BOOLEAN DEFAULT TRUE,
    compliance_flags JSON,
    risk_score FLOAT DEFAULT 0.0,
    tags JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP
);

-- Create indexes for audit_events table
CREATE INDEX IF NOT EXISTS idx_audit_events_event_id ON audit_events(event_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_events_severity ON audit_events(severity);
CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit_events(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_service_name ON audit_events(service_name);
CREATE INDEX IF NOT EXISTS idx_audit_events_operation ON audit_events(operation);
CREATE INDEX IF NOT EXISTS idx_audit_events_resource_type ON audit_events(resource_type);
CREATE INDEX IF NOT EXISTS idx_audit_events_resource_id ON audit_events(resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_success ON audit_events(success);
CREATE INDEX IF NOT EXISTS idx_audit_events_risk_score ON audit_events(risk_score);
CREATE INDEX IF NOT EXISTS idx_audit_events_created_at ON audit_events(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_events_compliance_flags ON audit_events USING GIN(compliance_flags);

-- Archived audit events
CREATE TABLE IF NOT EXISTS audit_archive (
    id VARCHAR(100) PRIMARY KEY,
    archive_id VARCHAR(100) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    record_count INTEGER NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    checksum VARCHAR(128) NOT NULL,
    compression_type VARCHAR(20) DEFAULT 'gzip',
    retention_period_days INTEGER NOT NULL,
    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for audit_archive table
CREATE INDEX IF NOT EXISTS idx_audit_archive_archive_id ON audit_archive(archive_id);
CREATE INDEX IF NOT EXISTS idx_audit_archive_start_date ON audit_archive(start_date);
CREATE INDEX IF NOT EXISTS idx_audit_archive_end_date ON audit_archive(end_date);
CREATE INDEX IF NOT EXISTS idx_audit_archive_archived_at ON audit_archive(archived_at);

-- Compliance reports
CREATE TABLE IF NOT EXISTS compliance_reports (
    id VARCHAR(100) PRIMARY KEY,
    report_type VARCHAR(50) NOT NULL,
    report_period VARCHAR(20) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    total_events INTEGER DEFAULT 0,
    compliant_events INTEGER DEFAULT 0,
    non_compliant_events INTEGER DEFAULT 0,
    risk_events INTEGER DEFAULT 0,
    findings JSON,
    recommendations JSON,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP
);

-- Create indexes for compliance_reports table
CREATE INDEX IF NOT EXISTS idx_compliance_reports_report_type ON compliance_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_compliance_reports_start_date ON compliance_reports(start_date);
CREATE INDEX IF NOT EXISTS idx_compliance_reports_end_date ON compliance_reports(end_date);
CREATE INDEX IF NOT EXISTS idx_compliance_reports_generated_at ON compliance_reports(generated_at);

-- Security alert rules
CREATE TABLE IF NOT EXISTS alert_rules (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    conditions JSON NOT NULL,
    severity VARCHAR(20) NOT NULL,
    threshold INTEGER NOT NULL,
    time_window_minutes INTEGER NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    notification_channels JSON,
    last_triggered TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for alert_rules table
CREATE INDEX IF NOT EXISTS idx_alert_rules_event_type ON alert_rules(event_type);
CREATE INDEX IF NOT EXISTS idx_alert_rules_is_active ON alert_rules(is_active);
CREATE INDEX IF NOT EXISTS idx_alert_rules_last_triggered ON alert_rules(last_triggered);

-- Alert instances
CREATE TABLE IF NOT EXISTS alert_instances (
    id VARCHAR(100) PRIMARY KEY,
    rule_id VARCHAR(100) NOT NULL,
    alert_message TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    event_count INTEGER NOT NULL,
    events JSON NOT NULL,
    notified_channels JSON,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for alert_instances table
CREATE INDEX IF NOT EXISTS idx_alert_instances_rule_id ON alert_instances(rule_id);
CREATE INDEX IF NOT EXISTS idx_alert_instances_severity ON alert_instances(severity);
CREATE INDEX IF NOT EXISTS idx_alert_instances_created_at ON alert_instances(created_at);
CREATE INDEX IF NOT EXISTS idx_alert_instances_resolved_at ON alert_instances(resolved_at);

-- Data retention policies
CREATE TABLE IF NOT EXISTS data_retention_policies (
    id VARCHAR(100) PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    retention_days INTEGER NOT NULL,
    archive_after_days INTEGER NOT NULL,
    delete_after_days INTEGER NOT NULL,
    compliance_requirements JSON,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for data_retention_policies table
CREATE INDEX IF NOT EXISTS idx_data_retention_policies_event_type ON data_retention_policies(event_type);
CREATE INDEX IF NOT EXISTS idx_data_retention_policies_is_active ON data_retention_policies(is_active);

-- Insert default data retention policies
INSERT INTO data_retention_policies (id, event_type, retention_days, archive_after_days, delete_after_days, compliance_requirements, is_active) VALUES
('policy_authentication', 'authentication', 365, 90, 2555, '{"gdpr": true, "sox": true}', true),
('policy_authorization', 'authorization', 365, 90, 2555, '{"gdpr": true, "sox": true}', true),
('policy_operation', 'operation', 2555, 365, 2555, '{"gdpr": true, "sox": true, "hipaa": true}', true),
('policy_security', 'security', 2555, 365, 2555, '{"gdpr": true, "sox": true, "pci": true}', true),
('policy_compliance', 'compliance', 2555, 365, 2555, '{"gdpr": true, "sox": true, "hipaa": true, "pci": true}', true)
ON CONFLICT (event_type) DO NOTHING;

-- Insert default alert rules
INSERT INTO alert_rules (id, name, description, event_type, conditions, severity, threshold, time_window_minutes, is_active, notification_channels) VALUES
('rule_failed_logins', 'Multiple Failed Authentication Attempts', 'Alert when multiple failed login attempts occur from the same source', 'authentication', '{"success": false, "operation": "login"}', 'warning', 5, 10, true, '["email"]'),
('rule_high_risk_ops', 'High Risk Operations', 'Alert on operations with high risk scores', 'operation', '{"risk_score": {"gt": 8.0}}', 'critical', 1, 5, true, '["email"]'),
('rule_admin_changes', 'Administrative Privilege Changes', 'Alert on changes to administrative privileges', 'authorization', '{"operation": "role_change", "resource_type": "user"}', 'warning', 1, 1, true, '["email"]'),
('rule_unauthorized_access', 'Unauthorized Access Attempts', 'Alert on repeated unauthorized access attempts', 'authorization', '{"success": false, "operation": "access_check"}', 'warning', 3, 15, true, '["email"]')
ON CONFLICT (name) DO NOTHING;

-- Create views for audit analytics
CREATE OR REPLACE VIEW audit_event_summary AS
SELECT
    DATE(created_at) as date,
    event_type,
    severity,
    service_name,
    COUNT(*) as total_events,
    COUNT(CASE WHEN success = true THEN 1 END) as successful_events,
    COUNT(CASE WHEN success = false THEN 1 END) as failed_events,
    AVG(execution_time_ms) as avg_execution_time,
    MAX(risk_score) as max_risk_score,
    COUNT(DISTINCT user_id) as unique_users
FROM audit_events
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY DATE(created_at), event_type, severity, service_name
ORDER BY date DESC, total_events DESC;

CREATE OR REPLACE VIEW compliance_violations AS
SELECT
    ae.*,
    jsonb_object_keys(ae.compliance_flags) as compliance_type
FROM audit_events ae
WHERE ae.compliance_flags IS NOT NULL
AND ae.success = false
AND ae.created_at >= CURRENT_TIMESTAMP - INTERVAL '90 days'
ORDER BY ae.created_at DESC;

CREATE OR REPLACE VIEW security_incidents AS
SELECT
    ae.*,
    ai.alert_message,
    ai.severity as alert_severity
FROM audit_events ae
JOIN alert_instances ai ON ai.events::jsonb ? ae.event_id
WHERE ae.event_type = 'security'
AND ae.created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
ORDER BY ae.created_at DESC;

CREATE OR REPLACE VIEW audit_user_activity AS
SELECT
    user_id,
    COUNT(*) as total_events,
    COUNT(CASE WHEN success = true THEN 1 END) as successful_events,
    COUNT(CASE WHEN success = false THEN 1 END) as failed_events,
    COUNT(DISTINCT DATE(created_at)) as active_days,
    MAX(created_at) as last_activity,
    AVG(risk_score) as avg_risk_score,
    MAX(risk_score) as max_risk_score
FROM audit_events
WHERE user_id IS NOT NULL
AND created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY user_id
ORDER BY total_events DESC;

-- =============================================================================
-- MONITORING METRICS TABLES
-- =============================================================================

-- Historical metrics snapshots
CREATE TABLE IF NOT EXISTS metrics_snapshots (
    id VARCHAR(100) PRIMARY KEY,
    snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    labels JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for metrics_snapshots table
CREATE INDEX IF NOT EXISTS idx_metrics_snapshots_snapshot_time ON metrics_snapshots(snapshot_time);
CREATE INDEX IF NOT EXISTS idx_metrics_snapshots_metric_type ON metrics_snapshots(metric_type);
CREATE INDEX IF NOT EXISTS idx_metrics_snapshots_metric_name ON metrics_snapshots(metric_name);

-- SLA compliance tracking
CREATE TABLE IF NOT EXISTS sla_compliance (
    id VARCHAR(100) PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    sla_metric VARCHAR(100) NOT NULL,
    target_value FLOAT NOT NULL,
    actual_value FLOAT NOT NULL,
    compliance_percentage FLOAT NOT NULL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for sla_compliance table
CREATE INDEX IF NOT EXISTS idx_sla_compliance_service_name ON sla_compliance(service_name);
CREATE INDEX IF NOT EXISTS idx_sla_compliance_period_start ON sla_compliance(period_start);
CREATE INDEX IF NOT EXISTS idx_sla_compliance_period_end ON sla_compliance(period_end);
CREATE INDEX IF NOT EXISTS idx_sla_compliance_status ON sla_compliance(status);

-- Performance baselines
CREATE TABLE IF NOT EXISTS performance_baselines (
    id VARCHAR(100) PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    baseline_value FLOAT NOT NULL,
    standard_deviation FLOAT NOT NULL,
    sample_size INTEGER NOT NULL,
    calculation_period_days INTEGER NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance_baselines table
CREATE INDEX IF NOT EXISTS idx_performance_baselines_metric_name ON performance_baselines(metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_baselines_last_updated ON performance_baselines(last_updated);

-- Default SLA configurations
INSERT INTO sla_compliance (id, service_name, sla_metric, target_value, actual_value, compliance_percentage, period_start, period_end, status) VALUES
('sla_agent_orchestrator_response_time', 'agent_orchestrator', 'response_time_p95', 500, 450, 100, CURRENT_TIMESTAMP - INTERVAL '30 days', CURRENT_TIMESTAMP, 'compliant'),
('sla_brain_factory_success_rate', 'brain_factory', 'success_rate', 99.5, 99.2, 99.7, CURRENT_TIMESTAMP - INTERVAL '30 days', CURRENT_TIMESTAMP, 'compliant'),
('sla_workflow_engine_throughput', 'workflow_engine', 'tasks_per_minute', 100, 95, 95, CURRENT_TIMESTAMP - INTERVAL '30 days', CURRENT_TIMESTAMP, 'warning')
ON CONFLICT (id) DO NOTHING;

-- Default performance baselines
INSERT INTO performance_baselines (id, metric_name, baseline_value, standard_deviation, sample_size, calculation_period_days) VALUES
('baseline_agent_active_count', 'agent_count_active', 5.0, 1.5, 100, 30),
('baseline_workflow_success_rate', 'workflow_success_rate', 0.95, 0.02, 100, 30),
('baseline_response_time_p95', 'response_time_p95', 450, 50, 100, 30),
('baseline_task_completion_rate', 'task_completion_rate', 0.98, 0.01, 100, 30)
ON CONFLICT (id) DO NOTHING;

-- Create views for monitoring analytics
CREATE OR REPLACE VIEW metrics_summary AS
SELECT
    metric_type,
    metric_name,
    COUNT(*) as total_snapshots,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value,
    STDDEV(metric_value) as std_deviation,
    MAX(snapshot_time) as last_updated
FROM metrics_snapshots
WHERE snapshot_time >= CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY metric_type, metric_name
ORDER BY metric_type, metric_name;

CREATE OR REPLACE VIEW service_performance AS
SELECT
    ms.metric_name,
    ms.avg_value,
    ms.std_deviation,
    pb.baseline_value,
    CASE
        WHEN pb.baseline_value IS NOT NULL AND ABS(ms.avg_value - pb.baseline_value) > (2 * pb.standard_deviation)
        THEN 'anomaly'
        WHEN pb.baseline_value IS NOT NULL AND ABS(ms.avg_value - pb.baseline_value) > pb.standard_deviation
        THEN 'warning'
        ELSE 'normal'
    END as status,
    ms.last_updated
FROM metrics_summary ms
LEFT JOIN performance_baselines pb ON ms.metric_name = pb.metric_name
WHERE ms.metric_type = 'agent_brain'
ORDER BY
    CASE
        WHEN pb.baseline_value IS NOT NULL AND ABS(ms.avg_value - pb.baseline_value) > (2 * pb.standard_deviation) THEN 1
        WHEN pb.baseline_value IS NOT NULL AND ABS(ms.avg_value - pb.baseline_value) > pb.standard_deviation THEN 2
        ELSE 3
    END,
    ms.metric_name;

CREATE OR REPLACE VIEW sla_status_summary AS
SELECT
    service_name,
    sla_metric,
    AVG(compliance_percentage) as avg_compliance,
    MIN(compliance_percentage) as min_compliance,
    MAX(compliance_percentage) as max_compliance,
    COUNT(CASE WHEN status = 'compliant' THEN 1 END) as compliant_periods,
    COUNT(CASE WHEN status = 'warning' THEN 1 END) as warning_periods,
    COUNT(CASE WHEN status = 'breach' THEN 1 END) as breach_periods,
    COUNT(*) as total_periods,
    MAX(period_end) as last_updated
FROM sla_compliance
WHERE period_end >= CURRENT_TIMESTAMP - INTERVAL '90 days'
GROUP BY service_name, sla_metric
ORDER BY service_name, sla_metric;

CREATE OR REPLACE VIEW system_health_score AS
SELECT
    CURRENT_TIMESTAMP as calculated_at,
    AVG(CASE
        WHEN status = 'compliant' THEN 100
        WHEN status = 'warning' THEN 75
        WHEN status = 'breach' THEN 25
        ELSE 50
    END) as overall_sla_score,
    AVG(CASE
        WHEN status = 'normal' THEN 100
        WHEN status = 'warning' THEN 75
        WHEN status = 'anomaly' THEN 25
        ELSE 50
    END) as performance_score,
    COUNT(CASE WHEN status IN ('compliant', 'normal') THEN 1 END) as healthy_metrics,
    COUNT(*) as total_metrics,
    ROUND(
        (
            AVG(CASE
                WHEN status = 'compliant' THEN 100
                WHEN status = 'warning' THEN 75
                WHEN status = 'breach' THEN 25
                ELSE 50
            END) +
            AVG(CASE
                WHEN status = 'normal' THEN 100
                WHEN status = 'warning' THEN 75
                WHEN status = 'anomaly' THEN 25
                ELSE 50
            END)
        ) / 2, 2
    ) as overall_health_score
FROM (
    SELECT 'sla' as metric_type, service_name as entity_name, status
    FROM sla_compliance
    WHERE period_end >= CURRENT_TIMESTAMP - INTERVAL '7 days'

    UNION ALL

    SELECT 'performance' as metric_type, metric_name as entity_name, status
    FROM service_performance
) combined_metrics;

-- =============================================================================
-- ERROR HANDLING TABLES
-- =============================================================================

-- Error records storage
CREATE TABLE IF NOT EXISTS error_records (
    id VARCHAR(100) PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    error_type VARCHAR(100) NOT NULL,
    stack_trace TEXT,
    category VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    recovery_strategy VARCHAR(50) NOT NULL,
    recovery_attempts INTEGER DEFAULT 0,
    max_recovery_attempts INTEGER DEFAULT 3,
    recovery_status VARCHAR(20) DEFAULT 'pending',
    root_cause TEXT,
    impact_assessment TEXT,
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    request_id VARCHAR(100),
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_time_seconds INTEGER
);

-- Create indexes for error_records table
CREATE INDEX IF NOT EXISTS idx_error_records_service_name ON error_records(service_name);
CREATE INDEX IF NOT EXISTS idx_error_records_category ON error_records(category);
CREATE INDEX IF NOT EXISTS idx_error_records_severity ON error_records(severity);
CREATE INDEX IF NOT EXISTS idx_error_records_recovery_status ON error_records(recovery_status);
CREATE INDEX IF NOT EXISTS idx_error_records_created_at ON error_records(created_at);
CREATE INDEX IF NOT EXISTS idx_error_records_user_id ON error_records(user_id);

-- Recovery action records
CREATE TABLE IF NOT EXISTS recovery_actions (
    id VARCHAR(100) PRIMARY KEY,
    error_id VARCHAR(100) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    action_details JSON,
    success BOOLEAN DEFAULT FALSE,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_time_seconds FLOAT,
    FOREIGN KEY (error_id) REFERENCES error_records(id)
);

-- Create indexes for recovery_actions table
CREATE INDEX IF NOT EXISTS idx_recovery_actions_error_id ON recovery_actions(error_id);
CREATE INDEX IF NOT EXISTS idx_recovery_actions_action_type ON recovery_actions(action_type);
CREATE INDEX IF NOT EXISTS idx_recovery_actions_success ON recovery_actions(success);
CREATE INDEX IF NOT EXISTS idx_recovery_actions_executed_at ON recovery_actions(executed_at);

-- Error pattern definitions
CREATE TABLE IF NOT EXISTS error_patterns (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    error_keywords JSON,
    category VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    recovery_strategy VARCHAR(50) NOT NULL,
    max_retries INTEGER DEFAULT 3,
    retry_delay_seconds INTEGER DEFAULT 5,
    alert_required BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for error_patterns table
CREATE INDEX IF NOT EXISTS idx_error_patterns_category ON error_patterns(category);
CREATE INDEX IF NOT EXISTS idx_error_patterns_severity ON error_patterns(severity);
CREATE INDEX IF NOT EXISTS idx_error_patterns_recovery_strategy ON error_patterns(recovery_strategy);

-- Insert default error patterns
INSERT INTO error_patterns (id, name, description, error_keywords, category, severity, recovery_strategy, max_retries, retry_delay_seconds, alert_required) VALUES
('network_timeout', 'Network Timeout', 'Network request timeout errors', '["timeout", "connection", "network", "unreachable"]', 'network', 'medium', 'retry', 3, 5, true),
('network_connection_refused', 'Connection Refused', 'Network connection refused errors', '["connection refused", "connection reset", "broken pipe"]', 'network', 'high', 'failover', 2, 10, true),
('database_connection', 'Database Connection Error', 'Database connection failures', '["connection", "database", "postgres", "sql", "pool"]', 'database', 'high', 'restart', 1, 30, true),
('database_deadlock', 'Database Deadlock', 'Database deadlock detection', '["deadlock", "lock", "concurrent", "serialization"]', 'database', 'medium', 'retry', 2, 10, false),
('authentication_failure', 'Authentication Failure', 'User authentication failures', '["authentication", "credentials", "login", "password"]', 'authentication', 'high', 'notification', 1, 0, true),
('validation_error', 'Validation Error', 'Input validation failures', '["validation", "invalid", "required", "format"]', 'validation', 'low', 'ignore', 0, 0, false),
('business_logic', 'Business Logic Error', 'Application business logic errors', '["business", "logic", "rule", "constraint"]', 'business_logic', 'medium', 'manual_intervention', 1, 0, true),
('external_service', 'External Service Error', 'Third-party service failures', '["external", "service", "api", "third-party"]', 'external_service', 'high', 'failover', 2, 15, true),
('resource_limit', 'Resource Limit Exceeded', 'Resource usage limits exceeded', '["limit", "quota", "capacity", "memory", "cpu"]', 'resource_limit', 'high', 'restart', 1, 60, true),
('configuration', 'Configuration Error', 'Configuration-related errors', '["configuration", "config", "setting", "parameter"]', 'configuration', 'high', 'manual_intervention', 1, 0, true),
('security_violation', 'Security Violation', 'Security-related errors and violations', '["security", "unauthorized", "forbidden", "access"]', 'security', 'critical', 'manual_intervention', 0, 0, true)
ON CONFLICT (id) DO NOTHING;

-- Create views for error analytics
CREATE OR REPLACE VIEW error_summary AS
SELECT
    DATE(created_at) as date,
    service_name,
    category,
    severity,
    COUNT(*) as total_errors,
    COUNT(CASE WHEN recovery_status = 'completed' THEN 1 END) as resolved_errors,
    COUNT(CASE WHEN recovery_status = 'failed' THEN 1 END) as failed_recoveries,
    AVG(recovery_attempts) as avg_recovery_attempts,
    AVG(resolution_time_seconds) as avg_resolution_time,
    MAX(created_at) as last_error
FROM error_records
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY DATE(created_at), service_name, category, severity
ORDER BY date DESC, total_errors DESC;

CREATE OR REPLACE VIEW error_trends AS
SELECT
    DATE_TRUNC('hour', created_at) as hour,
    service_name,
    category,
    severity,
    COUNT(*) as error_count,
    AVG(CASE WHEN resolution_time_seconds IS NOT NULL THEN resolution_time_seconds END) as avg_resolution_time
FROM error_records
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', created_at), service_name, category, severity
ORDER BY hour DESC;

CREATE OR REPLACE VIEW recovery_effectiveness AS
SELECT
    service_name,
    recovery_strategy,
    COUNT(*) as total_attempts,
    COUNT(CASE WHEN success = true THEN 1 END) as successful_recoveries,
    ROUND(
        (COUNT(CASE WHEN success = true THEN 1 END)::FLOAT / COUNT(*)) * 100, 2
    ) as success_rate,
    AVG(execution_time_seconds) as avg_execution_time,
    MAX(executed_at) as last_attempt
FROM recovery_actions ra
JOIN error_records er ON ra.error_id = er.id
GROUP BY service_name, recovery_strategy
ORDER BY service_name, success_rate DESC;

CREATE OR REPLACE VIEW error_correlations AS
SELECT
    e1.service_name,
    e1.category as primary_category,
    e1.severity as primary_severity,
    e2.category as correlated_category,
    e2.severity as correlated_severity,
    COUNT(*) as correlation_count,
    AVG(EXTRACT(EPOCH FROM (e2.created_at - e1.created_at))) as avg_time_gap_seconds
FROM error_records e1
JOIN error_records e2 ON e1.service_name = e2.service_name
    AND e1.id != e2.id
    AND e2.created_at BETWEEN e1.created_at AND e1.created_at + INTERVAL '1 hour'
    AND e1.category != e2.category
WHERE e1.created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY e1.service_name, e1.category, e1.severity, e2.category, e2.severity
HAVING COUNT(*) >= 3
ORDER BY correlation_count DESC;

CREATE OR REPLACE VIEW error_impact_analysis AS
SELECT
    service_name,
    category,
    severity,
    COUNT(*) as error_count,
    AVG(CASE WHEN resolution_time_seconds IS NOT NULL THEN resolution_time_seconds END) as avg_mttr_seconds,
    COUNT(DISTINCT DATE(created_at)) as affected_days,
    MAX(created_at) as last_occurrence,
    MIN(created_at) as first_occurrence,
    ROUND(
        EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / 86400, 2
    ) as error_span_days,
    CASE
        WHEN COUNT(*) > 100 THEN 'high_impact'
        WHEN COUNT(*) > 50 THEN 'medium_impact'
        ELSE 'low_impact'
    END as impact_level
FROM error_records
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '90 days'
GROUP BY service_name, category, severity
ORDER BY error_count DESC;

CREATE OR REPLACE VIEW system_error_health AS
SELECT
    CURRENT_TIMESTAMP as calculated_at,
    (SELECT COUNT(*) FROM error_records WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours') as total_errors_24h,
    (SELECT COUNT(*) FROM error_records WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours' AND severity = 'critical') as critical_errors_24h,
    (SELECT COUNT(*) FROM error_records WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours' AND recovery_status = 'pending') as unresolved_errors_24h,
    (SELECT ROUND(AVG(resolution_time_seconds), 2) FROM error_records WHERE resolved_at IS NOT NULL AND created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours') as avg_resolution_time_24h,
    (SELECT ROUND((COUNT(CASE WHEN recovery_status = 'completed' THEN 1 END)::FLOAT / COUNT(*)) * 100, 2) FROM error_records WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours') as recovery_success_rate_24h,
    CASE
        WHEN (SELECT COUNT(*) FROM error_records WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour' AND severity IN ('critical', 'high')) > 10 THEN 'critical'
        WHEN (SELECT COUNT(*) FROM error_records WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour' AND severity = 'medium') > 20 THEN 'warning'
        ELSE 'healthy'
    END as overall_health_status;

-- =============================================================================
-- END-TO-END TESTING TABLES
-- =============================================================================

-- Test execution records
CREATE TABLE IF NOT EXISTS test_executions (
    id VARCHAR(100) PRIMARY KEY,
    scenario VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds FLOAT,
    steps_completed JSON DEFAULT '[]'::json,
    errors JSON DEFAULT '[]'::json,
    metrics JSON DEFAULT '{}'::json,
    performance_data JSON DEFAULT '{}'::json,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for test_executions table
CREATE INDEX IF NOT EXISTS idx_test_executions_scenario ON test_executions(scenario);
CREATE INDEX IF NOT EXISTS idx_test_executions_status ON test_executions(status);
CREATE INDEX IF NOT EXISTS idx_test_executions_start_time ON test_executions(start_time);

-- Test suite execution records
CREATE TABLE IF NOT EXISTS test_suite_executions (
    id VARCHAR(100) PRIMARY KEY,
    status VARCHAR(20) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds FLOAT,
    tests_executed JSON DEFAULT '[]'::json,
    summary JSON DEFAULT '{}'::json,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for test_suite_executions table
CREATE INDEX IF NOT EXISTS idx_test_suite_executions_status ON test_suite_executions(status);
CREATE INDEX IF NOT EXISTS idx_test_suite_executions_start_time ON test_suite_executions(start_time);

-- UI test screenshots and logs
CREATE TABLE IF NOT EXISTS ui_test_artifacts (
    id VARCHAR(100) PRIMARY KEY,
    test_id VARCHAR(100) NOT NULL,
    artifact_type VARCHAR(20) NOT NULL, -- screenshot, video, log, html
    file_path VARCHAR(500) NOT NULL,
    step_name VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON DEFAULT '{}'::json,
    FOREIGN KEY (test_id) REFERENCES test_executions(id) ON DELETE CASCADE
);

-- Create indexes for ui_test_artifacts table
CREATE INDEX IF NOT EXISTS idx_ui_test_artifacts_test_id ON ui_test_artifacts(test_id);
CREATE INDEX IF NOT EXISTS idx_ui_test_artifacts_type ON ui_test_artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_ui_test_artifacts_timestamp ON ui_test_artifacts(timestamp);

-- Performance test results
CREATE TABLE IF NOT EXISTS performance_test_results (
    id VARCHAR(100) PRIMARY KEY,
    test_id VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(20),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON DEFAULT '{}'::json,
    FOREIGN KEY (test_id) REFERENCES test_executions(id) ON DELETE CASCADE
);

-- Create indexes for performance_test_results table
CREATE INDEX IF NOT EXISTS idx_performance_test_results_test_id ON performance_test_results(test_id);
CREATE INDEX IF NOT EXISTS idx_performance_test_results_metric ON performance_test_results(metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_test_results_timestamp ON performance_test_results(timestamp);

-- Test scenario definitions
CREATE TABLE IF NOT EXISTS test_scenarios (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    category VARCHAR(50) NOT NULL,
    estimated_duration_seconds INTEGER DEFAULT 60,
    required_services JSON DEFAULT '[]'::json,
    test_steps JSON DEFAULT '[]'::json,
    success_criteria JSON DEFAULT '[]'::json,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for test_scenarios table
CREATE INDEX IF NOT EXISTS idx_test_scenarios_category ON test_scenarios(category);
CREATE INDEX IF NOT EXISTS idx_test_scenarios_enabled ON test_scenarios(enabled);

-- Insert default test scenarios
INSERT INTO test_scenarios (id, name, description, category, estimated_duration_seconds, required_services, test_steps, success_criteria, enabled) VALUES
('ui_workflow_builder', 'UI Workflow Builder', 'Tests drag-and-drop workflow creation interface', 'ui', 120, '["agent-builder-ui"]', '["navigate_to_ui", "verify_canvas_load", "test_component_drag_drop", "validate_workflow_creation"]', '["page_load_success", "canvas_loaded", "workflow_created", "validation_passed"]', true),
('end_to_end_pipeline', 'End-to-End Pipeline', 'Tests complete agent lifecycle from creation to task execution', 'integration', 300, '["brain-factory", "agent-orchestrator", "deployment-pipeline"]', '["create_agent_config", "generate_agent", "register_agent", "execute_task", "validate_results"]', '["agent_creation_success", "agent_registration_success", "task_execution_success", "cleanup_success"]', true),
('performance_load_test', 'Performance Load Test', 'Tests system performance under concurrent load', 'performance', 120, '["all"]', '["initialize_load_test", "execute_concurrent_requests", "collect_performance_metrics", "analyze_results"]', '["target_throughput_achieved", "response_time_within_limits", "error_rate_below_threshold"]', true),
('integration_test', 'Integration Test', 'Validates service-to-service integrations', 'integration', 180, '["all"]', '["test_service_communications", "validate_data_flow", "check_health_endpoints", "verify_consistency"]', '["all_services_communicating", "data_flow_valid", "health_checks_pass", "consistency_maintained"]', true),
('error_recovery_test', 'Error Recovery Test', 'Tests error handling and recovery mechanisms', 'resilience', 240, '["error-handling-service", "monitoring-metrics-service"]', '["simulate_service_failures", "trigger_error_conditions", "monitor_recovery_process", "validate_system_stability"]', '["errors_detected", "recovery_initiated", "system_stabilized", "minimal_data_loss"]', true),
('data_flow_validation', 'Data Flow Validation', 'Tests end-to-end data processing pipeline', 'data', 150, '["ingestion-services", "output-services", "qdrant-vector"]', '["ingest_test_data", "process_data_pipeline", "validate_transformations", "verify_output_storage"]', '["data_ingested_successfully", "transformations_applied", "output_stored_correctly", "data_integrity_maintained"]', true),
('security_test', 'Security Test', 'Validates security controls and authentication', 'security', 90, '["authentication-service", "audit-logging-service"]', '["test_authentication_flows", "validate_authorization", "check_security_headers", "audit_log_verification"]', '["authentication_works", "authorization_enforced", "security_headers_present", "audit_logs_complete"]', true)
ON CONFLICT (id) DO NOTHING;

-- Create views for test analytics
CREATE OR REPLACE VIEW test_execution_summary AS
SELECT
    DATE(start_time) as date,
    scenario,
    COUNT(*) as total_tests,
    COUNT(CASE WHEN status = 'passed' THEN 1 END) as passed_tests,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tests,
    COUNT(CASE WHEN status = 'error' THEN 1 END) as error_tests,
    ROUND(AVG(duration_seconds), 2) as avg_duration_seconds,
    ROUND(
        (COUNT(CASE WHEN status = 'passed' THEN 1 END)::FLOAT / COUNT(*)) * 100, 2
    ) as success_rate,
    MAX(start_time) as last_execution
FROM test_executions
WHERE start_time >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY DATE(start_time), scenario
ORDER BY date DESC, scenario;

CREATE OR REPLACE VIEW test_performance_trends AS
SELECT
    DATE_TRUNC('hour', start_time) as hour,
    scenario,
    COUNT(*) as tests_executed,
    ROUND(AVG(duration_seconds), 2) as avg_duration,
    ROUND(
        (COUNT(CASE WHEN status = 'passed' THEN 1 END)::FLOAT / COUNT(*)) * 100, 2
    ) as success_rate,
    ROUND(AVG((metrics->>'response_time_avg')::FLOAT), 2) as avg_response_time,
    ROUND(AVG((metrics->>'error_rate')::FLOAT), 4) as avg_error_rate
FROM test_executions
WHERE start_time >= CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', start_time), scenario
ORDER BY hour DESC;

CREATE OR REPLACE VIEW test_failure_analysis AS
SELECT
    scenario,
    jsonb_array_elements(errors) as error_details,
    COUNT(*) as occurrence_count,
    MAX(start_time) as last_occurrence,
    MIN(start_time) as first_occurrence
FROM test_executions
WHERE status IN ('failed', 'error')
    AND start_time >= CURRENT_TIMESTAMP - INTERVAL '30 days'
    AND jsonb_array_length(errors) > 0
GROUP BY scenario, jsonb_array_elements(errors)
ORDER BY occurrence_count DESC, scenario;

CREATE OR REPLACE VIEW test_scenario_health AS
SELECT
    ts.id,
    ts.name,
    ts.category,
    ts.estimated_duration_seconds,
    COALESCE(te.recent_success_rate, 0) as recent_success_rate,
    COALESCE(te.avg_execution_time, 0) as avg_execution_time,
    COALESCE(te.last_execution, NULL) as last_execution,
    CASE
        WHEN te.last_execution IS NULL THEN 'never_executed'
        WHEN te.last_execution < CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 'stale'
        WHEN te.recent_success_rate >= 95 THEN 'healthy'
        WHEN te.recent_success_rate >= 80 THEN 'warning'
        ELSE 'unhealthy'
    END as health_status
FROM test_scenarios ts
LEFT JOIN (
    SELECT
        scenario,
        ROUND(
            (COUNT(CASE WHEN status = 'passed' THEN 1 END)::FLOAT / COUNT(*)) * 100, 2
        ) as recent_success_rate,
        ROUND(AVG(duration_seconds), 2) as avg_execution_time,
        MAX(start_time) as last_execution
    FROM test_executions
    WHERE start_time >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    GROUP BY scenario
) te ON ts.id = te.scenario
WHERE ts.enabled = true
ORDER BY
    CASE
        WHEN te.last_execution IS NULL THEN 1
        WHEN te.recent_success_rate < 80 THEN 2
        WHEN te.last_execution < CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 3
        ELSE 4
    END,
    ts.category,
    ts.name;

CREATE OR REPLACE VIEW test_system_load_impact AS
SELECT
    DATE_TRUNC('hour', te.start_time) as hour,
    COUNT(*) as concurrent_tests,
    ROUND(AVG(te.duration_seconds), 2) as avg_test_duration,
    ROUND(AVG((te.performance_data->>'system_cpu_percent')::FLOAT), 2) as avg_system_cpu,
    ROUND(AVG((te.performance_data->>'system_memory_percent')::FLOAT), 2) as avg_system_memory,
    ROUND(AVG((te.metrics->>'response_time_avg')::FLOAT), 2) as avg_response_time,
    ROUND(AVG((te.metrics->>'throughput_rps')::FLOAT), 2) as avg_throughput
FROM test_executions te
WHERE te.start_time >= CURRENT_TIMESTAMP - INTERVAL '30 days'
    AND te.performance_data IS NOT NULL
GROUP BY DATE_TRUNC('hour', te.start_time)
ORDER BY hour DESC;

CREATE OR REPLACE VIEW test_coverage_analysis AS
SELECT
    'agent_services' as coverage_area,
    COUNT(DISTINCT CASE WHEN required_services::text LIKE '%agent%' THEN id END) as covered_services,
    COUNT(DISTINCT id) as total_scenarios,
    ROUND(
        (COUNT(DISTINCT CASE WHEN required_services::text LIKE '%agent%' THEN id END)::FLOAT /
         NULLIF(COUNT(DISTINCT id), 0)) * 100, 2
    ) as coverage_percentage
FROM test_scenarios
WHERE enabled = true

UNION ALL

SELECT
    'ui_components' as coverage_area,
    COUNT(DISTINCT CASE WHEN category = 'ui' THEN id END) as covered_services,
    COUNT(DISTINCT id) as total_scenarios,
    ROUND(
        (COUNT(DISTINCT CASE WHEN category = 'ui' THEN id END)::FLOAT /
         NULLIF(COUNT(DISTINCT id), 0)) * 100, 2
    ) as coverage_percentage
FROM test_scenarios
WHERE enabled = true

UNION ALL

SELECT
    'data_flow' as coverage_area,
    COUNT(DISTINCT CASE WHEN category = 'data' THEN id END) as covered_services,
    COUNT(DISTINCT id) as total_scenarios,
    ROUND(
        (COUNT(DISTINCT CASE WHEN category = 'data' THEN id END)::FLOAT /
         NULLIF(COUNT(DISTINCT id), 0)) * 100, 2
    ) as coverage_percentage
FROM test_scenarios
WHERE enabled = true

UNION ALL

SELECT
    'security' as coverage_area,
    COUNT(DISTINCT CASE WHEN category = 'security' THEN id END) as covered_services,
    COUNT(DISTINCT id) as total_scenarios,
    ROUND(
        (COUNT(DISTINCT CASE WHEN category = 'security' THEN id END)::FLOAT /
         NULLIF(COUNT(DISTINCT id), 0)) * 100, 2
    ) as coverage_percentage
FROM test_scenarios
WHERE enabled = true;
