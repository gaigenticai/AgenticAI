#!/usr/bin/env python3
"""
Automated Security Scanner for Agentic Platform

This script performs comprehensive security checks including:
1. Secret scanning in source code and configuration files
2. Security vulnerability detection
3. Compliance checks for security best practices
4. Automated security report generation

Features:
- Scans for hardcoded secrets, passwords, and API keys
- Checks for security misconfigurations
- Validates file permissions and access controls
- Generates security reports and alerts
- Integrates with CI/CD pipelines

Usage:
    python security_scanner.py [options] [scan_type]

Options:
    --full          Full security scan (default)
    --secrets       Scan for secrets only
    --config        Scan configuration files only
    --report FILE   Output report filename
    --fail-on-find  Exit with error if secrets found
    --verbose       Verbose output

Scan Types:
    source          Scan source code files
    config          Scan configuration files
    all             Scan everything (default)

Examples:
    python security_scanner.py --secrets --fail-on-find
    python security_scanner.py --full --report security_report.json
"""

import os
import re
import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import stat

# Security patterns to scan for
SECRET_PATTERNS = {
    'aws_access_key': re.compile(r'AKIA[0-9A-Z]{16}', re.IGNORECASE),
    'aws_secret_key': re.compile(r'(?i)aws_secret_access_key\s*[:=]\s*["\']?([A-Za-z0-9/+=]{40})["\']?', re.IGNORECASE),
    'generic_secret': re.compile(r'(?i)(secret|password|token|key)\s*[:=]\s*["\']?([A-Za-z0-9+/=]{20,})["\']?', re.IGNORECASE),
    'jwt_secret': re.compile(r'(?i)jwt.*secret\s*[:=]\s*["\']?([A-Za-z0-9+/=]{20,})["\']?', re.IGNORECASE),
    'private_key': re.compile(r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----', re.IGNORECASE),
    'api_key': re.compile(r'(?i)api[_-]?key\s*[:=]\s*["\']?([A-Za-z0-9+/=]{20,})["\']?', re.IGNORECASE),
    'database_url': re.compile(r'(?i)postgresql://([^:]+):([^@]+)@', re.IGNORECASE),
    'hardcoded_password': re.compile(r'password.*[:=].*["\'][^"\']{3,}["\']', re.IGNORECASE),
}

# Files to exclude from scanning
EXCLUDE_PATTERNS = [
    '.git/',
    '__pycache__/',
    'node_modules/',
    '*.pyc',
    '*.pyo',
    '*.log',
    '*.tmp',
    '.env*',
    'test-results/',
    'security_report*',
    '*.backup',
    '*.swp',
    '*.swo',
]

# File extensions to scan
SCAN_EXTENSIONS = [
    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
    '.php', '.rb', '.go', '.rs', '.sh', '.bash', '.zsh',
    '.yml', '.yaml', '.json', '.xml', '.sql', '.md',
    '.txt', '.conf', '.ini', '.cfg', '.properties'
]


@dataclass
class SecurityFinding:
    """Represents a security finding"""
    file_path: str
    line_number: int
    line_content: str
    finding_type: str
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'
    description: str
    pattern: str
    matched_text: str = ""
    recommendation: str = ""


@dataclass
class SecurityReport:
    """Comprehensive security report"""
    scan_start_time: datetime
    scan_end_time: Optional[datetime] = None
    total_files_scanned: int = 0
    findings: List[SecurityFinding] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def add_finding(self, finding: SecurityFinding):
        """Add a security finding"""
        self.findings.append(finding)

    def generate_summary(self):
        """Generate summary statistics"""
        severity_counts = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0,
            'INFO': 0
        }

        for finding in self.findings:
            severity_counts[finding.severity] += 1

        self.summary = {
            'total_findings': len(self.findings),
            'severity_breakdown': severity_counts,
            'scan_duration_seconds': (self.scan_end_time - self.scan_start_time).total_seconds() if self.scan_end_time else 0,
            'files_scanned': self.total_files_scanned
        }


class SecurityScanner:
    """Automated security scanner"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.report = SecurityReport(scan_start_time=datetime.now())
        self.base_path = Path.cwd()

    def should_scan_file(self, file_path: Path) -> bool:
        """Determine if a file should be scanned"""
        # Check exclude patterns
        file_str = str(file_path)
        for pattern in EXCLUDE_PATTERNS:
            if pattern in file_str or file_str.endswith(pattern):
                return False

        # Check file extension
        if file_path.suffix.lower() not in SCAN_EXTENSIONS:
            return False

        # Skip files larger than 10MB
        if file_path.stat().st_size > 10 * 1024 * 1024:
            return False

        return True

    def scan_file_for_secrets(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a file for potential secrets"""
        findings = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line_clean = line.strip()

                # Skip comments and empty lines
                if not line_clean or line_clean.startswith('#') or line_clean.startswith('//'):
                    continue

                for pattern_name, pattern in SECRET_PATTERNS.items():
                    matches = pattern.findall(line_clean)
                    if matches:
                        for match in matches:
                            finding = SecurityFinding(
                                file_path=str(file_path),
                                line_number=line_num,
                                line_content=line_clean,
                                finding_type='secret_pattern',
                                severity=self._determine_severity(pattern_name, match),
                                description=f"Potential {pattern_name.replace('_', ' ')} found",
                                pattern=pattern_name,
                                matched_text=str(match)[:50],  # Truncate for security
                                recommendation=self._get_recommendation(pattern_name)
                            )
                            findings.append(finding)

        except Exception as e:
            if self.verbose:
                print(f"Error scanning {file_path}: {e}")

        return findings

    def _determine_severity(self, pattern_name: str, match: str) -> str:
        """Determine severity level for a finding"""
        if pattern_name in ['private_key', 'aws_secret_key']:
            return 'CRITICAL'
        elif pattern_name in ['aws_access_key', 'database_url']:
            return 'HIGH'
        elif pattern_name in ['jwt_secret', 'api_key']:
            return 'HIGH'
        elif pattern_name in ['generic_secret', 'hardcoded_password']:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _get_recommendation(self, pattern_name: str) -> str:
        """Get security recommendation for a finding"""
        recommendations = {
            'aws_access_key': 'Use IAM roles or environment variables instead of hardcoded keys',
            'aws_secret_key': 'Store AWS secrets in AWS Secrets Manager or use IAM roles',
            'jwt_secret': 'Use environment variables or secure key management service',
            'private_key': 'Never commit private keys to version control',
            'database_url': 'Use environment variables for database credentials',
            'api_key': 'Store API keys in environment variables or secure vault',
            'generic_secret': 'Use environment variables or secure configuration management',
            'hardcoded_password': 'Use environment variables for passwords'
        }
        return recommendations.get(pattern_name, 'Review and remove sensitive information')

    def scan_configuration_security(self) -> List[SecurityFinding]:
        """Scan configuration files for security issues"""
        findings = []

        # Check .env files for security issues
        env_files = list(self.base_path.glob('.env*'))
        for env_file in env_files:
            if env_file.exists():
                findings.extend(self._check_env_file_security(env_file))

        # Check docker-compose.yml for security issues
        docker_compose = self.base_path / 'docker-compose.yml'
        if docker_compose.exists():
            findings.extend(self._check_docker_compose_security(docker_compose))

        # Check file permissions
        findings.extend(self._check_file_permissions())

        return findings

    def _check_env_file_security(self, env_file: Path) -> List[SecurityFinding]:
        """Check .env file for security issues"""
        findings = []

        try:
            with open(env_file, 'r') as f:
                content = f.read()

            # Check for insecure defaults
            if 'password=admin' in content or 'PASSWORD=admin' in content:
                findings.append(SecurityFinding(
                    file_path=str(env_file),
                    line_number=0,
                    line_content='',
                    finding_type='insecure_default',
                    severity='HIGH',
                    description='Insecure default password found',
                    pattern='insecure_default',
                    recommendation='Change default passwords before deployment'
                ))

            # Check for debug mode in production
            if 'DEBUG=true' in content or 'DEBUG=True' in content:
                findings.append(SecurityFinding(
                    file_path=str(env_file),
                    line_number=0,
                    line_content='',
                    finding_type='debug_enabled',
                    severity='MEDIUM',
                    description='Debug mode enabled in configuration',
                    pattern='debug_enabled',
                    recommendation='Disable debug mode in production'
                ))

        except Exception as e:
            findings.append(SecurityFinding(
                file_path=str(env_file),
                line_number=0,
                line_content='',
                finding_type='file_error',
                severity='LOW',
                description=f'Could not scan env file: {e}',
                pattern='file_error'
            ))

        return findings

    def _check_docker_compose_security(self, compose_file: Path) -> List[SecurityFinding]:
        """Check docker-compose.yml for security issues"""
        findings = []

        try:
            with open(compose_file, 'r') as f:
                content = f.read()

            # Check for privileged containers
            if 'privileged: true' in content:
                findings.append(SecurityFinding(
                    file_path=str(compose_file),
                    line_number=0,
                    line_content='',
                    finding_type='privileged_container',
                    severity='HIGH',
                    description='Privileged container found',
                    pattern='privileged_container',
                    recommendation='Avoid privileged containers in production'
                ))

            # Check for root user
            if 'user: root' in content or 'USER: root' in content:
                findings.append(SecurityFinding(
                    file_path=str(compose_file),
                    line_number=0,
                    line_content='',
                    finding_type='root_user',
                    severity='MEDIUM',
                    description='Container running as root user',
                    pattern='root_user',
                    recommendation='Use non-root user for better security'
                ))

        except Exception as e:
            findings.append(SecurityFinding(
                file_path=str(compose_file),
                line_number=0,
                line_content='',
                finding_type='file_error',
                severity='LOW',
                description=f'Could not scan docker-compose file: {e}',
                pattern='file_error'
            ))

        return findings

    def _check_file_permissions(self) -> List[SecurityFinding]:
        """Check file permissions for security issues"""
        findings = []

        # Check sensitive files
        sensitive_files = [
            '.env',
            '.env.local',
            '.env.production',
            'secrets/',
            'keys/',
            'private/',
            'ssl/',
            '*.key',
            '*.pem',
            '*.crt'
        ]

        for pattern in sensitive_files:
            for file_path in self.base_path.rglob(pattern):
                if file_path.is_file():
                    try:
                        stat_info = file_path.stat()
                        mode = stat.filemode(stat_info.st_mode)

                        # Check if file is world-readable
                        if stat_info.st_mode & 0o077:
                            findings.append(SecurityFinding(
                                file_path=str(file_path),
                                line_number=0,
                                line_content='',
                                finding_type='insecure_permissions',
                                severity='MEDIUM',
                                description=f'File has insecure permissions: {mode}',
                                pattern='insecure_permissions',
                                recommendation='Restrict file permissions (chmod 600)'
                            ))

                    except Exception as e:
                        findings.append(SecurityFinding(
                            file_path=str(file_path),
                            line_number=0,
                            line_content='',
                            finding_type='permission_check_error',
                            severity='LOW',
                            description=f'Could not check permissions: {e}',
                            pattern='permission_check_error'
                        ))

        return findings

    def scan_directory(self, scan_type: str = 'all') -> SecurityReport:
        """Scan directory for security issues"""
        print(f"🔍 Starting security scan (type: {scan_type})...")

        # Scan source files for secrets
        if scan_type in ['all', 'source']:
            print("📄 Scanning source files for secrets...")
            for file_path in self.base_path.rglob('*'):
                if file_path.is_file() and self.should_scan_file(file_path):
                    findings = self.scan_file_for_secrets(file_path)
                    for finding in findings:
                        self.report.add_finding(finding)
                    self.report.total_files_scanned += 1

        # Scan configuration files
        if scan_type in ['all', 'config']:
            print("⚙️ Scanning configuration files...")
            config_findings = self.scan_configuration_security()
            for finding in config_findings:
                self.report.add_finding(finding)

        self.report.scan_end_time = datetime.now()
        self.report.generate_summary()

        return self.report

    def print_report(self, report: SecurityReport):
        """Print security report to console"""
        print("\n" + "=" * 70)
        print("🔒 SECURITY SCAN REPORT")
        print("=" * 70)

        summary = report.summary
        print(f"Files Scanned:     {summary['files_scanned']}")
        print(f"Total Findings:    {summary['total_findings']}")
        print(f"Scan Duration:     {summary['scan_duration_seconds']:.2f}s")

        severity_breakdown = summary['severity_breakdown']
        print("\nSeverity Breakdown:")
        print(f"  Critical: {severity_breakdown['CRITICAL']} 🔴")
        print(f"  High:     {severity_breakdown['HIGH']} 🟠")
        print(f"  Medium:   {severity_breakdown['MEDIUM']} 🟡")
        print(f"  Low:      {severity_breakdown['LOW']} 🔵")
        print(f"  Info:     {severity_breakdown['INFO']} ℹ️")

        if report.findings:
            print("\n🚨 SECURITY FINDINGS:")
            for finding in report.findings[:10]:  # Show first 10 findings
                print(f"  {finding.severity}: {finding.description}")
                print(f"    File: {finding.file_path}:{finding.line_number}")
                if finding.recommendation:
                    print(f"    💡 {finding.recommendation}")
                print()

            if len(report.findings) > 10:
                print(f"  ... and {len(report.findings) - 10} more findings")

        # Security score
        critical_count = severity_breakdown['CRITICAL']
        high_count = severity_breakdown['HIGH']
        score = 100 - (critical_count * 20) - (high_count * 10) - (severity_breakdown['MEDIUM'] * 5)

        if score >= 90:
            print(f"🟢 Security Score: {score}/100 - Excellent")
        elif score >= 70:
            print(f"🟡 Security Score: {score}/100 - Good")
        elif score >= 50:
            print(f"🟠 Security Score: {score}/100 - Needs Attention")
        else:
            print(f"🔴 Security Score: {score}/100 - Critical Issues Found")

    def save_report(self, report: SecurityReport, filename: str = None) -> str:
        """Save security report to JSON file"""
        if filename is None:
            timestamp = int(datetime.now().timestamp())
            filename = f"security_report_{timestamp}.json"

        report_data = {
            'scan_start_time': report.scan_start_time.isoformat(),
            'scan_end_time': report.scan_end_time.isoformat() if report.scan_end_time else None,
            'total_files_scanned': report.total_files_scanned,
            'summary': report.summary,
            'findings': [
                {
                    'file_path': f.file_path,
                    'line_number': f.line_number,
                    'line_content': f.line_content,
                    'finding_type': f.finding_type,
                    'severity': f.severity,
                    'description': f.description,
                    'pattern': f.pattern,
                    'matched_text': f.matched_text,
                    'recommendation': f.recommendation
                }
                for f in report.findings
            ]
        }

        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"📄 Security report saved to: {filename}")
        return filename


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Automated Security Scanner")
    parser.add_argument('scan_type', nargs='?', default='all',
                       choices=['source', 'config', 'all'],
                       help='Type of scan to perform')
    parser.add_argument('--secrets', action='store_true',
                       help='Scan for secrets only')
    parser.add_argument('--config', action='store_true',
                       help='Scan configuration files only')
    parser.add_argument('--report', help='Output report filename')
    parser.add_argument('--fail-on-find', action='store_true',
                       help='Exit with error if secrets found')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Determine scan type
    if args.secrets:
        scan_type = 'source'
    elif args.config:
        scan_type = 'config'
    else:
        scan_type = args.scan_type

    # Initialize scanner
    scanner = SecurityScanner(verbose=args.verbose)

    try:
        # Perform scan
        report = scanner.scan_directory(scan_type)

        # Print report
        scanner.print_report(report)

        # Save report
        if args.report:
            scanner.save_report(report, args.report)

        # Determine exit code
        severity_breakdown = report.summary['severity_breakdown']
        critical_high_count = severity_breakdown['CRITICAL'] + severity_breakdown['HIGH']

        if args.fail_on_find and critical_high_count > 0:
            print("❌ Security scan failed: Critical/High severity findings detected")
            return 1
        elif critical_high_count > 0:
            print("⚠️ Security scan completed with findings")
            return 0
        else:
            print("✅ Security scan passed")
            return 0

    except Exception as e:
        print(f"💥 Security scan failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
