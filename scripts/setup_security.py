#!/usr/bin/env python3
"""
Security Setup Script for Agentic Platform

This script sets up automated security scanning and initializes
security monitoring for the platform.

Features:
- Initializes pre-commit hooks
- Creates baseline for secret detection
- Sets up security monitoring
- Validates security configuration

Usage:
    python scripts/setup_security.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def run_command(command, check=True, capture_output=False):
    """Run a shell command"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        print(f"Error: {e}")
        if not check:
            return e
        sys.exit(1)


def setup_pre_commit():
    """Setup pre-commit hooks"""
    print("🔧 Setting up pre-commit hooks...")

    if not Path('.pre-commit-config.yaml').exists():
        print("❌ .pre-commit-config.yaml not found")
        return False

    # Install pre-commit
    run_command("pip install pre-commit")

    # Install pre-commit hooks
    run_command("pre-commit install")

    # Run initial pre-commit check
    result = run_command("pre-commit run --all-files", check=False)

    if result.returncode == 0:
        print("✅ Pre-commit hooks setup successfully")
        return True
    else:
        print("⚠️ Some pre-commit hooks failed on initial run")
        print("This is normal for existing codebases")
        return True


def setup_detect_secrets():
    """Setup detect-secrets for secret detection"""
    print("🔐 Setting up detect-secrets...")

    try:
        # Install detect-secrets
        run_command("pip install detect-secrets")

        # Create baseline
        if not Path('.secrets.baseline').exists():
            print("📝 Creating secrets baseline...")
            run_command("detect-secrets scan --baseline .secrets.baseline")

        print("✅ Detect-secrets setup successfully")
        return True
    except Exception as e:
        print(f"⚠️ Detect-secrets setup failed: {e}")
        return False


def validate_security_config():
    """Validate security configuration"""
    print("🔍 Validating security configuration...")

    issues = []

    # Check if security scanner exists
    if not Path('scripts/security_scanner.py').exists():
        issues.append("Security scanner script not found")

    # Check if pre-commit config exists
    if not Path('.pre-commit-config.yaml').exists():
        issues.append("Pre-commit configuration not found")

    # Check if GitHub Actions workflow exists
    if not Path('.github/workflows/security-scan.yml').exists():
        issues.append("GitHub Actions security workflow not found")

    # Check for .env.example
    if not Path('.env.example').exists():
        issues.append("Environment template not found")
    else:
        # Check for insecure defaults in .env.example
        with open('.env.example', 'r') as f:
            content = f.read()

        if 'password=admin' in content or 'PASSWORD=admin' in content:
            issues.append("Insecure default password found in .env.example")

    if issues:
        print("❌ Security configuration issues found:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    else:
        print("✅ Security configuration is valid")
        return True


def run_initial_security_scan():
    """Run initial security scan"""
    print("🔍 Running initial security scan...")

    if not Path('scripts/security_scanner.py').exists():
        print("❌ Security scanner not found")
        return False

    try:
        result = run_command("python scripts/security_scanner.py --report initial_security_report.json", check=False)

        if result.returncode == 0:
            print("✅ Initial security scan completed successfully")
            return True
        else:
            print("⚠️ Initial security scan found issues")
            return True  # Don't fail setup for existing issues

    except Exception as e:
        print(f"⚠️ Initial security scan failed: {e}")
        return False


def create_security_documentation():
    """Create security documentation"""
    print("📝 Creating security documentation...")

    security_docs = """
# 🔒 Security Guidelines

## Automated Security Scanning

This project uses automated security scanning to detect potential security issues:

### Pre-commit Hooks
- Runs before every commit
- Checks for secrets, syntax errors, and code quality
- Use `pre-commit run --all-files` to run manually

### CI/CD Security Checks
- Runs on every push and pull request
- Scans for secrets and vulnerabilities
- Fails builds with critical findings

### Manual Security Scanning
```bash
# Scan for secrets
python scripts/security_scanner.py --secrets

# Full security scan
python scripts/security_scanner.py --full

# Generate report
python scripts/security_scanner.py --report security_report.json
```

## Security Best Practices

### 1. Never commit secrets
- Use environment variables for sensitive data
- Add sensitive files to .gitignore
- Use secure key management services

### 2. Use secure defaults
- Change default passwords before deployment
- Disable debug mode in production
- Use HTTPS in production

### 3. Regular security updates
- Keep dependencies updated
- Monitor for security advisories
- Run security scans regularly

## Security Contacts

For security issues, please contact:
- Security Team: security@agentic.ai
- Emergency: +1-555-0123

## Security Checklist

- [ ] No hardcoded secrets in source code
- [ ] Environment variables used for sensitive data
- [ ] Secure file permissions set
- [ ] Dependencies regularly updated
- [ ] Security scans passing
- [ ] Pre-commit hooks configured
"""

    with open('SECURITY.md', 'w') as f:
        f.write(security_docs)

    print("✅ Security documentation created")


def main():
    """Main setup function"""
    print("🚀 Agentic Platform Security Setup")
    print("=" * 50)

    success_count = 0
    total_steps = 5

    # Step 1: Validate security configuration
    if validate_security_config():
        success_count += 1

    # Step 2: Setup pre-commit hooks
    if setup_pre_commit():
        success_count += 1

    # Step 3: Setup detect-secrets
    if setup_detect_secrets():
        success_count += 1

    # Step 4: Run initial security scan
    if run_initial_security_scan():
        success_count += 1

    # Step 5: Create security documentation
    create_security_documentation()
    success_count += 1

    # Summary
    print("\n" + "=" * 50)
    print("📊 SECURITY SETUP SUMMARY")
    print("=" * 50)
    print(f"Completed: {success_count}/{total_steps} steps")

    if success_count == total_steps:
        print("🎉 Security setup completed successfully!")
        print("\nNext steps:")
        print("1. Review and fix any security findings in initial_security_report.json")
        print("2. Commit the security configuration: git add . && git commit -m 'Add security scanning'")
        print("3. Enable GitHub Actions workflow for CI/CD security checks")
        return 0
    else:
        print("⚠️ Security setup completed with some issues")
        print("Review the output above and fix any problems")
        return 1


if __name__ == "__main__":
    sys.exit(main())
