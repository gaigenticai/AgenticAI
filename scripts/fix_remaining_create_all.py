#!/usr/bin/env python3
"""
Script to systematically remove create_all calls from remaining services

This script identifies services that still have SQLAlchemy create_all calls
and removes them, ensuring all database schema management goes through schema.sql.
"""

import os
import re
from pathlib import Path


def find_services_with_create_all():
    """Find all services that still have create_all calls"""
    services_dir = Path(__file__).parent.parent / "services"
    services_with_create_all = []

    for service_dir in services_dir.iterdir():
        if service_dir.is_dir() and not service_dir.name.startswith('.'):
            main_file = service_dir / "main.py"
            if main_file.exists():
                with open(main_file, 'r') as f:
                    content = f.read()
                    if 'Base.metadata.create_all' in content and 'create_all(bind=' in content:
                        services_with_create_all.append(service_dir.name)

    return services_with_create_all


def remove_create_all_from_service(service_name):
    """Remove create_all call from a specific service"""
    services_dir = Path(__file__).parent.parent / "services"
    main_file = services_dir / service_name / "main.py"

    if not main_file.exists():
        print(f"❌ Main file not found for service: {service_name}")
        return False

    with open(main_file, 'r') as f:
        content = f.read()

    # Replace create_all calls
    original_content = content
    content = re.sub(
        r'Base\.metadata\.create_all\(bind=[^)]+\)',
        '# Base.metadata.create_all(bind=...)  # Removed - use schema.sql instead',
        content
    )

    if content != original_content:
        with open(main_file, 'w') as f:
            f.write(content)
        print(f"✅ Removed create_all from {service_name}")
        return True
    else:
        print(f"ℹ️  No create_all found in {service_name}")
        return False


def main():
    """Main function"""
    print("🔧 Fixing remaining create_all calls...")
    print("=" * 50)

    services_with_create_all = find_services_with_create_all()

    if not services_with_create_all:
        print("✅ No services with create_all calls found!")
        return

    print(f"Found {len(services_with_create_all)} services with create_all calls:")
    for service in services_with_create_all:
        print(f"  • {service}")

    print("\nRemoving create_all calls...")
    fixed_count = 0

    for service in services_with_create_all:
        if remove_create_all_from_service(service):
            fixed_count += 1

    print("\n" + "=" * 50)
    print(f"✅ Fixed {fixed_count} services")
    print(f"📋 Remaining services: {len(services_with_create_all) - fixed_count}")

    if fixed_count > 0:
        print("\n⚠️  IMPORTANT: Ensure all table definitions from these services")
        print("   are added to schema.sql before deploying!")


if __name__ == "__main__":
    main()
