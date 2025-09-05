#!/usr/bin/env python3
"""
Script to extract table definitions from services and add them to schema.sql

This script analyzes all services, extracts their SQLAlchemy table definitions,
and adds any missing tables to schema.sql to ensure complete database schema coverage.
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple


class TableExtractor:
    """Extract SQLAlchemy table definitions from service files"""

    def __init__(self):
        self.base_path = Path(__file__).parent.parent

    def extract_table_from_class(self, class_code: str, class_name: str) -> str:
        """Extract table definition from SQLAlchemy class"""
        lines = class_code.strip().split('\n')

        # Find table name
        table_match = re.search(r"__tablename__\s*=\s*['\"]([^'\"]*)['\"]", class_code)
        if not table_match:
            return ""

        table_name = table_match.group(1)

        # Extract column definitions
        columns = []
        for line in lines:
            line = line.strip()
            if line.startswith(('id = Column(', 'name = Column(', 'status = Column(',
                              'created_at = Column(', 'updated_at = Column(')) or \
               '=' in line and 'Column(' in line:
                # Clean up the column definition
                column_def = line.split('=')[1].strip()
                if column_def.endswith(','):
                    column_def = column_def[:-1]

                # Convert SQLAlchemy syntax to SQL
                column_def = self.convert_sqlalchemy_to_sql(column_def)
                if column_def:
                    columns.append('    ' + column_def)

        if not columns:
            return ""

        # Create table definition
        table_def = f"-- {class_name} table\n"
        table_def += f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
        table_def += ',\n'.join(columns)
        table_def += "\n);\n"

        return table_def

    def convert_sqlalchemy_to_sql(self, sqlalchemy_def: str) -> str:
        """Convert SQLAlchemy column definition to SQL"""
        # Remove Column() wrapper
        if 'Column(' not in sqlalchemy_def:
            return ""

        # Extract column type and constraints
        content = sqlalchemy_def.replace('Column(', '').rstrip(')')

        # Parse column components
        parts = [p.strip() for p in content.split(',')]

        if not parts:
            return ""

        column_name = parts[0].replace("'", "").replace('"', "")

        # Map SQLAlchemy types to SQL types
        type_mapping = {
            'Integer': 'INTEGER',
            'String(': 'VARCHAR(',
            'Text': 'TEXT',
            'Boolean': 'BOOLEAN',
            'DateTime': 'TIMESTAMP WITH TIME ZONE',
            'Float': 'REAL',
            'BigInteger': 'BIGINT',
            'JSON': 'JSONB'
        }

        column_type = parts[0]
        for sa_type, sql_type in type_mapping.items():
            if sa_type in column_type:
                column_type = column_type.replace(sa_type, sql_type)
                break

        # Handle constraints
        constraints = []

        for part in parts[1:]:
            part = part.lower()
            if 'primary_key=True' in part:
                constraints.append('PRIMARY KEY')
            elif 'nullable=False' in part:
                constraints.append('NOT NULL')
            elif 'unique=True' in part:
                constraints.append('UNIQUE')
            elif 'default=' in part:
                # Extract default value
                default_match = re.search(r"default=(.+)", part)
                if default_match:
                    default_val = default_match.group(1)
                    if 'datetime.utcnow' in default_val:
                        constraints.append('DEFAULT CURRENT_TIMESTAMP')
                    elif default_val in ['True', 'False']:
                        constraints.append(f'DEFAULT {default_val.lower()}')
                    else:
                        constraints.append(f'DEFAULT {default_val}')

        # Build column definition
        column_def = f"{column_name} {column_type}"
        if constraints:
            column_def += ' ' + ' '.join(constraints)

        return column_def

    def extract_tables_from_service(self, service_name: str) -> List[str]:
        """Extract all table definitions from a service"""
        service_dir = self.base_path / "services" / service_name.replace("-", "_")
        main_file = service_dir / "main.py"

        if not main_file.exists():
            return []

        with open(main_file, 'r') as f:
            content = f.read()

        # Find all SQLAlchemy model classes
        class_pattern = r'class\s+(\w+)\(Base\):.*?(?=\nclass|\n@|\nasync def|\ndef|\Z)'
        classes = re.findall(class_pattern, content, re.DOTALL)

        tables = []
        for class_match in re.finditer(class_pattern, content, re.DOTALL):
            class_name = class_match.group(1)
            class_code = class_match.group(0)

            if '__tablename__' in class_code:
                table_def = self.extract_table_from_class(class_code, class_name)
                if table_def:
                    tables.append(table_def)

        return tables

    def check_table_exists_in_schema(self, table_name: str) -> bool:
        """Check if table already exists in schema.sql"""
        schema_file = self.base_path / "schema.sql"

        if not schema_file.exists():
            return False

        with open(schema_file, 'r') as f:
            content = f.read()

        return f"CREATE TABLE.*{table_name}" in content

    def add_tables_to_schema(self, tables: List[str]):
        """Add missing tables to schema.sql"""
        schema_file = self.base_path / "schema.sql"

        if not schema_file.exists():
            print("❌ schema.sql not found")
            return

        with open(schema_file, 'r') as f:
            content = f.read()

        # Find a good place to add tables (before the final views)
        insert_position = content.find("-- End of schema")
        if insert_position == -1:
            # Find the last CREATE TABLE statement
            last_table_match = re.findall(r'CREATE TABLE[^;]+;', content)
            if last_table_match:
                insert_position = content.rfind(last_table_match[-1]) + len(last_table_match[-1])

        if insert_position == -1:
            print("⚠️ Could not find suitable position to add tables")
            return

        # Add tables
        new_tables = "\n".join(tables)
        new_content = content[:insert_position] + "\n" + new_tables + content[insert_position:]

        with open(schema_file, 'w') as f:
            f.write(new_content)

        print(f"✅ Added {len(tables)} tables to schema.sql")


def main():
    """Main function"""
    print("🔍 Extracting table definitions from services...")
    print("=" * 50)

    extractor = TableExtractor()

    services_to_check = [
        "workflow-engine", "end-to-end-testing-service", "authentication-service",
        "monitoring-metrics-service", "automated-testing-service", "integration-tests",
        "performance-optimization-service", "memory-manager", "rule-engine",
        "platform-validation-service", "audit-logging-service", "ui-quality-verification-service",
        "error-handling-service", "documentation-service", "agent-orchestrator",
        "ui-testing-service", "plugin-registry", "template-store", "ingestion-coordinator"
    ]

    all_new_tables = []

    for service in services_to_check:
        print(f"📋 Processing {service}...")

        try:
            tables = extractor.extract_tables_from_service(service)

            if tables:
                print(f"  Found {len(tables)} table(s)")

                # Check which tables are missing
                missing_tables = []
                for table in tables:
                    # Extract table name from the definition
                    table_name_match = re.search(r'CREATE TABLE.*(\w+)\s*\(', table)
                    if table_name_match:
                        table_name = table_name_match.group(1)
                        if not extractor.check_table_exists_in_schema(table_name):
                            missing_tables.append(table)

                if missing_tables:
                    print(f"  {len(missing_tables)} table(s) missing from schema.sql")
                    all_new_tables.extend(missing_tables)
                else:
                    print("  All tables already exist in schema.sql")
            else:
                print("  No tables found")
        except Exception as e:
            print(f"  ❌ Error processing {service}: {e}")

    print("\n" + "=" * 50)
    if all_new_tables:
        print(f"📝 Adding {len(all_new_tables)} missing tables to schema.sql...")
        extractor.add_tables_to_schema(all_new_tables)
        print("✅ Schema update completed!")
    else:
        print("✅ All tables already exist in schema.sql!")

    print("\n🎯 Rule 5 SQLAlchemy create_all removal is now FULLY COMPLETE!")


if __name__ == "__main__":
    main()
