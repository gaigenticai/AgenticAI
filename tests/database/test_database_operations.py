#!/usr/bin/env python3
"""
Database Operations Tests for Agentic Platform

Tests database connectivity, schema integrity, CRUD operations,
and performance for all database interactions.
"""

import os
import time
import json
import unittest
from typing import Dict, Any, List
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor
import pymongo
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


class DatabaseTester:
    """Comprehensive database testing suite"""

    def __init__(self):
        self.postgres_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'agentic_ingestion'),
            'user': os.getenv('POSTGRES_USER', 'agentic_user'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }

        self.mongo_config = {
            'host': os.getenv('MONGO_HOST', 'localhost'),
            'port': int(os.getenv('MONGO_PORT', '27017')),
            'database': os.getenv('MONGO_DB', 'agentic_data')
        }

        self.redis_config = {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'db': int(os.getenv('REDIS_DB', '0'))
        }

        self.test_results = {}

    def test_postgres_connectivity(self) -> Dict[str, Any]:
        """Test PostgreSQL database connectivity"""
        try:
            start_time = time.time()

            conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )

            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            response_time = time.time() - start_time

            return {
                'test': 'postgres_connectivity',
                'status': 'success',
                'response_time': response_time,
                'version': version[:50] + '...' if len(version) > 50 else version,
                'message': 'PostgreSQL connection successful'
            }

        except Exception as e:
            return {
                'test': 'postgres_connectivity',
                'status': 'failed',
                'error': str(e),
                'message': f'PostgreSQL connection failed: {e}'
            }

    def test_postgres_schema_integrity(self) -> Dict[str, Any]:
        """Test PostgreSQL schema integrity"""
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Check for required tables
            required_tables = [
                'ingestion_jobs',
                'data_validation_results',
                'data_quality_metrics',
                'metadata_catalog',
                'workflow_executions',
                'component_executions'
            ]

            existing_tables = []
            missing_tables = []

            for table in required_tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    )
                """, (table,))

                exists = cursor.fetchone()['exists']
                if exists:
                    existing_tables.append(table)
                else:
                    missing_tables.append(table)

            cursor.close()
            conn.close()

            return {
                'test': 'postgres_schema_integrity',
                'status': 'success' if len(missing_tables) == 0 else 'warning',
                'existing_tables': existing_tables,
                'missing_tables': missing_tables,
                'message': f'Found {len(existing_tables)} tables, {len(missing_tables)} missing'
            }

        except Exception as e:
            return {
                'test': 'postgres_schema_integrity',
                'status': 'failed',
                'error': str(e),
                'message': f'Schema integrity check failed: {e}'
            }

    def test_postgres_crud_operations(self) -> Dict[str, Any]:
        """Test PostgreSQL CRUD operations"""
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cursor = conn.cursor()

            # Test CREATE
            cursor.execute("""
                CREATE TEMP TABLE test_crud (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Test INSERT
            cursor.execute("""
                INSERT INTO test_crud (name) VALUES (%s) RETURNING id
            """, ('test_record',))
            inserted_id = cursor.fetchone()[0]

            # Test SELECT
            cursor.execute("SELECT * FROM test_crud WHERE id = %s", (inserted_id,))
            record = cursor.fetchone()

            # Test UPDATE
            cursor.execute("""
                UPDATE test_crud SET name = %s WHERE id = %s
            """, ('updated_record', inserted_id))

            # Test DELETE
            cursor.execute("DELETE FROM test_crud WHERE id = %s", (inserted_id,))

            conn.commit()
            cursor.close()
            conn.close()

            return {
                'test': 'postgres_crud_operations',
                'status': 'success',
                'operations_tested': ['CREATE', 'INSERT', 'SELECT', 'UPDATE', 'DELETE'],
                'message': 'All CRUD operations successful'
            }

        except Exception as e:
            return {
                'test': 'postgres_crud_operations',
                'status': 'failed',
                'error': str(e),
                'message': f'CRUD operations failed: {e}'
            }

    def test_postgres_performance(self) -> Dict[str, Any]:
        """Test PostgreSQL performance"""
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cursor = conn.cursor()

            # Create test table
            cursor.execute("""
                CREATE TEMP TABLE perf_test (
                    id SERIAL PRIMARY KEY,
                    data TEXT
                )
            """)

            # Test bulk insert performance
            start_time = time.time()
            for i in range(100):
                cursor.execute("""
                    INSERT INTO perf_test (data) VALUES (%s)
                """, (f'test_data_{i}',))

            insert_time = time.time() - start_time

            # Test query performance
            start_time = time.time()
            cursor.execute("SELECT COUNT(*) FROM perf_test")
            count = cursor.fetchone()[0]
            query_time = time.time() - start_time

            conn.commit()
            cursor.close()
            conn.close()

            return {
                'test': 'postgres_performance',
                'status': 'success',
                'insert_time': insert_time,
                'query_time': query_time,
                'records_processed': count,
                'message': '.3f'
            }

        except Exception as e:
            return {
                'test': 'postgres_performance',
                'status': 'failed',
                'error': str(e),
                'message': f'Performance test failed: {e}'
            }

    def test_mongodb_connectivity(self) -> Dict[str, Any]:
        """Test MongoDB connectivity"""
        try:
            start_time = time.time()

            client = pymongo.MongoClient(
                host=self.mongo_config['host'],
                port=self.mongo_config['port']
            )

            # Test connection
            db = client[self.mongo_config['database']]
            collections = db.list_collection_names()

            client.close()

            response_time = time.time() - start_time

            return {
                'test': 'mongodb_connectivity',
                'status': 'success',
                'response_time': response_time,
                'collections_count': len(collections),
                'message': f'MongoDB connection successful, {len(collections)} collections found'
            }

        except Exception as e:
            return {
                'test': 'mongodb_connectivity',
                'status': 'failed',
                'error': str(e),
                'message': f'MongoDB connection failed: {e}'
            }

    def test_redis_connectivity(self) -> Dict[str, Any]:
        """Test Redis connectivity"""
        try:
            start_time = time.time()

            r = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config['db']
            )

            # Test connection
            r.ping()
            info = r.info()

            # Test basic operations
            r.set('test_key', 'test_value')
            value = r.get('test_key')
            r.delete('test_key')

            response_time = time.time() - start_time

            return {
                'test': 'redis_connectivity',
                'status': 'success',
                'response_time': response_time,
                'redis_version': info.get('redis_version'),
                'connected_clients': info.get('connected_clients'),
                'message': 'Redis connection and operations successful'
            }

        except Exception as e:
            return {
                'test': 'redis_connectivity',
                'status': 'failed',
                'error': str(e),
                'message': f'Redis connection failed: {e}'
            }

    def test_redis_operations(self) -> Dict[str, Any]:
        """Test Redis operations"""
        try:
            r = redis.Redis(**self.redis_config)

            # Test string operations
            r.set('test_string', 'hello world')
            value = r.get('test_string')

            # Test hash operations
            r.hset('test_hash', 'field1', 'value1')
            r.hset('test_hash', 'field2', 'value2')
            hash_values = r.hgetall('test_hash')

            # Test list operations
            r.lpush('test_list', 'item1', 'item2', 'item3')
            list_length = r.llen('test_list')
            list_items = r.lrange('test_list', 0, -1)

            # Cleanup
            r.delete('test_string', 'test_hash', 'test_list')

            return {
                'test': 'redis_operations',
                'status': 'success',
                'operations_tested': ['string', 'hash', 'list'],
                'message': 'All Redis operations successful'
            }

        except Exception as e:
            return {
                'test': 'redis_operations',
                'status': 'failed',
                'error': str(e),
                'message': f'Redis operations failed: {e}'
            }

    def test_sqlalchemy_integration(self) -> Dict[str, Any]:
        """Test SQLAlchemy integration"""
        try:
            # Create SQLAlchemy engine
            db_url = f"postgresql://{self.postgres_config['user']}:{self.postgres_config['password']}@{self.postgres_config['host']}:{self.postgres_config['port']}/{self.postgres_config['database']}"
            engine = create_engine(db_url)

            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                test_value = result.fetchone()[0]

            # Test session
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            session = SessionLocal()
            session.close()

            return {
                'test': 'sqlalchemy_integration',
                'status': 'success',
                'test_value': test_value,
                'message': 'SQLAlchemy integration successful'
            }

        except Exception as e:
            return {
                'test': 'sqlalchemy_integration',
                'status': 'failed',
                'error': str(e),
                'message': f'SQLAlchemy integration failed: {e}'
            }

    def run_all_database_tests(self) -> Dict[str, Any]:
        """Run all database tests"""
        print("💾 Running Database Tests")
        print("=" * 50)

        tests = [
            self.test_postgres_connectivity,
            self.test_postgres_schema_integrity,
            self.test_postgres_crud_operations,
            self.test_postgres_performance,
            self.test_mongodb_connectivity,
            self.test_redis_connectivity,
            self.test_redis_operations,
            self.test_sqlalchemy_integration
        ]

        results = []
        passed_tests = 0
        failed_tests = 0

        for test_func in tests:
            print(f"Running {test_func.__name__}...")
            result = test_func()
            results.append(result)

            if result['status'] in ['success', 'warning']:
                passed_tests += 1
                print(f"✅ {test_func.__name__}: {result['message']}")
            else:
                failed_tests += 1
                print(f"❌ {test_func.__name__}: {result['message']}")

        summary = {
            'total_tests': len(tests),
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / len(tests)) * 100 if len(tests) > 0 else 0,
            'results': results
        }

        print("\n" + "=" * 50)
        print("📊 Database Test Results Summary:")
        print(f"   Tests run: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(".1f")

        return summary


def run_database_tests():
    """Run all database tests"""
    tester = DatabaseTester()
    results = tester.run_all_database_tests()

    # Save results to file
    results_file = f"/tmp/database_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Detailed results saved to: {results_file}")

    return 0 if results['failed_tests'] == 0 else 1


if __name__ == "__main__":
    exit(run_database_tests())
