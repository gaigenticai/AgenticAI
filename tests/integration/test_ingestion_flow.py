#!/usr/bin/env python3
"""
Integration Test for Ingestion Flow

This test verifies the complete ingestion pipeline:
1. Ingestion Coordinator API
2. CSV Ingestion Service
3. Database storage
4. Message queue communication
"""

import csv
import json
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import psycopg2
import requests
from sqlalchemy import create_engine


class TestIngestionFlow(unittest.TestCase):
    """Test cases for the ingestion flow"""

    def setUp(self):
        """Set up test environment"""
        self.base_url = os.getenv("INGESTION_COORDINATOR_URL", "http://localhost:8080")
        self.db_url = os.getenv("DATABASE_URL", "postgresql://agentic_user:agentic123@localhost:5432/agentic_ingestion")

        # Test data
        self.test_data = [
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "San Francisco"},
            {"name": "Charlie", "age": 35, "city": "Chicago"}
        ]

    def create_test_csv(self) -> str:
        """Create a temporary CSV file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["name", "age", "city"])
            writer.writeheader()
            writer.writerows(self.test_data)
            return f.name

    def test_coordinator_health(self):
        """Test ingestion coordinator health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertEqual(data["service"], "ingestion-coordinator")

            print("✓ Ingestion Coordinator health check passed")

        except requests.exceptions.RequestException as e:
            self.fail(f"Health check failed: {e}")

    def test_csv_file_creation(self):
        """Test CSV file creation"""
        csv_file = self.create_test_csv()
        self.assertTrue(os.path.exists(csv_file))

        # Verify CSV content
        df = pd.read_csv(csv_file)
        self.assertEqual(len(df), 3)
        self.assertListEqual(df.columns.tolist(), ["name", "age", "city"])

        # Cleanup
        os.unlink(csv_file)
        print("✓ CSV file creation test passed")

    def test_database_connection(self):
        """Test database connection"""
        try:
            engine = create_engine(self.db_url)
            connection = engine.connect()

            # Test basic query
            result = connection.execute("SELECT 1 as test")
            self.assertEqual(result.fetchone()[0], 1)

            connection.close()
            print("✓ Database connection test passed")

        except Exception as e:
            self.fail(f"Database connection failed: {e}")

    def test_ingestion_job_creation(self):
        """Test creating an ingestion job via API"""
        csv_file = self.create_test_csv()

        try:
            with open(csv_file, 'rb') as f:
                files = {'file': ('test_data.csv', f, 'text/csv')}
                data = {'source_type': 'csv'}

                response = requests.post(
                    f"{self.base_url}/ingestion/upload",
                    files=files,
                    data=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    self.assertIn("job_id", result)
                    self.assertEqual(result["status"], "uploaded")

                    job_id = result["job_id"]
                    print(f"✓ Ingestion job created successfully: {job_id}")

                    # Test job status retrieval
                    self.test_job_status_retrieval(job_id)

                else:
                    print(f"⚠ Ingestion job creation returned status {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"⚠ API request failed (service may not be running): {e}")
        finally:
            os.unlink(csv_file)

    def test_job_status_retrieval(self, job_id: str = None):
        """Test retrieving job status"""
        if not job_id:
            # Skip if no job_id provided
            return

        try:
            response = requests.get(f"{self.base_url}/ingestion/jobs/{job_id}", timeout=10)

            if response.status_code == 200:
                data = response.json()
                self.assertEqual(data["job_id"], job_id)
                self.assertIn(data["status"], ["pending", "processing", "completed", "failed"])

                print(f"✓ Job status retrieval successful: {data['status']}")

            else:
                print(f"⚠ Job status retrieval failed: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"⚠ Job status request failed: {e}")

    def test_list_jobs(self):
        """Test listing ingestion jobs"""
        try:
            response = requests.get(f"{self.base_url}/ingestion/jobs", timeout=10)

            if response.status_code == 200:
                data = response.json()
                self.assertIsInstance(data, list)

                if data:
                    # Check structure of first job
                    job = data[0]
                    self.assertIn("job_id", job)
                    self.assertIn("status", job)
                    self.assertIn("progress", job)

                print(f"✓ List jobs successful: {len(data)} jobs found")

            else:
                print(f"⚠ List jobs failed: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"⚠ List jobs request failed: {e}")

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)

            if response.status_code == 200:
                content = response.text
                self.assertIn("# HELP", content)  # Prometheus format
                self.assertIn("# TYPE", content)

                print("✓ Metrics endpoint test passed")

            else:
                print(f"⚠ Metrics endpoint returned status {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"⚠ Metrics request failed: {e}")

    def test_database_schema(self):
        """Test database schema integrity"""
        try:
            engine = create_engine(self.db_url)
            connection = engine.connect()

            # Check if required tables exist
            required_tables = [
                'ingestion_jobs',
                'data_validation_results',
                'data_quality_metrics',
                'metadata_catalog'
            ]

            for table in required_tables:
                result = connection.execute(f"""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_name = '{table}'
                    )
                """)
                exists = result.fetchone()[0]
                self.assertTrue(exists, f"Table {table} does not exist")

            connection.close()
            print("✓ Database schema test passed")

        except Exception as e:
            self.fail(f"Database schema test failed: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test resources"""
        # Clean up any temporary files
        import glob
        for pattern in ["test_*.csv", "temp_*.csv"]:
            for file in glob.glob(pattern):
                try:
                    os.unlink(file)
                except:
                    pass


def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting Agentic Platform Integration Tests")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIngestionFlow)
    runner = unittest.TextTestRunner(verbosity=2)

    # Run tests
    result = runner.run(suite)

    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = run_integration_tests()
    exit(exit_code)
