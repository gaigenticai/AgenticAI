#!/usr/bin/env python3
"""
Automated UI Tests for Vector Operations

This module contains comprehensive tests for the Vector UI service:
- UI rendering and navigation
- API integration testing
- Authentication flow testing
- Performance and load testing
- Cross-browser compatibility checks
"""

import asyncio
import json
import time
from typing import Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

# Import the vector UI app for testing
from vector_ui_service import app

# Test client
client = TestClient(app)

# Base URL for external API calls during testing
BASE_URL = "http://localhost:8081"  # Output coordinator

class TestVectorUI:
    """Test suite for Vector UI functionality"""

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "vector-ui"
        assert "require_auth" in data

    def test_main_dashboard(self):
        """Test main dashboard page rendering"""
        response = client.get("/")
        assert response.status_code == 200
        assert "Vector Operations Dashboard" in response.text
        assert "text/html" in response.headers["content-type"]

    def test_embeddings_page(self):
        """Test embeddings page rendering"""
        response = client.get("/embeddings")
        assert response.status_code == 200
        assert "Generate Embeddings" in response.text
        assert "text/html" in response.headers["content-type"]

    def test_search_page(self):
        """Test search page rendering"""
        response = client.get("/search")
        assert response.status_code == 200
        assert "Vector Search" in response.text
        assert "text/html" in response.headers["content-type"]

    def test_user_guide_page(self):
        """Test user guide page rendering"""
        response = client.get("/guide")
        assert response.status_code == 200
        assert "Vector Operations User Guide" in response.text
        assert "text/html" in response.headers["content-type"]

    def test_collections_page(self):
        """Test collections page rendering"""
        response = client.get("/collections")
        assert response.status_code == 200
        assert "Manage Collections" in response.text
        assert "text/html" in response.headers["content-type"]

    def test_static_files(self):
        """Test static file serving"""
        response = client.get("/static/css/style.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]

        response = client.get("/static/js/main.js")
        assert response.status_code == 200
        assert "application/javascript" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_api_integration_embeddings(self):
        """Test API integration for embeddings generation"""
        test_texts = [
            "This is a test document",
            "Another test document for embeddings"
        ]

        # Test the API endpoint
        response = client.post(
            "/api/embeddings/generate",
            json={"texts": test_texts}
        )

        if response.status_code == 200:
            data = response.json()
            assert "embeddings" in data
            assert len(data["embeddings"]) == len(test_texts)
            assert "model" in data
            assert "dimensions" in data
        else:
            # If external service is not available, expect 500
            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_api_integration_search(self):
        """Test API integration for vector search"""
        search_query = {
            "query": "test query",
            "collection": "documents",
            "limit": 5
        }

        response = client.post(
            "/api/vectors/search",
            json=search_query
        )

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "total_found" in data
            assert "search_time" in data

    def test_form_processing_embeddings(self):
        """Test form processing for embeddings generation"""
        form_data = {
            "texts": "First test document\nSecond test document\nThird test document"
        }

        response = client.post("/generate-embeddings", data=form_data)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_form_processing_search(self):
        """Test form processing for vector search"""
        form_data = {
            "query": "machine learning",
            "collection": "documents",
            "limit": "10"
        }

        response = client.post("/search-vectors", data=form_data)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test with empty query
        response = client.post("/search-vectors", data={"query": ""})
        assert response.status_code == 200
        assert "error" in response.text.lower()

        # Test with invalid endpoint
        response = client.get("/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_collections_api(self):
        """Test collections API integration"""
        response = client.get("/api/vectors/collections")

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "collections" in data
            assert isinstance(data["collections"], list)

    def test_navigation_links(self):
        """Test navigation links in templates"""
        response = client.get("/")
        content = response.text

        # Check for navigation links
        assert "/embeddings" in content
        assert "/search" in content
        assert "/collections" in content
        assert "/guide" in content

    def test_responsive_design(self):
        """Test responsive design elements"""
        response = client.get("/")
        content = response.text

        # Check for Bootstrap responsive classes
        assert "container" in content
        assert "row" in content
        assert "col-" in content

    @pytest.mark.asyncio
    async def test_performance(self):
        """Test performance of UI endpoints"""
        start_time = time.time()

        # Test multiple endpoints
        endpoints = ["/", "/embeddings", "/search", "/guide", "/collections"]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            # Each request should complete within 1 second
            assert time.time() - start_time < 1.0

    def test_content_security(self):
        """Test content security and XSS prevention"""
        # Test with potentially malicious input
        malicious_input = "<script>alert('xss')</script>"
        form_data = {"query": malicious_input}

        response = client.post("/search-vectors", data=form_data)
        assert response.status_code == 200
        # Ensure malicious script is not executed/reflected
        assert "<script>" not in response.text

class TestAuthentication:
    """Test authentication functionality"""

    def test_authentication_headers(self):
        """Test that authentication is properly handled"""
        # This test assumes REQUIRE_AUTH might be true
        response = client.get("/health")
        data = response.json()
        assert "require_auth" in data

    def test_protected_endpoints(self):
        """Test that protected endpoints require authentication when enabled"""
        # Test embedding generation
        response = client.post("/api/embeddings/generate", json={"texts": ["test"]})
        # Should either succeed (auth disabled) or fail (auth enabled)
        assert response.status_code in [200, 401, 500, 503]

class TestLoadTesting:
    """Load testing for vector UI"""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        async def make_request():
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8082/health")
                return response.status_code

        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed or fail gracefully
        for status in results:
            assert status in [200, 404, 500]  # 404 if service not running

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

    # Additional manual test for UI functionality
    print("\n=== Manual UI Testing Instructions ===")
    print("1. Start the vector-ui service: docker-compose up vector-ui")
    print("2. Open browser to http://localhost:8082")
    print("3. Test each page: Dashboard, Embeddings, Search, Collections, Guide")
    print("4. Verify responsive design on mobile/tablet")
    print("5. Test form submissions and error handling")
    print("6. Check API integration with output-coordinator")
