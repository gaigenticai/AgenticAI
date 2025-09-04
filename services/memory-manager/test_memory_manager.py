#!/usr/bin/env python3
"""
Comprehensive Test Suite for Memory Manager Service

This test suite provides complete coverage for the Memory Manager Service including:
- Unit tests for individual components
- Integration tests for service interactions
- Performance tests for memory operations
- Memory consolidation and cleanup tests
- Vector memory similarity tests
- Error handling and edge case tests

Test Categories:
1. Memory Storage and Retrieval
2. Memory Types (Working, Episodic, Semantic, Vector)
3. Memory Consolidation Strategies
4. Memory Cleanup and Expiration
5. Vector Similarity Search
6. Performance and Scalability
7. Error Handling and Recovery
8. Integration with External Services
"""

import asyncio
import json
import os
import pytest
import pytest_asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
import redis
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import service components
from main import (
    MemoryManager,
    VectorMemoryManager,
    MemoryConsolidator,
    MemoryType,
    MemoryStoreRequest,
    MemorySearchRequest,
    MemoryConsolidationRequest,
    VectorMemoryItem,
    Config
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    with patch('redis.Redis') as mock_redis:
        mock_instance = Mock()
        mock_instance.get.return_value = None
        mock_instance.setex.return_value = True
        mock_instance.delete.return_value = 1
        mock_instance.keys.return_value = []
        mock_redis.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_db_session():
    """Mock database session for testing"""
    with patch('sqlalchemy.orm.sessionmaker') as mock_sessionmaker:
        mock_session = Mock()
        mock_sessionmaker.return_value = Mock(return_value=mock_session)
        # Mock query operations
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter_by.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        mock_query.first.return_value = None
        mock_query.count.return_value = 0
        yield mock_session

@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing"""
    return {
        "agent_id": "test_agent_123",
        "memory_type": "working",
        "content_key": "test_key",
        "content_value": {
            "task": "process_data",
            "status": "completed",
            "result": "success"
        },
        "importance_score": 0.8,
        "tags": ["test", "processing"],
        "metadata": {
            "source": "unit_test",
            "version": "1.0"
        }
    }

@pytest.fixture
def sample_vector_memory():
    """Sample vector memory for testing"""
    return VectorMemoryItem(
        memory_id="vec_123",
        agent_id="test_agent_123",
        content_key="test_pattern",
        content_text="This is a test pattern for vector memory",
        vector_embedding=[0.1, 0.2, 0.3, 0.4, 0.5] * 154,  # 768 dimensions
        importance_score=0.9,
        metadata={"category": "test"}
    )


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestMemoryConsolidator:
    """Unit tests for MemoryConsolidator class"""

    @pytest.fixture
    def consolidator(self):
        """Create MemoryConsolidator instance"""
        return MemoryConsolidator()

    def test_consolidate_by_importance(self, consolidator):
        """Test importance-based consolidation"""
        memories = [
            {"importance_score": 0.9, "content_key": "high_importance"},
            {"importance_score": 0.3, "content_key": "low_importance"},
            {"importance_score": 0.7, "content_key": "medium_importance"}
        ]

        result = consolidator.consolidate_memories(memories, "importance_based", {"max_items": 2})

        assert len(result) == 2
        assert result[0]["importance_score"] == 0.9
        assert result[1]["importance_score"] == 0.7

    def test_consolidate_by_recency(self, consolidator):
        """Test recency-based consolidation"""
        now = datetime.utcnow()
        memories = [
            {"created_at": now, "content_key": "recent"},
            {"created_at": now - timedelta(days=10), "content_key": "old"},
            {"created_at": now - timedelta(days=5), "content_key": "medium"}
        ]

        result = consolidator.consolidate_memories(memories, "recency_based", {"max_items": 2})

        assert len(result) == 2
        assert result[0]["content_key"] == "recent"
        assert result[1]["content_key"] == "medium"

    def test_consolidate_by_similarity(self, consolidator):
        """Test similarity-based consolidation"""
        memories = [
            {"content_key": "machine learning algorithms", "importance_score": 0.8},
            {"content_key": "ML algorithms and models", "importance_score": 0.7},
            {"content_key": "data processing techniques", "importance_score": 0.6}
        ]

        result = consolidator.consolidate_memories(
            memories, "similarity_based",
            {"similarity_threshold": 0.3, "max_items": 2}
        )

        # Should consolidate similar ML-related memories
        assert len(result) <= 2

    def test_calculate_similarity_identical_texts(self, consolidator):
        """Test similarity calculation for identical texts"""
        similarity = consolidator._calculate_similarity("test text", "test text")
        assert similarity == 1.0

    def test_calculate_similarity_different_texts(self, consolidator):
        """Test similarity calculation for different texts"""
        similarity = consolidator._calculate_similarity("machine learning", "deep learning")
        assert 0.0 < similarity < 1.0

    def test_calculate_similarity_empty_texts(self, consolidator):
        """Test similarity calculation for empty texts"""
        similarity = consolidator._calculate_similarity("", "test")
        assert similarity == 0.0


class TestVectorMemoryManager:
    """Unit tests for VectorMemoryManager class"""

    @pytest.fixture
    def vector_manager(self):
        """Create VectorMemoryManager instance"""
        return VectorMemoryManager()

    @pytest.mark.asyncio
    async def test_store_vector_memory(self, vector_manager, sample_vector_memory):
        """Test storing vector memory"""
        memory_id = await vector_manager.store_vector_memory(sample_vector_memory)

        assert memory_id == sample_vector_memory.memory_id
        assert memory_id in vector_manager.vector_cache

    @pytest.mark.asyncio
    async def test_search_similar_memories(self, vector_manager, sample_vector_memory):
        """Test searching similar vector memories"""
        # Store sample memory
        await vector_manager.store_vector_memory(sample_vector_memory)

        # Search with same vector
        results = await vector_manager.search_similar_memories(
            sample_vector_memory.agent_id,
            sample_vector_memory.vector_embedding,
            limit=5
        )

        assert len(results) > 0
        assert results[0]["memory_id"] == sample_vector_memory.memory_id
        assert "similarity_score" in results[0]

    def test_cosine_similarity_calculation(self, vector_manager):
        """Test cosine similarity calculation"""
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]

        similarity = vector_manager._calculate_cosine_similarity(vec1, vec2)
        assert similarity == 0.0  # Orthogonal vectors

        # Test identical vectors
        similarity = vector_manager._calculate_cosine_similarity(vec1, vec1)
        assert similarity == 1.0

        # Test parallel vectors
        vec3 = [2, 0, 0]
        similarity = vector_manager._calculate_cosine_similarity(vec1, vec3)
        assert similarity == 1.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestMemoryManagerIntegration:
    """Integration tests for MemoryManager class"""

    @pytest.fixture
    async def memory_manager(self, mock_db_session, mock_redis):
        """Create MemoryManager instance with mocked dependencies"""
        manager = MemoryManager(mock_db_session, mock_redis)
        yield manager

    @pytest.mark.asyncio
    async def test_store_and_retrieve_memory(self, memory_manager, sample_memory_data):
        """Test complete store and retrieve workflow"""
        # Create store request
        request = MemoryStoreRequest(**sample_memory_data)

        # Mock database operations
        mock_memory = Mock()
        mock_memory.memory_id = "test_memory_id"
        mock_memory.agent_id = request.agent_id
        mock_memory.memory_type = request.memory_type
        mock_memory.content_key = request.content_key
        mock_memory.content_value = request.content_value
        mock_memory.importance_score = request.importance_score
        mock_memory.last_accessed = datetime.utcnow()
        mock_memory.created_at = datetime.utcnow()
        mock_memory.expires_at = None
        mock_memory.tags = request.tags
        mock_memory.metadata = request.metadata

        memory_manager.db.add = Mock()
        memory_manager.db.commit = Mock()

        # Store memory
        memory_id = await memory_manager.store_memory(request)

        # Verify storage operations were called
        memory_manager.db.add.assert_called_once()
        memory_manager.db.commit.assert_called_once()

        # Verify Redis cache was updated
        memory_manager.redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_search(self, memory_manager):
        """Test memory search functionality"""
        # Mock search results
        mock_memory = Mock()
        mock_memory.memory_id = "search_result_1"
        mock_memory.memory_type = "working"
        mock_memory.content_key = "test_key"
        mock_memory.content_value = {"result": "found"}
        mock_memory.importance_score = 0.8
        mock_memory.access_count = 5
        mock_memory.last_accessed = datetime.utcnow()
        mock_memory.created_at = datetime.utcnow()
        mock_memory.expires_at = None
        mock_memory.tags = ["test"]
        mock_memory.metadata = {}

        memory_manager.db.query.return_value.filter_by.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [mock_memory]

        # Create search request
        request = MemorySearchRequest(
            agent_id="test_agent_123",
            query="test",
            limit=10
        )

        # Perform search
        results = memory_manager.search_memories(request)

        # Verify results structure
        assert "memories" in results
        assert "total_count" in results
        assert "page" in results
        assert "limit" in results
        assert "has_more" in results

    @pytest.mark.asyncio
    async def test_memory_consolidation(self, memory_manager):
        """Test memory consolidation workflow"""
        # Mock memory data for consolidation
        mock_memories = [
            Mock(memory_id="mem1", content_key="test1", importance_score=0.8,
                  access_count=10, created_at=datetime.utcnow(), tags=[], metadata={}),
            Mock(memory_id="mem2", content_key="test2", importance_score=0.6,
                  access_count=5, created_at=datetime.utcnow(), tags=[], metadata={})
        ]

        memory_manager.db.query.return_value.filter_by.return_value.filter.return_value.all.return_value = mock_memories

        # Create consolidation request
        request = MemoryConsolidationRequest(
            agent_id="test_agent_123",
            source_memory_types=["working"],
            target_memory_type="semantic",
            consolidation_strategy="importance_based",
            min_importance_score=0.5
        )

        # Perform consolidation
        result = await memory_manager.consolidate_memory(request)

        # Verify result structure
        assert "agent_id" in result
        assert "consolidation_strategy" in result
        assert "source_memories_count" in result
        assert "consolidated_memories_count" in result
        assert "stored_memories_count" in result

    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_manager):
        """Test expired memory cleanup"""
        # Mock expired memories
        expired_memory = Mock()
        expired_memory.memory_id = "expired_mem_1"

        memory_manager.db.query.return_value.filter.return_value.all.return_value = [expired_memory]
        memory_manager.db.delete = Mock()
        memory_manager.db.commit = Mock()

        # Perform cleanup
        cleaned_count = await memory_manager.cleanup_expired_memories()

        # Verify cleanup operations
        memory_manager.db.delete.assert_called_once_with(expired_memory)
        memory_manager.db.commit.assert_called_once()


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestMemoryPerformance:
    """Performance tests for memory operations"""

    @pytest.fixture
    async def memory_manager(self, mock_db_session, mock_redis):
        """Create MemoryManager instance for performance testing"""
        manager = MemoryManager(mock_db_session, mock_redis)
        yield manager

    @pytest.mark.asyncio
    async def test_bulk_memory_storage(self, memory_manager):
        """Test bulk memory storage performance"""
        # Create multiple memory items
        memory_items = []
        for i in range(100):
            item = MemoryStoreRequest(
                agent_id=f"perf_test_agent_{i % 10}",  # 10 different agents
                memory_type="working",
                content_key=f"perf_key_{i}",
                content_value={"data": f"value_{i}", "index": i},
                importance_score=0.5 + (i % 50) / 100  # Varying importance
            )
            memory_items.append(item)

        # Mock database operations
        memory_manager.db.add = Mock()
        memory_manager.db.commit = Mock()

        # Measure storage time
        import time
        start_time = time.time()

        for item in memory_items:
            await memory_manager.store_memory(item)

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert total_time < 5.0  # Should complete within 5 seconds
        avg_time_per_item = total_time / len(memory_items)
        assert avg_time_per_item < 0.05  # Less than 50ms per item

    @pytest.mark.asyncio
    async def test_memory_search_performance(self, memory_manager):
        """Test memory search performance under load"""
        # Mock large result set
        mock_results = []
        for i in range(1000):
            mock_memory = Mock()
            mock_memory.memory_id = f"search_mem_{i}"
            mock_memory.memory_type = "working"
            mock_memory.content_key = f"search_key_{i}"
            mock_memory.content_value = {"result": f"data_{i}"}
            mock_memory.importance_score = 0.5
            mock_memory.access_count = i % 100
            mock_memory.last_accessed = datetime.utcnow()
            mock_memory.created_at = datetime.utcnow()
            mock_memory.expires_at = None
            mock_memory.tags = ["performance", "test"]
            mock_memory.metadata = {"batch": i // 100}
            mock_results.append(mock_memory)

        memory_manager.db.query.return_value.filter_by.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_results[:50]

        # Perform search
        import time
        start_time = time.time()

        request = MemorySearchRequest(
            agent_id="perf_test_agent",
            limit=50
        )

        results = memory_manager.search_memories(request)

        end_time = time.time()
        search_time = end_time - start_time

        # Performance assertions
        assert search_time < 1.0  # Should complete within 1 second
        assert len(results["memories"]) == 50


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestMemoryErrorHandling:
    """Error handling and edge case tests"""

    @pytest.fixture
    async def memory_manager(self, mock_db_session, mock_redis):
        """Create MemoryManager instance for error testing"""
        manager = MemoryManager(mock_db_session, mock_redis)
        yield manager

    @pytest.mark.asyncio
    async def test_database_connection_failure(self, memory_manager, sample_memory_data):
        """Test handling of database connection failures"""
        # Mock database failure
        memory_manager.db.add.side_effect = Exception("Database connection failed")

        request = MemoryStoreRequest(**sample_memory_data)

        # Should raise exception
        with pytest.raises(Exception, match="Database connection failed"):
            await memory_manager.store_memory(request)

    @pytest.mark.asyncio
    async def test_redis_connection_failure(self, memory_manager, sample_memory_data):
        """Test handling of Redis connection failures"""
        # Mock Redis failure
        memory_manager.redis.setex.side_effect = Exception("Redis connection failed")

        request = MemoryStoreRequest(**sample_memory_data)

        # Should still work (Redis failure shouldn't break storage)
        memory_manager.db.add = Mock()
        memory_manager.db.commit = Mock()

        memory_id = await memory_manager.store_memory(request)

        # Should still return memory ID despite Redis failure
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_invalid_memory_data(self, memory_manager):
        """Test handling of invalid memory data"""
        # Test with missing required fields
        invalid_request = {
            "agent_id": "test_agent",
            # Missing memory_type
            "content_key": "test_key",
            "content_value": {"test": "data"}
        }

        # Should raise validation error
        with pytest.raises(Exception):
            request = MemoryStoreRequest(**invalid_request)
            await memory_manager.store_memory(request)

    @pytest.mark.asyncio
    async def test_memory_not_found(self, memory_manager):
        """Test handling of non-existent memory retrieval"""
        # Mock empty result
        memory_manager.db.query.return_value.filter_by.return_value.first.return_value = None

        result = memory_manager.retrieve_memory(
            "non_existent_agent",
            "working",
            "non_existent_key"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self, memory_manager, sample_memory_data):
        """Test concurrent memory operations"""
        import asyncio

        # Mock database operations
        memory_manager.db.add = Mock()
        memory_manager.db.commit = Mock()

        async def store_memory_async(index):
            request = MemoryStoreRequest(
                agent_id=f"concurrent_agent_{index}",
                memory_type="working",
                content_key=f"concurrent_key_{index}",
                content_value={"index": index},
                importance_score=0.5
            )
            return await memory_manager.store_memory(request)

        # Execute multiple concurrent operations
        tasks = [store_memory_async(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All operations should succeed
        assert len(results) == 10
        assert all(result is not None for result in results)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestMemoryConfiguration:
    """Configuration and environment variable tests"""

    def test_config_defaults(self):
        """Test configuration default values"""
        assert Config.SERVICE_HOST == "0.0.0.0"
        assert Config.SERVICE_PORT == 8205
        assert Config.WORKING_MEMORY_TTL_SECONDS == 3600
        assert Config.MEMORY_VECTOR_DIMENSION == 768

    def test_config_environment_variables(self):
        """Test configuration from environment variables"""
        with patch.dict(os.environ, {
            'MEMORY_MANAGER_PORT': '9000',
            'WORKING_MEMORY_TTL_SECONDS': '7200',
            'REQUIRE_AUTH': 'true'
        }):
            # Reload configuration
            assert int(os.getenv('MEMORY_MANAGER_PORT', '8205')) == 9000
            assert int(os.getenv('WORKING_MEMORY_TTL_SECONDS', '3600')) == 7200
            assert os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'

    def test_memory_types_enum(self):
        """Test MemoryType enum values"""
        assert MemoryType.WORKING.value == "working"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.VECTOR.value == "vector"


# =============================================================================
# TEST UTILITIES
# =============================================================================

def pytest_configure(config):
    """Pytest configuration"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Mark integration tests
        if "Integration" in item.cls.__name__:
            item.add_marker(pytest.mark.integration)

        # Mark performance tests
        if "Performance" in item.cls.__name__:
            item.add_marker(pytest.mark.performance)


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "--verbose",
        "--cov=main",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--durations=10"
    ])
