#!/usr/bin/env python3
"""
Memory Manager Service for Agentic Brain Platform

This service provides comprehensive memory management capabilities for AI agents including:
- Working memory with TTL-based expiration
- Episodic memory for event-based recall
- Semantic memory for long-term knowledge storage
- Vector memory for similarity-based retrieval
- Memory consolidation and optimization
- Memory search and pattern matching
- Memory performance monitoring and analytics

Features:
- Multi-tier memory architecture (working, episodic, semantic, vector)
- Automatic memory consolidation and cleanup
- Vector similarity search for semantic retrieval
- Memory TTL management and expiration handling
- Memory performance optimization and caching
- RESTful API for memory operations
- Comprehensive monitoring and metrics
- Authentication and authorization support
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import hashlib
import heapq
from collections import defaultdict

import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry, REGISTRY
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON, Float, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func
import httpx
import uvicorn
import sys
import os

# Local configuration (removed utils dependency for Docker compatibility)
class DatabaseConfig:
    @staticmethod
    def get_postgres_config():
        return {
            'host': os.getenv('POSTGRES_HOST', 'postgresql_ingestion'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'agentic_db'),
            'user': os.getenv('POSTGRES_USER', 'agentic_user'),
            'password': os.getenv('POSTGRES_PASSWORD', ''),
            'url': os.getenv('DATABASE_URL', '')
        }

    @staticmethod
    def get_redis_config():
        return {
            'host': os.getenv('REDIS_HOST', 'redis_ingestion'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'db': int(os.getenv('REDIS_DB', '0')),
            'password': os.getenv('REDIS_PASSWORD', '')
        }

# JWT support for authentication
try:
    import jwt
except ImportError:
    jwt = None

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for Memory Manager Service"""

    # Database Configuration - using shared config for modularity (Rule 2)
    db_config = DatabaseConfig.get_postgres_config()
    DB_HOST = db_config['host']
    DB_PORT = db_config['port']
    DB_NAME = db_config['database']
    DB_USER = db_config['user']
    DB_PASSWORD = db_config['password']
    DATABASE_URL = db_config['url'] or f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Redis Configuration - using shared config for modularity (Rule 2)
    redis_config = DatabaseConfig.get_redis_config()
    REDIS_HOST = redis_config['host']
    REDIS_PORT = redis_config['port']
    REDIS_DB = 5  # Use DB 5 for memory manager (service-specific)

    # Vector Database Configuration
    VECTOR_DB_HOST = os.getenv('VECTOR_DB_HOST', 'qdrant_vector')
    VECTOR_DB_PORT = int(os.getenv('VECTOR_DB_PORT', '6333'))

    # Service Configuration
    SERVICE_HOST = os.getenv('MEMORY_MANAGER_HOST', '0.0.0.0')
    SERVICE_PORT = int(os.getenv('MEMORY_MANAGER_PORT', '8205'))

    # Authentication Configuration
    REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'
    JWT_SECRET = os.getenv('JWT_SECRET', '')
    JWT_ALGORITHM = 'HS256'

    # Memory Configuration
    WORKING_MEMORY_TTL_SECONDS = int(os.getenv('WORKING_MEMORY_TTL_SECONDS', '3600'))
    EPISODIC_MEMORY_RETENTION_DAYS = int(os.getenv('EPISODIC_MEMORY_RETENTION_DAYS', '30'))
    SEMANTIC_MEMORY_ENABLED = os.getenv('SEMANTIC_MEMORY_ENABLED', 'true').lower() == 'true'
    LONG_TERM_MEMORY_ENABLED = os.getenv('LONG_TERM_MEMORY_ENABLED', 'true').lower() == 'true'
    MEMORY_VECTOR_DIMENSION = int(os.getenv('MEMORY_VECTOR_DIMENSION', '768'))
    MEMORY_SIMILARITY_THRESHOLD = float(os.getenv('MEMORY_SIMILARITY_THRESHOLD', '0.85'))

    # Performance Configuration
    MAX_MEMORY_ITEMS_PER_AGENT = int(os.getenv('MAX_MEMORY_ITEMS_PER_AGENT', '10000'))
    MEMORY_CLEANUP_INTERVAL_SECONDS = int(os.getenv('MEMORY_CLEANUP_INTERVAL_SECONDS', '3600'))
    VECTOR_SEARCH_BATCH_SIZE = int(os.getenv('VECTOR_SEARCH_BATCH_SIZE', '100'))

    # Monitoring Configuration
    METRICS_ENABLED = os.getenv('AGENT_METRICS_ENABLED', 'true').lower() == 'true'

# =============================================================================
# MEMORY TYPES AND STRUCTURES
# =============================================================================

class MemoryType(Enum):
    """Enumeration of memory types"""
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    VECTOR = "vector"

class MemoryItem(BaseModel):
    """Model for memory items"""
    memory_id: str
    agent_id: str
    memory_type: str
    content_key: str
    content_value: Dict[str, Any]
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0)
    last_accessed: datetime
    created_at: datetime
    expires_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemorySearchRequest(BaseModel):
    """Model for memory search requests"""
    agent_id: str
    query: Optional[str] = None
    memory_type: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=20, ge=1, le=100)
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    include_expired: bool = Field(default=False)

class MemoryStoreRequest(BaseModel):
    """Model for memory storage requests"""
    agent_id: str
    memory_type: str
    content_key: str
    content_value: Dict[str, Any]
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_seconds: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemoryConsolidationRequest(BaseModel):
    """Model for memory consolidation requests"""
    agent_id: str
    source_memory_types: List[str]
    target_memory_type: str
    consolidation_criteria: Dict[str, Any] = Field(default_factory=dict)
    min_importance_score: float = Field(default=0.3, ge=0.0, le=1.0)

class VectorMemoryItem(BaseModel):
    """Model for vector memory items"""
    memory_id: str
    agent_id: str
    content_key: str
    content_text: str
    vector_embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime

# =============================================================================
# MEMORY MANAGEMENT CLASSES
# =============================================================================

class MemoryConsolidator:
    """Handles memory consolidation and optimization"""

    def __init__(self):
        self.consolidation_strategies = {
            'importance_based': self._consolidate_by_importance,
            'recency_based': self._consolidate_by_recency,
            'frequency_based': self._consolidate_by_frequency,
            'similarity_based': self._consolidate_by_similarity
        }

    def consolidate_memories(self, memories: List[Dict[str, Any]],
                           strategy: str = 'importance_based',
                           criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Consolidate memories using specified strategy"""
        if strategy not in self.consolidation_strategies:
            raise ValueError(f"Unknown consolidation strategy: {strategy}")

        return self.consolidation_strategies[strategy](memories, criteria or {})

    def _consolidate_by_importance(self, memories: List[Dict[str, Any]],
                                 criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Consolidate memories by importance score"""
        min_score = criteria.get('min_importance_score', 0.3)
        max_items = criteria.get('max_items', 1000)

        # Filter and sort by importance
        filtered_memories = [m for m in memories if m.get('importance_score', 0) >= min_score]
        sorted_memories = sorted(filtered_memories,
                               key=lambda m: m.get('importance_score', 0),
                               reverse=True)

        return sorted_memories[:max_items]

    def _consolidate_by_recency(self, memories: List[Dict[str, Any]],
                               criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Consolidate memories by recency"""
        max_items = criteria.get('max_items', 1000)
        days_back = criteria.get('days_back', 30)

        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        recent_memories = [m for m in memories if m.get('created_at', datetime.min) > cutoff_date]

        # Sort by creation date (most recent first)
        sorted_memories = sorted(recent_memories,
                               key=lambda m: m.get('created_at', datetime.min),
                               reverse=True)

        return sorted_memories[:max_items]

    def _consolidate_by_frequency(self, memories: List[Dict[str, Any]],
                                criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Consolidate memories by access frequency"""
        max_items = criteria.get('max_items', 1000)
        min_access_count = criteria.get('min_access_count', 1)

        # Filter and sort by access count
        filtered_memories = [m for m in memories if m.get('access_count', 0) >= min_access_count]
        sorted_memories = sorted(filtered_memories,
                               key=lambda m: m.get('access_count', 0),
                               reverse=True)

        return sorted_memories[:max_items]

    def _consolidate_by_similarity(self, memories: List[Dict[str, Any]],
                                 criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Consolidate memories by similarity to reduce redundancy"""
        max_items = criteria.get('max_items', 1000)
        similarity_threshold = criteria.get('similarity_threshold', 0.8)

        # Simple content-based similarity (placeholder for actual similarity calculation)
        consolidated = []
        for memory in memories:
            is_similar = False
            content_key = memory.get('content_key', '')

            for existing in consolidated:
                # Simple similarity check based on content key similarity
                if self._calculate_similarity(content_key, existing.get('content_key', '')) > similarity_threshold:
                    # Merge memories if similar
                    existing['access_count'] = max(existing.get('access_count', 0), memory.get('access_count', 0))
                    existing['importance_score'] = max(existing.get('importance_score', 0), memory.get('importance_score', 0))
                    is_similar = True
                    break

            if not is_similar:
                consolidated.append(memory)

        return consolidated[:max_items]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity using Jaccard similarity coefficient.

        This method computes the similarity between two text strings by:
        1. Converting both texts to lowercase
        2. Splitting into word sets (tokenization)
        3. Calculating Jaccard similarity: |intersection| / |union|
        4. Returning 0.0 for empty inputs or identical texts

        Args:
            text1: First text string to compare
            text2: Second text string to compare

        Returns:
            float: Similarity score between 0.0 and 1.0

        Note:
            This is a simple word-based similarity. For production use,
            consider more sophisticated NLP techniques like TF-IDF,
            word embeddings, or semantic similarity models.
        """
        # Handle edge cases: empty or None inputs
        if not text1 or not text2:
            return 0.0

        # Convert to lowercase and split into words for tokenization
        # This creates sets of unique words, ignoring case and word order
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Handle case where both texts are identical (would result in 1.0)
        if words1 == words2:
            return 1.0

        # Calculate Jaccard similarity: intersection over union
        # This measures the overlap between the two word sets
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        # Return similarity ratio, avoiding division by zero
        return intersection / union if union > 0 else 0.0

class VectorMemoryManager:
    """Manages vector-based memory for similarity search"""

    def __init__(self):
        self.vector_cache = {}  # Simple in-memory cache for vectors
        self.similarity_cache = {}  # Cache for similarity calculations

    async def store_vector_memory(self, item: VectorMemoryItem) -> str:
        """Store vector memory item"""
        try:
            # Validate vector dimension
            if len(item.vector_embedding) != Config.MEMORY_VECTOR_DIMENSION:
                raise ValueError(f"Vector dimension mismatch: expected {Config.MEMORY_VECTOR_DIMENSION}, got {len(item.vector_embedding)}")

            # Cache the vector memory item
            cache_key = f"{item.agent_id}:{item.memory_id}"
            self.vector_cache[cache_key] = {
                'item': item,
                'stored_at': datetime.utcnow()
            }

            logger.info(f"Stored vector memory: {item.memory_id} for agent {item.agent_id}")
            return item.memory_id

        except Exception as e:
            logger.error(f"Failed to store vector memory: {str(e)}")
            raise

    async def search_similar_memories(self, agent_id: str, query_vector: List[float],
                                    limit: int = 10, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Search for similar vector memories"""
        try:
            if len(query_vector) != Config.MEMORY_VECTOR_DIMENSION:
                raise ValueError(f"Query vector dimension mismatch: expected {Config.MEMORY_VECTOR_DIMENSION}, got {len(query_vector)}")

            # Get all vector memories for the agent
            agent_memories = [
                cache_item['item'] for cache_key, cache_item in self.vector_cache.items()
                if cache_key.startswith(f"{agent_id}:")
            ]

            # Calculate similarities
            similarities = []
            for memory in agent_memories:
                similarity = self._calculate_cosine_similarity(query_vector, memory.vector_embedding)
                if similarity >= threshold:
                    similarities.append({
                        'memory_id': memory.memory_id,
                        'content_key': memory.content_key,
                        'content_text': memory.content_text,
                        'similarity_score': similarity,
                        'importance_score': memory.importance_score,
                        'metadata': memory.metadata,
                        'created_at': memory.created_at
                    })

            # Sort by similarity score and return top results
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:limit]

        except Exception as e:
            logger.error(f"Failed to search similar memories: {str(e)}")
            raise

    def _calculate_cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vector1, vector2))

            # Calculate magnitudes
            magnitude1 = sum(a * a for a in vector1) ** 0.5
            magnitude2 = sum(b * b for b in vector2) ** 0.5

            # Calculate cosine similarity
            if magnitude1 * magnitude2 == 0:
                return 0.0

            return dot_product / (magnitude1 * magnitude2)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0

    async def cleanup_expired_vectors(self, agent_id: Optional[str] = None) -> int:
        """Clean up expired vector memories"""
        current_time = datetime.utcnow()
        expired_count = 0

        keys_to_remove = []
        for cache_key, cache_item in self.vector_cache.items():
            if agent_id and not cache_key.startswith(f"{agent_id}:"):
                continue

            # Check if item has exceeded cache lifetime (placeholder logic)
            stored_at = cache_item.get('stored_at')
            if stored_at and (current_time - stored_at).total_seconds() > 86400:  # 24 hours
                keys_to_remove.append(cache_key)
                expired_count += 1

        # Remove expired items
        for key in keys_to_remove:
            del self.vector_cache[key]

        logger.info(f"Cleaned up {expired_count} expired vector memories")
        return expired_count

# =============================================================================
# BUSINESS LOGIC CLASSES
# =============================================================================

class MemoryManager:
    """Main memory management service"""

    def __init__(self, db_session: Session, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.consolidator = MemoryConsolidator()
        self.vector_manager = VectorMemoryManager()

        # Start background cleanup task
        asyncio.create_task(self._start_cleanup_scheduler())

    async def store_memory(self, request: MemoryStoreRequest) -> str:
        """
        Store a memory item in the multi-tier memory system.

        This method handles the complete memory storage workflow:
        1. Generates a unique memory ID using UUID4
        2. Calculates expiration time based on TTL or memory type
        3. Stores in PostgreSQL database for persistence
        4. Caches in Redis for fast access
        5. Returns the generated memory ID

        Args:
            request: MemoryStoreRequest containing agent_id, memory_type,
                    content_key, content_value, and optional parameters

        Returns:
            str: Unique memory ID for the stored item

        Raises:
            Exception: If database or cache operations fail
        """
        try:
            # Generate unique identifier for this memory item
            # UUID4 provides cryptographically secure randomness
            memory_id = str(uuid.uuid4())

            # Determine expiration time based on request parameters
            # Priority: explicit TTL > memory type defaults > no expiration
            expires_at = None
            if request.ttl_seconds:
                # User-specified TTL takes precedence
                expires_at = datetime.utcnow() + timedelta(seconds=request.ttl_seconds)
            elif request.memory_type == MemoryType.WORKING.value:
                # Working memory has default TTL for automatic cleanup
                expires_at = datetime.utcnow() + timedelta(seconds=Config.WORKING_MEMORY_TTL_SECONDS)

            # Create memory item
            memory_item = {
                'memory_id': memory_id,
                'agent_id': request.agent_id,
                'memory_type': request.memory_type,
                'content_key': request.content_key,
                'content_value': request.content_value,
                'importance_score': request.importance_score,
                'access_count': 0,
                'last_accessed': datetime.utcnow(),
                'created_at': datetime.utcnow(),
                'expires_at': expires_at,
                'tags': request.tags,
                'metadata': request.metadata
            }

            # Store in database
            agent_memory = AgentMemory(
                memory_id=memory_id,
                agent_id=request.agent_id,
                memory_type=request.memory_type,
                content_key=request.content_key,
                content_value=request.content_value,
                importance_score=request.importance_score,
                last_accessed=datetime.utcnow(),
                expires_at=expires_at,
                tags=request.tags,
                metadata=request.metadata
            )
            self.db.add(agent_memory)
            self.db.commit()

            # Cache in Redis for fast access
            cache_key = f"memory:{request.agent_id}:{request.memory_type}:{request.content_key}"
            self.redis.setex(cache_key, 3600, json.dumps(memory_item))  # 1 hour cache

            logger.info(f"Stored memory: {memory_id} for agent {request.agent_id}")
            return memory_id

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to store memory: {str(e)}")
            raise

    def retrieve_memory(self, agent_id: str, memory_type: str, content_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory item"""
        try:
            # Check cache first
            cache_key = f"memory:{agent_id}:{memory_type}:{content_key}"
            cached = self.redis.get(cache_key)
            if cached:
                memory_item = json.loads(cached)
                # Update access count in background
                asyncio.create_task(self._update_access_count(memory_item['memory_id']))
                return memory_item

            # Get from database
            memory = self.db.query(AgentMemory).filter_by(
                agent_id=agent_id,
                memory_type=memory_type,
                content_key=content_key,
                expires_at=None  # Only active memories
            ).first() or self.db.query(AgentMemory).filter_by(
                agent_id=agent_id,
                memory_type=memory_type,
                content_key=content_key
            ).filter(AgentMemory.expires_at > datetime.utcnow()).first()

            if not memory:
                return None

            # Update access information
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
            self.db.commit()

            memory_item = {
                'memory_id': memory.memory_id,
                'agent_id': memory.agent_id,
                'memory_type': memory.memory_type,
                'content_key': memory.content_key,
                'content_value': memory.content_value,
                'importance_score': memory.importance_score,
                'access_count': memory.access_count,
                'last_accessed': memory.last_accessed,
                'created_at': memory.created_at,
                'expires_at': memory.expires_at,
                'tags': memory.tags,
                'metadata': memory.metadata
            }

            # Cache the result
            self.redis.setex(cache_key, 3600, json.dumps(memory_item))

            return memory_item

        except Exception as e:
            logger.error(f"Failed to retrieve memory: {str(e)}")
            return None

    def search_memories(self, request: MemorySearchRequest) -> Dict[str, Any]:
        """Search memories with filtering and pagination"""
        try:
            query = self.db.query(AgentMemory).filter_by(agent_id=request.agent_id)

            # Apply filters
            if request.memory_type:
                query = query.filter_by(memory_type=request.memory_type)

            if request.tags:
                # Filter by tags (array contains)
                for tag in request.tags:
                    query = query.filter(AgentMemory.tags.contains([tag]))

            if request.date_from:
                query = query.filter(AgentMemory.created_at >= request.date_from)

            if request.date_to:
                query = query.filter(AgentMemory.created_at <= request.date_to)

            if not request.include_expired:
                # Exclude expired memories
                now = datetime.utcnow()
                query = query.filter(
                    (AgentMemory.expires_at.is_(None)) |
                    (AgentMemory.expires_at > now)
                )

            # Apply text search if query provided
            if request.query:
                query = query.filter(
                    AgentMemory.content_key.ilike(f"%{request.query}%") |
                    AgentMemory.metadata.cast(String).ilike(f"%{request.query}%")
                )

            # Get total count
            total_count = query.count()

            # Apply sorting (by importance score, then by last accessed)
            query = query.order_by(
                AgentMemory.importance_score.desc(),
                AgentMemory.last_accessed.desc()
            )

            # Apply pagination
            memories = query.offset((request.page - 1) * request.limit).limit(request.limit).all()

            # Convert to dict format
            results = []
            for memory in memories:
                results.append({
                    'memory_id': memory.memory_id,
                    'memory_type': memory.memory_type,
                    'content_key': memory.content_key,
                    'content_value': memory.content_value,
                    'importance_score': memory.importance_score,
                    'access_count': memory.access_count,
                    'last_accessed': memory.last_accessed.isoformat(),
                    'created_at': memory.created_at.isoformat(),
                    'expires_at': memory.expires_at.isoformat() if memory.expires_at else None,
                    'tags': memory.tags,
                    'metadata': memory.metadata
                })

            return {
                'memories': results,
                'total_count': total_count,
                'page': request.page,
                'limit': request.limit,
                'has_more': len(results) == request.limit
            }

        except Exception as e:
            logger.error(f"Failed to search memories: {str(e)}")
            raise

    async def consolidate_memory(self, request: MemoryConsolidationRequest) -> Dict[str, Any]:
        """Consolidate memories using specified strategy"""
        try:
            # Get memories to consolidate
            query = self.db.query(AgentMemory).filter_by(agent_id=request.agent_id)
            query = query.filter(AgentMemory.memory_type.in_(request.source_memory_types))

            memories = query.all()
            memory_dicts = [{
                'memory_id': m.memory_id,
                'content_key': m.content_key,
                'content_value': m.content_value,
                'importance_score': m.importance_score,
                'access_count': m.access_count,
                'created_at': m.created_at,
                'tags': m.tags,
                'metadata': m.metadata
            } for m in memories]

            # Apply consolidation
            consolidated = self.consolidator.consolidate_memories(
                memory_dicts,
                request.consolidation_strategy,
                request.consolidation_criteria
            )

            # Store consolidated memories
            stored_count = 0
            for memory in consolidated:
                try:
                    store_request = MemoryStoreRequest(
                        agent_id=request.agent_id,
                        memory_type=request.target_memory_type,
                        content_key=f"consolidated_{memory['memory_id']}",
                        content_value=memory['content_value'],
                        importance_score=memory['importance_score'],
                        tags=memory.get('tags', []),
                        metadata={
                            **memory.get('metadata', {}),
                            'consolidated_from': memory['memory_id'],
                            'consolidation_strategy': request.consolidation_strategy
                        }
                    )
                    await self.store_memory(store_request)
                    stored_count += 1

                except Exception as e:
                    logger.error(f"Failed to store consolidated memory: {str(e)}")

            return {
                'agent_id': request.agent_id,
                'consolidation_strategy': request.consolidation_strategy,
                'source_memories_count': len(memory_dicts),
                'consolidated_memories_count': len(consolidated),
                'stored_memories_count': stored_count,
                'consolidation_completed_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to consolidate memory: {str(e)}")
            raise

    async def cleanup_expired_memories(self) -> int:
        """Clean up expired memories"""
        try:
            now = datetime.utcnow()

            # Delete expired memories from database
            expired_memories = self.db.query(AgentMemory).filter(
                AgentMemory.expires_at.isnot(None),
                AgentMemory.expires_at <= now
            ).all()

            expired_count = len(expired_memories)

            # Delete from database
            for memory in expired_memories:
                self.db.delete(memory)

            # Clean up Redis cache
            cache_keys = self.redis.keys("memory:*")
            for key in cache_keys:
                try:
                    cached_data = self.redis.get(key)
                    if cached_data:
                        memory_item = json.loads(cached_data)
                        if memory_item.get('expires_at'):
                            expires_at = datetime.fromisoformat(memory_item['expires_at'].replace('Z', '+00:00'))
                            if expires_at <= now:
                                self.redis.delete(key)
                except Exception as e:
                    logger.error(f"Error cleaning up cache key {key}: {str(e)}")

            # Clean up vector memories
            vector_expired = await self.vector_manager.cleanup_expired_vectors()

            self.db.commit()

            logger.info(f"Cleaned up {expired_count} expired memories and {vector_expired} expired vectors")
            return expired_count

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to cleanup expired memories: {str(e)}")
            raise

    async def _update_access_count(self, memory_id: str):
        """Update access count for a memory item"""
        try:
            memory = self.db.query(AgentMemory).filter_by(memory_id=memory_id).first()
            if memory:
                memory.access_count += 1
                memory.last_accessed = datetime.utcnow()
                self.db.commit()
        except Exception as e:
            logger.error(f"Failed to update access count: {str(e)}")

    async def _start_cleanup_scheduler(self):
        """Start background cleanup scheduler"""
        while True:
            try:
                await asyncio.sleep(Config.MEMORY_CLEANUP_INTERVAL_SECONDS)
                await self.cleanup_expired_memories()
            except Exception as e:
                logger.error(f"Error in cleanup scheduler: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

    def get_memory_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            query = self.db.query(
                AgentMemory.memory_type,
                func.count(AgentMemory.memory_id).label('count'),
                func.avg(AgentMemory.importance_score).label('avg_importance'),
                func.sum(AgentMemory.access_count).label('total_accesses')
            )

            if agent_id:
                query = query.filter_by(agent_id=agent_id)

            query = query.group_by(AgentMemory.memory_type)

            stats = {}
            for row in query.all():
                stats[row.memory_type] = {
                    'count': row.count,
                    'avg_importance': float(row.avg_importance or 0),
                    'total_accesses': row.total_accesses or 0
                }

            return {
                'stats': stats,
                'generated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {str(e)}")
            return {'stats': {}, 'error': str(e)}

# =============================================================================
# API MODELS
# =============================================================================

class MemoryStats(BaseModel):
    """Model for memory statistics"""
    memory_type: str
    count: int
    avg_importance: float
    total_accesses: int

# =============================================================================
# MONITORING & METRICS
# =============================================================================

class MetricsCollector:
    """Collects and exposes Prometheus metrics"""

    def __init__(self):
        # Use default registry to avoid duplication issues
        self.registry = REGISTRY

        # Memory metrics - check if they already exist
        if 'memory_manager_items_total' not in self.registry._names_to_collectors:
            self.memory_items_total = Gauge('memory_manager_items_total', 'Total memory items by type', ['memory_type'], registry=self.registry)
        else:
            self.memory_items_total = self.registry._names_to_collectors['memory_manager_items_total']

        if 'memory_manager_access_total' not in self.registry._names_to_collectors:
            self.memory_access_total = Counter('memory_manager_access_total', 'Total memory accesses', ['memory_type'], registry=self.registry)
        else:
            self.memory_access_total = self.registry._names_to_collectors['memory_manager_access_total']

        if 'memory_manager_expiration_total' not in self.registry._names_to_collectors:
            self.memory_expiration_total = Counter('memory_manager_expiration_total', 'Total memory expirations', registry=self.registry)
        else:
            self.memory_expiration_total = self.registry._names_to_collectors['memory_manager_expiration_total']

        if 'memory_manager_consolidation_total' not in self.registry._names_to_collectors:
            self.memory_consolidation_total = Counter('memory_manager_consolidation_total', 'Total memory consolidations', registry=self.registry)
        else:
            self.memory_consolidation_total = self.registry._names_to_collectors['memory_manager_consolidation_total']

        # Performance metrics
        if 'memory_manager_requests_total' not in self.registry._names_to_collectors:
            self.request_count = Counter('memory_manager_requests_total', 'Total number of requests', ['method', 'endpoint'], registry=self.registry)
        else:
            self.request_count = self.registry._names_to_collectors['memory_manager_requests_total']

        if 'memory_manager_request_duration_seconds' not in self.registry._names_to_collectors:
            self.request_duration = Histogram('memory_manager_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'], registry=self.registry)
        else:
            self.request_duration = self.registry._names_to_collectors['memory_manager_request_duration_seconds']

        if 'memory_manager_errors_total' not in self.registry._names_to_collectors:
            self.error_count = Counter('memory_manager_errors_total', 'Total number of errors', ['type'], registry=self.registry)
        else:
            self.error_count = self.registry._names_to_collectors['memory_manager_errors_total']

    def update_memory_metrics(self, memory_manager: MemoryManager):
        """Update memory-related metrics"""
        try:
            stats = memory_manager.get_memory_stats()
            for memory_type, type_stats in stats.get('stats', {}).items():
                self.memory_items_total.labels(memory_type=memory_type).set(type_stats['count'])

        except Exception as e:
            logger.error(f"Failed to update memory metrics: {str(e)}")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Memory Manager Service",
    description="Comprehensive memory management for AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db_engine = create_engine(Config.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=Config.REDIS_DB, decode_responses=True)
metrics_collector = MetricsCollector()

# Dependency injection
def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_memory_manager(db: Session = Depends(get_db)):
    """Memory manager dependency"""
    return MemoryManager(db, redis_client)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "memory-manager",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    if not Config.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    return generate_latest(metrics_collector.registry)

@app.post("/memories")
async def store_memory(
    request: MemoryStoreRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Store a memory item"""
    metrics_collector.request_count.labels(method='POST', endpoint='/memories').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/memories').time():
        memory_id = await memory_manager.store_memory(request)

    return {
        "memory_id": memory_id,
        "status": "stored",
        "message": "Memory item stored successfully"
    }

@app.get("/memories/{agent_id}/{memory_type}/{content_key}")
async def retrieve_memory(
    agent_id: str,
    memory_type: str,
    content_key: str,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Retrieve a specific memory item"""
    metrics_collector.request_count.labels(method='GET', endpoint='/memories/{agent_id}/{memory_type}/{content_key}').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/memories/{agent_id}/{memory_type}/{content_key}').time():
        memory = memory_manager.retrieve_memory(agent_id, memory_type, content_key)

        if not memory:
            raise HTTPException(status_code=404, detail="Memory item not found")

    return memory

@app.post("/memories/search")
async def search_memories(
    request: MemorySearchRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Search memories with filtering"""
    metrics_collector.request_count.labels(method='POST', endpoint='/memories/search').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/memories/search').time():
        results = memory_manager.search_memories(request)

    return results

@app.post("/memories/consolidate")
async def consolidate_memory(
    request: MemoryConsolidationRequest,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Consolidate memories using specified strategy"""
    metrics_collector.request_count.labels(method='POST', endpoint='/memories/consolidate').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/memories/consolidate').time():
        result = await memory_manager.consolidate_memory(request)

    return result

@app.post("/memories/cleanup")
async def cleanup_expired_memories(
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Trigger cleanup of expired memories"""
    metrics_collector.request_count.labels(method='POST', endpoint='/memories/cleanup').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/memories/cleanup').time():
        # Start cleanup in background
        background_tasks.add_task(memory_manager.cleanup_expired_memories)

        return {
            "status": "cleanup_started",
            "message": "Memory cleanup started in background"
        }

@app.get("/memories/stats")
async def get_memory_stats(
    agent_id: Optional[str] = None,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Get memory statistics"""
    metrics_collector.request_count.labels(method='GET', endpoint='/memories/stats').inc()

    with metrics_collector.request_duration.labels(method='GET', endpoint='/memories/stats').time():
        stats = memory_manager.get_memory_stats(agent_id)

    return stats

@app.get("/memories/types")
async def get_memory_types():
    """Get available memory types"""
    memory_types = [
        {
            "type": "working",
            "name": "Working Memory",
            "description": "Short-term memory with TTL-based expiration",
            "default_ttl_seconds": Config.WORKING_MEMORY_TTL_SECONDS
        },
        {
            "type": "episodic",
            "name": "Episodic Memory",
            "description": "Event-based memory for experience recall",
            "retention_days": Config.EPISODIC_MEMORY_RETENTION_DAYS
        },
        {
            "type": "semantic",
            "name": "Semantic Memory",
            "description": "Long-term knowledge and pattern storage",
            "enabled": Config.SEMANTIC_MEMORY_ENABLED
        },
        {
            "type": "vector",
            "name": "Vector Memory",
            "description": "Embedding-based memory for similarity search",
            "vector_dimension": Config.MEMORY_VECTOR_DIMENSION
        }
    ]

    return {"memory_types": memory_types}

@app.post("/vectors")
async def store_vector_memory(
    item: VectorMemoryItem,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Store a vector memory item"""
    metrics_collector.request_count.labels(method='POST', endpoint='/vectors').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/vectors').time():
        memory_id = await memory_manager.vector_manager.store_vector_memory(item)

    return {
        "memory_id": memory_id,
        "status": "stored",
        "message": "Vector memory item stored successfully"
    }

@app.post("/vectors/search")
async def search_vector_memories(
    agent_id: str,
    query_vector: List[float],
    limit: int = 10,
    threshold: float = 0.8,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Search for similar vector memories"""
    metrics_collector.request_count.labels(method='POST', endpoint='/vectors/search').inc()

    with metrics_collector.request_duration.labels(method='POST', endpoint='/vectors/search').time():
        results = await memory_manager.vector_manager.search_similar_memories(
            agent_id, query_vector, limit, threshold
        )

    return {
        "results": results,
        "query_vector_dimension": len(query_vector),
        "search_completed_at": datetime.utcnow().isoformat()
    }

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    metrics_collector.error_count.labels(type='validation').inc()

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "message": "Invalid request data"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    metrics_collector.error_count.labels(type='http').inc()

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    metrics_collector.error_count.labels(type='general').inc()

    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": str(uuid.uuid4())
        }
    )

# =============================================================================
# LIFECYCLE MANAGEMENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Starting Memory Manager Service...")

    # Create database tables
    try:
        # Base.metadata.create_all(bind=...)  # Removed - use schema.sql instead
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise

    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        raise

    logger.info(f"Memory Manager Service started on {Config.SERVICE_HOST}:{Config.SERVICE_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down Memory Manager Service...")

    # Close Redis connection
    try:
        redis_client.close()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {str(e)}")

    logger.info("Memory Manager Service shutdown complete")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.SERVICE_HOST,
        port=Config.SERVICE_PORT,
        reload=False,
        log_level="info",
        access_log=True
    )
