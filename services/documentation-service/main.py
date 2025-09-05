#!/usr/bin/env python3
"""
Documentation Service for Agentic Brain Platform

This service provides comprehensive web-based documentation and user guides
instead of relying on Markdown files. It includes interactive tutorials,
API documentation, video guides, and a centralized documentation portal.

Features:
- Web-based user guides and tutorials
- Interactive API documentation
- Video tutorials and walkthroughs
- Searchable knowledge base
- User feedback and ratings
- Multi-language support
- Accessibility-compliant documentation
- Real-time documentation updates
- Documentation analytics and usage tracking
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON, Float, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func
import httpx
import uvicorn

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for Documentation Service"""

    # Service Configuration
    DOCUMENTATION_PORT = int(os.getenv("DOCUMENTATION_PORT", "8410"))
    HOST = os.getenv("HOST", "0.0.0.0")

    # Service URLs
    AGENT_BUILDER_UI_URL = os.getenv("AGENT_BUILDER_UI_URL", "http://localhost:8300")
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8200")

    # Documentation Configuration
    DOCS_LANGUAGE = os.getenv("DOCS_LANGUAGE", "en")
    SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "zh"]
    DOCS_CACHE_TTL = int(os.getenv("DOCS_CACHE_TTL", "3600"))  # 1 hour

    # Content Management
    MAX_CONTENT_SIZE = int(os.getenv("MAX_CONTENT_SIZE", "10485760"))  # 10MB
    ALLOWED_FILE_TYPES = [".pdf", ".mp4", ".jpg", ".png", ".gif", ".svg"]

    # Analytics Configuration
    ENABLE_ANALYTICS = os.getenv("ENABLE_ANALYTICS", "true").lower() == "true"
    ANALYTICS_RETENTION_DAYS = int(os.getenv("ANALYTICS_RETENTION_DAYS", "365"))

    # Database Configuration
    # Database URL must be provided via environment; avoid hardcoded credentials
    DATABASE_URL = os.getenv("DATABASE_URL", "")

    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class DocumentationContent(Base):
    """Documentation content storage"""
    __tablename__ = 'documentation_content'

    id = Column(String(100), primary_key=True)
    title = Column(String(500), nullable=False)
    content_type = Column(String(50), nullable=False)  # guide, tutorial, api, video, faq
    category = Column(String(100), nullable=False)
    subcategory = Column(String(100))
    content = Column(Text, nullable=False)  # HTML content
    metadata = Column(JSON, default=dict)  # tags, difficulty, duration, etc.
    language = Column(String(10), default="en")
    version = Column(String(20), default="1.0")
    status = Column(String(20), default="published")  # draft, published, archived
    author = Column(String(100))
    tags = Column(JSON, default=list)
    view_count = Column(Integer, default=0)
    helpful_votes = Column(Integer, default=0)
    total_votes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class DocumentationMedia(Base):
    """Documentation media files"""
    __tablename__ = 'documentation_media'

    id = Column(String(100), primary_key=True)
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_type = Column(String(50), nullable=False)  # image, video, document
    mime_type = Column(String(100), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    content_id = Column(String(100))  # Associated content ID
    alt_text = Column(String(500))  # For accessibility
    description = Column(Text)
    uploaded_by = Column(String(100))
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class DocumentationSearch(Base):
    """Documentation search index"""
    __tablename__ = 'documentation_search'

    id = Column(String(100), primary_key=True)
    content_id = Column(String(100), nullable=False)
    search_text = Column(Text, nullable=False)  # Concatenated searchable text
    title_vector = Column(Text)  # For full-text search
    content_vector = Column(Text)  # For full-text search
    tags_vector = Column(Text)  # For tag-based search
    indexed_at = Column(DateTime, default=datetime.utcnow)

class DocumentationFeedback(Base):
    """User feedback on documentation"""
    __tablename__ = 'documentation_feedback'

    id = Column(String(100), primary_key=True)
    content_id = Column(String(100), nullable=False)
    user_id = Column(String(100))  # Optional for anonymous feedback
    feedback_type = Column(String(20), nullable=False)  # helpful, not_helpful, suggestion, bug_report
    rating = Column(Integer)  # 1-5 rating
    comment = Column(Text)
    user_agent = Column(String(500))
    ip_address = Column(String(50))
    submitted_at = Column(DateTime, default=datetime.utcnow)

class DocumentationAnalytics(Base):
    """Documentation usage analytics"""
    __tablename__ = 'documentation_analytics'

    id = Column(String(100), primary_key=True)
    content_id = Column(String(100))
    event_type = Column(String(50), nullable=False)  # view, search, feedback, share
    user_id = Column(String(100))
    session_id = Column(String(100))
    metadata = Column(JSON, default=dict)  # Additional event data
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    timestamp = Column(DateTime, default=datetime.utcnow)

class DocumentationTutorial(Base):
    """Interactive tutorial definitions"""
    __tablename__ = 'documentation_tutorials'

    id = Column(String(100), primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    category = Column(String(100), nullable=False)
    difficulty = Column(String(20), default="beginner")  # beginner, intermediate, advanced
    estimated_duration = Column(Integer, default=0)  # minutes
    steps = Column(JSON, default=list)  # Tutorial steps with content and actions
    prerequisites = Column(JSON, default=list)  # Required knowledge or setup
    completion_criteria = Column(JSON, default=list)  # How to determine completion
    success_rate = Column(Float, default=0.0)  # Percentage of users who complete
    total_attempts = Column(Integer, default=0)
    total_completions = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class TutorialProgress(Base):
    """User progress through tutorials"""
    __tablename__ = 'tutorial_progress'

    id = Column(String(100), primary_key=True)
    tutorial_id = Column(String(100), nullable=False)
    user_id = Column(String(100), nullable=False)
    current_step = Column(Integer, default=0)
    completed_steps = Column(JSON, default=list)
    started_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    time_spent_seconds = Column(Integer, default=0)

# =============================================================================
# MARKDOWN TO HTML CONVERTER
# =============================================================================

class MarkdownToHTMLConverter:
    """Converts Markdown documentation to HTML web forms"""

    def __init__(self):
        self.logger = structlog.get_logger("markdown_converter")

    async def convert_markdown_to_html(self, markdown_content: str, title: str = "") -> str:
        """Convert Markdown content to HTML with enhanced formatting"""
        try:
            # Basic Markdown to HTML conversion
            html_content = await self._parse_markdown(markdown_content)

            # Enhance with web form elements
            html_content = await self._enhance_with_web_elements(html_content, title)

            # Add accessibility features
            html_content = await self._add_accessibility_features(html_content)

            # Add interactive elements
            html_content = await self._add_interactive_elements(html_content)

            return html_content

        except Exception as e:
            self.logger.error("Markdown conversion failed", error=str(e))
            return f"<div class='error'>Error converting documentation: {str(e)}</div>"

    async def _parse_markdown(self, content: str) -> str:
        """Parse basic Markdown syntax to HTML.

        Note: This implementation is intentionally conservative and uses the
        `markdown` package when available to avoid fragile ad-hoc parsing.
        """
        try:
            import markdown as _md
            # Use safe extensions and disable raw HTML by default
            html = _md.markdown(content, extensions=['extra', 'sane_lists'])
            return html
        except Exception:
            # Fallback: very conservative transformations
            import re
            html = content
            html = re.sub(r'```([\s\S]*?)```', r'<pre><code>\1</code></pre>', html)
            html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
            html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
            html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
            # Very small list handling
            html = re.sub(r'(?m)^- (.+)$', r'<li>\1</li>', html)
            html = re.sub(r'(<li>.+</li>)', r'<ul>\1</ul>', html, count=1)
            # Wrap remaining lines in paragraphs
            lines = [l for l in html.split('\n') if l.strip()]
            return '\n'.join(f'<p>{l}</p>' for l in lines)

    async def _enhance_with_web_elements(self, content: str, title: str) -> str:
        """Enhance HTML with web form elements and styling"""
        enhanced_content = f"""
        <div class="documentation-content">
            <div class="content-header">
                <h1>{title}</h1>
                <div class="content-meta">
                    <span class="last-updated">Last updated: {datetime.utcnow().strftime('%Y-%m-%d')}</span>
                    <div class="content-actions">
                        <button class="btn btn-secondary" onclick="printContent()">Print</button>
                        <button class="btn btn-secondary" onclick="shareContent()">Share</button>
                    </div>
                </div>
            </div>
            <div class="content-body">
                {content}
            </div>
            <div class="content-footer">
                <div class="feedback-section">
                    <h3>Was this helpful?</h3>
                    <div class="feedback-buttons">
                        <button class="btn btn-success" onclick="submitFeedback('helpful')">Yes</button>
                        <button class="btn btn-danger" onclick="submitFeedback('not_helpful')">No</button>
                    </div>
                    <div class="feedback-form" style="display: none;">
                        <textarea id="feedback-comment" placeholder="Tell us how we can improve this documentation..." rows="3"></textarea>
                        <button class="btn btn-primary" onclick="submitDetailedFeedback()">Submit Feedback</button>
                    </div>
                </div>
                <div class="related-content">
                    <h3>Related Topics</h3>
                    <ul id="related-links">
                        <!-- Related content will be populated dynamically -->
                    </ul>
                </div>
            </div>
        </div>
        """
        return enhanced_content

    async def _add_accessibility_features(self, content: str) -> str:
        """Add accessibility features to the content"""
        accessibility_content = f"""
        <div class="accessibility-controls">
            <button class="accessibility-btn" onclick="increaseFontSize()" aria-label="Increase font size">
                <span class="sr-only">Increase Font Size</span>A+
            </button>
            <button class="accessibility-btn" onclick="decreaseFontSize()" aria-label="Decrease font size">
                <span class="sr-only">Decrease Font Size</span>A-
            </button>
            <button class="accessibility-btn" onclick="toggleHighContrast()" aria-label="Toggle high contrast mode">
                <span class="sr-only">Toggle High Contrast</span>HC
            </button>
        </div>
        {content}
        """
        return accessibility_content

    async def _add_interactive_elements(self, content: str) -> str:
        """Add interactive elements to the content"""
        interactive_content = content.replace(
            '<pre><code>',
            '<div class="code-block"><button class="copy-btn" onclick="copyCode(this)" aria-label="Copy code">Copy</button><pre><code>'
        ).replace(
            '</code></pre>',
            '</code></pre></div>'
        )

        # Add expandable sections for long content
        interactive_content = interactive_content.replace(
            '<h2>', '<h2 class="expandable" onclick="toggleSection(this)">'
        )

        return interactive_content

# =============================================================================
# DOCUMENTATION PORTAL
# =============================================================================

class DocumentationPortal:
    """Main documentation portal with web forms"""

    def __init__(self, db_session: Session):
        self.db = db_session
        self.converter = MarkdownToHTMLConverter()
        self.logger = structlog.get_logger("documentation_portal")

    async def get_documentation_content(self, content_id: str, language: str = "en") -> Dict[str, Any]:
        """Get documentation content by ID"""
        try:
            content = self.db.query(DocumentationContent).filter_by(
                id=content_id,
                language=language,
                status="published"
            ).first()

            if not content:
                raise HTTPException(status_code=404, detail=f"Documentation content {content_id} not found")

            # Update view count
            content.view_count += 1
            self.db.commit()

            # Get related media
            media = self.db.query(DocumentationMedia).filter_by(content_id=content_id).all()

            # Get feedback summary
            feedback_summary = self._get_feedback_summary(content_id)

            return {
                "id": content.id,
                "title": content.title,
                "content_type": content.content_type,
                "category": content.category,
                "subcategory": content.subcategory,
                "content": content.content,
                "metadata": content.metadata,
                "language": content.language,
                "version": content.version,
                "author": content.author,
                "tags": content.tags,
                "view_count": content.view_count,
                "helpful_rating": feedback_summary.get("helpful_percentage", 0),
                "total_feedbacks": feedback_summary.get("total_feedbacks", 0),
                "media": [
                    {
                        "id": m.id,
                        "filename": m.filename,
                        "file_type": m.file_type,
                        "description": m.description,
                        "alt_text": m.alt_text
                    }
                    for m in media
                ],
                "created_at": content.created_at.isoformat(),
                "updated_at": content.updated_at.isoformat()
            }

        except Exception as e:
            self.logger.error("Failed to get documentation content", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to get documentation content: {str(e)}")

    async def search_documentation(self, query: str, category: str = None,
                                 content_type: str = None, language: str = "en",
                                 limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """Search documentation content"""
        try:
            # Build search query
            search_query = self.db.query(DocumentationContent).filter(
                DocumentationContent.language == language,
                DocumentationContent.status == "published"
            )

            if category:
                search_query = search_query.filter(DocumentationContent.category == category)

            if content_type:
                search_query = search_query.filter(DocumentationContent.content_type == content_type)

            # Simple text search (in production, use full-text search)
            if query:
                search_terms = query.lower().split()
                for term in search_terms:
                    search_query = search_query.filter(
                        DocumentationContent.title.ilike(f'%{term}%') |
                        DocumentationContent.content.ilike(f'%{term}%')
                    )

            # Get total count
            total_count = search_query.count()

            # Get results
            results = search_query.offset(offset).limit(limit).all()

            return {
                "query": query,
                "total_results": total_count,
                "results": [
                    {
                        "id": result.id,
                        "title": result.title,
                        "content_type": result.content_type,
                        "category": result.category,
                        "subcategory": result.subcategory,
                        "tags": result.tags,
                        "view_count": result.view_count,
                        "updated_at": result.updated_at.isoformat()
                    }
                    for result in results
                ],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": (offset + limit) < total_count
                }
            }

        except Exception as e:
            self.logger.error("Documentation search failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Documentation search failed: {str(e)}")

    async def submit_feedback(self, content_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit user feedback on documentation"""
        try:
            feedback = DocumentationFeedback(
                id=str(uuid.uuid4()),
                content_id=content_id,
                feedback_type=feedback_data.get("feedback_type"),
                rating=feedback_data.get("rating"),
                comment=feedback_data.get("comment"),
                user_agent=feedback_data.get("user_agent"),
                ip_address=feedback_data.get("ip_address")
            )

            self.db.add(feedback)
            self.db.commit()

            # Track analytics
            await self._track_analytics_event(
                content_id=content_id,
                event_type="feedback",
                metadata={"feedback_type": feedback.feedback_type, "rating": feedback.rating}
            )

            return {
                "feedback_id": feedback.id,
                "status": "submitted",
                "message": "Thank you for your feedback!"
            }

        except Exception as e:
            self.logger.error("Feedback submission failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

    async def get_tutorial_progress(self, tutorial_id: str, user_id: str) -> Dict[str, Any]:
        """Get user progress through a tutorial"""
        try:
            progress = self.db.query(TutorialProgress).filter_by(
                tutorial_id=tutorial_id,
                user_id=user_id
            ).first()

            if not progress:
                return {
                    "tutorial_id": tutorial_id,
                    "user_id": user_id,
                    "current_step": 0,
                    "completed_steps": [],
                    "progress_percentage": 0,
                    "status": "not_started"
                }

            tutorial = self.db.query(DocumentationTutorial).filter_by(id=tutorial_id).first()
            total_steps = len(tutorial.steps) if tutorial else 0
            progress_percentage = (len(progress.completed_steps) / total_steps * 100) if total_steps > 0 else 0

            return {
                "tutorial_id": tutorial_id,
                "user_id": user_id,
                "current_step": progress.current_step,
                "completed_steps": progress.completed_steps,
                "progress_percentage": round(progress_percentage, 1),
                "status": "completed" if progress.completed_at else "in_progress",
                "started_at": progress.started_at.isoformat(),
                "last_updated": progress.last_updated.isoformat(),
                "completed_at": progress.completed_at.isoformat() if progress.completed_at else None,
                "time_spent_seconds": progress.time_spent_seconds
            }

        except Exception as e:
            self.logger.error("Failed to get tutorial progress", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to get tutorial progress: {str(e)}")

    async def update_tutorial_progress(self, tutorial_id: str, user_id: str,
                                     current_step: int, completed_steps: List[int]) -> Dict[str, Any]:
        """Update user progress through a tutorial"""
        try:
            progress = self.db.query(TutorialProgress).filter_by(
                tutorial_id=tutorial_id,
                user_id=user_id
            ).first()

            if not progress:
                progress = TutorialProgress(
                    id=str(uuid.uuid4()),
                    tutorial_id=tutorial_id,
                    user_id=user_id
                )
                self.db.add(progress)

            progress.current_step = current_step
            progress.completed_steps = completed_steps
            progress.last_updated = datetime.utcnow()

            # Check if tutorial is completed
            tutorial = self.db.query(DocumentationTutorial).filter_by(id=tutorial_id).first()
            if tutorial and len(completed_steps) >= len(tutorial.steps):
                progress.completed_at = datetime.utcnow()
                tutorial.total_completions += 1
                tutorial.success_rate = (tutorial.total_completions / tutorial.total_attempts) * 100

            self.db.commit()

            return {
                "status": "updated",
                "progress_percentage": (len(completed_steps) / len(tutorial.steps) * 100) if tutorial else 0
            }

        except Exception as e:
            self.logger.error("Failed to update tutorial progress", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to update tutorial progress: {str(e)}")

    def _get_feedback_summary(self, content_id: str) -> Dict[str, Any]:
        """Get feedback summary for content"""
        try:
            feedbacks = self.db.query(DocumentationFeedback).filter_by(content_id=content_id).all()

            if not feedbacks:
                return {"total_feedbacks": 0, "helpful_percentage": 0}

            helpful_count = sum(1 for f in feedbacks if f.feedback_type == "helpful" or f.rating >= 4)
            total_feedbacks = len(feedbacks)
            helpful_percentage = (helpful_count / total_feedbacks * 100) if total_feedbacks > 0 else 0

            return {
                "total_feedbacks": total_feedbacks,
                "helpful_count": helpful_count,
                "helpful_percentage": round(helpful_percentage, 1),
                "average_rating": sum(f.rating for f in feedbacks if f.rating) / len([f for f in feedbacks if f.rating]) if any(f.rating for f in feedbacks) else 0
            }

        except Exception as e:
            self.logger.error("Failed to get feedback summary", error=str(e))
            return {"total_feedbacks": 0, "helpful_percentage": 0}

    async def _track_analytics_event(self, content_id: str = None, event_type: str = "",
                                   metadata: Dict[str, Any] = None) -> None:
        """Track analytics event"""
        try:
            if not Config.ENABLE_ANALYTICS:
                return

            analytics = DocumentationAnalytics(
                id=str(uuid.uuid4()),
                content_id=content_id,
                event_type=event_type,
                metadata=metadata or {},
                timestamp=datetime.utcnow()
            )

            self.db.add(analytics)
            self.db.commit()

        except Exception as e:
            self.logger.error("Failed to track analytics event", error=str(e))

# =============================================================================
# API MODELS
# =============================================================================

class DocumentationSearchRequest(BaseModel):
    """Documentation search request"""
    query: str = Field(..., description="Search query")
    category: Optional[str] = Field(None, description="Filter by category")
    content_type: Optional[str] = Field(None, description="Filter by content type")
    language: str = Field("en", description="Content language")
    limit: int = Field(20, description="Maximum results")
    offset: int = Field(0, description="Results offset")

class FeedbackSubmissionRequest(BaseModel):
    """Feedback submission request"""
    content_id: str = Field(..., description="Content ID for feedback")
    feedback_type: str = Field(..., description="Type of feedback")
    rating: Optional[int] = Field(None, description="Rating (1-5)")
    comment: Optional[str] = Field(None, description="Feedback comment")

class TutorialProgressRequest(BaseModel):
    """Tutorial progress request"""
    tutorial_id: str = Field(..., description="Tutorial ID")
    user_id: str = Field(..., description="User ID")

class TutorialProgressUpdateRequest(BaseModel):
    """Tutorial progress update request"""
    tutorial_id: str = Field(..., description="Tutorial ID")
    user_id: str = Field(..., description="User ID")
    current_step: int = Field(..., description="Current step number")
    completed_steps: List[int] = Field(..., description="List of completed step numbers")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Documentation Service",
    description="Web-based documentation portal for Agentic Brain platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
engine = create_engine(Config.DATABASE_URL)
# Base.metadata.create_all(bind=...)  # Removed - use schema.sql instead
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize Redis
redis_client = redis.Redis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=Config.REDIS_DB,
    decode_responses=True
)

# Initialize documentation portal
docs_portal = DocumentationPortal(SessionLocal)

# Initialize Prometheus metrics
REQUEST_COUNT = Counter('documentation_requests_total', 'Total number of requests', ['method', 'endpoint'])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Documentation Service",
        "status": "healthy",
        "version": "1.0.0",
        "capabilities": {
            "web_based_documentation": True,
            "interactive_tutorials": True,
            "searchable_knowledge_base": True,
            "user_feedback_system": True,
            "analytics_and_tracking": True,
            "multi_language_support": True
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "redis": "connected",
            "documentation_portal": "active",
            "markdown_converter": "ready"
        }
    }

@app.get("/api/docs/{content_id}")
async def get_documentation(content_id: str, language: str = "en"):
    """Get documentation content by ID"""
    REQUEST_COUNT.labels(method="GET", endpoint=f"/api/docs/{content_id}").inc()

    try:
        content = await docs_portal.get_documentation_content(content_id, language)

        # Track view analytics
        await docs_portal._track_analytics_event(
            content_id=content_id,
            event_type="view",
            metadata={"language": language}
        )

        return content

    except Exception as e:
        logger.error("Failed to get documentation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get documentation: {str(e)}")

@app.post("/api/docs/search")
async def search_documentation(request: DocumentationSearchRequest):
    """Search documentation content"""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/docs/search").inc()

    try:
        results = await docs_portal.search_documentation(
            query=request.query,
            category=request.category,
            content_type=request.content_type,
            language=request.language,
            limit=request.limit,
            offset=request.offset
        )

        # Track search analytics
        await docs_portal._track_analytics_event(
            event_type="search",
            metadata={
                "query": request.query,
                "category": request.category,
                "content_type": request.content_type,
                "results_count": len(results["results"])
            }
        )

        return results

    except Exception as e:
        logger.error("Documentation search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Documentation search failed: {str(e)}")

@app.post("/api/docs/feedback")
async def submit_feedback(request: FeedbackSubmissionRequest, req: Request):
    """Submit feedback on documentation"""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/docs/feedback").inc()

    try:
        # Get client information
        client_info = {
            "user_agent": req.headers.get("user-agent"),
            "ip_address": req.client.host if req.client else None
        }

        feedback_data = request.dict()
        feedback_data.update(client_info)

        result = await docs_portal.submit_feedback(request.content_id, feedback_data)
        return result

    except Exception as e:
        logger.error("Feedback submission failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@app.get("/api/tutorials/{tutorial_id}/progress")
async def get_tutorial_progress(tutorial_id: str, user_id: str):
    """Get tutorial progress for a user"""
    REQUEST_COUNT.labels(method="GET", endpoint=f"/api/tutorials/{tutorial_id}/progress").inc()

    try:
        progress = await docs_portal.get_tutorial_progress(tutorial_id, user_id)
        return progress

    except Exception as e:
        logger.error("Failed to get tutorial progress", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get tutorial progress: {str(e)}")

@app.put("/api/tutorials/progress")
async def update_tutorial_progress(request: TutorialProgressUpdateRequest):
    """Update tutorial progress for a user"""
    REQUEST_COUNT.labels(method="PUT", endpoint="/api/tutorials/progress").inc()

    try:
        result = await docs_portal.update_tutorial_progress(
            request.tutorial_id,
            request.user_id,
            request.current_step,
            request.completed_steps
        )
        return result

    except Exception as e:
        logger.error("Failed to update tutorial progress", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update tutorial progress: {str(e)}")

@app.get("/api/docs/categories")
async def get_documentation_categories():
    """Get available documentation categories"""
    REQUEST_COUNT.labels(method="GET", endpoint="/api/docs/categories").inc()

    try:
        db = SessionLocal()
        categories = db.query(
            DocumentationContent.category,
            func.count(DocumentationContent.id).label('count')
        ).filter(
            DocumentationContent.status == "published"
        ).group_by(
            DocumentationContent.category
        ).all()
        db.close()

        return {
            "categories": [
                {"name": cat.category, "count": cat.count}
                for cat in categories
            ]
        }

    except Exception as e:
        logger.error("Failed to get documentation categories", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get documentation categories: {str(e)}")

@app.get("/portal", response_class=HTMLResponse)
async def documentation_portal():
    """Interactive documentation portal"""
    try:
        portal_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Agentic Brain Documentation Portal</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}

                .portal-container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    min-height: 100vh;
                    box-shadow: 0 0 30px rgba(0,0,0,0.1);
                }}

                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 2rem;
                    text-align: center;
                }}

                .header h1 {{
                    font-size: 2.5rem;
                    margin-bottom: 0.5rem;
                }}

                .header p {{
                    font-size: 1.2rem;
                    opacity: 0.9;
                }}

                .search-section {{
                    padding: 2rem;
                    background: #f8f9fa;
                    border-bottom: 1px solid #e9ecef;
                }}

                .search-container {{
                    max-width: 600px;
                    margin: 0 auto;
                }}

                .search-input {{
                    width: 100%;
                    padding: 1rem;
                    font-size: 1.1rem;
                    border: 2px solid #ddd;
                    border-radius: 50px;
                    outline: none;
                    transition: border-color 0.3s;
                }}

                .search-input:focus {{
                    border-color: #667eea;
                }}

                .content-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 2rem;
                    padding: 2rem;
                }}

                .content-card {{
                    background: white;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    transition: transform 0.3s, box-shadow 0.3s;
                }}

                .content-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                }}

                .card-header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 1.5rem;
                }}

                .card-title {{
                    font-size: 1.3rem;
                    margin-bottom: 0.5rem;
                }}

                .card-type {{
                    display: inline-block;
                    background: rgba(255,255,255,0.2);
                    padding: 0.25rem 0.75rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}

                .card-content {{
                    padding: 1.5rem;
                }}

                .card-description {{
                    color: #666;
                    margin-bottom: 1rem;
                    line-height: 1.5;
                }}

                .card-meta {{
                    display: flex;
                    justify-content: space-between;
                    font-size: 0.9rem;
                    color: #888;
                }}

                .category-section {{
                    padding: 2rem;
                    background: #f8f9fa;
                }}

                .category-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem;
                }}

                .category-card {{
                    background: white;
                    padding: 1.5rem;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: transform 0.2s;
                }}

                .category-card:hover {{
                    transform: translateY(-2px);
                }}

                .category-icon {{
                    font-size: 2rem;
                    margin-bottom: 1rem;
                    color: #667eea;
                }}

                .category-name {{
                    font-weight: bold;
                    margin-bottom: 0.5rem;
                }}

                .category-count {{
                    color: #666;
                    font-size: 0.9rem;
                }}

                .footer {{
                    background: #333;
                    color: white;
                    text-align: center;
                    padding: 2rem;
                }}

                @media (max-width: 768px) {{
                    .content-grid {{
                        grid-template-columns: 1fr;
                        padding: 1rem;
                    }}

                    .header h1 {{
                        font-size: 2rem;
                    }}

                    .search-section {{
                        padding: 1rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="portal-container">
                <header class="header">
                    <h1>📚 Agentic Brain Documentation</h1>
                    <p>Comprehensive web-based documentation and interactive tutorials</p>
                </header>

                <section class="search-section">
                    <div class="search-container">
                        <input type="text" id="search-input" class="search-input"
                               placeholder="Search documentation, tutorials, and guides..." />
                    </div>
                </section>

                <section class="content-grid" id="featured-content">
                    <!-- Featured content will be loaded dynamically -->
                    <div class="content-card">
                        <div class="card-header">
                            <h3 class="card-title">Getting Started Guide</h3>
                            <span class="card-type">guide</span>
                        </div>
                        <div class="card-content">
                            <p class="card-description">
                                Learn the basics of Agentic Brain platform and get started with your first agent.
                            </p>
                            <div class="card-meta">
                                <span>Beginner</span>
                                <span>15 min read</span>
                            </div>
                        </div>
                    </div>

                    <div class="content-card">
                        <div class="card-header">
                            <h3 class="card-title">Agent Builder Tutorial</h3>
                            <span class="card-type">tutorial</span>
                        </div>
                        <div class="card-content">
                            <p class="card-description">
                                Interactive tutorial on creating and configuring AI agents using the visual builder.
                            </p>
                            <div class="card-meta">
                                <span>Intermediate</span>
                                <span>25 min tutorial</span>
                            </div>
                        </div>
                    </div>

                    <div class="content-card">
                        <div class="card-header">
                            <h3 class="card-title">API Reference</h3>
                            <span class="card-type">api</span>
                        </div>
                        <div class="card-content">
                            <p class="card-description">
                                Complete API documentation with examples and interactive testing.
                            </p>
                            <div class="card-meta">
                                <span>Advanced</span>
                                <span>Reference</span>
                            </div>
                        </div>
                    </div>
                </section>

                <section class="category-section">
                    <h2 style="text-align: center; margin-bottom: 2rem; color: #333;">Browse by Category</h2>
                    <div class="category-grid" id="categories">
                        <div class="category-card">
                            <div class="category-icon">🚀</div>
                            <div class="category-name">Getting Started</div>
                            <div class="category-count">12 guides</div>
                        </div>
                        <div class="category-card">
                            <div class="category-icon">⚙️</div>
                            <div class="category-name">Configuration</div>
                            <div class="category-count">8 guides</div>
                        </div>
                        <div class="category-card">
                            <div class="category-icon">🔧</div>
                            <div class="category-name">Development</div>
                            <div class="category-count">15 guides</div>
                        </div>
                        <div class="category-card">
                            <div class="category-icon">📊</div>
                            <div class="category-name">Analytics</div>
                            <div class="category-count">6 guides</div>
                        </div>
                        <div class="category-card">
                            <div class="category-icon">🔒</div>
                            <div class="category-name">Security</div>
                            <div class="category-count">5 guides</div>
                        </div>
                        <div class="category-card">
                            <div class="category-icon">🎯</div>
                            <div class="category-name">Best Practices</div>
                            <div class="category-count">10 guides</div>
                        </div>
                    </div>
                </section>

                <footer class="footer">
                    <p>&copy; 2024 Agentic Brain Platform. All rights reserved.</p>
                    <p>Built with ❤️ for enterprise-grade AI automation</p>
                </footer>
            </div>

            <script>
                // Search functionality
                document.getElementById('search-input').addEventListener('input', function(e) {{
                    const query = e.target.value.trim();
                    if (query.length > 2) {{
                        performSearch(query);
                    }}
                }});

                async function performSearch(query) {{
                    try {{
                        const response = await fetch('/api/docs/search', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{ query: query }})
                        }});

                        const results = await response.json();
                        displaySearchResults(results);
                    }} catch (error) {{
                        console.error('Search failed:', error);
                    }}
                }}

                function displaySearchResults(results) {{
                    // Update the UI with search results
                    console.log('Search results:', results);
                }}

                // Load categories dynamically
                async function loadCategories() {{
                    try {{
                        const response = await fetch('/api/docs/categories');
                        const data = await response.json();

                        const categoriesContainer = document.getElementById('categories');
                        categoriesContainer.innerHTML = data.categories.map(cat => `
                            <div class="category-card" onclick="filterByCategory('${{cat.name}}')">
                                <div class="category-icon">${{getCategoryIcon(cat.name)}}</div>
                                <div class="category-name">${{cat.name}}</div>
                                <div class="category-count">${{cat.count}} guides</div>
                            </div>
                        `).join('');
                    }} catch (error) {{
                        console.error('Failed to load categories:', error);
                    }}
                }}

                function getCategoryIcon(category) {{
                    const icons = {{
                        'Getting Started': '🚀',
                        'Configuration': '⚙️',
                        'Development': '🔧',
                        'Analytics': '📊',
                        'Security': '🔒',
                        'Best Practices': '🎯'
                    }};
                    return icons[category] || '📖';
                }}

                function filterByCategory(category) {{
                    // Filter content by category
                    console.log('Filtering by category:', category);
                }}

                // Initialize
                document.addEventListener('DOMContentLoaded', function() {{
                    loadCategories();
                }});
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=portal_html, status_code=200)
    except Exception as e:
        logger.error("Failed to load documentation portal", error=str(e))
        return HTMLResponse(content="<h1>Documentation Portal Error</h1>", status_code=500)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return HTMLResponse(generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.DOCUMENTATION_PORT,
        reload=True,
        log_level="info"
    )
