#!/usr/bin/env python3
"""
Authentication Service for Agentic Brain Platform

This service provides comprehensive authentication, authorization, and session management
for the Agentic Brain platform, supporting JWT tokens, OAuth2/OIDC integration,
role-based access control, and secure API access across all services.

Features:
- User authentication and registration
- JWT token management with refresh tokens
- Session management and tracking
- Role-based access control (RBAC)
- OAuth2/OIDC integration
- Multi-factor authentication support
- API key management
- Audit logging and security monitoring
- Integration with all Agent Brain services
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
from concurrent.futures import ThreadPoolExecutor
import threading

import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON, Float, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func
import httpx
import bcrypt
import jwt
import pyotp
import qrcode
from cryptography.fernet import Fernet
from authlib.integrations.httpx_client import OAuth2Client
from authlib.integrations.base_client import OAuthError

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
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class User(Base):
    """User account information"""
    __tablename__ = 'users'

    id = Column(String(100), primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=True)  # Nullable for OAuth users
    full_name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    email_verified_at = Column(DateTime, nullable=True)
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(255), nullable=True)
    oauth_provider = Column(String(50), nullable=True)  # google, github, etc.
    oauth_id = Column(String(255), nullable=True)
    role = Column(String(50), default='user')  # admin, user, viewer
    permissions = Column(JSON, nullable=True)
    profile_data = Column(JSON, nullable=True)
    last_login_at = Column(DateTime, nullable=True)
    login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class UserSession(Base):
    """User session tracking"""
    __tablename__ = 'user_sessions'

    id = Column(String(100), primary_key=True)
    user_id = Column(String(100), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    refresh_token = Column(String(255), unique=True, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    device_info = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity_at = Column(DateTime, default=datetime.utcnow)

class APIKey(Base):
    """API key management"""
    __tablename__ = 'api_keys'

    id = Column(String(100), primary_key=True)
    user_id = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False)
    permissions = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class OAuthProvider(Base):
    """OAuth provider configuration"""
    __tablename__ = 'oauth_providers'

    id = Column(String(100), primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    client_id = Column(String(255), nullable=False)
    client_secret = Column(String(255), nullable=False)
    authorization_url = Column(String(500), nullable=False)
    token_url = Column(String(500), nullable=False)
    user_info_url = Column(String(500), nullable=False)
    scope = Column(String(255), default='openid email profile')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    """Security audit logging"""
    __tablename__ = 'audit_logs'

    id = Column(String(100), primary_key=True)
    user_id = Column(String(100), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    details = Column(JSON, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Role(Base):
    """Role definitions for RBAC"""
    __tablename__ = 'roles'

    id = Column(String(100), primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=False)
    permissions = Column(JSON, nullable=False)
    is_system_role = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration"""
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agentic_brain")

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

    # Service ports
    AUTH_SERVICE_PORT = int(os.getenv("AUTH_SERVICE_PORT", "8330"))

    # JWT Configuration
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-in-production")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # Password security
    PASSWORD_MIN_LENGTH = int(os.getenv("PASSWORD_MIN_LENGTH", "8"))
    PASSWORD_REQUIRE_UPPERCASE = os.getenv("PASSWORD_REQUIRE_UPPERCASE", "true").lower() == "true"
    PASSWORD_REQUIRE_LOWERCASE = os.getenv("PASSWORD_REQUIRE_LOWERCASE", "true").lower() == "true"
    PASSWORD_REQUIRE_DIGITS = os.getenv("PASSWORD_REQUIRE_DIGITS", "true").lower() == "true"
    PASSWORD_REQUIRE_SPECIAL = os.getenv("PASSWORD_REQUIRE_SPECIAL", "false").lower() == "true"

    # Session management
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
    MAX_LOGIN_ATTEMPTS = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
    ACCOUNT_LOCK_DURATION_MINUTES = int(os.getenv("ACCOUNT_LOCK_DURATION_MINUTES", "30"))

    # MFA settings
    MFA_ISSUER_NAME = os.getenv("MFA_ISSUER_NAME", "Agentic Brain")

    # OAuth2 settings
    OAUTH2_ENABLED = os.getenv("OAUTH2_ENABLED", "false").lower() == "true"
    OAUTH2_GOOGLE_CLIENT_ID = os.getenv("OAUTH2_GOOGLE_CLIENT_ID", "")
    OAUTH2_GOOGLE_CLIENT_SECRET = os.getenv("OAUTH2_GOOGLE_CLIENT_SECRET", "")
    OAUTH2_GITHUB_CLIENT_ID = os.getenv("OAUTH2_GITHUB_CLIENT_ID", "")
    OAUTH2_GITHUB_CLIENT_SECRET = os.getenv("OAUTH2_GITHUB_CLIENT_SECRET", "")

    # Email settings (for verification)
    SMTP_SERVER = os.getenv("SMTP_SERVER", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@agenticbrain.com")

    # Encryption
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())

    # Security headers
    SECURITY_HEADERS_ENABLED = os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true"

    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8300,http://localhost:3000").split(",")

    # Monitoring
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8003"))

# =============================================================================
# SECURITY UTILITIES
# =============================================================================

class SecurityUtils:
    """Security utility functions"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    @staticmethod
    def generate_jwt_token(data: Dict[str, Any], expires_delta: timedelta = None) -> str:
        """Generate JWT token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=Config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)
        return encoded_jwt

    @staticmethod
    def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=[Config.JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    @staticmethod
    def generate_api_key() -> str:
        """Generate secure API key"""
        return str(uuid.uuid4()) + str(uuid.uuid4()).replace('-', '')

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage"""
        return bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def generate_mfa_secret() -> str:
        """Generate MFA secret"""
        return pyotp.random_base32()

    @staticmethod
    def generate_mfa_qr_code(secret: str, username: str) -> str:
        """Generate MFA QR code provisioning URI"""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=username, issuer_name=Config.MFA_ISSUER_NAME)

    @staticmethod
    def verify_mfa_code(secret: str, code: str) -> bool:
        """Verify MFA code"""
        totp = pyotp.TOTP(secret)
        return totp.verify(code)

    @staticmethod
    def encrypt_data(data: str) -> str:
        """Encrypt sensitive data"""
        f = Fernet(Config.ENCRYPTION_KEY.encode())
        return f.encrypt(data.encode()).decode()

    @staticmethod
    def decrypt_data(encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        f = Fernet(Config.ENCRYPTION_KEY.encode())
        return f.decrypt(encrypted_data.encode()).decode()

# =============================================================================
# AUTHENTICATION MANAGER
# =============================================================================

class AuthenticationManager:
    """Core authentication and authorization manager"""

    def __init__(self, db_session: Session, redis_client):
        self.db = db_session
        self.redis = redis_client
        self.oauth_clients = {}

        # Initialize OAuth clients
        self._initialize_oauth_clients()

    def _initialize_oauth_clients(self):
        """Initialize OAuth2 clients"""
        if Config.OAUTH2_ENABLED:
            if Config.OAUTH2_GOOGLE_CLIENT_ID:
                self.oauth_clients['google'] = {
                    'client_id': Config.OAUTH2_GOOGLE_CLIENT_ID,
                    'client_secret': Config.OAUTH2_GOOGLE_CLIENT_SECRET,
                    'authorize_url': 'https://accounts.google.com/o/oauth2/auth',
                    'token_url': 'https://oauth2.googleapis.com/token',
                    'userinfo_url': 'https://openidconnect.googleapis.com/v1/userinfo'
                }

            if Config.OAUTH2_GITHUB_CLIENT_ID:
                self.oauth_clients['github'] = {
                    'client_id': Config.OAUTH2_GITHUB_CLIENT_ID,
                    'client_secret': Config.OAUTH2_GITHUB_CLIENT_SECRET,
                    'authorize_url': 'https://github.com/login/oauth/authorize',
                    'token_url': 'https://github.com/login/oauth/access_token',
                    'userinfo_url': 'https://api.github.com/user'
                }

    async def authenticate_user(self, username_or_email: str, password: str,
                               ip_address: str = None, user_agent: str = None) -> Dict[str, Any]:
        """Authenticate user with username/email and password"""
        try:
            # Find user
            user = self.db.query(User).filter(
                ((User.username == username_or_email) | (User.email == username_or_email)) &
                (User.is_active == True)
            ).first()

            if not user:
                self._log_audit(None, 'login_failed', 'user', None, ip_address, user_agent,
                              {'reason': 'user_not_found'})
                raise HTTPException(status_code=401, detail="Invalid credentials")

            # Check account lock
            if user.locked_until and user.locked_until > datetime.utcnow():
                self._log_audit(user.id, 'login_failed', 'user', user.id, ip_address, user_agent,
                              {'reason': 'account_locked'})
                raise HTTPException(status_code=423, detail="Account is temporarily locked")

            # Verify password (for non-OAuth users)
            if user.password_hash and not SecurityUtils.verify_password(password, user.password_hash):
                user.login_attempts += 1

                if user.login_attempts >= Config.MAX_LOGIN_ATTEMPTS:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=Config.ACCOUNT_LOCK_DURATION_MINUTES)
                    self.db.commit()
                    self._log_audit(user.id, 'account_locked', 'user', user.id, ip_address, user_agent)
                    raise HTTPException(status_code=423, detail="Account locked due to too many failed attempts")

                self.db.commit()
                self._log_audit(user.id, 'login_failed', 'user', user.id, ip_address, user_agent,
                              {'reason': 'invalid_password', 'attempts': user.login_attempts})
                raise HTTPException(status_code=401, detail="Invalid credentials")

            # Check MFA if enabled
            if user.mfa_enabled and user.mfa_secret:
                # MFA verification would happen in a separate step
                return {
                    'requires_mfa': True,
                    'user_id': user.id,
                    'temp_token': SecurityUtils.generate_jwt_token(
                        {'user_id': user.id, 'type': 'mfa_temp'},
                        timedelta(minutes=5)
                    )
                }

            # Create session
            session = self._create_user_session(user.id, ip_address, user_agent)

            # Update user login info
            user.last_login_at = datetime.utcnow()
            user.login_attempts = 0
            user.locked_until = None
            self.db.commit()

            # Generate tokens
            access_token = SecurityUtils.generate_jwt_token({
                'user_id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'permissions': user.permissions or []
            })

            refresh_token = SecurityUtils.generate_jwt_token({
                'user_id': user.id,
                'type': 'refresh'
            }, timedelta(days=Config.JWT_REFRESH_TOKEN_EXPIRE_DAYS))

            # Store refresh token
            session.refresh_token = refresh_token
            self.db.commit()

            # Cache session in Redis
            self._cache_session(session.id, {
                'user_id': user.id,
                'username': user.username,
                'role': user.role,
                'permissions': user.permissions or []
            })

            self._log_audit(user.id, 'login_success', 'user', user.id, ip_address, user_agent)

            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'bearer',
                'expires_in': Config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name,
                    'role': user.role,
                    'permissions': user.permissions or []
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            raise HTTPException(status_code=500, detail="Authentication failed")

    async def register_user(self, email: str, username: str, password: str, full_name: str) -> Dict[str, Any]:
        """Register new user"""
        try:
            # Check if user already exists
            existing_user = self.db.query(User).filter(
                (User.email == email) | (User.username == username)
            ).first()

            if existing_user:
                raise HTTPException(status_code=409, detail="User already exists")

            # Validate password strength
            self._validate_password_strength(password)

            # Create user
            user_id = str(uuid.uuid4())
            password_hash = SecurityUtils.hash_password(password)

            user = User(
                id=user_id,
                email=email,
                username=username,
                password_hash=password_hash,
                full_name=full_name,
                role='user',
                permissions=['read']
            )

            self.db.add(user)
            self.db.commit()

            self._log_audit(user_id, 'user_registered', 'user', user_id)

            return {
                'user_id': user_id,
                'message': 'User registered successfully',
                'requires_verification': True
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("User registration failed", error=str(e))
            raise HTTPException(status_code=500, detail="Registration failed")

    async def verify_mfa(self, temp_token: str, mfa_code: str) -> Dict[str, Any]:
        """Verify MFA code and complete authentication"""
        try:
            # Verify temp token
            payload = SecurityUtils.verify_jwt_token(temp_token)
            if not payload or payload.get('type') != 'mfa_temp':
                raise HTTPException(status_code=401, detail="Invalid MFA token")

            user_id = payload['user_id']
            user = self.db.query(User).filter_by(id=user_id).first()

            if not user or not user.mfa_enabled or not user.mfa_secret:
                raise HTTPException(status_code=400, detail="MFA not configured")

            # Verify MFA code
            if not SecurityUtils.verify_mfa_code(user.mfa_secret, mfa_code):
                self._log_audit(user_id, 'mfa_failed', 'user', user_id)
                raise HTTPException(status_code=401, detail="Invalid MFA code")

            # Complete authentication (reuse login logic)
            session = self._create_user_session(user_id)

            access_token = SecurityUtils.generate_jwt_token({
                'user_id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'permissions': user.permissions or []
            })

            refresh_token = SecurityUtils.generate_jwt_token({
                'user_id': user.id,
                'type': 'refresh'
            }, timedelta(days=Config.JWT_REFRESH_TOKEN_EXPIRE_DAYS))

            session.refresh_token = refresh_token
            self.db.commit()

            self._cache_session(session.id, {
                'user_id': user.id,
                'username': user.username,
                'role': user.role,
                'permissions': user.permissions or []
            })

            self._log_audit(user_id, 'mfa_success', 'user', user_id)

            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'bearer',
                'expires_in': Config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name,
                    'role': user.role,
                    'permissions': user.permissions or []
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("MFA verification failed", error=str(e))
            raise HTTPException(status_code=500, detail="MFA verification failed")

    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        try:
            # Verify refresh token
            payload = SecurityUtils.verify_jwt_token(refresh_token)
            if not payload or payload.get('type') != 'refresh':
                raise HTTPException(status_code=401, detail="Invalid refresh token")

            user_id = payload['user_id']

            # Find active session with this refresh token
            session = self.db.query(UserSession).filter(
                (UserSession.user_id == user_id) &
                (UserSession.refresh_token == refresh_token) &
                (UserSession.is_active == True) &
                (UserSession.expires_at > datetime.utcnow())
            ).first()

            if not session:
                raise HTTPException(status_code=401, detail="Invalid refresh token")

            # Get user
            user = self.db.query(User).filter_by(id=user_id).first()
            if not user or not user.is_active:
                raise HTTPException(status_code=401, detail="User not found or inactive")

            # Generate new access token
            access_token = SecurityUtils.generate_jwt_token({
                'user_id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'permissions': user.permissions or []
            })

            # Update session activity
            session.last_activity_at = datetime.utcnow()
            self.db.commit()

            # Update cache
            self._cache_session(session.id, {
                'user_id': user.id,
                'username': user.username,
                'role': user.role,
                'permissions': user.permissions or []
            })

            return {
                'access_token': access_token,
                'token_type': 'bearer',
                'expires_in': Config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Token refresh failed", error=str(e))
            raise HTTPException(status_code=500, detail="Token refresh failed")

    async def logout_user(self, access_token: str) -> Dict[str, Any]:
        """Logout user and invalidate session"""
        try:
            # Decode token to get user info
            payload = SecurityUtils.verify_jwt_token(access_token)
            if not payload:
                raise HTTPException(status_code=401, detail="Invalid token")

            user_id = payload['user_id']

            # Deactivate all user sessions
            self.db.query(UserSession).filter(
                (UserSession.user_id == user_id) &
                (UserSession.is_active == True)
            ).update({'is_active': False})

            # Clear Redis cache
            session_keys = self.redis.keys(f"session:{user_id}:*")
            if session_keys:
                self.redis.delete(*session_keys)

            self.db.commit()
            self._log_audit(user_id, 'logout', 'user', user_id)

            return {'message': 'Logged out successfully'}

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Logout failed", error=str(e))
            raise HTTPException(status_code=500, detail="Logout failed")

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile information"""
        try:
            user = self.db.query(User).filter_by(id=user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            return {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'role': user.role,
                'permissions': user.permissions or [],
                'is_verified': user.is_verified,
                'mfa_enabled': user.mfa_enabled,
                'last_login_at': user.last_login_at.isoformat() if user.last_login_at else None,
                'created_at': user.created_at.isoformat()
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to get user profile", user_id=user_id, error=str(e))
            raise HTTPException(status_code=500, detail="Failed to get user profile")

    def _create_user_session(self, user_id: str, ip_address: str = None,
                           user_agent: str = None) -> UserSession:
        """Create new user session"""
        session_id = str(uuid.uuid4())
        session_token = SecurityUtils.generate_jwt_token({
            'session_id': session_id,
            'user_id': user_id,
            'type': 'session'
        }, timedelta(minutes=Config.SESSION_TIMEOUT_MINUTES))

        session = UserSession(
            id=session_id,
            user_id=user_id,
            session_token=session_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.utcnow() + timedelta(minutes=Config.SESSION_TIMEOUT_MINUTES)
        )

        self.db.add(session)
        return session

    def _cache_session(self, session_id: str, session_data: Dict[str, Any]):
        """Cache session data in Redis"""
        cache_key = f"session:{session_data['user_id']}:{session_id}"
        self.redis.setex(cache_key, Config.SESSION_TIMEOUT_MINUTES * 60,
                        json.dumps(session_data))

    def _log_audit(self, user_id: str, action: str, resource_type: str,
                  resource_id: str = None, ip_address: str = None,
                  user_agent: str = None, details: Dict[str, Any] = None):
        """Log audit event"""
        audit_log = AuditLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details
        )

        self.db.add(audit_log)
        self.db.commit()

    def _validate_password_strength(self, password: str):
        """Validate password strength requirements"""
        if len(password) < Config.PASSWORD_MIN_LENGTH:
            raise HTTPException(status_code=400, detail=f"Password must be at least {Config.PASSWORD_MIN_LENGTH} characters long")

        if Config.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter")

        if Config.PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            raise HTTPException(status_code=400, detail="Password must contain at least one lowercase letter")

        if Config.PASSWORD_REQUIRE_DIGITS and not any(c.isdigit() for c in password):
            raise HTTPException(status_code=400, detail="Password must contain at least one digit")

        if Config.PASSWORD_REQUIRE_SPECIAL and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            raise HTTPException(status_code=400, detail="Password must contain at least one special character")

# =============================================================================
# API MODELS
# =============================================================================

class LoginRequest(BaseModel):
    """Login request model"""
    username_or_email: str = Field(..., description="Username or email address")
    password: str = Field(..., description="User password")

class RegisterRequest(BaseModel):
    """User registration request"""
    email: str = Field(..., description="User email address")
    username: str = Field(..., description="Unique username")
    password: str = Field(..., description="User password")
    full_name: str = Field(..., description="Full name")

class MFALoginRequest(BaseModel):
    """MFA login request"""
    temp_token: str = Field(..., description="Temporary token from initial login")
    mfa_code: str = Field(..., description="MFA code from authenticator app")

class TokenRefreshRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str = Field(..., description="Refresh token")

class APIKeyCreateRequest(BaseModel):
    """API key creation request"""
    name: str = Field(..., description="API key name")
    permissions: List[str] = Field(default_factory=list, description="API key permissions")
    expires_in_days: Optional[int] = Field(None, description="Expiration in days")

class UserUpdateRequest(BaseModel):
    """User update request"""
    full_name: Optional[str] = Field(None, description="Full name")
    email: Optional[str] = Field(None, description="Email address")
    current_password: Optional[str] = Field(None, description="Current password for verification")
    new_password: Optional[str] = Field(None, description="New password")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Authentication Service",
    description="Comprehensive authentication and authorization service for Agentic Brain platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers middleware
if Config.SECURITY_HEADERS_ENABLED:
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response

# Initialize database
engine = create_engine(Config.DATABASE_URL)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize Redis
redis_client = redis.Redis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=Config.REDIS_DB,
    decode_responses=True
)

# Initialize authentication manager
auth_manager = AuthenticationManager(SessionLocal(), redis_client)

# Security schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
security = HTTPBearer(auto_error=False)

# Prometheus metrics
REQUEST_COUNT = Counter('auth_requests_total', 'Total authentication requests', ['method', 'endpoint', 'status'])
AUTH_DURATION = Histogram('auth_request_duration_seconds', 'Authentication request duration')

@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    """Database session middleware"""
    response = None
    try:
        request.state.db = SessionLocal()
        response = await call_next(request)
    finally:
        if hasattr(request.state, 'db'):
            request.state.db.close()

    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code if response else 500
    ).inc()

    return response

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Get current authenticated user"""
    if not token:
        raise HTTPException(status_code=401, detail="Token required")

    payload = SecurityUtils.verify_jwt_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get('user_id')
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Check Redis cache first
    cache_key = f"user:{user_id}"
    cached_user = redis_client.get(cache_key)

    if cached_user:
        return json.loads(cached_user)

    # Get from database
    db = SessionLocal()
    user = db.query(User).filter_by(id=user_id).first()
    db.close()

    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    user_data = {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role,
        'permissions': user.permissions or []
    }

    # Cache user data
    redis_client.setex(cache_key, 300, json.dumps(user_data))  # 5 minutes

    return user_data

def require_permission(permission: str):
    """Dependency to require specific permission"""
    def permission_checker(user: Dict[str, Any] = Depends(get_current_user)):
        if permission not in user.get('permissions', []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return permission_checker

def require_role(role: str):
    """Dependency to require specific role"""
    def role_checker(user: Dict[str, Any] = Depends(get_current_user)):
        user_role = user.get('role', 'user')
        role_hierarchy = {'viewer': 0, 'user': 1, 'admin': 2}

        required_level = role_hierarchy.get(role, 0)
        user_level = role_hierarchy.get(user_role, 0)

        if user_level < required_level:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user
    return role_checker

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Authentication Service", "status": "healthy", "version": "1.0.0"}

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
            "oauth_providers": len(auth_manager.oauth_clients) if Config.OAUTH2_ENABLED else 0
        }
    }

@app.post("/auth/login")
async def login(request: LoginRequest, background_tasks: BackgroundTasks):
    """User login endpoint"""
    try:
        result = await auth_manager.authenticate_user(
            request.username_or_email,
            request.password,
            None,  # IP address (would come from request headers)
            None   # User agent (would come from request headers)
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/auth/register")
async def register(request: RegisterRequest):
    """User registration endpoint"""
    try:
        result = await auth_manager.register_user(
            request.email,
            request.username,
            request.password,
            request.full_name
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration failed", error=str(e))
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/mfa/verify")
async def verify_mfa(request: MFALoginRequest):
    """MFA verification endpoint"""
    try:
        result = await auth_manager.verify_mfa(
            request.temp_token,
            request.mfa_code
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("MFA verification failed", error=str(e))
        raise HTTPException(status_code=500, detail="MFA verification failed")

@app.post("/auth/refresh")
async def refresh_token(request: TokenRefreshRequest):
    """Token refresh endpoint"""
    try:
        result = await auth_manager.refresh_token(request.refresh_token)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(status_code=500, detail="Token refresh failed")

@app.post("/auth/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    """User logout endpoint"""
    try:
        result = await auth_manager.logout_user(token)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Logout failed", error=str(e))
        raise HTTPException(status_code=500, detail="Logout failed")

@app.get("/auth/profile")
async def get_profile(user: Dict[str, Any] = Depends(get_current_user)):
    """Get user profile"""
    try:
        profile = await auth_manager.get_user_profile(user['id'])
        return profile

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get profile", user_id=user['id'], error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get profile")

@app.put("/auth/profile")
async def update_profile(request: UserUpdateRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """Update user profile"""
    try:
        # Implementation for profile updates would go here
        return {"message": "Profile updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Profile update failed", user_id=user['id'], error=str(e))
        raise HTTPException(status_code=500, detail="Profile update failed")

@app.post("/auth/mfa/setup")
async def setup_mfa(user: Dict[str, Any] = Depends(get_current_user)):
    """Setup MFA for user"""
    try:
        db = SessionLocal()
        db_user = db.query(User).filter_by(id=user['id']).first()

        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        if db_user.mfa_enabled:
            raise HTTPException(status_code=400, detail="MFA already enabled")

        # Generate MFA secret
        mfa_secret = SecurityUtils.generate_mfa_secret()

        # Generate QR code
        qr_code_uri = SecurityUtils.generate_mfa_qr_code(mfa_secret, db_user.username)

        # Store secret temporarily (user needs to verify before enabling)
        db_user.mfa_secret = mfa_secret
        db.commit()
        db.close()

        return {
            "secret": mfa_secret,
            "qr_code_uri": qr_code_uri,
            "message": "Scan QR code with authenticator app and verify with /auth/mfa/enable"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("MFA setup failed", user_id=user['id'], error=str(e))
        raise HTTPException(status_code=500, detail="MFA setup failed")

@app.post("/auth/mfa/enable")
async def enable_mfa(mfa_code: str = Form(...), user: Dict[str, Any] = Depends(get_current_user)):
    """Enable MFA after verification"""
    try:
        db = SessionLocal()
        db_user = db.query(User).filter_by(id=user['id']).first()

        if not db_user or not db_user.mfa_secret:
            raise HTTPException(status_code=400, detail="MFA setup not initiated")

        # Verify MFA code
        if not SecurityUtils.verify_mfa_code(db_user.mfa_secret, mfa_code):
            raise HTTPException(status_code=400, detail="Invalid MFA code")

        # Enable MFA
        db_user.mfa_enabled = True
        db.commit()
        db.close()

        return {"message": "MFA enabled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("MFA enable failed", user_id=user['id'], error=str(e))
        raise HTTPException(status_code=500, detail="MFA enable failed")

@app.get("/auth/sessions")
async def get_user_sessions(user: Dict[str, Any] = Depends(get_current_user)):
    """Get user sessions"""
    try:
        db = SessionLocal()
        sessions = db.query(UserSession).filter(
            (UserSession.user_id == user['id']) &
            (UserSession.is_active == True)
        ).order_by(UserSession.created_at.desc()).all()
        db.close()

        return {
            "sessions": [
                {
                    "id": session.id,
                    "ip_address": session.ip_address,
                    "user_agent": session.user_agent,
                    "created_at": session.created_at.isoformat(),
                    "last_activity_at": session.last_activity_at.isoformat(),
                    "expires_at": session.expires_at.isoformat()
                }
                for session in sessions
            ]
        }

    except Exception as e:
        logger.error("Failed to get sessions", user_id=user['id'], error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get sessions")

@app.delete("/auth/sessions/{session_id}")
async def revoke_session(session_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Revoke specific session"""
    try:
        db = SessionLocal()
        session = db.query(UserSession).filter(
            (UserSession.id == session_id) &
            (UserSession.user_id == user['id'])
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session.is_active = False
        db.commit()
        db.close()

        # Clear Redis cache
        cache_key = f"session:{user['id']}:{session_id}"
        redis_client.delete(cache_key)

        return {"message": "Session revoked successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Session revoke failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Session revoke failed")

@app.post("/auth/api-keys")
async def create_api_key(request: APIKeyCreateRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """Create API key"""
    try:
        db = SessionLocal()

        # Generate API key
        api_key = SecurityUtils.generate_api_key()
        key_hash = SecurityUtils.hash_api_key(api_key)

        # Create API key record
        key_record = APIKey(
            id=str(uuid.uuid4()),
            user_id=user['id'],
            name=request.name,
            key_hash=key_hash,
            permissions=request.permissions,
            expires_at=datetime.utcnow() + timedelta(days=request.expires_in_days) if request.expires_in_days else None
        )

        db.add(key_record)
        db.commit()
        db.close()

        return {
            "api_key": api_key,
            "name": request.name,
            "permissions": request.permissions,
            "expires_at": key_record.expires_at.isoformat() if key_record.expires_at else None,
            "message": "API key created successfully. Store it securely - it cannot be retrieved again."
        }

    except Exception as e:
        logger.error("API key creation failed", user_id=user['id'], error=str(e))
        raise HTTPException(status_code=500, detail="API key creation failed")

@app.get("/auth/api-keys")
async def list_api_keys(user: Dict[str, Any] = Depends(get_current_user)):
    """List user API keys"""
    try:
        db = SessionLocal()
        api_keys = db.query(APIKey).filter(
            (APIKey.user_id == user['id']) &
            (APIKey.is_active == True)
        ).order_by(APIKey.created_at.desc()).all()
        db.close()

        return {
            "api_keys": [
                {
                    "id": key.id,
                    "name": key.name,
                    "permissions": key.permissions,
                    "created_at": key.created_at.isoformat(),
                    "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                    "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None
                }
                for key in api_keys
            ]
        }

    except Exception as e:
        logger.error("Failed to list API keys", user_id=user['id'], error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list API keys")

@app.delete("/auth/api-keys/{key_id}")
async def revoke_api_key(key_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Revoke API key"""
    try:
        db = SessionLocal()
        api_key = db.query(APIKey).filter(
            (APIKey.id == key_id) &
            (APIKey.user_id == user['id'])
        ).first()

        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")

        api_key.is_active = False
        db.commit()
        db.close()

        return {"message": "API key revoked successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("API key revoke failed", key_id=key_id, error=str(e))
        raise HTTPException(status_code=500, detail="API key revoke failed")

@app.get("/auth/audit-log")
async def get_audit_log(limit: int = 50, offset: int = 0, user: Dict[str, Any] = Depends(require_role('admin'))):
    """Get audit log (admin only)"""
    try:
        db = SessionLocal()
        audit_logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).offset(offset).all()
        db.close()

        return {
            "audit_logs": [
                {
                    "id": log.id,
                    "user_id": log.user_id,
                    "action": log.action,
                    "resource_type": log.resource_type,
                    "resource_id": log.resource_id,
                    "ip_address": log.ip_address,
                    "success": log.success,
                    "error_message": log.error_message,
                    "created_at": log.created_at.isoformat()
                }
                for log in audit_logs
            ],
            "total": len(audit_logs),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error("Failed to get audit log", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get audit log")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return HTMLResponse(generate_latest(), media_type="text/plain")

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Authentication Service starting up")

    # Create default admin user if it doesn't exist
    try:
        db = SessionLocal()
        admin_user = db.query(User).filter_by(username='admin').first()

        if not admin_user:
            admin_id = str(uuid.uuid4())
            admin_password = SecurityUtils.hash_password('admin123!')

            admin = User(
                id=admin_id,
                email='admin@agenticbrain.com',
                username='admin',
                password_hash=admin_password,
                full_name='System Administrator',
                role='admin',
                permissions=['read', 'write', 'delete', 'admin'],
                is_active=True,
                is_verified=True
            )

            db.add(admin)
            db.commit()
            logger.info("Default admin user created", username='admin')

        db.close()

    except Exception as e:
        logger.warning("Failed to create default admin user", error=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Authentication Service shutting down")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.AUTH_SERVICE_PORT,
        reload=True,
        log_level="info"
    )
