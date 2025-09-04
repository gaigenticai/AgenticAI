# Authentication Service

A comprehensive authentication and authorization service for the Agentic Brain platform that provides secure user management, JWT token handling, OAuth2/OIDC integration, role-based access control, and enterprise-grade security features.

## 🎯 Features

### Core Authentication
- **User Registration & Login**: Secure user account creation and authentication
- **JWT Token Management**: Access tokens and refresh token rotation
- **Session Management**: Secure session tracking and timeout handling
- **Password Security**: Bcrypt hashing with configurable strength requirements
- **Account Security**: Login attempt monitoring and account lockout protection

### Authorization & Access Control
- **Role-Based Access Control (RBAC)**: Hierarchical permission system
- **Permission Management**: Granular resource-level permissions
- **API Key Support**: Secure API access for integrations
- **Multi-Factor Authentication (MFA)**: TOTP-based two-factor authentication
- **Session Security**: Secure session management with activity tracking

### OAuth2 & Social Login
- **OAuth2 Integration**: Support for Google, GitHub, and custom providers
- **OIDC Compliance**: OpenID Connect protocol implementation
- **Social Login**: Seamless authentication via external providers
- **Provider Management**: Configurable OAuth provider settings
- **Token Exchange**: Secure token exchange and validation

### Enterprise Security
- **Audit Logging**: Comprehensive security event logging
- **Data Encryption**: Sensitive data encryption at rest and in transit
- **Security Headers**: OWASP security headers and CSRF protection
- **Rate Limiting**: Configurable request rate limiting
- **Compliance**: GDPR, CCPA, and SOX compliance features

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- PostgreSQL database
- Redis instance
- SMTP server (for email verification)

### Environment Setup
```bash
# Copy environment file
cp .env.example .env

# Configure environment variables
nano .env
```

### Docker Deployment
```bash
# Build and start the service
docker-compose up -d authentication-service

# Check service health
curl http://localhost:8330/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

## 📡 API Endpoints

### Authentication Endpoints
```http
POST /auth/login
Content-Type: application/json

{
  "username_or_email": "user@example.com",
  "password": "securepassword123"
}
```

```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "SecurePass123!",
  "full_name": "John Doe"
}
```

```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### User Management
```http
GET  /auth/profile          # Get user profile
PUT  /auth/profile          # Update user profile
POST /auth/logout           # Logout user
GET  /auth/sessions         # Get user sessions
DELETE /auth/sessions/{id}  # Revoke session
```

### Multi-Factor Authentication
```http
POST /auth/mfa/setup        # Setup MFA
POST /auth/mfa/enable       # Enable MFA after verification
POST /auth/mfa/verify       # Verify MFA code
```

### API Key Management
```http
POST /auth/api-keys         # Create API key
GET  /auth/api-keys         # List API keys
DELETE /auth/api-keys/{id}  # Revoke API key
```

### Administrative Endpoints
```http
GET /auth/audit-log         # Get audit log (admin only)
GET /metrics                # Prometheus metrics
GET /health                 # Service health check
```

## ⚙️ Configuration

### Environment Variables
```bash
# Service Configuration
AUTH_SERVICE_PORT=8330
AUTH_SERVICE_HOST=0.0.0.0

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Password Security
PASSWORD_MIN_LENGTH=8
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_DIGITS=true
PASSWORD_REQUIRE_SPECIAL=false

# Session Management
SESSION_TIMEOUT_MINUTES=60
MAX_LOGIN_ATTEMPTS=5
ACCOUNT_LOCK_DURATION_MINUTES=30

# MFA Settings
MFA_ISSUER_NAME=Agentic Brain

# OAuth2 Configuration
OAUTH2_ENABLED=false
OAUTH2_GOOGLE_CLIENT_ID=your-google-client-id
OAUTH2_GOOGLE_CLIENT_SECRET=your-google-client-secret
OAUTH2_GITHUB_CLIENT_ID=your-github-client-id
OAUTH2_GITHUB_CLIENT_SECRET=your-github-client-secret

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=noreply@agenticbrain.com

# Security
ENCRYPTION_KEY=your-32-byte-encryption-key
SECURITY_HEADERS_ENABLED=true

# CORS
CORS_ORIGINS=http://localhost:8300,http://localhost:3000

# Database & Redis
DATABASE_URL=postgresql://user:password@postgres:5432/agentic_brain
REDIS_HOST=redis
REDIS_PORT=6379

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8003
```

### Database Schema
```sql
-- User accounts
CREATE TABLE users (
    id VARCHAR(100) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    full_name VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    email_verified_at TIMESTAMP,
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    oauth_provider VARCHAR(50),
    oauth_id VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',
    permissions JSON,
    profile_data JSON,
    last_login_at TIMESTAMP,
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User sessions
CREATE TABLE user_sessions (
    id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address VARCHAR(45),
    user_agent TEXT,
    device_info JSON,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API keys
CREATE TABLE api_keys (
    id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    permissions JSON,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- OAuth providers
CREATE TABLE oauth_providers (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    client_id VARCHAR(255) NOT NULL,
    client_secret VARCHAR(255) NOT NULL,
    authorization_url VARCHAR(500) NOT NULL,
    token_url VARCHAR(500) NOT NULL,
    user_info_url VARCHAR(500) NOT NULL,
    scope VARCHAR(255) DEFAULT 'openid email profile',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit logging
CREATE TABLE audit_logs (
    id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    ip_address VARCHAR(45),
    user_agent TEXT,
    details JSON,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Roles and permissions
CREATE TABLE roles (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    permissions JSON NOT NULL,
    is_system_role BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔐 Authentication Flow

### Standard Login Flow
```mermaid
sequenceDiagram
    participant U as User
    participant A as Auth Service
    participant DB as Database

    U->>A: POST /auth/login
    A->>DB: Validate credentials
    DB-->>A: User data
    A->>A: Generate JWT tokens
    A->>DB: Create session
    A-->>U: Access & refresh tokens
```

### MFA Authentication Flow
```mermaid
sequenceDiagram
    participant U as User
    participant A as Auth Service
    participant DB as Database

    U->>A: POST /auth/login
    A->>DB: Validate credentials
    DB-->>A: User data (MFA enabled)
    A-->>U: MFA required + temp token

    U->>A: POST /auth/mfa/verify
    A->>A: Verify MFA code
    A->>A: Generate JWT tokens
    A->>DB: Create session
    A-->>U: Access & refresh tokens
```

### OAuth2 Flow
```mermaid
sequenceDiagram
    participant U as User
    participant A as Auth Service
    participant OP as OAuth Provider
    participant DB as Database

    U->>A: GET /auth/oauth/{provider}
    A->>OP: Redirect to OAuth provider
    OP-->>U: Authorization page
    U->>OP: Grant permission
    OP->>A: Authorization code
    A->>OP: Exchange code for tokens
    OP-->>A: Access token + user info
    A->>DB: Create/update user
    A->>A: Generate JWT tokens
    A-->>U: Access & refresh tokens
```

## 🛡️ Security Features

### Password Security
- **Bcrypt Hashing**: Industry-standard password hashing
- **Strength Requirements**: Configurable complexity rules
- **Account Lockout**: Progressive delay after failed attempts
- **Password History**: Prevent reuse of recent passwords

### Session Security
- **JWT Tokens**: Secure, stateless authentication
- **Refresh Tokens**: Secure token rotation
- **Session Timeout**: Automatic session expiration
- **Concurrent Session Limits**: Prevent session abuse
- **Device Tracking**: Monitor login devices and locations

### API Security
- **Rate Limiting**: Prevent brute force attacks
- **Request Validation**: Comprehensive input validation
- **CORS Protection**: Cross-origin request protection
- **Security Headers**: OWASP security headers
- **Audit Logging**: Complete request/response logging

## 🔧 Integration Examples

### FastAPI Integration
```python
from fastapi import Depends, HTTPException
from httpx import AsyncClient

# Dependency for protected routes
async def get_current_user(token: str = Depends(oauth2_scheme)):
    async with AsyncClient() as client:
        response = await client.get(
            "http://localhost:8330/auth/verify",
            headers={"Authorization": f"Bearer {token}"}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid token")

        return response.json()

# Protected route
@app.get("/protected")
async def protected_route(user = Depends(get_current_user)):
    return {"message": f"Hello {user['username']}", "user": user}
```

### API Key Authentication
```python
from fastapi import Header, HTTPException

# API key dependency
async def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")

    async with AsyncClient() as client:
        response = await client.post(
            "http://localhost:8330/auth/verify-api-key",
            json={"api_key": x_api_key}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return response.json()

# API key protected route
@app.get("/api/data")
async def get_data(api_user = Depends(verify_api_key)):
    return {"data": "Protected resource", "user": api_user}
```

### Role-Based Access Control
```python
from fastapi import Depends, HTTPException

# Role dependency
def require_role(required_role: str):
    async def role_checker(user = Depends(get_current_user)):
        user_role = user.get('role', 'user')
        role_hierarchy = {'viewer': 0, 'user': 1, 'admin': 2}

        if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 0):
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        return user
    return role_checker

# Admin-only route
@app.get("/admin/dashboard")
async def admin_dashboard(user = Depends(require_role('admin'))):
    return {"message": "Admin dashboard", "user": user}
```

## 📊 Monitoring & Analytics

### Metrics Collection
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Authentication metrics
AUTH_REQUESTS = Counter('auth_requests_total', 'Total authentication requests', ['method', 'status'])
LOGIN_DURATION = Histogram('auth_login_duration_seconds', 'Login request duration')
ACTIVE_SESSIONS = Gauge('auth_active_sessions', 'Number of active sessions')
FAILED_LOGIN_ATTEMPTS = Counter('auth_failed_logins_total', 'Total failed login attempts')

# API metrics
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
RESPONSE_TIME = Histogram('api_response_time_seconds', 'API response time', ['endpoint'])
```

### Audit Logging
```python
# Comprehensive audit logging
async def log_audit_event(user_id, action, resource_type, resource_id, details=None):
    audit_data = {
        "user_id": user_id,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "ip_address": get_client_ip(),
        "user_agent": get_user_agent(),
        "timestamp": datetime.utcnow().isoformat(),
        "details": details
    }

    # Store in database
    await store_audit_log(audit_data)

    # Send to monitoring system
    await send_to_monitoring(audit_data)
```

### Health Monitoring
```python
# Service health checks
@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {
            "database": await check_database_health(),
            "redis": await check_redis_health(),
            "oauth_providers": await check_oauth_providers(),
            "email_service": await check_email_service()
        }
    }

    # Calculate overall health
    all_healthy = all(check['status'] == 'healthy' for check in health_status['checks'].values())
    health_status['status'] = 'healthy' if all_healthy else 'unhealthy'

    return health_status
```

## 🔧 Advanced Configuration

### OAuth2 Provider Setup
```python
# Google OAuth2 configuration
GOOGLE_OAUTH_CONFIG = {
    "client_id": os.getenv("OAUTH2_GOOGLE_CLIENT_ID"),
    "client_secret": os.getenv("OAUTH2_GOOGLE_CLIENT_SECRET"),
    "authorize_url": "https://accounts.google.com/o/oauth2/auth",
    "token_url": "https://oauth2.googleapis.com/token",
    "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
    "scope": "openid email profile",
    "redirect_uri": "http://localhost:8330/auth/oauth/google/callback"
}

# GitHub OAuth2 configuration
GITHUB_OAUTH_CONFIG = {
    "client_id": os.getenv("OAUTH2_GITHUB_CLIENT_ID"),
    "client_secret": os.getenv("OAUTH2_GITHUB_CLIENT_SECRET"),
    "authorize_url": "https://github.com/login/oauth/authorize",
    "token_url": "https://github.com/login/oauth/access_token",
    "userinfo_url": "https://api.github.com/user",
    "scope": "user:email",
    "redirect_uri": "http://localhost:8330/auth/oauth/github/callback"
}
```

### Custom Authentication Middleware
```python
# Custom authentication middleware
class AuthenticationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public routes
        if request.url.path in ['/health', '/metrics', '/auth/login', '/auth/register']:
            return await call_next(request)

        # Extract token from header
        authorization = request.headers.get('Authorization')
        if not authorization or not authorization.startswith('Bearer '):
            return JSONResponse(
                status_code=401,
                content={"detail": "Authorization header missing or invalid"}
            )

        token = authorization[7:]  # Remove 'Bearer ' prefix

        # Verify token with auth service
        async with AsyncClient() as client:
            response = await client.post(
                "http://localhost:8330/auth/verify-token",
                json={"token": token}
            )

            if response.status_code != 200:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or expired token"}
                )

            # Add user info to request state
            request.state.user = response.json()

        return await call_next(request)
```

### Database Migration
```python
# Alembic migration for database schema updates
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from authentication_service.models import Base

# this is the Alembic Config object
config = context.config

# add your model's MetaData object here
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

## 🚨 Troubleshooting

### Common Issues

#### Token Expiration
```bash
# Refresh token before expiration
curl -X POST http://localhost:8330/auth/refresh \
     -H "Content-Type: application/json" \
     -d '{"refresh_token": "your-refresh-token"}'
```

#### MFA Setup Issues
```bash
# Reset MFA for user (admin only)
curl -X POST http://localhost:8330/admin/users/{user_id}/mfa/reset \
     -H "Authorization: Bearer {admin-token}"
```

#### OAuth2 Configuration
```bash
# Test OAuth2 configuration
curl -X GET http://localhost:8330/auth/oauth/{provider}/test

# Debug OAuth2 flow
curl -X GET http://localhost:8330/auth/oauth/{provider}/debug
```

#### Database Connection Issues
```bash
# Test database connection
python -c "from authentication_service.main import SessionLocal; db = SessionLocal(); db.execute('SELECT 1'); print('Database connected')"

# Check database migrations
alembic current

# Run pending migrations
alembic upgrade head
```

#### Redis Connection Issues
```bash
# Test Redis connection
python -c "import redis; r = redis.Redis(host='localhost', port=6379); r.ping(); print('Redis connected')"

# Check Redis memory usage
redis-cli info memory
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export PYTHONPATH=/app

# Run with verbose output
python main.py --debug --verbose

# Enable request/response logging
export REQUEST_LOGGING=true
export RESPONSE_LOGGING=true
```

### Performance Tuning
```bash
# Database connection pool tuning
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_RECYCLE=3600

# Redis connection pool
REDIS_MAX_CONNECTIONS=20
REDIS_SOCKET_TIMEOUT=5

# JWT token caching
JWT_CACHE_TTL=300
JWT_CACHE_SIZE=10000

# Session management
SESSION_CLEANUP_INTERVAL=3600
MAX_SESSIONS_PER_USER=10
```

## 📚 API Documentation

### Complete API Reference
- [Authentication API](./docs/api/authentication.md)
- [User Management API](./docs/api/user-management.md)
- [OAuth2 Integration API](./docs/api/oauth2-integration.md)
- [MFA API](./docs/api/mfa.md)
- [API Keys API](./docs/api/api-keys.md)
- [Audit & Monitoring API](./docs/api/audit-monitoring.md)

### SDKs and Libraries
- **Python SDK**: `pip install agentic-brain-auth`
- **JavaScript SDK**: `npm install @agentic-brain/auth`
- **REST API Client**: Comprehensive HTTP client libraries

## 🤝 Contributing

### Development Setup
```bash
# Clone the repository
git clone https://github.com/agentic-brain/authentication-service.git
cd authentication-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v --asyncio-mode=auto

# Start development server
python main.py --reload --debug
```

### Code Standards
- Follow PEP 8 style guidelines for Python code
- Write comprehensive docstrings for all functions
- Include type hints for function parameters and return values
- Add unit tests for all new functionality
- Update documentation for API changes

### Pull Request Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/oauth2-integration`)
3. Make your changes with tests
4. Ensure all tests pass (`pytest tests/ -v`)
5. Update documentation if needed
6. Submit a pull request with detailed description

### Security Considerations
- Never commit secrets or private keys
- Use environment variables for configuration
- Implement proper input validation
- Follow OWASP security guidelines
- Regular security code reviews

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: [docs.agenticbrain.com/authentication](https://docs.agenticbrain.com/authentication)
- **Community Forum**: [community.agenticbrain.com](https://community.agenticbrain.com)
- **Issue Tracker**: [github.com/agentic-brain/authentication-service/issues](https://github.com/agentic-brain/authentication-service/issues)
- **Email Support**: support@agenticbrain.com

### Service Level Agreements
- **Response Time**: < 4 hours for critical security issues
- **Resolution Time**: < 24 hours for standard issues
- **Uptime**: 99.9% service availability
- **Security Updates**: < 72 hours for critical vulnerabilities
- **Support Hours**: 24/7 enterprise support available

---

**Built with ❤️ for the Agentic Brain Platform**

*Securing the future of AI-driven automation with enterprise-grade authentication*
