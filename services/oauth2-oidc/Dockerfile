# OAuth2/OIDC Authentication Service
# Enterprise-grade authentication and authorization service

FROM python:3.11-slim

# Install system dependencies for cryptography and security
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash oauthservice && \
    chown -R oauthservice:oauthservice /app
USER oauthservice

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8093/health || exit 1

# Expose port
EXPOSE 8093

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "oauth2_oidc_service.py"]
