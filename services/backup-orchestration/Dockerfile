# Backup Orchestration Service
# Comprehensive backup and disaster recovery orchestration

FROM python:3.11-slim

# Install system dependencies for backup tools
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-client \
    mysql-client \
    mongodb-clients \
    redis-tools \
    curl \
    wget \
    gnupg \
    lsb-release \
    openssh-client \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash backup && \
    chown -R backup:backup /app
USER backup

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8097/health || exit 1

# Expose port
EXPOSE 8097

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "backup_orchestration_service.py"]
