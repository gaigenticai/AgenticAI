# Port Manager Service for Automatic Port Conflict Resolution
FROM python:3.11-slim

# Install required packages
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY port_manager.py .

# Create non-root user
RUN useradd --create-home --shell /bin/bash portmanager && \
    chown -R portmanager:portmanager /app
USER portmanager

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1

# Expose port
EXPOSE 8082

# Run the port manager
CMD ["python", "port_manager.py"]
