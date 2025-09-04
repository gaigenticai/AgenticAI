# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for testing frameworks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    curl \
    wget \
    gnupg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create directories for test workspaces and results
RUN mkdir -p /app/test_workspace /app/test_results /app/test_logs

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create test directories with proper permissions
RUN chmod -R 755 /app/test_workspace /app/test_results /app/test_logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8390/health || exit 1

# Expose port
EXPOSE 8390

# Run the application
CMD ["python", "main.py"]
