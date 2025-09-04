# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for performance monitoring
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    curl \
    wget \
    gnupg \
    procps \
    sysstat \
    && rm -rf /var/lib/apt/lists/*

# Create directories for performance data and logs
RUN mkdir -p /app/performance_data /app/optimization_logs

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create directories with proper permissions
RUN chmod -R 755 /app/performance_data /app/optimization_logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8420/health || exit 1

# Expose port
EXPOSE 8420

# Run the application
CMD ["python", "main.py"]
