# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy templates and static files
COPY templates/ ./templates/
COPY static/ ./static/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash agentic && \
    chown -R agentic:agentic /app
USER agentic

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8300/health || exit 1

# Expose port
EXPOSE 8300

# Run the application
CMD ["python", "app.py"]
