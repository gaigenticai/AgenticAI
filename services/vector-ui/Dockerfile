# Vector UI Service Dockerfile
# Professional web interface for vector operations testing

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create templates and static directories if they don't exist
RUN mkdir -p templates static/css static/js

# Create non-root user
RUN useradd --create-home --shell /bin/bash vectorui && \
    chown -R vectorui:vectorui /app

# Create static files
RUN echo '/* Vector UI Custom Styles */' > static/css/style.css && \
    echo '// Vector UI Custom JavaScript' > static/js/main.js

# Switch to non-root user
USER vectorui

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${VECTOR_UI_PORT:-8082}/health || exit 1

# Expose port
EXPOSE ${VECTOR_UI_PORT:-8082}

# Environment variables
ENV PYTHONPATH=/app
ENV VECTOR_UI_PORT=${VECTOR_UI_PORT:-8082}
ENV OUTPUT_COORDINATOR_URL=${OUTPUT_COORDINATOR_URL:-http://output-coordinator:8081}

# Start the application
CMD ["python", "vector_ui_service.py"]
