# PostgreSQL Output Service
# Handles data output to PostgreSQL databases with advanced features

FROM python:3.11-slim

# Install system dependencies for PostgreSQL and data processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash postgresqlservice && \
    chown -R postgresqlservice:postgresqlservice /app
USER postgresqlservice

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import psycopg2; psycopg2.connect('${POSTGRES_OUTPUT_URL}').close()" || exit 1

# Expose port
EXPOSE 8083

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "postgresql_output_service.py"]
