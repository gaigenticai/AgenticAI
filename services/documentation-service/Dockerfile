# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for documentation processing
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
    imagemagick \
    libmagic1 \
    libmagic-dev \
    file \
    && rm -rf /var/lib/apt/lists/*

# Create directories for documentation content and media
RUN mkdir -p /app/docs_content /app/docs_media /app/templates

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data for text processing
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY main.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create directories with proper permissions
RUN chmod -R 755 /app/docs_content /app/docs_media /app/templates

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8410/health || exit 1

# Expose port
EXPOSE 8410

# Run the application
CMD ["python", "main.py"]
