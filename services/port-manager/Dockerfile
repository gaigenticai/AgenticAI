# Port Manager Service Dockerfile
# Automatic port conflict resolution for Docker services

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy port resolver script
COPY docker/port_resolver.py docker/port_resolver.py
COPY docker-compose.yml docker-compose.yml
COPY .env.example .env.example

# Make script executable
RUN chmod +x docker/port_resolver.py

# Environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python", "docker/port_resolver.py"]