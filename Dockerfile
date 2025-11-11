FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (excluding shared-lib)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/shared-storage/processed /app/logs

# Install shared-lib at runtime via volume mount
# The shared-lib will be available at /app/shared-lib via the volume mount
RUN pip install --no-cache-dir -e /app/shared-lib

# Expose port
EXPOSE 9004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9004/health || exit 1

# Start the service
CMD ["uvicorn", "core.service:app", "--host", "0.0.0.0", "--port", "9004"]