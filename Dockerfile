# COMPAS Bias Audit — Dockerfile
# Builds a container image for Google Cloud Run deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn google-generativeai

# Copy all project files
COPY . .

# Create required directories
RUN mkdir -p data output/charts output/metrics report

# Cloud Run listens on port 8080
EXPOSE 8080

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

