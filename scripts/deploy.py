#!/usr/bin/env python3
"""
Deployment Script for Medical Multimodal Retrieval System

This script handles deployment to different environments:
- Development
- Staging  
- Production
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def deploy_development():
    """Deploy to development environment"""
    print("🔧 Deploying to development...")
    
    # Start API server
    api_process = subprocess.Popen([
        sys.executable, "app/api/main.py"
    ])
    
    # Start Streamlit app
    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "app/frontend/streamlit_app.py", "--server.port", "8501"
    ])
    
    print("✅ Development servers started!")
    print("  API: http://127.0.0.1:8000")
    print("  Streamlit: http://127.0.0.1:8501")
    print("  Press Ctrl+C to stop")
    
    try:
        # Wait for processes
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        api_process.terminate()
        streamlit_process.terminate()
        print("✅ Servers stopped")

def deploy_staging():
    """Deploy to staging environment"""
    print("🚀 Deploying to staging...")
    
    # Build Docker image
    subprocess.check_call([
        "docker", "build", "-t", "medical-retrieval:staging", "."
    ])
    
    # Run staging container
    subprocess.check_call([
        "docker", "run", "-d", 
        "--name", "medical-retrieval-staging",
        "-p", "8001:8000",
        "medical-retrieval:staging"
    ])
    
    print("✅ Staging deployed!")
    print("  URL: http://localhost:8001")

def deploy_production():
    """Deploy to production environment"""
    print("🌍 Deploying to production...")
    
    # Production checks
    print("Running pre-deployment checks...")
    
    # 1. Check if model exists
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # 2. Check if data exists
    data_path = "data/processed/test_processed.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        return False
    
    # 3. Run tests
    print("Running production tests...")
    test_result = subprocess.run([
        sys.executable, "scripts/run_tests.py", "--unit"
    ], capture_output=True)
    
    if test_result.returncode != 0:
        print("❌ Production tests failed")
        print(test_result.stderr.decode())
        return False
    
    # Build production image
    print("Building production image...")
    subprocess.check_call([
        "docker", "build", "-f", "Dockerfile.prod", 
        "-t", "medical-retrieval:latest", "."
    ])
    
    # Tag for production
    subprocess.check_call([
        "docker", "tag", "medical-retrieval:latest",
        "your-registry/medical-retrieval:latest"
    ])
    
    # Push to registry (placeholder)
    print("Pushing to registry...")
    # subprocess.check_call([
    #     "docker", "push", "your-registry/medical-retrieval:latest"
    # ])
    
    print("✅ Production deployment completed!")
    print("  Image: medical-retrieval:latest")
    print("  Note: Update push command with your registry")

def setup_monitoring():
    """Set up monitoring and logging"""
    print("📊 Setting up monitoring...")
    
    # Create monitoring directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("monitoring", exist_ok=True)
    
    # Create log configuration
    log_config = """
version: 1
disable_existing_loggers: false

formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: default

  error_file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/error.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    level: ERROR
    formatter: default

loggers:
  app:
    level: INFO
    handlers: [file, error_file]

root:
  level: INFO
  handlers: [file]
"""
    
    with open("monitoring/logging.yaml", "w") as f:
        f.write(log_config)
    
    print("✅ Monitoring configured!")

def create_docker_files():
    """Create Docker configuration files"""
    print("🐳 Creating Docker files...")
    
    # Development Dockerfile
    dockerfile_dev = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "scripts/deploy.py", "--dev"]
"""
    
    with open("Dockerfile.dev", "w") as f:
        f.write(dockerfile_dev)
    
    # Production Dockerfile
    dockerfile_prod = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open("Dockerfile.prod", "w") as f:
        f.write(dockerfile_prod)
    
    # Docker Compose
    docker_compose = """
version: '3.8'

services:
  medical-retrieval:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - medical-retrieval
    restart: unless-stopped
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    print("✅ Docker files created!")

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(
        description="Deploy Medical Multimodal Retrieval System"
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="Deploy to development environment"
    )
    parser.add_argument(
        "--staging", action="store_true", 
        help="Deploy to staging environment"
    )
    parser.add_argument(
        "--prod", action="store_true",
        help="Deploy to production environment"
    )
    parser.add_argument(
        "--setup-monitoring", action="store_true",
        help="Set up monitoring and logging"
    )
    parser.add_argument(
        "--create-docker", action="store_true",
        help="Create Docker configuration files"
    )
    
    args = parser.parse_args()
    
    print("🚀 Medical Multimodal Retrieval System Deployment")
    print("=" * 60)
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    if args.create_docker:
        create_docker_files()
    
    if args.setup_monitoring:
        setup_monitoring()
    
    if args.dev:
        deploy_development()
    elif args.staging:
        deploy_staging()
    elif args.prod:
        deploy_production()
    else:
        print("Please specify deployment environment:")
        print("  --dev        Deploy to development")
        print("  --staging    Deploy to staging")
        print("  --prod        Deploy to production")
        print("  --setup-monitoring  Set up monitoring")
        print("  --create-docker      Create Docker files")

if __name__ == "__main__":
    main()
