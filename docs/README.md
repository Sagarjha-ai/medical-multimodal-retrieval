# Medical Multimodal Retrieval System - Documentation

## Overview

This documentation provides comprehensive information about the Medical Multimodal Retrieval System, an AI-powered platform for retrieving clinically similar chest X-ray images using advanced multimodal embeddings.

## Table of Contents

- [System Architecture](#system-architecture)
- [Installation Guide](#installation-guide)
- [API Documentation](#api-documentation)
- [Model Architecture](#model-architecture)
- [Data Pipeline](#data-pipeline)
- [Training Guide](#training-guide)
- [Deployment Guide](#deployment-guide)
- [Troubleshooting](#troubleshooting)

## System Architecture

### Core Components

1. **Image Encoder**: CLIP Vision Transformer for extracting visual features
2. **Text Encoder**: BERT for processing clinical reports
3. **Projection Heads**: Neural networks that map features to shared embedding space
4. **Retrieval Engine**: FAISS-based similarity search
5. **API Layer**: FastAPI backend for serving requests
6. **Frontend**: Streamlit application for interactive retrieval

### Data Flow

```
Image Input → CLIP Encoder → Projection → 512D Embedding → FAISS Search → Results
Text Input  → BERT Encoder  → Projection → 512D Embedding → FAISS Search → Results
```

## Installation Guide

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 50GB+ storage

### Setup Steps

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd medical-multimodal-retrieval
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Models**
   ```bash
   # Models will be downloaded automatically on first run
   # Or manually download to checkpoints/
   ```

5. **Configure Paths**
   - Update `BASE_DIR` in configuration files
   - Ensure data paths are correct

## API Documentation

### Endpoints

#### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "device": "cuda",
  "database_size": 10000
}
```

#### Image Search
```http
POST /search/image
Content-Type: multipart/form-data
```

Parameters:
- `file`: Image file (PNG, JPG, JPEG)
- `top_k`: Number of results (default: 5)

Response:
```json
{
  "success": true,
  "query_image": "xray.png",
  "results": [
    {
      "rank": 1,
      "similarity": 0.85,
      "caption": "Normal chest X-ray...",
      "image_url": "http://127.0.0.1:8000/static/retrieved/uuid.png"
    }
  ]
}
```

#### Text Search
```http
POST /search/text
Content-Type: application/json
```

Request Body:
```json
{
  "query": "pneumonia with pleural effusion",
  "top_k": 5
}
```

Response:
```json
{
  "success": true,
  "query": "pneumonia with pleural effusion",
  "results": [...]
}
```

## Model Architecture

### Multimodal Model

The system uses a dual-encoder architecture:

#### Image Encoder
- **Base Model**: OpenAI CLIP ViT-B/32
- **Input**: 224×224 RGB images
- **Output**: 768-dimensional features
- **Enhancement**: CLAHE preprocessing

#### Text Encoder
- **Base Model**: BERT-base-uncased
- **Input**: Tokenized clinical text (max 128 tokens)
- **Output**: 768-dimensional features

#### Projection Heads
Both encoders project features to a shared 512-dimensional space:
```python
Sequential(
    Linear(768, 512),
    GELU(),
    Dropout(0.2),
    Linear(512, 512)
)
```

#### Training Strategy
- **Loss Function**: Contrastive Loss with temperature scaling
- **Optimization**: AdamW with cosine annealing
- **Fine-tuning**: Last 2 transformer layers unfrozen
- **Mixed Precision**: Automatic Mixed Precision (AMP)

## Data Pipeline

### Data Sources

1. **MIMIC-CXR Dataset**: Chest X-ray images with reports
2. **Preprocessing**: Clinical entity extraction, text cleaning
3. **Augmentation**: Random flips, rotations for training

### Processing Steps

1. **Image Enhancement**: CLAHE for contrast improvement
2. **Text Processing**: Entity extraction, structured captioning
3. **Tokenization**: BERT tokenization with padding
4. **Normalization**: ImageNet statistics for images

## Training Guide

### Configuration

Key hyperparameters:
- **Batch Size**: 16
- **Learning Rate**: 1e-5
- **Epochs**: 20
- **Temperature**: 0.07
- **Weight Decay**: 1e-4

### Training Commands

```bash
# Train model
python app/training/training.py

# Generate embeddings
python app/backend/retrieval_engine.py

# Run interpretability
python app/interpertability/inter.py
```

### Monitoring

- **Loss Curves**: Saved to `training_plots_v2/`
- **Validation**: Every epoch with early stopping
- **Checkpoints**: Best model saved automatically

## Deployment Guide

### Production Deployment

#### API Server
```bash
cd app/api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Streamlit App
```bash
cd app/frontend
streamlit run streamlit_app.py --server.port 8501
```

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /static/ {
        alias /app/static/;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 8  # Instead of 16

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

#### Model Loading Errors
```python
# Check model paths
import os
assert os.path.exists(MODEL_PATH), "Model file not found"

# Verify CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
```

#### Slow Retrieval
```python
# Use GPU FAISS if available
import faiss
if faiss.get_num_gpus() > 0:
    index = faiss.IndexGPUToCPU(gpu_index)
```

#### API Errors
```python
# Check file size limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Validate image formats
ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg']
```

### Performance Optimization

1. **Enable Caching**: Use Redis for frequent queries
2. **Batch Processing**: Process multiple requests together
3. **Model Quantization**: Use INT8 for inference
4. **Load Balancing**: Multiple API instances behind load balancer

### Monitoring

Set up monitoring for:
- **GPU Memory**: NVIDIA DCGM
- **API Response Times**: Prometheus + Grafana
- **Error Rates**: Sentry or similar
- **Resource Usage**: System metrics

## Contributing

Please refer to the main README.md for contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
