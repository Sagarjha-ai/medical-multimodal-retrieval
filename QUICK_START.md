# Medical Multimodal Retrieval System - Quick Start Guide

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and setup
git clone <repository-url>
cd medical-multimodal-retrieval
python scripts/setup_environment.py
```

### 2. Train Model
```bash
# Train the multimodal model
python app/training/training.py

# Generate embeddings and FAISS index
python app/backend/retrieval_engine.py
```

### 3. Launch Applications

#### API Server
```bash
cd app/api
python main.py
# Access: http://127.0.0.1:8000
```

#### Streamlit Frontend
```bash
streamlit run app/frontend/streamlit_app.py
# Access: http://127.0.0.1:8501
```

### 4. Test the System
```bash
# Run all tests
python scripts/run_tests.py --all

# Run specific tests
python scripts/run_tests.py --unit
python scripts/run_tests.py --api
```

## 📁 Project Structure

```
medical-multimodal-retrieval/
├── 📂 app/                    # Application code
│   ├── 📂 api/               # FastAPI backend
│   ├── 📂 frontend/           # Streamlit app  
│   ├── 📂 models/            # Model definitions
│   ├── 📂 backend/           # Retrieval engine
│   ├── 📂 training/          # Training scripts
│   ├── 📂 preprocessing/      # Data preprocessing
│   └── 📂 interpertability/  # Model interpretability
├── 📂 configs/               # Configuration files
├── 📂 data/                 # Data directories
├── 📂 docs/                 # Documentation
├── 📂 notebooks/             # Jupyter notebooks
├── 📂 scripts/              # Utility scripts
├── 📂 tests/                # Test suite
├── 📄 requirements.txt       # Dependencies
├── 📄 README.md            # Main documentation
└── 📄 QUICK_START.md       # This file
```

## 🔧 Configuration

### Update Paths
Edit `configs/retrieval_config.yaml`:
```yaml
paths:
  base_dir: "C:/Users/sagar/OneDrive/Desktop/IISC/data/mimic_cxr_project"
  model_checkpoint: "checkpoints_v2/best_model_v2.pth"
  test_csv: "data/mimic_cxr_project/processed/test_processed.csv"
```

### Model Settings
```yaml
model:
  image_encoder: "openai/clip-vit-base-patch32"
  text_encoder: "bert-base-uncased"
  embedding_dim: 512
  max_length: 128
```

## 🎯 Key Features

### 🖼️ Image Retrieval
- Upload chest X-ray images
- Get clinically similar cases
- Visual similarity scoring

### 📝 Text Retrieval  
- Search with clinical queries
- Natural language processing
- Entity-based matching

### 🧬 Embedding Analysis
- t-SNE visualization
- Similarity distributions
- Model interpretability

### ⚡ Performance
- FAISS vector search
- GPU acceleration
- Batch processing

## 🧪 Testing

### Run Tests
```bash
# All tests
python scripts/run_tests.py --all

# Specific categories
python scripts/run_tests.py --unit      # Unit tests
python scripts/run_tests.py --api       # API tests  
python scripts/run_tests.py --integration # Integration tests
python scripts/run_tests.py --coverage  # Coverage report
```

### Test Coverage
Coverage reports generated in `htmlcov/` directory.

## 🚀 Deployment

### Development
```bash
python scripts/deploy.py --dev
```

### Production
```bash
python scripts/deploy.py --prod
```

### Docker
```bash
# Create Docker files
python scripts/deploy.py --create-docker

# Build and run
docker-compose up --build
```

## 📊 Monitoring

### Health Checks
- API: `GET /health`
- System metrics: `GET /stats`

### Logging
- Application logs: `logs/app.log`
- Error logs: `logs/error.log`

## 🔍 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size in configs/train_config.yaml
batch_size: 8  # Instead of 16
```

#### Model Loading Errors
```python
# Check file paths and permissions
import os
print(os.path.exists("checkpoints/best_model_v2.pth"))
```

#### API Connection Issues
```bash
# Check if port is available
netstat -an | grep 8000

# Kill existing processes
pkill -f "uvicorn"
```

### Performance Tips

1. **Use GPU**: Ensure CUDA is available
2. **Batch Processing**: Process multiple requests together
3. **Caching**: Enable Redis for frequent queries
4. **Load Balancing**: Multiple API instances

## 📚 Documentation

- **Full Documentation**: `docs/README.md`
- **API Reference**: `docs/api.md`
- **Model Details**: `docs/model.md`
- **Deployment Guide**: `docs/deployment.md`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Run test suite
5. Submit pull request

## 📞 Support

For issues and questions:
- Create GitHub issue
- Check documentation
- Review troubleshooting guide

## 🎉 Success!

Your Medical Multimodal Retrieval System is now ready! 

**Next Steps:**
1. Configure your data paths
2. Train or load a model
3. Start the API server
4. Launch the Streamlit frontend
5. Test with sample images

**Happy Retrieving! 🏥**
