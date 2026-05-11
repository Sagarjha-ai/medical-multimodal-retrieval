#!/usr/bin/env python3
"""
Environment Setup Script for Medical Multimodal Retrieval System

This script sets up the development environment by:
1. Creating necessary directories
2. Installing dependencies
3. Downloading required models
4. Setting up configuration files
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def create_directories():
    """Create necessary directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/sample_images",
        "checkpoints",
        "embeddings",
        "faiss_index",
        "outputs/logs",
        "outputs/plots",
        "outputs/retrieval_results",
        "static/retrieved",
        "uploads",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("Directory structure created successfully!\n")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("Dependencies installed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def verify_installation():
    """Verify critical installations"""
    print("🔍 Verifying installation...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
        
        import faiss
        print(f"  ✓ FAISS {faiss.__version__}")
        
        import streamlit
        print(f"  ✓ Streamlit {streamlit.__version__}")
        
        import fastapi
        print(f"  ✓ FastAPI {fastapi.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available - will use CPU")
            
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        sys.exit(1)
    
    print("Installation verified!\n")

def create_sample_config():
    """Create sample configuration files"""
    print("⚙️ Creating sample configuration...")
    
    config_content = """
# Sample Configuration File
# Copy to configs/config.yaml and modify as needed

paths:
  base_dir: "C:/Users/sagar/OneDrive/Desktop/IISC/data/mimic_cxr_project"
  model_checkpoint: "checkpoints/best_model.pth"
  faiss_index: "faiss_index/image_index.faiss"

model:
  image_encoder: "openai/clip-vit-base-patch32"
  text_encoder: "bert-base-uncased"
  embedding_dim: 512
  max_length: 128

training:
  batch_size: 16
  learning_rate: 1e-5
  epochs: 20
  temperature: 0.07

api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
"""
    
    with open("configs/sample_config.yaml", "w") as f:
        f.write(config_content)
    
    print("Sample configuration created!\n")

def create_gitignore():
    """Create .gitignore file"""
    print("📝 Creating .gitignore...")
    
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Data and Models
data/raw/*
!data/raw/.gitkeep
checkpoints/*
!checkpoints/.gitkeep
embeddings/*
!embeddings/.gitkeep
faiss_index/*
!faiss_index/.gitkeep

# Logs and Outputs
logs/
outputs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
.cache/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print(".gitignore created!\n")

def create_readme():
    """Create quick start README"""
    print("📖 Creating quick start guide...")
    
    readme_content = """# Medical Multimodal Retrieval System

## Quick Start

1. **Setup Environment**
   ```bash
   python scripts/setup_environment.py
   ```

2. **Train Model**
   ```bash
   python app/training/training.py
   ```

3. **Start API**
   ```bash
   python app/api/main.py
   ```

4. **Start Streamlit App**
   ```bash
   streamlit run app/frontend/streamlit_app.py
   ```

## Directory Structure

```
medical-multimodal-retrieval/
├── app/                    # Application code
│   ├── api/               # FastAPI backend
│   ├── frontend/           # Streamlit app
│   ├── models/            # Model definitions
│   ├── backend/           # Retrieval engine
│   ├── training/          # Training scripts
│   ├── preprocessing/      # Data preprocessing
│   └── interpertability/  # Model interpretability
├── configs/               # Configuration files
├── data/                 # Data directories
├── docs/                 # Documentation
├── notebooks/             # Jupyter notebooks
├── scripts/              # Utility scripts
├── tests/                # Test suite
├── requirements.txt       # Dependencies
└── README.md            # This file
```

## Next Steps

1. Configure paths in `configs/`
2. Prepare your dataset
3. Run training pipeline
4. Deploy API and frontend

For detailed documentation, see `docs/README.md`.
"""
    
    with open("QUICK_START.md", "w") as f:
        f.write(readme_content)
    
    print("Quick start guide created!\n")

def main():
    """Main setup function"""
    print("🚀 Medical Multimodal Retrieval System Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if "--skip-deps" not in sys.argv:
        install_dependencies()
    else:
        print("⏭ Skipping dependency installation\n")
    
    # Verify installation
    verify_installation()
    
    # Create configuration files
    create_sample_config()
    create_gitignore()
    create_readme()
    
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Review and update configs/sample_config.yaml")
    print("2. Prepare your data in data/raw/")
    print("3. Run: python app/training/training.py")
    print("4. Start API: python app/api/main.py")
    print("5. Launch frontend: streamlit run app/frontend/streamlit_app.py")
    print("\nFor detailed documentation, see docs/README.md")

if __name__ == "__main__":
    main()
