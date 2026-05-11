"""
FastAPI Main Application for Medical Multimodal Retrieval System
"""

import os
import faiss
import torch
import shutil
import uuid
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPVisionModel,
    CLIPImageProcessor
)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Medical Multimodal Retrieval API",
    description="AI-powered medical image retrieval system",
    version="2.0.0"
)

# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs("uploads", exist_ok=True)
os.makedirs("static/retrieved", exist_ok=True)

# ============================================================
# STATIC FILES
# ============================================================

app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\sagar\OneDrive\Desktop\IISC\data\mimic_cxr_project"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints_v2",
    "best_model_v2.pth"
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "processed",
    "test_processed.csv"
)

FAISS_PATH = os.path.join(
    BASE_DIR,
    "retrieval_results_v2",
    "faiss_index",
    "image_index_v2.faiss"
)

# ============================================================
# LOAD RESOURCES
# ============================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("Loading image processor...")
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Loading test data...")
df = pd.read_csv(TEST_CSV)

print("Loading FAISS index...")
index = faiss.read_index(FAISS_PATH)

# ============================================================
# MODEL DEFINITION
# ============================================================

class RetrievalModel(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.image_encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.text_encoder = AutoModel.from_pretrained(
            "bert-base-uncased"
        )

        self.image_projection = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 512)
        )

        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 512)
        )

    def encode_image(self, pixel_values):

        outputs = self.image_encoder(
            pixel_values=pixel_values
        )

        features = outputs.pooler_output

        embeddings = self.image_projection(features)

        embeddings = torch.nn.functional.normalize(
            embeddings,
            dim=-1
        )

        return embeddings

    def encode_text(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        features = outputs.last_hidden_state[:,0,:]

        embeddings = self.text_projection(features)

        embeddings = torch.nn.functional.normalize(
            embeddings,
            dim=-1
        )

        return embeddings

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")
model = RetrievalModel().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully!")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_caption(row):

    if "structured_caption" in row.index:
        return str(row["structured_caption"])

    elif "caption" in row.index:
        return str(row["caption"])

    elif "impression" in row.index:
        return str(row["impression"])

    return "No clinical findings available."

def encode_uploaded_image(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():

        embeddings = model.encode_image(
            pixel_values
        )

    return embeddings.cpu().numpy()

def encode_query_text(text):

    tokens = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = tokens["input_ids"].to(device)

    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():

        embeddings = model.encode_text(
            input_ids,
            attention_mask
        )

    return embeddings.cpu().numpy()

# ============================================================
# PYDANTIC MODELS
# ============================================================

class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    rank: int
    similarity: float
    caption: str
    image_url: str

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Multimodal Retrieval API",
        "version": "2.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "database_size": len(df)
    }

@app.post("/search/image", response_model=dict)
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Search for similar images by uploading an image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read and process image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Encode image
        query_embedding = encode_uploaded_image(image)
        
        # Search in FAISS
        similarities, indices = index.search(
            query_embedding.astype(np.float32),
            top_k
        )
        
        # Format results
        results = []
        
        for rank, idx in enumerate(indices[0]):
            
            row = df.iloc[idx]
            
            # Copy retrieved image to static folder
            filename = f"{uuid.uuid4()}.png"
            destination = os.path.join(
                "static/retrieved",
                filename
            )
            shutil.copy(row["image_path"], destination)
            image_url = f"http://127.0.0.1:8000/static/retrieved/{filename}"
            
            similarity = float(similarities[0][rank])
            caption = str(row["structured_caption"])
            
            results.append({
                "rank": rank + 1,
                "similarity": similarity,
                "caption": caption,
                "image_url": image_url
            })
        
        return {
            "success": True,
            "query_image": file.filename,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/search/text", response_model=dict)
async def search_by_text(request: TextSearchRequest):
    """
    Search for similar images by text query
    """
    try:
        # Validate query
        if len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 3 characters long"
            )
        
        # Encode text
        query_embedding = encode_query_text(request.query)
        
        # Search in FAISS
        similarities, indices = index.search(
            query_embedding.astype(np.float32),
            request.top_k
        )
        
        # Format results
        results = []
        
        for rank, idx in enumerate(indices[0]):
            
            row = df.iloc[idx]
            
            # Copy retrieved image to static folder
            filename = f"{uuid.uuid4()}.png"
            destination = os.path.join(
                "static/retrieved",
                filename
            )
            shutil.copy(row["image_path"], destination)
            image_url = f"http://127.0.0.1:8000/static/retrieved/{filename}"
            
            similarity = float(similarities[0][rank])
            caption = str(row["structured_caption"])
            
            results.append({
                "rank": rank + 1,
                "similarity": similarity,
                "caption": caption,
                "image_url": image_url
            })
        
        return {
            "success": True,
            "query": request.query,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text query: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "database_size": len(df),
        "embedding_dimension": 512,
        "index_type": "IndexFlatIP",
        "device": str(device),
        "model_path": MODEL_PATH
    }

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("MEDICAL MULTIMODAL RETRIEVAL API")
    print("="*60)
    
    print(f"\nServer starting on http://127.0.0.1:8000")
    print(f"Device: {device}")
    print(f"Database size: {len(df)} images")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
