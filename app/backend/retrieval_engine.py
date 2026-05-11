
import os
import cv2
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPVisionModel
)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n" + "="*60)
print("GPU INFORMATION")
print("="*60)

print(f"\nUsing Device: {DEVICE}")

if torch.cuda.is_available():

    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\sagar\OneDrive\Desktop\IISC\data\mimic_cxr_project"

TEST_CSV = os.path.join(
    BASE_DIR,
    "processed",
    "test_processed.csv"
)

# ============================================================
# IMPORTANT UPDATED MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints_v2",
    "best_model_v2.pth"
)

# ============================================================
# NEW RESULT FOLDERS
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "retrieval_results_v2"
)

FAISS_DIR = os.path.join(
    RESULTS_DIR,
    "faiss_index"
)

VIS_DIR = os.path.join(
    RESULTS_DIR,
    "visualizations"
)

EMBED_DIR = os.path.join(
    RESULTS_DIR,
    "embeddings"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224
MAX_LENGTH = 128
EMBED_DIM = 512
BATCH_SIZE = 32

# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

# ============================================================
# CLAHE
# ============================================================

def apply_clahe(pil_image):

    image = np.array(pil_image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB
    )

    return Image.fromarray(image)

# ============================================================
# TRANSFORMS
# ============================================================

test_transforms = transforms.Compose([

    transforms.Lambda(apply_clahe),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# LOAD TEST CSV
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_CSV)

print(f"\nTest Samples: {len(test_df)}")

# ============================================================
# DATASET
# ============================================================

class ChestXrayDataset(Dataset):

    def __init__(self, dataframe, transforms=None):

        self.dataframe = dataframe
        self.transforms = transforms

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image_path = row["image_path"]

        caption = row["structured_caption"]

        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        encoding = tokenizer(

            caption,

            padding="max_length",

            truncation=True,

            max_length=MAX_LENGTH,

            return_tensors="pt"
        )

        return {

            "image": image,

            "input_ids":
            encoding["input_ids"].squeeze(0),

            "attention_mask":
            encoding["attention_mask"].squeeze(0),

            "caption": caption,

            "image_path": image_path
        }

# ============================================================
# DATASET + DATALOADER
# ============================================================

test_dataset = ChestXrayDataset(
    test_df,
    transforms=test_transforms
)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=True
)

# ============================================================
# LOAD ENCODERS
# ============================================================

print("\nLoading encoders...")

image_encoder = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

text_encoder = AutoModel.from_pretrained(
    "bert-base-uncased"
)

# ============================================================
# MODEL
# ============================================================

class MultimodalModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.image_encoder = image_encoder

        self.text_encoder = text_encoder

        self.image_projection = nn.Sequential(

            nn.Linear(768, EMBED_DIM),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(EMBED_DIM, EMBED_DIM)
        )

        self.text_projection = nn.Sequential(

            nn.Linear(768, EMBED_DIM),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(EMBED_DIM, EMBED_DIM)
        )

    def forward(
        self,
        images,
        input_ids,
        attention_mask
    ):

        # IMAGE

        image_outputs = self.image_encoder(
            pixel_values=images
        )

        image_cls = image_outputs.last_hidden_state[:, 0]

        image_embeddings = self.image_projection(
            image_cls
        )

        # TEXT

        text_outputs = self.text_encoder(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        text_cls = text_outputs.last_hidden_state[:, 0]

        text_embeddings = self.text_projection(
            text_cls
        )

        # NORMALIZATION

        image_embeddings = F.normalize(
            image_embeddings,
            dim=-1
        )

        text_embeddings = F.normalize(
            text_embeddings,
            dim=-1
        )

        return image_embeddings, text_embeddings

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = MultimodalModel().to(DEVICE)

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("\nModel loaded successfully!")

# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

print("\nExtracting embeddings...")

all_image_embeddings = []
all_text_embeddings = []

all_captions = []
all_image_paths = []

with torch.no_grad():

    for batch in tqdm(test_loader):

        images = batch["image"].to(DEVICE)

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        image_embeddings, text_embeddings = model(

            images,
            input_ids,
            attention_mask
        )

        all_image_embeddings.append(
            image_embeddings.cpu().numpy()
        )

        all_text_embeddings.append(
            text_embeddings.cpu().numpy()
        )

        all_captions.extend(batch["caption"])

        all_image_paths.extend(batch["image_path"])

# ============================================================
# CONCATENATE EMBEDDINGS
# ============================================================

all_image_embeddings = np.concatenate(
    all_image_embeddings,
    axis=0
)

all_text_embeddings = np.concatenate(
    all_text_embeddings,
    axis=0
)

print("\nEmbedding Shapes:")

print("Image:", all_image_embeddings.shape)

print("Text :", all_text_embeddings.shape)

# ============================================================
# CREATE FAISS INDEX
# ============================================================

print("\nCreating FAISS index...")

index = faiss.IndexFlatIP(
    EMBED_DIM
)

index.add(
    all_image_embeddings.astype(np.float32)
)

# ============================================================
# SAVE INDEX
# ============================================================

index_path = os.path.join(
    FAISS_DIR,
    "image_index_v2.faiss"
)

faiss.write_index(
    index,
    index_path
)

print("\nFAISS index saved!")

# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_top_k(query_embedding, k=5):

    distances, indices = index.search(

        query_embedding.astype(np.float32),

        k
    )

    return distances, indices

# ============================================================
# SAMPLE RETRIEVAL
# ============================================================

print("\nRunning sample retrieval...")

sample_query = all_text_embeddings[0].reshape(1, -1)

distances, indices = retrieve_top_k(
    sample_query,
    k=5
)

print("\nTop Retrieved Results:\n")

for rank, idx in enumerate(indices[0]):

    print(f"Rank {rank+1}")

    print(f"Similarity Score: {distances[0][rank]:.4f}")

    print(f"Image Path: {all_image_paths[idx]}")

    print(f"Caption: {all_captions[idx][:150]}")

    print("-"*60)

# ============================================================
# RETRIEVAL VISUALIZATION
# ============================================================

print("\nGenerating retrieval visualization...")

fig, axes = plt.subplots(
    1,
    5,
    figsize=(22, 5)
)

for i, idx in enumerate(indices[0]):

    image = Image.open(
        all_image_paths[idx]
    )

    axes[i].imshow(image, cmap="gray")

    axes[i].set_title(
        f"Rank {i+1}\nScore: {distances[0][i]:.2f}",
        fontsize=12
    )

    axes[i].axis("off")

plt.tight_layout()

retrieval_vis_path = os.path.join(
    VIS_DIR,
    "top5_retrieval_v2.png"
)

plt.savefig(
    retrieval_vis_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# TSNE VISUALIZATION
# ============================================================

print("\nGenerating t-SNE visualization...")

combined_embeddings = np.concatenate([

    all_image_embeddings[:200],

    all_text_embeddings[:200]

])

tsne = TSNE(

    n_components=2,

    perplexity=30,

    random_state=42
)

tsne_results = tsne.fit_transform(
    combined_embeddings
)

plt.figure(figsize=(10, 8))

plt.scatter(

    tsne_results[:200, 0],

    tsne_results[:200, 1],

    label="Images"
)

plt.scatter(

    tsne_results[200:, 0],

    tsne_results[200:, 1],

    label="Texts"
)

plt.legend()

plt.title("Improved t-SNE Embedding Space")

tsne_path = os.path.join(
    VIS_DIR,
    "tsne_embeddings_v2.png"
)

plt.savefig(
    tsne_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# RETRIEVAL ACCURACY
# ============================================================

print("\nCalculating retrieval accuracy...")

correct = 0
top_k = 5

for i in range(len(all_text_embeddings)):

    query = all_text_embeddings[i].reshape(1, -1)

    _, retrieved_indices = retrieve_top_k(
        query,
        k=top_k
    )

    if i in retrieved_indices[0]:

        correct += 1

retrieval_accuracy = correct / len(all_text_embeddings)

print(f"\nTop-{top_k} Retrieval Accuracy: {retrieval_accuracy:.4f}")

# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(

    os.path.join(
        EMBED_DIR,
        "image_embeddings_v2.npy"
    ),

    all_image_embeddings
)

np.save(

    os.path.join(
        EMBED_DIR,
        "text_embeddings_v2.npy"
    ),

    all_text_embeddings
)

print("\nEmbeddings saved!")

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "="*60)
print("STEP 5 COMPLETED SUCCESSFULLY!")
print("="*60)

print(f"\nBest Model Used:")
print(MODEL_PATH)

print(f"\nFAISS Index Saved At:")
print(index_path)

print(f"\nRetrieval Visualization Saved At:")
print(retrieval_vis_path)

print(f"\nt-SNE Visualization Saved At:")
print(tsne_path)

print(f"\nTop-{top_k} Retrieval Accuracy:")
print(f"{retrieval_accuracy:.4f}")
