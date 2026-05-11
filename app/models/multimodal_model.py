# ============================================================
# IMPORTS
# ============================================================

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPVisionModel
)

from torch.amp import autocast, GradScaler

# ============================================================
# GPU
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n" + "=" * 60)
print("GPU INFORMATION")
print("=" * 60)

print(f"\nUsing Device: {DEVICE}")

if torch.cuda.is_available():

    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    print(f"CUDA Version: {torch.version.cuda}")

torch.backends.cudnn.benchmark = True

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\sagar\OneDrive\Desktop\IISC\data\mimic_cxr_project"

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "processed",
    "train_processed.csv"
)

VAL_CSV = os.path.join(
    BASE_DIR,
    "processed",
    "val_processed.csv"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints_v2"
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "training_plots_v2"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading processed CSV files...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print(f"\nTrain Samples : {len(train_df)}")
print(f"Validation Samples : {len(val_df)}")

# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
MAX_LENGTH = 128
EMBED_DIM = 512
EPOCHS = 20
LR = 1e-5

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

train_transforms = transforms.Compose([

    transforms.Lambda(apply_clahe),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([

    transforms.Lambda(apply_clahe),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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
            encoding["attention_mask"].squeeze(0)
        }

# ============================================================
# DATASETS
# ============================================================

train_dataset = ChestXrayDataset(
    train_df,
    transforms=train_transforms
)

val_dataset = ChestXrayDataset(
    val_df,
    transforms=val_transforms
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading CLIP image encoder...")

image_encoder = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

print("\nLoading BERT text encoder...")

text_encoder = AutoModel.from_pretrained(
    "bert-base-uncased"
)

# ============================================================
# MULTIMODAL MODEL
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
# MODEL
# ============================================================

model = MultimodalModel().to(DEVICE)

print("\nModel loaded successfully!")

# ============================================================
# FREEZE ALL FIRST
# ============================================================

for param in model.image_encoder.parameters():
    param.requires_grad = False

for param in model.text_encoder.parameters():
    param.requires_grad = False

# ============================================================
# UNFREEZE LAST 2 CLIP LAYERS
# ============================================================

for param in model.image_encoder.vision_model.encoder.layers[-2:].parameters():
    param.requires_grad = True

# ============================================================
# UNFREEZE LAST 2 BERT LAYERS
# ============================================================

for param in model.text_encoder.encoder.layer[-2:].parameters():
    param.requires_grad = True

# ============================================================
# PROJECTION HEADS TRAINABLE
# ============================================================

for param in model.image_projection.parameters():
    param.requires_grad = True

for param in model.text_projection.parameters():
    param.requires_grad = True

# ============================================================
# LOSS
# ============================================================

class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07):

        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        image_embeddings,
        text_embeddings
    ):

        logits = (
            image_embeddings @ text_embeddings.T
        ) / self.temperature

        labels = torch.arange(
            logits.shape[0]
        ).to(DEVICE)

        loss_i = F.cross_entropy(
            logits,
            labels
        )

        loss_t = F.cross_entropy(
            logits.T,
            labels
        )

        return (loss_i + loss_t) / 2

criterion = ContrastiveLoss()

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=LR,
    weight_decay=1e-4
)

# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# ============================================================
# MIXED PRECISION
# ============================================================

scaler = GradScaler("cuda")

# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_epoch():

    model.train()

    total_loss = 0

    loop = tqdm(train_loader)

    for batch in loop:

        images = batch["image"].to(DEVICE)

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        optimizer.zero_grad()

        with autocast("cuda"):

            image_embeddings, text_embeddings = model(

                images,
                input_ids,
                attention_mask
            )

            loss = criterion(
                image_embeddings,
                text_embeddings
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

        loop.set_description(
            f"Train Loss: {loss.item():.4f}"
        )

    return total_loss / len(train_loader)

# ============================================================
# VALIDATION
# ============================================================

def validate_epoch():

    model.eval()

    total_loss = 0

    with torch.no_grad():

        loop = tqdm(val_loader)

        for batch in loop:

            images = batch["image"].to(DEVICE)

            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch["attention_mask"].to(DEVICE)

            with autocast("cuda"):

                image_embeddings, text_embeddings = model(

                    images,
                    input_ids,
                    attention_mask
                )

                loss = criterion(
                    image_embeddings,
                    text_embeddings
                )

            total_loss += loss.item()

            loop.set_description(
                f"Val Loss: {loss.item():.4f}"
            )

    return total_loss / len(val_loader)

# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("STARTING IMPROVED TRAINING")
print("=" * 60)

train_losses = []
val_losses = []

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)
    print(f"EPOCH {epoch+1}/{EPOCHS}")
    print("=" * 60)

    train_loss = train_epoch()

    val_loss = validate_epoch()

    scheduler.step()

    train_losses.append(train_loss)

    val_losses.append(val_loss)

    print(f"\nTrain Loss: {train_loss:.4f}")

    print(f"Validation Loss: {val_loss:.4f}")

    # SAVE BEST MODEL

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model_v2.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("\nBest model updated!")

# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    CHECKPOINT_DIR,
    "final_model_v2.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)

# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    train_losses,
    label="Train Loss",
    linewidth=3
)

plt.plot(
    val_losses,
    label="Validation Loss",
    linewidth=3
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Improved Training Curve")

plt.legend()

plot_path = os.path.join(
    PLOT_DIR,
    "improved_loss_curve.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED TRAINING COMPLETED!")
print("=" * 60)

print(f"\nBest Model Saved At:")
print(best_model_path)

print(f"\nFinal Model Saved At:")
print(final_model_path)

print(f"\nLoss Curve Saved At:")
print(plot_path)

print("\nNOW RUN STEP 5 AGAIN USING:")
print(best_model_path)