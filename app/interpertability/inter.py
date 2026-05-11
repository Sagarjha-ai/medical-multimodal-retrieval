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
print("STEP 6 — INTERPRETABILITY")
print("="*60)

print(f"\nUsing Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU Name: {torch.cuda.get_device_name(0)}"
    )

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\sagar\OneDrive\Desktop\IISC\data\mimic_cxr_project"

TEST_CSV = os.path.join(
    BASE_DIR,
    "processed",
    "test_processed.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints_v2",
    "best_model_v2.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "interpretability_results_v2"
)

GRADCAM_DIR = os.path.join(
    OUTPUT_DIR,
    "gradcam"
)

ATTENTION_DIR = os.path.join(
    OUTPUT_DIR,
    "attention_maps"
)

SIMILARITY_DIR = os.path.join(
    OUTPUT_DIR,
    "similarity_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(ATTENTION_DIR, exist_ok=True)
os.makedirs(SIMILARITY_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224
MAX_LENGTH = 128
EMBED_DIM = 512
NUM_SAMPLES = 10

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
# LOAD TEST DATA
# ============================================================

print("\nLoading test CSV...")

test_df = pd.read_csv(TEST_CSV)

print(f"\nTotal Test Samples: {len(test_df)}")

# ============================================================
# LOAD ENCODERS
# ============================================================

print("\nLoading encoders...")

image_encoder = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    attn_implementation="eager"
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
        image,
        input_ids,
        attention_mask
    ):

        image_outputs = self.image_encoder(
            pixel_values=image,
            output_attentions=True
        )

        image_cls = image_outputs.last_hidden_state[:, 0]

        image_embedding = self.image_projection(
            image_cls
        )

        text_outputs = self.text_encoder(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        text_cls = text_outputs.last_hidden_state[:, 0]

        text_embedding = self.text_projection(
            text_cls
        )

        image_embedding = F.normalize(
            image_embedding,
            dim=-1
        )

        text_embedding = F.normalize(
            text_embedding,
            dim=-1
        )

        return (

            image_embedding,

            text_embedding,

            image_outputs
        )

# ============================================================
# LOAD MODEL
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
# HOOKS
# ============================================================

gradients = []
activations = []

def backward_hook(module, grad_input, grad_output):

    gradients.append(grad_output[0])

def forward_hook(module, input, output):

    activations.append(output[0])

# ============================================================
# TARGET LAYER
# ============================================================

target_layer = model.image_encoder.vision_model.encoder.layers[-1]

target_layer.register_forward_hook(forward_hook)

target_layer.register_full_backward_hook(backward_hook)

# ============================================================
# START PROCESSING
# ============================================================

print("\nGenerating GradCAM visualizations...")

for idx in range(NUM_SAMPLES):

    print(f"\nProcessing Sample {idx+1}/{NUM_SAMPLES}")

    row = test_df.iloc[idx]

    image_path = row["image_path"]

    caption = row["structured_caption"]

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    pil_image = Image.open(
        image_path
    ).convert("RGB")

    original_image = np.array(pil_image)

    original_image_resized = cv2.resize(
        original_image,
        (224, 224)
    )

    image_tensor = test_transforms(
        pil_image
    ).unsqueeze(0).to(DEVICE)

    # ========================================================
    # TOKENIZE TEXT
    # ========================================================

    encoding = tokenizer(

        caption,

        padding="max_length",

        truncation=True,

        max_length=MAX_LENGTH,

        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)

    attention_mask = encoding["attention_mask"].to(DEVICE)

    # ========================================================
    # CLEAR PREVIOUS HOOK DATA
    # ========================================================

    gradients.clear()
    activations.clear()

    # ========================================================
    # FORWARD
    # ========================================================

    image_embedding, text_embedding, image_outputs = model(

        image_tensor,
        input_ids,
        attention_mask
    )

    similarity = torch.sum(
        image_embedding * text_embedding
    )

    # ========================================================
    # BACKWARD
    # ========================================================

    model.zero_grad()

    similarity.backward()

    # ========================================================
    # GET ACTIVATIONS + GRADIENTS
    # ========================================================

    grads = gradients[0]

    acts = activations[0]

    # REMOVE CLS TOKEN

    grads = grads[:, 1:, :]

    acts = acts[:, 1:, :]

    # ========================================================
    # COMPUTE CAM
    # ========================================================

    weights = grads.mean(dim=1)

    cam = (
        weights.unsqueeze(1) * acts
    ).sum(dim=-1)

    cam = cam.squeeze().detach().cpu().numpy()

    # ========================================================
    # RESHAPE CAM
    # ========================================================

    grid_size = int(np.sqrt(cam.shape[0]))

    cam = cam.reshape(
        grid_size,
        grid_size
    )

    cam = cv2.resize(
        cam,
        (224, 224)
    )

    cam = np.maximum(cam, 0)

    cam = cam / (cam.max() + 1e-8)

    # ========================================================
    # HEATMAP
    # ========================================================

    heatmap = cv2.applyColorMap(

        np.uint8(255 * cam),

        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # ========================================================
    # OVERLAY
    # ========================================================

    overlay = (
        0.5 * original_image_resized +
        0.5 * heatmap
    ).astype(np.uint8)

    # ========================================================
    # SAVE GRADCAM
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    axes[0].imshow(
        original_image_resized
    )

    axes[0].set_title(
        "Original X-ray"
    )

    axes[0].axis("off")

    axes[1].imshow(
        cam,
        cmap="jet"
    )

    axes[1].set_title(
        "GradCAM Heatmap"
    )

    axes[1].axis("off")

    axes[2].imshow(
        overlay
    )

    axes[2].set_title(
        f"Overlay\nSimilarity={similarity.item():.3f}"
    )

    axes[2].axis("off")

    plt.suptitle(
        caption[:120],
        fontsize=10
    )

    gradcam_path = os.path.join(

        GRADCAM_DIR,

        f"gradcam_{idx}.png"
    )

    plt.tight_layout()

    plt.savefig(
        gradcam_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # ATTENTION MAP
    # ========================================================

    attentions = image_outputs.attentions[-1]

    attention = attentions[0].mean(dim=0)

    cls_attention = attention[0, 1:]

    cls_attention = cls_attention.detach().cpu().numpy()

    grid_size = int(np.sqrt(cls_attention.shape[0]))

    cls_attention = cls_attention.reshape(
        grid_size,
        grid_size
    )

    cls_attention = cv2.resize(
        cls_attention,
        (224, 224)
    )

    plt.figure(figsize=(6, 6))

    plt.imshow(
        cls_attention,
        cmap="viridis"
    )

    plt.title(
        "CLS Attention Map"
    )

    plt.axis("off")

    attention_path = os.path.join(

        ATTENTION_DIR,

        f"attention_{idx}.png"
    )

    plt.savefig(
        attention_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # SIMILARITY HISTOGRAM
    # ========================================================

    similarity_scores = []

    for j in range(20):

        row2 = test_df.iloc[j]

        caption2 = row2["structured_caption"]

        enc2 = tokenizer(

            caption2,

            padding="max_length",

            truncation=True,

            max_length=MAX_LENGTH,

            return_tensors="pt"
        )

        input_ids2 = enc2["input_ids"].to(DEVICE)

        attention_mask2 = enc2["attention_mask"].to(DEVICE)

        with torch.no_grad():

            _, text_emb2, _ = model(

                image_tensor,

                input_ids2,

                attention_mask2
            )

        score = torch.sum(
            image_embedding * text_emb2
        ).item()

        similarity_scores.append(score)

    plt.figure(figsize=(8, 5))

    plt.hist(
        similarity_scores,
        bins=10
    )

    plt.title(
        "Similarity Score Distribution"
    )

    plt.xlabel(
        "Similarity Score"
    )

    plt.ylabel(
        "Frequency"
    )

    hist_path = os.path.join(

        SIMILARITY_DIR,

        f"similarity_hist_{idx}.png"
    )

    plt.savefig(
        hist_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "="*60)
print("STEP 6 COMPLETED SUCCESSFULLY!")
print("="*60)

print("\nGenerated Outputs:")

print("\n1. GradCAM Visualizations")
print("2. Attention Maps")
print("3. Similarity Histograms")
print("4. Overlay Heatmaps")
print("5. Interpretability Analysis")

print(f"\nSaved At:")
print(OUTPUT_DIR)

print("\nREADY FOR FINAL STEP → STREAMLIT APP")