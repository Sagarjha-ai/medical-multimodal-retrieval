import os
import faiss
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from PIL import Image

from sklearn.manifold import TSNE

from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPVisionModel,
    CLIPImageProcessor
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MedVision AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fb;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.hero {
    padding: 70px;
    border-radius: 35px;
    background:
    linear-gradient(135deg,#f8fbff 0%,#edf6ff 100%);
    border: 1px solid #e5edf8;
    margin-bottom: 40px;
}

.hero-title {
    font-size: 72px;
    font-weight: 900;
    color: #0f172a;
    line-height: 1.05;
}

.hero-sub {
    font-size: 22px;
    color: #64748b;
    line-height: 1.7;
    margin-top: 25px;
    max-width: 850px;
}

.badge {
    display:inline-block;
    background:#dbeafe;
    color:#2563eb;
    padding:10px 18px;
    border-radius:999px;
    font-weight:700;
    margin-bottom:20px;
}

.metric-card {
    background:white;
    border-radius:24px;
    padding:30px;
    border:1px solid #e2e8f0;
    text-align:center;
}

.metric-number {
    font-size:40px;
    font-weight:800;
    color:#0f172a;
}

.metric-label {
    color:#64748b;
    margin-top:5px;
}

.section-title {
    font-size:42px;
    font-weight:800;
    color:#0f172a;
}

.section-sub {
    font-size:18px;
    color:#64748b;
    margin-top:8px;
    margin-bottom:30px;
}

.result-card {
    background:white;
    border-radius:24px;
    overflow:hidden;
    border:1px solid #e2e8f0;
    margin-bottom:25px;
    transition:0.25s;
}

.result-card:hover {
    transform: translateY(-4px);
    box-shadow:0 20px 40px rgba(0,0,0,0.08);
}

.feature {
    background:white;
    border-radius:26px;
    padding:35px;
    border:1px solid #e2e8f0;
    height:100%;
}

.feature-title {
    font-size:28px;
    font-weight:700;
    color:#0f172a;
    margin-top:20px;
}

.feature-text {
    color:#64748b;
    line-height:1.8;
    margin-top:12px;
}

img {
    border-radius:20px;
}

.stButton>button {
    width:100%;
    border:none;
    border-radius:16px;
    background:linear-gradient(135deg,#06b6d4,#2563eb);
    color:white;
    padding:15px;
    font-weight:700;
    font-size:17px;
}

.stTabs [data-baseweb="tab-list"] {
    gap:20px;
}

.stTabs [data-baseweb="tab"] {
    background:white;
    border-radius:14px;
    padding:14px 22px;
}

.footer {
    text-align:center;
    color:#64748b;
    padding:40px;
}

</style>
""", unsafe_allow_html=True)

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

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "retrieval_results_v2",
    "embeddings"
)

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="badge">
🫁 Next Generation Medical AI
</div>

<div class="hero-title">
Redefining<br>
Diagnostics with AI
</div>

<div class="hero-sub">
Multimodal medical retrieval system powered by CLIP,
BERT, and FAISS vector search. Retrieve clinically similar
chest X-rays instantly using embedding-space retrieval.
</div>

</div>
""", unsafe_allow_html=True)

# ============================================================
# METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-number">⬢</div>
    <div class="metric-label">CLIP Vision Encoder</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-number">📄</div>
    <div class="metric-label">BERT Text Encoder</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-number">⚡</div>
    <div class="metric-label">FAISS Retrieval</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-number">512D</div>
    <div class="metric-label">Embedding Space</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# LOAD CSV
# ============================================================

@st.cache_data
def load_csv():
    return pd.read_csv(TEST_CSV)

df = load_csv()

# ============================================================
# TOKENIZER
# ============================================================

@st.cache_resource
def load_tokenizer():

    return AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )

tokenizer = load_tokenizer()

# ============================================================
# PROCESSOR
# ============================================================

@st.cache_resource
def load_processor():

    return CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

processor = load_processor()

# ============================================================
# MODEL
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

@st.cache_resource
def load_model():

    model = RetrievalModel()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()

    return model

model = load_model()

# ============================================================
# LOAD FAISS
# ============================================================

@st.cache_resource
def load_faiss():

    return faiss.read_index(FAISS_PATH)

index = load_faiss()

# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_data
def load_embeddings():

    image_embeddings = np.load(
        os.path.join(
            EMBEDDING_DIR,
            "image_embeddings_v2.npy"
        )
    )

    text_embeddings = np.load(
        os.path.join(
            EMBEDDING_DIR,
            "text_embeddings_v2.npy"
        )
    )

    return image_embeddings, text_embeddings

image_embeddings, text_embeddings = load_embeddings()

# ============================================================
# SAFE CAPTION
# ============================================================

def get_caption(row):

    if "structured_caption" in row.index:
        return str(row["structured_caption"])

    elif "caption" in row.index:
        return str(row["caption"])

    elif "impression" in row.index:
        return str(row["impression"])

    return "No clinical findings available."

# ============================================================
# XRAY VALIDATION
# ============================================================

def is_valid_xray(image):

    img = np.array(image)

    if len(img.shape) != 3:
        return False

    diff_rg = np.mean(np.abs(img[:,:,0] - img[:,:,1]))
    diff_gb = np.mean(np.abs(img[:,:,1] - img[:,:,2]))

    grayscale_score = (diff_rg + diff_gb) / 2

    if grayscale_score > 14:
        return False

    brightness = np.mean(img)

    if brightness < 20 or brightness > 245:
        return False

    return True

# ============================================================
# MEDICAL QUERY VALIDATION
# ============================================================

def is_valid_medical_query(text):

    keywords = [

        "pneumonia",
        "opacity",
        "effusion",
        "edema",
        "atelectasis",
        "cardiomegaly",
        "pleural",
        "pulmonary",
        "thorax",
        "lung",
        "lungs",
        "xray",
        "radiograph",
        "consolidation",
        "infiltrate",
        "pneumothorax",
        "fibrosis",
        "nodule",
        "mass",
        "emphysema"

    ]

    text = text.lower()

    matches = 0

    for word in keywords:

        if word in text:
            matches += 1

    return matches >= 1

# ============================================================
# IMAGE ENCODING
# ============================================================

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

# ============================================================
# TEXT ENCODING
# ============================================================

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
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🫁 Image Retrieval",
    "📄 Text Retrieval",
    "🧬 Embedding Space"
])

# ============================================================
# IMAGE RETRIEVAL
# ============================================================

with tab1:

    st.markdown("""
    <div class="section-title">
    Image Retrieval
    </div>

    <div class="section-sub">
    Upload a chest X-ray and retrieve the most clinically similar cases.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=["png","jpg","jpeg"]
    )

    top_k = st.slider(
        "Top K Results",
        1,
        10,
        5
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, width=450)

        if not is_valid_xray(image):

            st.markdown("""
            <div style="
            background:#fef2f2;
            border:1px solid #fecaca;
            padding:28px;
            border-radius:22px;
            margin-top:25px;
            ">

            <div style="
            color:#991b1b;
            font-size:26px;
            font-weight:800;
            ">
            Invalid Medical Image
            </div>

            <div style="
            color:#7f1d1d;
            margin-top:14px;
            font-size:17px;
            line-height:1.8;
            ">
            This system only supports chest X-ray radiographs.
            Please upload a valid grayscale chest X-ray image.
            </div>

            </div>
            """, unsafe_allow_html=True)

            st.stop()

        st.success("Valid Chest X-ray Detected")

        query_embedding = encode_uploaded_image(image)

        similarities, indices = index.search(
            query_embedding.astype(np.float32),
            top_k
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("""
        <div class="section-title">
        Retrieved Similar Cases
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(3)

        for i, idx in enumerate(indices[0]):

            row = df.iloc[idx]

            retrieved_image = Image.open(
                row["image_path"]
            )

            similarity = min(
                round(float(similarities[0][i]) * 100, 2),
                99.9
            )

            caption = get_caption(row)

            with cols[i % 3]:

                st.markdown("""
                <div class="result-card">
                """, unsafe_allow_html=True)

                st.image(
                    retrieved_image,
                    width=320
                )

                st.markdown(f"""
                <div style="padding:20px;">

                <div style="
                font-size:30px;
                font-weight:800;
                color:#06b6d4;
                ">
                {similarity}%
                </div>

                <div style="
                margin-top:16px;
                font-size:16px;
                color:#475569;
                line-height:1.9;
                ">
                {caption[:240]}
                </div>

                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TEXT RETRIEVAL
# ============================================================

with tab2:

    st.markdown("""
    <div class="section-title">
    Text Retrieval
    </div>

    <div class="section-sub">
    Search chest X-rays using clinical findings.
    </div>
    """, unsafe_allow_html=True)

    query = st.text_area(
        "Clinical Query",
        placeholder="Example: pneumonia with pleural effusion"
    )

    if st.button("Retrieve Similar X-rays"):

        if len(query.strip()) < 5:

            st.warning(
                "Please enter a clinical radiology query."
            )

            st.stop()

        if not is_valid_medical_query(query):

            st.markdown("""
            <div style="
            background:#fef2f2;
            border:1px solid #fecaca;
            padding:28px;
            border-radius:22px;
            margin-top:25px;
            ">

            <div style="
            color:#991b1b;
            font-size:26px;
            font-weight:800;
            ">
            Invalid Clinical Query
            </div>

            <div style="
            color:#7f1d1d;
            margin-top:14px;
            font-size:17px;
            line-height:1.8;
            ">
            Please enter valid radiology terminology.
            <br><br>
            Example:
            pneumonia with pleural effusion
            </div>

            </div>
            """, unsafe_allow_html=True)

            st.stop()

        query_embedding = encode_query_text(query)

        similarities, indices = index.search(
            query_embedding.astype(np.float32),
            5
        )

        cols = st.columns(3)

        for i, idx in enumerate(indices[0]):

            row = df.iloc[idx]

            retrieved_image = Image.open(
                row["image_path"]
            )

            similarity = min(
                round(float(similarities[0][i]) * 100, 2),
                99.9
            )

            caption = get_caption(row)

            with cols[i % 3]:

                st.image(
                    retrieved_image,
                    width=320
                )

                st.success(
                    f"Similarity: {similarity}%"
                )

                st.caption(
                    caption[:240]
                )

# ============================================================
# EMBEDDING SPACE
# ============================================================

with tab3:

    st.markdown("""
    <div class="section-title">
    Embedding Space Visualization
    </div>

    <div class="section-sub">
    Visualizing image-text alignment in multimodal latent space.
    </div>
    """, unsafe_allow_html=True)

    sample_size = 150

    combined_embeddings = np.concatenate([
        image_embeddings[:sample_size],
        text_embeddings[:sample_size]
    ])

    labels = (
        ["Images"] * sample_size +
        ["Texts"] * sample_size
    )

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    )

    reduced = tsne.fit_transform(
        combined_embeddings
    )

    plot_df = pd.DataFrame({

        "x": reduced[:,0],
        "y": reduced[:,1],
        "Type": labels
    })

    fig = px.scatter(

        plot_df,

        x="x",

        y="y",

        color="Type",

        template="plotly_white",

        title="t-SNE Embedding Space"
    )

    fig.update_layout(
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# FEATURES
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">
Powered by State-of-the-Art Architecture
</div>

<div class="section-sub">
Multimodal medical retrieval using CLIP, BERT and FAISS.
</div>
""", unsafe_allow_html=True)

f1, f2, f3 = st.columns(3)

with f1:

    st.markdown("""
    <div class="feature">

    <div style="font-size:46px;">
    ⬢
    </div>

    <div class="feature-title">
    CLIP Vision Encoder
    </div>

    <div class="feature-text">
    Extracts powerful semantic image embeddings
    from chest X-rays.
    </div>

    </div>
    """, unsafe_allow_html=True)

with f2:

    st.markdown("""
    <div class="feature">

    <div style="font-size:46px;">
    📄
    </div>

    <div class="feature-title">
    BERT Text Encoder
    </div>

    <div class="feature-text">
    Processes radiology findings into
    semantic clinical representations.
    </div>

    </div>
    """, unsafe_allow_html=True)

with f3:

    st.markdown("""
    <div class="feature">

    <div style="font-size:46px;">
    ⚡
    </div>

    <div class="feature-title">
    FAISS Retrieval
    </div>

    <div class="feature-text">
    Performs ultra-fast vector similarity
    search across embedding space.
    </div>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

Medical Multimodal Retrieval AI • Streamlit • CLIP • BERT • FAISS

</div>
""", unsafe_allow_html=True)