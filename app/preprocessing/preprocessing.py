import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm
import spacy
import re
import os
from collections import Counter

# ============================================================
# SETTINGS
# ============================================================

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

sns.set_style("whitegrid")

# ============================================================
# LOAD SCISPACY MODEL
# ============================================================

print("Loading SciSpaCy model...")

nlp = spacy.load("en_core_sci_sm")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\sagar\OneDrive\Desktop\IISC\data\mimic_cxr_project"

TRAIN_CSV = os.path.join(BASE_DIR, "metadata", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "metadata", "val.csv")
TEST_CSV = os.path.join(BASE_DIR, "metadata", "test.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

GRAPH_DIR = os.path.join(BASE_DIR, "graphs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# ============================================================
# LOAD CSV FILES
# ============================================================

print("\nLoading CSV files...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Train Samples : {len(train_df)}")
print(f"Val Samples   : {len(val_df)}")
print(f"Test Samples  : {len(test_df)}")

# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[^a-zA-Z0-9., ]", "", text)

    return text.strip()

# ============================================================
# APPLY CLEANING
# ============================================================

print("\nCleaning reports...")

for df in [train_df, val_df, test_df]:

    df["findings_clean"] = df["findings"].apply(clean_text)

    df["impression_clean"] = df["impression"].apply(clean_text)

# ============================================================
# CLINICAL ENTITIES
# ============================================================

clinical_keywords = [
    "opacity",
    "edema",
    "effusion",
    "cardiomegaly",
    "atelectasis",
    "pneumothorax",
    "consolidation",
    "nodule",
    "mass",
    "pneumonia",
]

# ============================================================
# ENTITY EXTRACTION FUNCTION
# ============================================================

def extract_entities(text):

    entities = []

    doc = nlp(text)

    for ent in doc.ents:

        ent_text = ent.text.lower()

        for keyword in clinical_keywords:

            if keyword in ent_text:
                entities.append(keyword)

    # fallback keyword matching
    for keyword in clinical_keywords:

        if keyword in text:
            entities.append(keyword)

    entities = list(set(entities))

    return ", ".join(entities)

# ============================================================
# APPLY ENTITY EXTRACTION
# ============================================================

print("\nExtracting clinical entities...")

tqdm.pandas()

for df in [train_df, val_df, test_df]:

    df["entities"] = df["impression_clean"].progress_apply(
        extract_entities
    )

# ============================================================
# CREATE STRUCTURED CAPTION
# ============================================================

def create_caption(row):

    return (
        f"Impression: {row['impression_clean']}. "
        f"Entities: {row['entities']}"
    )

for df in [train_df, val_df, test_df]:

    df["structured_caption"] = df.apply(
        create_caption,
        axis=1
    )

# ============================================================
# SAVE PROCESSED CSV FILES
# ============================================================

train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train_processed.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(OUTPUT_DIR, "val_processed.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_processed.csv"),
    index=False
)

print("\nProcessed CSV files saved!")

# ============================================================
# ================= DATA VISUALIZATION =======================
# ============================================================

print("\nGenerating graphs...")

# ============================================================
# COMBINE ALL TEXT
# ============================================================

all_text = " ".join(
    train_df["impression_clean"].astype(str).tolist()
)

# ============================================================
# GRAPH 1 — REPORT LENGTH DISTRIBUTION
# ============================================================

train_df["report_length"] = train_df[
    "impression_clean"
].apply(lambda x: len(str(x).split()))

plt.figure()

sns.histplot(
    train_df["report_length"],
    bins=50
)

plt.title("Report Length Distribution")

plt.xlabel("Number of Words")

plt.ylabel("Frequency")

plt.savefig(
    os.path.join(GRAPH_DIR, "report_length_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# GRAPH 2 — TOP CLINICAL ENTITIES
# ============================================================

entity_list = []

for entities in train_df["entities"]:

    entity_list.extend(
        [e.strip() for e in entities.split(",") if e.strip()]
    )

entity_counts = Counter(entity_list)

top_entities = entity_counts.most_common(10)

entities = [x[0] for x in top_entities]
counts = [x[1] for x in top_entities]

plt.figure()

sns.barplot(
    x=counts,
    y=entities
)

plt.title("Top Clinical Entities")

plt.xlabel("Count")

plt.ylabel("Entity")

plt.savefig(
    os.path.join(GRAPH_DIR, "top_entities.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# GRAPH 3 — WORD CLOUD
# ============================================================

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(all_text)

plt.figure(figsize=(14, 7))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Radiology Report WordCloud")

plt.savefig(
    os.path.join(GRAPH_DIR, "wordcloud.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# GRAPH 4 — FINDINGS VS IMPRESSION LENGTH
# ============================================================

train_df["findings_length"] = train_df[
    "findings_clean"
].apply(lambda x: len(str(x).split()))

train_df["impression_length"] = train_df[
    "impression_clean"
].apply(lambda x: len(str(x).split()))

plt.figure()

sns.scatterplot(
    x=train_df["findings_length"],
    y=train_df["impression_length"]
)

plt.title("Findings vs Impression Length")

plt.xlabel("Findings Length")

plt.ylabel("Impression Length")

plt.savefig(
    os.path.join(GRAPH_DIR, "findings_vs_impression.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# GRAPH 5 — ENTITY COUNT DISTRIBUTION
# ============================================================

train_df["entity_count"] = train_df["entities"].apply(
    lambda x: len([e for e in str(x).split(",") if e.strip()])
)

plt.figure()

sns.countplot(
    x=train_df["entity_count"]
)

plt.title("Entity Count Distribution")

plt.xlabel("Number of Entities")

plt.ylabel("Frequency")

plt.savefig(
    os.path.join(GRAPH_DIR, "entity_count_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# GRAPH 6 — TOP WORDS
# ============================================================

words = all_text.split()

word_counts = Counter(words)

top_words = word_counts.most_common(20)

word_labels = [x[0] for x in top_words]
word_values = [x[1] for x in top_words]

plt.figure(figsize=(12, 8))

sns.barplot(
    x=word_values,
    y=word_labels
)

plt.title("Top 20 Most Common Words")

plt.xlabel("Count")

plt.ylabel("Word")

plt.savefig(
    os.path.join(GRAPH_DIR, "top_words.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# GRAPH 7 — REPORT LENGTH BOXPLOT
# ============================================================

plt.figure()

sns.boxplot(
    x=train_df["report_length"]
)

plt.title("Report Length Boxplot")

plt.savefig(
    os.path.join(GRAPH_DIR, "report_boxplot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# GRAPH 8 — ENTITY CORRELATION HEATMAP
# ============================================================

heatmap_df = pd.DataFrame()

for entity in clinical_keywords:

    heatmap_df[entity] = train_df["entities"].apply(
        lambda x: 1 if entity in str(x) else 0
    )

corr = heatmap_df.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Clinical Entity Correlation Heatmap")

plt.savefig(
    os.path.join(GRAPH_DIR, "entity_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# GRAPH 9 — TRAIN/VAL/TEST SPLIT PIE CHART
# ============================================================

sizes = [
    len(train_df),
    len(val_df),
    len(test_df)
]

labels = [
    "Train",
    "Validation",
    "Test"
]

plt.figure()

plt.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%"
)

plt.title("Dataset Split Distribution")

plt.savefig(
    os.path.join(GRAPH_DIR, "dataset_split.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# GRAPH 10 — ENTITY FREQUENCY HISTOGRAM
# ============================================================

plt.figure(figsize=(12, 6))

sns.histplot(
    counts,
    bins=10
)

plt.title("Clinical Entity Frequency Histogram")

plt.xlabel("Frequency")

plt.ylabel("Count")

plt.savefig(
    os.path.join(GRAPH_DIR, "entity_frequency_histogram.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("STEP 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nProcessed Files Saved In:")
print(OUTPUT_DIR)

print("\nGraphs Saved In:")
print(GRAPH_DIR)

print("\nGenerated Graphs:")

graph_names = [
    "report_length_distribution.png",
    "top_entities.png",
    "wordcloud.png",
    "findings_vs_impression.png",
    "entity_count_distribution.png",
    "top_words.png",
    "report_boxplot.png",
    "entity_heatmap.png",
    "dataset_split.png",
    "entity_frequency_histogram.png"
]

for g in graph_names:
    print(g)

print("\nDONE!")