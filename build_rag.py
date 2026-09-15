from pathlib import Path
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

KB_FILE = Path(r"D:\7th sem\bepractical\Retail\knowledge_base\knowledge_base.csv")

INDEX_FILE = Path("rag_index.faiss")
DOCUMENTS_FILE = Path("rag_documents.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# BUILD RAG INDEX
# ============================================================

def build():

    if not KB_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge base CSV not found: {KB_FILE}"
        )

    print(f"Loading knowledge base: {KB_FILE}")

    df = pd.read_csv(
        KB_FILE,
        encoding="utf-8"
    )

    required_columns = [
        "chunk_id",
        "document",
        "document_id",
        "document_title",
        "section",
        "topic",
        "content",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(missing)
        )

    # Remove empty content rows
    df["content"] = (
        df["content"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["content"] != ""
    ].copy()

    if df.empty:
        raise ValueError(
            "Knowledge base CSV contains no usable content."
        )

    # --------------------------------------------------------
    # Preserve the managed order from the CSV
    # --------------------------------------------------------

    sort_columns = [
        "document_order",
        "section_order",
        "chunk_id",
    ]

    existing_sort_columns = [
        col
        for col in sort_columns
        if col in df.columns
    ]

    if existing_sort_columns:

        df = df.sort_values(
            existing_sort_columns
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # Build text used for semantic search
    # --------------------------------------------------------

    texts = []

    for _, row in df.iterrows():

        text = (
            f"Document: {row['document_title']}\n"
            f"Section: {row['section']}\n"
            f"Topic: {row['topic']}\n"
            f"Content: {row['content']}"
        )

        texts.append(text)

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print(
        f"Creating embeddings for {len(texts)} knowledge entries..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype(
        "float32"
    )

    # --------------------------------------------------------
    # FAISS index
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # --------------------------------------------------------
    # Save structured documents
    # --------------------------------------------------------

    documents = []

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        documents.append(
            {
                "index": i,
                "chunk_id": str(
                    row["chunk_id"]
                ),
                "document": str(
                    row["document"]
                ),
                "document_id": str(
                    row["document_id"]
                ),
                "document_title": str(
                    row["document_title"]
                ),
                "section": str(
                    row["section"]
                ),
                "topic": str(
                    row["topic"]
                ),
                "content": str(
                    row["content"]
                ),
            }
        )

    with open(
        DOCUMENTS_FILE,
        "wb"
    ) as f:

        pickle.dump(
            documents,
            f
        )

    print("\nRAG build completed successfully.")
    print(
        f"Knowledge entries: {len(documents)}"
    )
    print(
        f"FAISS index:       {INDEX_FILE}"
    )
    print(
        f"Documents file:    {DOCUMENTS_FILE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    build()