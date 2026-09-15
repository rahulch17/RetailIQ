from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_FILE = Path("rag_index.faiss")
DOCUMENTS_FILE = Path("rag_documents.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:

    def __init__(
        self,
        index_file=INDEX_FILE,
        documents_file=DOCUMENTS_FILE,
        model_name=MODEL_NAME,
    ):

        self.index_file = Path(index_file)
        self.documents_file = Path(documents_file)

        if not self.index_file.exists():
            raise FileNotFoundError(
                f"RAG index not found: {self.index_file}\n"
                "Run: python build_rag.py"
            )

        if not self.documents_file.exists():
            raise FileNotFoundError(
                f"RAG documents file not found: {self.documents_file}\n"
                "Run: python build_rag.py"
            )

        self.index = faiss.read_index(
            str(self.index_file)
        )

        with open(
            self.documents_file,
            "rb"
        ) as f:
            self.documents = pickle.load(f)

        self.model = SentenceTransformer(
            model_name
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=3
    ):

        if not query or not str(query).strip():
            return []

        query_embedding = self.model.encode(
            [str(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            if index >= len(self.documents):
                continue

            item = self.documents[index]

            chunk_id = str(
                item.get(
                    "chunk_id",
                    ""
                )
            )

            document = str(
                item.get(
                    "document",
                    "knowledge_base.csv"
                )
            )

            document_id = str(
                item.get(
                    "document_id",
                    ""
                )
            )

            document_title = str(
                item.get(
                    "document_title",
                    document
                )
            )

            section = str(
                item.get(
                    "section",
                    ""
                )
            )

            topic = str(
                item.get(
                    "topic",
                    ""
                )
            )

            content = str(
                item.get(
                    "content",
                    ""
                )
            )

            source = document

            # ------------------------------------------------
            # IMPORTANT:
            # Provide BOTH text and content.
            # Older agent.py uses text.
            # Newer code uses content.
            # ------------------------------------------------

            results.append(
                {
                    "score": float(score),
                    "chunk_id": chunk_id,
                    "document": document,
                    "document_id": document_id,
                    "document_title": document_title,
                    "section": section,
                    "topic": topic,
                    "content": content,
                    "text": content,
                    "source": source,
                    "citation": (
                        f"{document_title}"
                        + (
                            f" — {section}"
                            if section
                            else ""
                        )
                    ),
                }
            )

        return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Testing RetailIQ RAG retriever..."
    )

    retriever = Retriever()

    results = retriever.search(
        "What is the inventory reorder policy?",
        top_k=3
    )

    print(
        f"\nRetrieved {len(results)} results.\n"
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"Result {i}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Document: {result['document_title']}"
        )

        print(
            f"Section: {result['section']}"
        )

        print(
            f"Topic: {result['topic']}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Content: {result['content'][:300]}"
        )

        print(
            "-" * 60
        )