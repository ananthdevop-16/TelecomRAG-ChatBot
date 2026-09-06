"""
Ingests data/faq.csv into the 'faq' Chroma collection.
Run once (or whenever the CSV changes): python ingest_faq.py
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma

# ✅ NEW: lightweight embeddings (NO torch, NO transformers)
from fastembed import TextEmbedding


# -------------------------
# CONFIG
# -------------------------
CHROMA_DIR = "chroma_store"
COLLECTION = "faq"
CSV_PATH = os.path.join("data", "faq.csv")


# -------------------------
# FAST EMBEDDING WRAPPER
# -------------------------
class FastEmbedWrapper:
    def __init__(self):
        # small + fast + CPU friendly model
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


# -------------------------
# LOAD FAQ DATA
# -------------------------
def load_faq_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)

    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}"

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "faq",
                    "category": row["category"],
                    "faq_id": str(row["id"]),
                },
            )
        )

    return docs


# -------------------------
# MAIN INGESTION PIPELINE
# -------------------------
def main():
    print("Loading FAQ documents...")
    docs = load_faq_documents(CSV_PATH)
    print(f"  Loaded {len(docs)} FAQ entries.")

    print("Initializing FAST embedding model (no torch)...")
    embeddings = FastEmbedWrapper()

    print(f"Storing embeddings into Chroma collection '{COLLECTION}'...")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    print(f"Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()