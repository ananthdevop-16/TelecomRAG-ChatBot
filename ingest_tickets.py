"""
Ingests resolved tickets from data/tickets.db into the 'tickets' Chroma collection.
Run once (or after adding new tickets): python ingest_tickets.py
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sqlite3
from langchain_core.documents import Document
from langchain_chroma import Chroma

# ✅ Lightweight embedding (NO torch, NO transformers)
from fastembed import TextEmbedding


CHROMA_DIR = "chroma_store"
COLLECTION = "tickets"
DB_PATH = os.path.join("data", "tickets.db")


# -------------------------
# FAST EMBEDDING WRAPPER
# -------------------------
class FastEmbedWrapper:
    def __init__(self):
        # small + fast CPU model
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


# -------------------------
# LOAD SQLITE TICKETS
# -------------------------
def load_ticket_documents(db_path: str) -> list[Document]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT * FROM tickets WHERE status = 'resolved'"
    ).fetchall()

    conn.close()

    docs = []

    for row in rows:
        content = (
            f"Issue: {row['issue_type']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "ticket",
                    "ticket_id": row["ticket_id"],
                    "category": row["category"],
                    "status": row["status"],
                },
            )
        )

    return docs


# -------------------------
# MAIN INGESTION
# -------------------------
def main():
    print("Loading ticket documents from SQLite...")
    docs = load_ticket_documents(DB_PATH)
    print(f"  {len(docs)} resolved tickets loaded.")

    print("Initializing FAST embedding model (no torch)...")
    embeddings = FastEmbedWrapper()

    print(f"Storing in Chroma collection '{COLLECTION}'...")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    print(f"Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()