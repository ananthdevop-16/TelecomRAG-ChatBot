
"""
Ingests resolved tickets from data/tickets.db into 'tickets' Chroma collection.
Run: python ingest_tickets.py
"""

import os
import sqlite3
from langchain_core.documents import Document
from langchain_chroma import Chroma
from fastembed import TextEmbedding

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

CHROMA_DIR = "chroma_store"
COLLECTION = "tickets"
DB_PATH = os.path.join("data", "tickets.db")


# -------------------------
# FAST EMBEDDING WRAPPER (FIXED)
# -------------------------
class FastEmbedWrapper:
    def __init__(self):
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        return [list(vec) for vec in self.model.embed(texts)]

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


# -------------------------
# CLEAN TEXT (IMPORTANT FIX)
# -------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    # remove excessive whitespace
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 3]

    return " ".join(lines)


# -------------------------
# LOAD TICKETS
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
        issue = clean_text(row["issue_type"])
        desc = clean_text(row["description"])
        resolution = clean_text(row["resolution"])

        # 🔥 compact + cleaner embedding format
        content = (
            f"Issue: {issue}\n"
            f"Problem: {desc}\n"
            f"Fix: {resolution}"
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "ticket",
                    "ticket_id": str(row["ticket_id"]),
                    "category": str(row["category"]),
                    "status": str(row["status"]),
                },
            )
        )

    return docs


# -------------------------
# REMOVE DUPLICATES
# -------------------------
def remove_duplicates(docs):
    seen = set()
    unique = []

    for d in docs:
        key = d.page_content[:200]

        if key not in seen:
            seen.add(key)
            unique.append(d)

    return unique


# -------------------------
# MAIN PIPELINE
# -------------------------
def main():
    print("Loading ticket documents from SQLite...")

    docs = load_ticket_documents(DB_PATH)

    # FIX: remove duplicate tickets
    docs = remove_duplicates(docs)

    print(f"{len(docs)} unique resolved tickets loaded.")

    print("Initializing FastEmbed model...")
    embeddings = FastEmbedWrapper()

    print(f"Storing into Chroma collection '{COLLECTION}'...")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    print(f"Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()
# """
# Ingests resolved tickets from data/tickets.db into the 'tickets' Chroma collection.
# Run once (or after adding new tickets): python ingest_tickets.py
# """

# import os
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# import sqlite3
# from langchain_core.documents import Document
# from langchain_chroma import Chroma

# # ✅ Lightweight embedding (NO torch, NO transformers)
# from fastembed import TextEmbedding


# CHROMA_DIR = "chroma_store"
# COLLECTION = "tickets"
# DB_PATH = os.path.join("data", "tickets.db")


# # -------------------------
# # FAST EMBEDDING WRAPPER
# # -------------------------
# class FastEmbedWrapper:
#     def __init__(self):
#         # small + fast CPU model
#         self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

#     def embed_documents(self, texts):
#         return list(self.model.embed(texts))

#     def embed_query(self, text):
#         return list(self.model.embed([text]))[0]


# # -------------------------
# # LOAD SQLITE TICKETS
# # -------------------------
# def load_ticket_documents(db_path: str) -> list[Document]:
#     conn = sqlite3.connect(db_path)
#     conn.row_factory = sqlite3.Row

#     rows = conn.execute(
#         "SELECT * FROM tickets WHERE status = 'resolved'"
#     ).fetchall()

#     conn.close()

#     docs = []

#     for row in rows:
#         content = (
#             f"Issue: {row['issue_type']}\n"
#             f"Description: {row['description']}\n"
#             f"Resolution: {row['resolution']}"
#         )

#         docs.append(
#             Document(
#                 page_content=content,
#                 metadata={
#                     "source": "ticket",
#                     "ticket_id": row["ticket_id"],
#                     "category": row["category"],
#                     "status": row["status"],
#                 },
#             )
#         )

#     return docs


# # -------------------------
# # MAIN INGESTION
# # -------------------------
# def main():
#     print("Loading ticket documents from SQLite...")
#     docs = load_ticket_documents(DB_PATH)
#     print(f"  {len(docs)} resolved tickets loaded.")

#     print("Initializing FAST embedding model (no torch)...")
#     embeddings = FastEmbedWrapper()

#     print(f"Storing in Chroma collection '{COLLECTION}'...")

#     vectorstore = Chroma.from_documents(
#         documents=docs,
#         embedding=embeddings,
#         collection_name=COLLECTION,
#         persist_directory=CHROMA_DIR,
#     )

#     print(f"Done. {vectorstore._collection.count()} vectors stored.")


# if __name__ == "__main__":
#     main()