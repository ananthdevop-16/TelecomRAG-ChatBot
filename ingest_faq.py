"""
Ingests data/faq.csv into the 'faq' Chroma collection.
Run: python ingest_faq.py
"""

import os
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from fastembed import TextEmbedding

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

CHROMA_DIR = "chroma_store"
COLLECTION = "faq"
CSV_PATH = os.path.join("data", "faq.csv")


# -------------------------
# FAST EMBEDDING WRAPPER (FIXED)
# -------------------------
class FastEmbedWrapper:
    def __init__(self):
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        # FIX: ensure strict list conversion
        return [list(vec) for vec in self.model.embed(texts)]

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


# -------------------------
# LOAD FAQ DOCUMENTS (IMPROVED)
# -------------------------
def load_faq_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)

    docs = []

    for _, row in df.iterrows():
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()

        # 🔥 compact format (reduces embedding noise)
        content = f"Q: {question}\nA: {answer}"

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "faq",
                    "category": str(row.get("category", "general")),
                    "faq_id": str(row["id"]),
                },
            )
        )

    return docs


# -------------------------
# REMOVE DUPLICATES (IMPORTANT)
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
# MAIN INGESTION PIPELINE
# -------------------------
def main():
    print("Loading FAQ documents...")
    docs = load_faq_documents(CSV_PATH)

    # FIX: prevent duplicate ingestion issues
    docs = remove_duplicates(docs)

    print(f"Loaded {len(docs)} unique FAQ entries.")

    print("Initializing FastEmbed model...")
    embeddings = FastEmbedWrapper()

    print(f"Storing into Chroma collection '{COLLECTION}'...")

    # ⚠ OPTIONAL SAFE MODE: uncomment if you want fresh rebuild each time
    # import shutil
    # shutil.rmtree(CHROMA_DIR, ignore_errors=True)

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
# Ingests data/faq.csv into the 'faq' Chroma collection.
# Run once (or whenever the CSV changes): python ingest_faq.py
# """

# import os
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# import pandas as pd
# from langchain_core.documents import Document
# from langchain_chroma import Chroma

# # ✅ NEW: lightweight embeddings (NO torch, NO transformers)
# from fastembed import TextEmbedding


# # -------------------------
# # CONFIG
# # -------------------------
# CHROMA_DIR = "chroma_store"
# COLLECTION = "faq"
# CSV_PATH = os.path.join("data", "faq.csv")


# # -------------------------
# # FAST EMBEDDING WRAPPER
# # -------------------------
# class FastEmbedWrapper:
#     def __init__(self):
#         # small + fast + CPU friendly model
#         self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

#     def embed_documents(self, texts):
#         return list(self.model.embed(texts))

#     def embed_query(self, text):
#         return list(self.model.embed([text]))[0]


# # -------------------------
# # LOAD FAQ DATA
# # -------------------------
# def load_faq_documents(csv_path: str) -> list[Document]:
#     df = pd.read_csv(csv_path)

#     docs = []
#     for _, row in df.iterrows():
#         content = f"Q: {row['question']}\nA: {row['answer']}"

#         docs.append(
#             Document(
#                 page_content=content,
#                 metadata={
#                     "source": "faq",
#                     "category": row["category"],
#                     "faq_id": str(row["id"]),
#                 },
#             )
#         )

#     return docs


# # -------------------------
# # MAIN INGESTION PIPELINE
# # -------------------------
# def main():
#     print("Loading FAQ documents...")
#     docs = load_faq_documents(CSV_PATH)
#     print(f"  Loaded {len(docs)} FAQ entries.")

#     print("Initializing FAST embedding model (no torch)...")
#     embeddings = FastEmbedWrapper()

#     print(f"Storing embeddings into Chroma collection '{COLLECTION}'...")

#     vectorstore = Chroma.from_documents(
#         documents=docs,
#         embedding=embeddings,
#         collection_name=COLLECTION,
#         persist_directory=CHROMA_DIR,
#     )

#     print(f"Done. {vectorstore._collection.count()} vectors stored.")


# if __name__ == "__main__":
#     main()