"""
Ingests data/telecom_guide.pdf into the 'guides' Chroma collection.
Applies RecursiveCharacterTextSplitter to break the long document into chunks.
Run once (or after regenerating the PDF): python ingest_pdf.py
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ✅ lightweight embedding (NO torch, NO transformers)
from fastembed import TextEmbedding

CHROMA_DIR = "chroma_store"
COLLECTION = "guides"
PDF_PATH = os.path.join("data", "telecom_guide.pdf")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


# -------------------------
# FAST EMBEDDING WRAPPER
# -------------------------
class FastEmbedWrapper:
    def __init__(self):
        # small, fast, CPU-friendly model
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


def main():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"  {len(pages)} pages loaded.")

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(pages)

    # Tag metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = i

    print(f"  {len(chunks)} chunks produced.")

    print("Initializing FAST embedding model (no torch)...")
    embeddings = FastEmbedWrapper()

    print(f"Storing in Chroma collection '{COLLECTION}'...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    print(f"Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()