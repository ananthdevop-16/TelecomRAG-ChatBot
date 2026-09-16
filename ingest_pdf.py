"""
Ingests data/telecom_guide.pdf into the 'guides' Chroma collection.
Run: python ingest_pdf.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from fastembed import TextEmbedding

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

CHROMA_DIR = "chroma_store"
COLLECTION = "guides"
PDF_PATH = os.path.join("data", "telecom_guide.pdf")

# 🔥 OPTIMIZED CHUNK SETTINGS (IMPORTANT FIX)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# -------------------------
# FAST EMBEDDING WRAPPER (FIXED)
# -------------------------
class FastEmbedWrapper:
    def __init__(self):
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        # FIX: ensure proper list conversion
        return [list(vec) for vec in self.model.embed(texts)]

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


# -------------------------
# CLEAN TEXT FUNCTION (IMPORTANT FIX)
# -------------------------
def clean_text(text: str) -> str:
    """
    Removes PDF noise (headers, page numbers, extra spaces)
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        # remove junk
        if len(line) < 3:
            continue
        if line.isdigit():  # page numbers
            continue

        cleaned.append(line)

    return " ".join(cleaned)


# -------------------------
# MAIN PIPELINE
# -------------------------
def main():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    print(f"{len(pages)} pages loaded.")

    # Clean pages
    for p in pages:
        p.page_content = clean_text(p.page_content)

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(pages)

    # Metadata tagging
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = i

    print(f"{len(chunks)} cleaned chunks produced.")

    print("Initializing FastEmbed model...")
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

# """
# Ingests data/telecom_guide.pdf into the 'guides' Chroma collection.
# Applies RecursiveCharacterTextSplitter to break the long document into chunks.
# Run once (or after regenerating the PDF): python ingest_pdf.py
# """

# import os
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma

# # ✅ lightweight embedding (NO torch, NO transformers)
# from fastembed import TextEmbedding

# CHROMA_DIR = "chroma_store"
# COLLECTION = "guides"
# PDF_PATH = os.path.join("data", "telecom_guide.pdf")

# CHUNK_SIZE = 300
# CHUNK_OVERLAP = 50


# # -------------------------
# # FAST EMBEDDING WRAPPER
# # -------------------------
# class FastEmbedWrapper:
#     def __init__(self):
#         # small, fast, CPU-friendly model
#         self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

#     def embed_documents(self, texts):
#         return list(self.model.embed(texts))

#     def embed_query(self, text):
#         return list(self.model.embed([text]))[0]


# def main():
#     print("Loading PDF...")
#     loader = PyPDFLoader(PDF_PATH)
#     pages = loader.load()
#     print(f"  {len(pages)} pages loaded.")

#     print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ".", " "],
#     )

#     chunks = splitter.split_documents(pages)

#     # Tag metadata
#     for i, chunk in enumerate(chunks):
#         chunk.metadata["source"] = "guide"
#         chunk.metadata["chunk_index"] = i

#     print(f"  {len(chunks)} chunks produced.")

#     print("Initializing FAST embedding model (no torch)...")
#     embeddings = FastEmbedWrapper()

#     print(f"Storing in Chroma collection '{COLLECTION}'...")

#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         collection_name=COLLECTION,
#         persist_directory=CHROMA_DIR,
#     )

#     print(f"Done. {vectorstore._collection.count()} vectors stored.")


# if __name__ == "__main__":
#     main()