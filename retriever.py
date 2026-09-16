"""
Builds a merged retriever across:
  - faq
  - tickets
  - guides
"""

from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from fastembed import TextEmbedding

CHROMA_DIR = "chroma_store"


# -------------------------
# FAST EMBEDDING WRAPPER (STABLE FIX)
# -------------------------
class FastEmbedWrapper:
    def __init__(self):
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        # FIX: force strict list + safe conversion
        return [list(vec) for vec in self.model.embed(list(texts))]

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


# -------------------------
# DEDUP
# -------------------------
def deduplicate_docs(docs):
    seen = set()
    unique = []

    for d in docs:
        key = (d.page_content or "")[:200]

        if key not in seen:
            seen.add(key)
            unique.append(d)

    return unique


# -------------------------
# TEXT TRIM (IMPORTANT FIX)
# -------------------------
def trim_docs(docs, max_chars=2000):
    """
    Prevents Gemini 500 INTERNAL caused by oversized context
    """
    result = []
    total = 0

    for d in docs:
        text = (d.page_content or "")[:400]

        if total + len(text) > max_chars:
            break

        d.page_content = text
        result.append(d)
        total += len(text)

    return result


# -------------------------
# BUILD RETRIEVER
# -------------------------
def build_retriever(
    k_faq: int = 2,
    k_tickets: int = 2,
    k_guides: int = 2,
    max_docs: int = 5
) -> RunnableLambda:

    embeddings = FastEmbedWrapper()

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    faq_retriever = faq_store.as_retriever(search_kwargs={"k": k_faq})
    tickets_retriever = tickets_store.as_retriever(search_kwargs={"k": k_tickets})
    guides_retriever = guides_store.as_retriever(search_kwargs={"k": k_guides})

    def retrieve(query: str) -> list[Document]:
        try:
            docs = []

            # weighted retrieval (IMPORTANT IMPROVEMENT)
            docs.extend(faq_retriever.invoke(query) or [])
            docs.extend(tickets_retriever.invoke(query) or [])
            docs.extend(guides_retriever.invoke(query) or [])

            # deduplicate
            docs = deduplicate_docs(docs)

            # trim content (CRITICAL for Gemini stability)
            docs = trim_docs(docs)

            # final cap
            return docs[:max_docs]

        except Exception as e:
            print("Retriever error:", e)
            return []

    return RunnableLambda(retrieve)
# """
# Builds a merged retriever across three Chroma collections:
#   - faq
#   - tickets
#   - guides
# """

# from langchain_chroma import Chroma
# from langchain_core.runnables import RunnableLambda
# from langchain_core.documents import Document
# from fastembed import TextEmbedding

# CHROMA_DIR = "chroma_store"


# # -------------------------
# # FAST EMBEDDING WRAPPER (FIXED)
# # -------------------------
# class FastEmbedWrapper:
#     def __init__(self):
#         self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

#     def embed_documents(self, texts):
#         # FIX: ensure strict list conversion
#         return [list(vec) for vec in self.model.embed(texts)]

#     def embed_query(self, text):
#         return list(self.model.embed([text]))[0]


# # -------------------------
# # DEDUP HELPERS
# # -------------------------
# def deduplicate_docs(docs):
#     seen = set()
#     unique_docs = []

#     for doc in docs:
#         key = doc.page_content[:200]  # lightweight fingerprint
#         if key not in seen:
#             seen.add(key)
#             unique_docs.append(doc)

#     return unique_docs


# # -------------------------
# # BUILD RETRIEVER
# # -------------------------
# def build_retriever(
#     k_faq: int = 2,
#     k_tickets: int = 2,
#     k_guides: int = 2,
#     max_docs: int = 6
# ) -> RunnableLambda:

#     embeddings = FastEmbedWrapper()

#     # Load Chroma collections
#     faq_store = Chroma(
#         collection_name="faq",
#         embedding_function=embeddings,
#         persist_directory=CHROMA_DIR,
#     )

#     tickets_store = Chroma(
#         collection_name="tickets",
#         embedding_function=embeddings,
#         persist_directory=CHROMA_DIR,
#     )

#     guides_store = Chroma(
#         collection_name="guides",
#         embedding_function=embeddings,
#         persist_directory=CHROMA_DIR,
#     )

#     # Individual retrievers
#     faq_retriever = faq_store.as_retriever(search_kwargs={"k": k_faq})
#     tickets_retriever = tickets_store.as_retriever(search_kwargs={"k": k_tickets})
#     guides_retriever = guides_store.as_retriever(search_kwargs={"k": k_guides})

#     # -------------------------
#     # SAFE MERGED RETRIEVAL
#     # -------------------------
#     def retrieve(query: str) -> list[Document]:
#         try:
#             docs = []

#             docs += faq_retriever.invoke(query) or []
#             docs += tickets_retriever.invoke(query) or []
#             docs += guides_retriever.invoke(query) or []

#             # deduplicate
#             docs = deduplicate_docs(docs)

#             # limit context size (VERY IMPORTANT for Gemini stability)
#             return docs[:max_docs]

#         except Exception as e:
#             print("Retriever error:", e)
#             return []

#     return RunnableLambda(retrieve)