"""
Builds a merged retriever across all three Chroma collections:
  - faq
  - tickets
  - guides
"""

from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

# ✅ lightweight embeddings (NO torch, NO transformers)
from fastembed import TextEmbedding

CHROMA_DIR = "chroma_store"


# -------------------------
# FAST EMBEDDING WRAPPER
# -------------------------
class FastEmbedWrapper:
    def __init__(self):
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


# -------------------------
# BUILD RETRIEVER
# -------------------------
def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:

    embeddings = FastEmbedWrapper()

    # Load Chroma collections
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

    # Retrievers
    faq_retriever = faq_store.as_retriever(search_kwargs={"k": k_faq})
    tickets_retriever = tickets_store.as_retriever(search_kwargs={"k": k_tickets})
    guides_retriever = guides_store.as_retriever(search_kwargs={"k": k_guides})

    # Merge results
    def retrieve(query: str) -> list[Document]:
        return (
            faq_retriever.invoke(query)
            + tickets_retriever.invoke(query)
            + guides_retriever.invoke(query)
        )

    return RunnableLambda(retrieve)