from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import build_retriever


# -------------------------
# SYSTEM PROMPT (UNCHANGED BUT SAFE)
# -------------------------
SYSTEM_PROMPT = """
You are a helpful and professional telecom customer care assistant.

Use ONLY the provided context to answer the question.
If the context is insufficient, clearly say you don't know and suggest calling 611 or using the MyTelecom app.
"""


# -------------------------
# SAFE INPUT CLEANER (IMPORTANT FIX)
# -------------------------
def clean_input(question: str) -> str:
    if not question:
        return "no valid question provided"

    return question.strip()[:300]  # prevent prompt abuse


# -------------------------
# SAFE FORMATTER (HARD LIMIT FIX)
# -------------------------
def _format_docs(docs):
    if not docs:
        return "No relevant context found."

    sections = []
    total_chars = 0

    MAX_DOCS = 3        # 🔥 reduced (critical fix)
    MAX_CHARS = 1800    # 🔥 hard cap for Gemini safety

    for doc in docs[:MAX_DOCS]:
        text = (doc.page_content or "")[:400]  # strict cut
        source = doc.metadata.get("source", "unknown").upper()

        block = f"[{source}]\n{text}\n"

        if total_chars + len(block) > MAX_CHARS:
            break

        sections.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(sections) if sections else "No relevant context found."


# -------------------------
# BUILD CHAIN (FINAL SAFE VERSION)
# -------------------------
def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\nContext:\n{context}"),
        ("human", "{question}")
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",   # ✅ KEEP YOUR MODEL
        temperature=0,
        max_retries=3
    )

    chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnableLambda(clean_input)
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# """
# Builds the RAG chain:
#   merged retriever → prompt → gemma-4-31b-it  → string output
# """
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.documents import Document
# from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI

# from retriever import build_retriever

# SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
# Your job is to help customers resolve technical issues with their mobile service.

# Use ONLY the context below to answer the customer's question.
# The context comes from two sources:
# - FAQ entries (general policy and how-to information)
# - Past support tickets (real resolved cases with step-by-step resolutions)

# If the context does not contain enough information to answer confidently, say so clearly \
# and suggest the customer call 611 or use the MyTelecom app.

# Context:
# {context}
# """


# #Converts retrieved documents into a single text block
# # def _format_docs(docs: list[Document]) -> str:
# #     sections = []
# #     for doc in docs:
# #         source = doc.metadata.get("source", "unknown").upper()
# #         sections.append(f"[{source}]\n{doc.page_content}")
# #     return "\n\n---\n\n".join(sections)
# def _format_docs(docs):
#     sections = []
#     total_chars = 0
#     MAX_CHARS = 4000

#     for doc in docs:
#         text = doc.page_content[:800]  # limit per doc
#         source = doc.metadata.get("source", "unknown").upper()

#         chunk = f"[{source}]\n{text}\n"
#         if total_chars + len(chunk) > MAX_CHARS:
#             break

#         sections.append(chunk)
#         total_chars += len(chunk)

#     return "\n\n---\n\n".join(sections)

# # def build_chain():
# #     retriever = build_retriever()

# #     prompt = ChatPromptTemplate.from_messages([
# #         ("system", SYSTEM_PROMPT + "\n\nContext:\n{context}"),
# #         ("human", "{question}"),
# #     ])

# #     llm = ChatGoogleGenerativeAI(
# #         model="gemma-4-31b-it",
# #         # model="gemini-1.5-flash-latest",
# #         temperature=0,
# #         max_tokens=None,
# #         reasoning_format="parsed",
# #         timeout=None,
# #         max_retries=2,
# #     )

# #     chain = (
# #         {"context": retriever | _format_docs, "question": RunnablePassthrough()}
# #         | prompt #prompt is augmented with context +systsem instructions+question
# #         | llm
# #         | StrOutputParser() # convert model output to plain string
# #     )
# #     return chain
# def build_chain():
#     retriever = build_retriever()

#     # SYSTEM_PROMPT = """
#     # You are a helpful and professional telecom customer care assistant.

#     # Use ONLY the provided context to answer.
#     # If context is insufficient, say so and suggest calling 611 or using MyTelecom app.
#     # """

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", SYSTEM_PROMPT + "\n\nContext:\n{context}"),
#         ("human", "{question}"),
#     ])

#     llm = ChatGoogleGenerativeAI(
#         model="gemma-4-31b-it",
#         temperature=0,
#         max_retries=2,
#     )

#     chain = (
#         {
#             "context": retriever | _format_docs,
#             "question": RunnablePassthrough()
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return chain