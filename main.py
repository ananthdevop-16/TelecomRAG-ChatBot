"""
CLI entry point for the telecom RAG chatbot.
Usage: python main.py
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
from rag_chain import build_chain

load_dotenv()


# -------------------------
# INPUT CLEANER (IMPORTANT FIX)
# -------------------------
def clean_input(text: str) -> str:
    if not text:
        return ""
    return text.strip()[:300]


def main():
    print("=== Telecom Customer Care Chatbot (RAG) ===")
    print("Type your question. Type 'quit' to exit.\n")

    chain = build_chain()

    while True:
        question = input("Customer: ")

        question = clean_input(question)

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        try:
            print("\nAssistant: ", end="", flush=True)

            # -------------------------
            # SAFE STREAMING WRAPPER
            # -------------------------
            response_stream = chain.stream(question)

            for chunk in response_stream:
                if chunk:
                    print(chunk, end="", flush=True)

            print("\n")

        except Exception as e:
            # 🔥 CRITICAL FIX: prevents 500 crash propagation
            print("\n[Error: Temporary issue in model/API. Please try again.]\n")
            print(f"(Debug: {str(e)[:200]})\n")


if __name__ == "__main__":
    main()

# """
# CLI entry point for the telecom RAG chatbot.
# Usage: python main.py
# """
# import os
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# from dotenv import load_dotenv
# from rag_chain import build_chain

# load_dotenv()


# def main():
#     print("=== Telecom Customer Care Chatbot (RAG) ===")
#     print("Type your question and press Enter. Type 'quit' to exit.\n")

#     chain = build_chain()

#     while True:
#         question = input("Customer: ").strip()
#         if not question:
#             continue
#         if question.lower() in {"quit", "exit", "q"}:
#             print("Goodbye!")
#             break

#         print("\nAssistant: ", end="", flush=True)
#         for chunk in chain.stream(question):
#             print(chunk, end="", flush=True)
#         print("\n")


# if __name__ == "__main__":
#     main()
