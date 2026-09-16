import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv
from rag_chain import build_chain

load_dotenv()


# -------------------------
# CONFIG
# -------------------------
SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi calling?",
    "I was charged for roaming but had a bundle active",
    "How do I unlock my phone for another network?",
]

st.set_page_config(
    page_title="Telecom Support Chat",
    page_icon="📡",
    layout="centered",
)


# -------------------------
# CACHE CHAIN
# -------------------------
@st.cache_resource
def get_chain():
    return build_chain()


# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.title("📡 Telecom Support")
    st.caption("RAG-powered assistant")
    st.divider()

    st.markdown("**Sample questions**")

    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []


# -------------------------
# MAIN UI
# -------------------------
st.title("Customer Care Assistant")
st.caption("Ask about SIM, billing, roaming, internet issues, etc.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------
# INPUT HANDLING (FIXED)
# -------------------------
question = st.chat_input("Describe your issue…")

# sidebar override (safe)
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


# -------------------------
# SAFE INPUT CLEANER
# -------------------------
def clean_input(q: str) -> str:
    if not q:
        return ""
    return q.strip()[:300]


# -------------------------
# PROCESS QUERY
# -------------------------
if question:
    question = clean_input(question)

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            chain = get_chain()

            try:
                with st.spinner("Analyzing your issue..."):
                    response = chain.invoke(question)

                if not response:
                    response = "I couldn't find enough information to answer this."

                st.markdown(response)

            except Exception as e:
                response = (
                    "Sorry, I'm having trouble processing your request right now. "
                    "Please try again or contact support at 611."
                )
                st.error(response)

                # debug (optional)
                st.caption(f"Debug: {str(e)[:200]}")

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
# import os
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# import streamlit as st
# from dotenv import load_dotenv
# from rag_chain import build_chain

# load_dotenv()

# SAMPLE_QUESTIONS = [
#     "Why is my mobile internet so slow?",
#     "My calls keep dropping — what should I do?",
#     "How do I activate international roaming?",
#     "Why is my bill higher than usual this month?",
#     "My phone shows SIM not detected after a restart",
#     "How do I enable Wi-Fi calling?",
#     "I was charged for roaming but had a bundle active",
#     "How do I unlock my phone for another network?",
# ]

# st.set_page_config(
#     page_title="Telecom Support Chat",
#     page_icon="📡",
#     layout="centered",
# )

# @st.cache_resource
# def get_chain():
#     return build_chain()

# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "pending_question" not in st.session_state:
#     st.session_state.pending_question = None

# # ── Sidebar ──────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.title("📡 Telecom Support")
#     st.caption("Powered by RAG · Qwen3-32B on Groq")
#     st.divider()

#     st.markdown("**Sample questions**")
#     st.caption("Click one to send it instantly.")
#     for q in SAMPLE_QUESTIONS:
#         if st.button(q, use_container_width=True):
#             st.session_state.pending_question = q

#     st.divider()
#     if st.button("🗑️ Clear conversation", use_container_width=True):
#         st.session_state.messages = []

# # ── Main ─────────────────────────────────────────────────────────────────────
# st.title("Customer Care Assistant")
# st.caption("Ask me anything about your mobile service — connectivity, billing, SIM, roaming, and more.")

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # Resolve question from chat input or sidebar button click
# question = st.chat_input("Describe your issue…")
# if st.session_state.pending_question:
#     question = st.session_state.pending_question
#     st.session_state.pending_question = None

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.chat_message("assistant"):
#         chain = get_chain()
#         response = chain.invoke(question)
#         st.write(response)
#         #response = st.write_stream(chain.stream(question))

#     st.session_state.messages.append({"role": "assistant", "content": response})
