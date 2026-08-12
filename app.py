import streamlit as st
import time
import os

from loader.pdf_loader import PDFLoader
from services.vector_db import VectorDB
from services.rag import RAGService


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)


# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.title("📄 PDF RAG")

    st.markdown("---")

    st.write("### Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    # ---------------------------------------------------
    # Conversational Memory
    # ---------------------------------------------------

    st.markdown("---")

    st.write("### 💬 Conversation Memory")

    if st.session_state.messages:

        for message in st.session_state.messages:

            if message["role"] == "user":

                st.markdown(
                    f"**You:** {message['content']}"
                )

            elif message["role"] == "assistant":

                st.markdown(
                    f"**Bot:** {message['content']}"
                )

    else:

        st.info("No conversation yet.")

    # ---------------------------------------------------
    # Top-K
    # ---------------------------------------------------

    st.markdown("---")

    top_k = st.slider(
        "Top-K Retrieved Documents",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    # ---------------------------------------------------
    # Clear Chat
    # ---------------------------------------------------

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------
# Create Vector Database from Uploaded PDFs
# ---------------------------------------------------

chain = None

if uploaded_files:

    with st.spinner("Processing uploaded PDFs..."):

        try:

            docs = PDFLoader.load_uploaded_pdfs(uploaded_files)

            db = VectorDB.create(docs)

            db = VectorDB.create(docs)

            chain = RAGService.create_chain(
                db,
                docs,
                top_k=top_k
        )

            st.sidebar.success(
                f"{len(uploaded_files)} PDF(s) loaded successfully"
            )

        except Exception as e:

            st.sidebar.error(f"Error: {e}")


# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("📄 PDF Question Answering")

st.caption(
    "Upload PDF files and ask questions about them."
)


# ---------------------------------------------------
# Display Chat History
# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------
# User Input
# ---------------------------------------------------

if chain:

    question = st.chat_input(
        "Ask a question about your uploaded PDFs..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching documents..."):

                try:

                    start_time = time.perf_counter()

                    # Build conversation history
                    chat_history = []

                    for message in st.session_state.messages[:-1]:

                        chat_history.append(
                            (
                                message["role"],
                                message["content"]
                            )
                        )

                    response = chain.invoke(
                        {
                            "input": question,
                            "chat_history": chat_history
                        }
                    )

                    end_time = time.perf_counter()

                    response_time = end_time - start_time

                    # ---------------------------------------------------
                    # Answer
                    # ---------------------------------------------------

                    answer = response["answer"]

                    st.markdown(answer)

                    # ---------------------------------------------------
                    # Source Citations
                    # ---------------------------------------------------

                    sources = set()

                    for document in response.get("context", []):

                        source = document.metadata.get("source")

                        if source:

                            sources.add(
                                os.path.basename(source)
                            )

                    if sources:

                        st.markdown("### 📚 Sources")

                        for source in sorted(sources):

                            st.write(f"📄 {source}")

                    # ---------------------------------------------------
                    # Response Time
                    # ---------------------------------------------------

                    st.caption(
                        f"Response time: {response_time:.2f} seconds"
                    )

                    # ---------------------------------------------------
                    # Save Answer
                    # ---------------------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                    # Refresh sidebar so new conversation appears
                    st.rerun()

                except Exception as e:

                    st.error(f"Error: {e}")

else:

    st.info(
        "Please upload one or more PDF files from the sidebar."
    )