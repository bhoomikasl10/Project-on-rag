import streamlit as st
import time

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
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.title("📄 PDF RAG")

    st.markdown("---")

    st.write("### Settings")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.success("Vector Database Loaded")


# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------------------------
# Load Database
# ---------------------------------------------------

def load_chain():

    db = VectorDB.load()

    chain = RAGService.create_chain(db)

    return chain


chain = load_chain()


# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("📄 PDF Question Answering")

st.caption("Ask questions about your PDF")


# ---------------------------------------------------
# Display Chat History
# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------
# User Input
# ---------------------------------------------------

question = st.chat_input("Ask a question about your PDF...")


if question:

    # Show User Question

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # Assistant

    with st.chat_message("assistant"):

        with st.spinner("Searching document..."):

            try:

                # Start timer
                start_time = time.perf_counter()

                # Run RAG
                response = chain.invoke(
                    {
                        "input": question
                    }
                )

                # Stop timer
                end_time = time.perf_counter()

                # Calculate response time
                response_time = end_time - start_time

                # Get answer
                answer = response["answer"]

                # Display answer
                st.markdown(answer)

                # Display response time
                st.caption(
                    f"Response time: {response_time:.2f} seconds"
                )

                # Save answer
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(f"Error: {e}")

                print(e)