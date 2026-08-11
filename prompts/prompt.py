from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.
Answer the user's question using only the provided context.
If the answer cannot be found in the context, say that you don't know.

Context:
{context}"""
    ),
    ("human", "{input}"),
])