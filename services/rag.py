from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from prompts.prompt import RAG_PROMPT
from services.llm import LLM


class RAGService:

    @staticmethod
    def create_chain(vector_db, docs, top_k=3):

        # -----------------------------------------------
        # Vector Retriever
        # -----------------------------------------------

        vector_retriever = vector_db.as_retriever(
            search_kwargs={"k": top_k}
        )

        # -----------------------------------------------
        # Keyword Retriever
        # -----------------------------------------------

        keyword_retriever = BM25Retriever.from_documents(docs)

        keyword_retriever.k = top_k

        # -----------------------------------------------
        # Hybrid Retriever
        # -----------------------------------------------

        retriever = EnsembleRetriever(
            retrievers=[
                keyword_retriever,
                vector_retriever
            ],
            weights=[
                0.5,
                0.5
            ]
        )

        # -----------------------------------------------
        # Document Chain
        # -----------------------------------------------

        document_chain = create_stuff_documents_chain(
            llm=LLM.load(),
            prompt=RAG_PROMPT
        )

        # -----------------------------------------------
        # Retrieval Chain
        # -----------------------------------------------

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        return retrieval_chain