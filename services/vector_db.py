from langchain_chroma import Chroma
from services.embedding import EmbeddingModel
from config import DB_DIRECTORY
import os


class VectorDB:

    @staticmethod
    def create(docs):

        print("=" * 60)
        print("Current Working Directory :", os.getcwd())
        print("Database Path :", os.path.abspath(DB_DIRECTORY))
        print("Database Exists :", os.path.exists(DB_DIRECTORY))
        print("=" * 60)

        embeddings = EmbeddingModel.load_embeddings()

        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB_DIRECTORY
        )

        print("Vector Database Created Successfully")

        return db

    @staticmethod
    def load():

        embeddings = EmbeddingModel.load_embeddings()

        db = Chroma(
            persist_directory=DB_DIRECTORY,
            embedding_function=embeddings
        )

        print("Vector Database Loaded Successfully")

        return db