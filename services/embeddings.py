from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:

    @staticmethod
    def load_embeddings():
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )