from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile


class PDFLoader:

    @staticmethod
    def load_pdf(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents

    @staticmethod
    def load_multiple_pdfs(pdf_directory: str):
        all_documents = []

        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_directory, filename)

                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                all_documents.extend(documents)

        return all_documents

    @staticmethod
    def load_uploaded_pdfs(uploaded_files):
        all_documents = []

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            all_documents.extend(documents)

            os.remove(temp_path)

        return all_documents