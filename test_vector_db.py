from loader.pdf_loader import PDFLoader
from services.vector_db import VectorDB

docs = PDFLoader.load_multiple_pdfs(
    r"C:\Users\Administrator\Bhoomi\data"
)

db = VectorDB.create(docs)

print("Database Created Successfully")
