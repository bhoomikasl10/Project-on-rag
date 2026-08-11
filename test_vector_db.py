from loaders.pdf_loader import PDFLoader
from services.vector_db import VectorDB

docs = PDFLoader.load_pdf(r"C:\Users\Administrator\Bhoomi\data\Bhoomika S L 2027.pdf")

db = VectorDB.create(docs)

print("Database Created Successfully")
