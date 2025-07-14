import os
from PyPDF2 import PdfReader

def load_documents(doc_dir="docs"):
    documents = []
    for filename in os.listdir(doc_dir):
        file_path = os.path.join(doc_dir, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append((filename, content))
        elif filename.endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                content = "\n".join(page.extract_text() or "" for page in reader.pages)
                documents.append((filename, content))
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
    return documents 