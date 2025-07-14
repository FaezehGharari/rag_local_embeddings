import os
from document_loader import load_documents
from vector_store import VectorStore
from transformers import pipeline

# Load documents from the docs directory (absolute path)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs_path = os.path.join(project_root, 'rag_project', 'docs')
documents = load_documents(docs_path)

# Initialize the vector store and add documents
vector_store = VectorStore()
vector_store.add_documents(documents)

# Get the user query
query = input("Enter your question: ")

# Retrieve relevant documents
retrieved_docs = vector_store.query(query, top_k=3)
context = "\n".join([content for _, content in retrieved_docs])

# Generate a natural language answer using HuggingFace transformers pipeline
qa_pipeline = pipeline("text-generation", model="gpt2")
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer in detail:"
response = qa_pipeline(prompt, max_length=256, do_sample=True, temperature=0.7)
answer = response[0]["generated_text"][len(prompt):].strip()
print("\nAnswer:\n", answer) 