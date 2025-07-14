import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []

    def embed(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    def add_documents(self, documents):
        texts = [content for _, content in documents]
        embeddings = self.embed(texts)
        self.index.add(embeddings)
        self.documents.extend(documents)

    def query(self, query_text, top_k=3):
        query_embedding = self.embed([query_text])
        D, I = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in I[0]] 