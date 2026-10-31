"""Vector search (FAISS) with an automatic TF-IDF fallback so the app runs without any API key."""
from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _embeddings(mode: str):
    mode = mode.lower()
    if mode == "tfidf":
        return None
    if mode == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.getenv("GOOGLE_API_KEY") and os.getenv("LLM_PROVIDER", "gemini") == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"))
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return None


class CampusRetriever:
    def __init__(self, docs: list[Document]):
        self.chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
        self.mode, self.index, self.note = "tfidf", None, ""
        if not self.chunks:
            return
        try:
            emb = _embeddings(os.getenv("EMBEDDINGS", "auto"))
            if emb:
                from langchain_community.vectorstores import FAISS
                self.index = FAISS.from_documents(self.chunks, emb)
                self.mode = "faiss"
        except Exception as e:  # bad key, quota, offline...
            self.index, self.note = None, f"Embeddings unavailable, using keyword search ({type(e).__name__})."
        if self.index is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
            self.matrix = self.vec.fit_transform([c.page_content for c in self.chunks])

    def search(self, query: str, k: int = 4, category: str | None = None,
               source_type: str | None = None) -> list[Document]:
        if not self.chunks:
            return []
        filtered = bool(category or source_type)
        pool = min(len(self.chunks), k * 6 if filtered else k)
        if self.index is not None:
            hits = self.index.similarity_search(query, k=pool)
        else:
            from sklearn.metrics.pairwise import linear_kernel
            sims = linear_kernel(self.vec.transform([query]), self.matrix).ravel()
            hits = [self.chunks[i] for i in sims.argsort()[::-1][:pool] if sims[i] > 0]
        if category:
            hits = [h for h in hits if h.metadata.get("category") == category]
        if source_type:
            hits = [h for h in hits if h.metadata.get("type") == source_type]
        return hits[:k]
