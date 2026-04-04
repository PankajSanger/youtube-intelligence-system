from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag.index_builder import build_index


def _index_is_complete(path="faiss_index") -> bool:
    index_path = Path(path)
    return (index_path / "index.faiss").exists() and (index_path / "index.pkl").exists()


def load_index(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_or_create_index(data_path, index_path="faiss_index", force_rebuild=False):
    if force_rebuild or not _index_is_complete(index_path):
        build_index(data_path, index_path)

    return load_index(index_path)


def retrieve_chunks(vectorstore, query, k=4):
    return vectorstore.similarity_search(query, k=k)


def build_context(results):
    chunks = []
    for doc in results:
        chunks.append(
            "\n".join(
                [
                    f"Video ID: {doc.metadata.get('video_id', '')}",
                    f"Title: {doc.metadata.get('title', '')}",
                    f"Channel: {doc.metadata.get('channel', '')}",
                    f"Published At: {doc.metadata.get('published_at', '')}",
                    f"Transcript Language: {doc.metadata.get('transcript_language', '')}",
                    f"Text: {doc.page_content}",
                ]
            )
        )
    return "\n\n".join(chunks)
