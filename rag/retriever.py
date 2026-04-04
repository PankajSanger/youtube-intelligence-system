import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag.index_builder import build_index


def load_index(path="faiss_index"):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def load_or_create_index(data_path, index_path="faiss_index"):

    if not os.path.exists(index_path):
        print("⚡ Building FAISS index...")
        build_index(data_path, index_path)

    return load_index(index_path)


def retrieve_chunks(vectorstore, query, k=3):
    return vectorstore.similarity_search(query, k=k)


def build_context(results):
    return "\n\n".join([
        f"Video ID: {doc.metadata['video_id']}\nText: {doc.page_content}"
        for doc in results
    ])