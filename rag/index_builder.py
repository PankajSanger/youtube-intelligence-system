from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.preprocessing import build_translation_chain, preprocess_text
from utils.data_manager import load_dataset


def build_index(file_path, save_path="faiss_index"):
    df = load_dataset(file_path)
    if df.empty:
        raise ValueError("No dataset found to index.")

    translation_chain = build_translation_chain()
    documents = []

    for row in df.itertuples(index=False):
        text = preprocess_text(row.transcript, translation_chain=translation_chain)
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "video_id": row.video_id,
                    "title": row.title,
                    "channel": row.channel,
                    "published_at": row.published_at,
                    "url": row.url,
                    "transcript_language": row.transcript_language,
                },
            )
        )

    if not documents:
        raise ValueError("No usable transcripts found. Fetch videos with transcripts before building the index.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", "।", ", ", " ", ""],
    )
    docs = splitter.split_documents(documents)

    for index, doc in enumerate(docs):
        doc.metadata["chunk_id"] = index

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_dir))
    return vectorstore
