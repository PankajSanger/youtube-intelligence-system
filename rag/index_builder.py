import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag.preprocessing import preprocess_text, build_translation_chain


def build_index(file_path, save_path="faiss_index"):

    df = pd.read_excel(file_path)

    translation_chain = build_translation_chain()

    documents = []

    for i in range(len(df)):
        raw_text = df.loc[i, "transcript"]

        text = preprocess_text(
            raw_text,
            translation_chain=translation_chain
        )

        if not text.strip():
            continue

        documents.append(Document(
            page_content=text,
            metadata={"video_id": df.loc[i, "video_id"]}
        ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "।", ", ", " ", ""]
    )

    docs = splitter.split_documents(documents)

    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = i

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_path)

    print("✅ FAISS index built and saved")