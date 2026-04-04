from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from rag.retriever import build_context, retrieve_chunks

load_dotenv()


def setup_chain():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return None

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=token,
        max_new_tokens=512,
        temperature=0.2,
    )

    model = ChatHuggingFace(llm=llm)
    parser = StrOutputParser()
    prompt = PromptTemplate(
        template=(
            "Answer using only the provided transcript context.\n"
            "If the answer is uncertain, say so clearly.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    return RunnableSequence(first=prompt, middle=[model], last=parser)


def fallback_answer(results):
    if not results:
        return "I could not find any relevant transcript chunks for that question."

    lines = []
    for index, doc in enumerate(results, start=1):
        snippet = doc.page_content.strip().replace("\n", " ")
        snippet = snippet[:320] + ("..." if len(snippet) > 320 else "")
        lines.append(
            f"{index}. {doc.metadata.get('title', 'Unknown title')} "
            f"({doc.metadata.get('channel', 'Unknown channel')}): {snippet}"
        )

    return (
        "Hugging Face generation is not configured, so here are the most relevant transcript excerpts:\n\n"
        + "\n".join(lines)
    )


def run_query(vectorstore, query):
    results = retrieve_chunks(vectorstore, query)
    if not results:
        return "No relevant transcript content was found for that question."

    chain = setup_chain()
    if chain is None:
        return fallback_answer(results)

    context = build_context(results)
    try:
        return chain.invoke({"context": context, "question": query})
    except Exception:
        return fallback_answer(results)
