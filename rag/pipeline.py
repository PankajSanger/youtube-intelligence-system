from __future__ import annotations

from dotenv import load_dotenv

from openai_service import get_openai_client, get_reasoning_model, openai_is_configured
from rag.retriever import build_context, retrieve_chunks

load_dotenv()


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
        "OpenAI generation is not configured, so here are the most relevant transcript excerpts:\n\n"
        + "\n".join(lines)
    )


def run_query(vectorstore, query):
    results = retrieve_chunks(vectorstore, query)
    if not results:
        return "No relevant transcript content was found for that question."

    if not openai_is_configured():
        return fallback_answer(results)

    context = build_context(results)
    client = get_openai_client()

    try:
        response = client.responses.create(
            model=get_reasoning_model(),
            input=(
                "Answer using only the provided transcript context. "
                "If the answer is uncertain, say so clearly.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{query}\n\n"
                "Answer:"
            ),
            reasoning={"effort": "none"},
            text={"verbosity": "medium"},
            max_output_tokens=900,
        )
        return response.output_text
    except Exception:
        return fallback_answer(results)
