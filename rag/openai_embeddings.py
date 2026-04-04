from __future__ import annotations

from openai_service import get_embedding_model, get_openai_client


class OpenAIEmbeddingFunction:
    def __init__(self, model_name: str | None = None) -> None:
        self.client = get_openai_client()
        self.model_name = model_name or get_embedding_model()

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding
