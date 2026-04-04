from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_REASONING_MODEL = "gpt-5.4"
DEFAULT_TRANSLATION_MODEL = "gpt-5.4-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def openai_is_configured() -> bool:
    return bool(_env("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    return OpenAI(api_key=api_key)


def get_reasoning_model() -> str:
    return _env("OPENAI_REASONING_MODEL", DEFAULT_REASONING_MODEL)


def get_translation_model() -> str:
    return _env("OPENAI_TRANSLATION_MODEL", DEFAULT_TRANSLATION_MODEL)


def get_embedding_model() -> str:
    return _env("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
