from __future__ import annotations

import re

from dotenv import load_dotenv

from openai_service import get_openai_client, get_translation_model, openai_is_configured

load_dotenv()


def clean_text(text):
    if not text:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_mostly_english(text: str, threshold: float = 0.85) -> bool:
    if not text:
        return True

    letters = [char for char in text if char.isalpha()]
    if not letters:
        return True

    ascii_letters = sum(1 for char in letters if ord(char) < 128)
    return (ascii_letters / len(letters)) >= threshold


def should_translate(language: str, text: str) -> bool:
    normalized = (language or "").strip().lower()
    if normalized in {"", "unknown"}:
        return not is_mostly_english(text)
    if normalized.startswith("en"):
        return False
    return True


def split_for_translation(text: str, max_chars: int = 1800) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


def build_translation_chain():
    if not openai_is_configured():
        return None
    return {
        "client": get_openai_client(),
        "model": get_translation_model(),
    }


def translate_text(text: str, translation_chain) -> str:
    if not translation_chain:
        return text

    client = translation_chain["client"]
    model = translation_chain["model"]
    translated_chunks: list[str] = []

    for chunk in split_for_translation(text):
        try:
            response = client.responses.create(
                model=model,
                input=(
                    "Translate the following transcript text to English. "
                    "Preserve names, meaning, and details. Do not summarize.\n\n"
                    f"Text:\n{chunk}"
                ),
                reasoning={"effort": "none"},
                text={"verbosity": "low"},
                max_output_tokens=1200,
            )
            translated_chunks.append(clean_text(response.output_text))
        except Exception:
            translated_chunks.append(chunk)

    return " ".join(part for part in translated_chunks if part).strip()


def preprocess_text(text, transcript_language: str = "", translation_chain=None):
    text = clean_text(text)
    if not text:
        return ""

    if should_translate(transcript_language, text):
        text = translate_text(text, translation_chain)

    return clean_text(text)
