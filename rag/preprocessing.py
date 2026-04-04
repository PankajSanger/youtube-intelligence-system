from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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


def build_translation_chain():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return None

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=token,
        max_new_tokens=512,
        temperature=0.1,
    )

    prompt = PromptTemplate(
        template=(
            "Translate the following text to English. "
            "Keep names and intent intact. Do not summarize.\n\n"
            "Text:\n{text}"
        ),
        input_variables=["text"],
    )

    model = ChatHuggingFace(llm=llm)
    parser = StrOutputParser()
    return RunnableSequence(first=prompt, middle=[model], last=parser)


def preprocess_text(text, translation_chain=None):
    text = clean_text(text)
    if not text:
        return ""

    if translation_chain and not is_mostly_english(text):
        try:
            text = translation_chain.invoke({"text": text})
        except Exception:
            pass

    return clean_text(text)
