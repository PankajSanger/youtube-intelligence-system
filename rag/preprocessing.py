import re
import os
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()


def clean_text(text):
    if not text:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_english(text):
    try:
        text.encode("ascii")
        return True
    except:
        return False


def build_translation_chain():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    prompt = PromptTemplate(
        template="""
Translate the following text to English.
Do not summarize.

Text:
{text}
""",
        input_variables=["text"]
    )

    model = ChatHuggingFace(llm=llm)
    parser = StrOutputParser()

    return RunnableSequence(
        first=prompt,
        middle=[model],
        last=parser
    )


def preprocess_text(text, translation_chain=None):

    text = clean_text(text)

    if translation_chain and not is_english(text):
        try:
            text = translation_chain.invoke({"text": text})
        except:
            pass

    return text