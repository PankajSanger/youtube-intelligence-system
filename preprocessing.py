# ==============================
# 1. IMPORTS
# ==============================
import os
import pandas as pd
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


# ==============================
# 2. CONFIGURATION
# ==============================
load_dotenv()

FILE_PATH = "C:\\Pankaj\\youtube_mvp\\youtube_data_for_project.xlsx"
CHUNK_SIZE = 100
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# ==============================
# 3. MODEL SETUP
# ==============================
def build_translation_chain():
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN
    )

    prompt = PromptTemplate(
        template="""Translate the following text to English.
Only return translated text.

Transcript: {transcript}
""",
        input_variables=['transcript']
    )

    model = ChatHuggingFace(llm=llm)
    parser = StrOutputParser()

    chain = RunnableSequence(
        first=prompt,
        middle=[model],
        last=parser
    )

    return chain


# ==============================
# 4. DATA HANDLING
# ==============================
def load_transcripts(file_path):
    df = pd.read_excel(file_path)
    return df['transcript'].dropna().astype(str).tolist()


def create_chunks(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


# ==============================
# 5. CORE LOGIC (PIPELINE)
# ==============================
def translate_transcripts(transcripts, chain, chunk_size):
    results = []

    for transcript in transcripts:
        chunks = create_chunks(transcript, chunk_size)

        for chunk in chunks:
            try:
                translated = chain.invoke({'transcript': chunk})
                print(translated)
                results.append(translated)
            except Exception as e:
                print(f"Error: {e}")
                continue

    return " ".join(results)


# ==============================
# 6. OUTPUT HANDLING
# ==============================
def save_to_txt(text, filename="output.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)


# ==============================
# 7. MAIN EXECUTION
# ==============================
def main():
    print("Loading data...")
    transcripts = load_transcripts(FILE_PATH)

    print("Building model...")
    chain = build_translation_chain()

    print("Translating...")
    final_text = translate_transcripts(transcripts, chain, CHUNK_SIZE)

    print("Saving output...")
    save_to_txt(final_text)

    print("Done.")

    return final_text


if __name__ == "__main__":
    main()