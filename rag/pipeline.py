import os
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

from rag.retriever import retrieve_chunks, build_context

load_dotenv()


def setup_chain():

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    model = ChatHuggingFace(llm=llm)
    parser = StrOutputParser()

    prompt = PromptTemplate(
        template="""
Answer using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    return RunnableSequence(
        first=prompt,
        middle=[model],
        last=parser
    )


def run_query(vectorstore, query):

    results = retrieve_chunks(vectorstore, query)
    context = build_context(results)
    chain = setup_chain()

    return chain.invoke({
        "context": context,
        "question": query
    })