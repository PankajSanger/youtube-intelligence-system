from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ======================
# 1. LOAD + CHUNK
# ======================

loader = TextLoader("output.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# ======================
# 2. EMBEDDINGS + VECTOR DB
# ======================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(docs, embeddings)

# ======================
# 3. QUERY + RETRIEVAL
# ======================
query = "What ingredients are being talked about and also give mention count of each ingredient"

results = vector_db.similarity_search(query, k=3)

# Build context for LLM
context = "\n\n".join([doc.page_content for doc in results])

# ======================
# 4. LLM SETUP
# ======================
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# ======================
# 5. PROMPT
# ======================
prompt = PromptTemplate(
    template="""
You are an expert analyst.

Determine what the user is asking (e.g., ingredients, causes, benefits, risks, summary).

Then answer using ONLY the provided context.

Rules:
- No external knowledge
- If missing → say "Not found in context"
- Structure answer appropriately (list, explanation, summary)

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


chain = RunnableSequence(
    first=prompt,
    middle=[model],
    last=parser
)

# ======================
# 6. LLM RESPONSE
# ======================
response = chain.invoke({
    "context": context,
    "question": query
})

print("\n--- Final Answer ---\n")
print(response)