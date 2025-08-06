# rag.py
import requests
import fitz  # PyMuPDF
import time
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

embedding_model_name = "intfloat/e5-small-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cpu"},  # set to "cuda" if GPU is available
    encode_kwargs={"normalize_embeddings": True}
)

def timed(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"🔄 Starting: {name}...")
            start = time.time()
            result = func(*args, **kwargs)
            duration = round(time.time() - start, 2)
            print(f"✅ Finished: {name} in {duration}s")
            return result
        return wrapper
    return decorator

@timed("PDF stream & text split")
def load_and_split_pdf(pdf_url):
    response = requests.get(pdf_url)
    doc = fitz.open(stream=response.content, filetype="pdf")

    docs = []
    for i, page in enumerate(doc):
        text = page.get_text()
        # Prepend 'passage:' for E5 model (improves quality)
        docs.append(Document(page_content=f"passage: {text}", metadata={"page": i + 1}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

@timed("Vector store creation (in-memory)")
def create_vectorstore(docs):
    return FAISS.from_documents(docs, embeddings)

def build_qa_chain(vectorstore):
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

@timed("LLM Q&A answering")
def get_answers(chain, questions):
    results = []
    for question in questions:
        start = time.time()
        result = chain(question)
        duration = round(time.time() - start, 2)
        results.append({
            "question": question,
            "answer": result["result"],
            "time_taken_seconds": duration
        })
    return results