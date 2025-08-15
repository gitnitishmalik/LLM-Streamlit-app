# ================== SQLite Version Fix ==================
# Must run BEFORE importing Chroma or anything using sqlite3 internally.
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    import sqlite3  # fallback if pysqlite3 is missing
# =========================================================

import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ----------------- Load API Key -----------------
load_dotenv()

google_api_key = None
try:
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
except Exception:
    pass

if not google_api_key:
    google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("❌ No GOOGLE_API_KEY found. Add it to `.streamlit/secrets.toml` or set as an environment variable.")
    st.stop()

genai.configure(api_key=google_api_key)

# ----------------- Utility Functions -----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="chroma_db")
        vector_store.persist()
        return vector_store
    except RuntimeError as e:
        st.error(f"Chroma error: {e}")
        st.stop()

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "Answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, prompt=prompt, chain_type="stuff")

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question})
    st.write("💬 Reply:", response["output_text"])

# ----------------- Streamlit App -----------------
def main():
    st.set_page_config(page_title="Chat with PDF using Gemini", layout="wide")
    st.header("📄 Chat with PDF using Gemini ✨")

    with st.sidebar:
        st.title("📚 Upload PDF Files")
        pdf_docs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ PDFs processed and indexed!")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

    user_question = st.text_input("Ask a question about your uploaded PDFs:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
