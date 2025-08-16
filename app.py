# ================== Streamlit PDF Chat App with FAISS & Google API ==================
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import pickle
import google.generativeai as genai

# ----------------- Load API Key -----------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if not google_api_key:
    st.error("❌ No GOOGLE_API_KEY found. Add it to `.streamlit/secrets.toml` or as an environment variable.")
    st.stop()

genai.configure(api_key=google_api_key)

# ----------------- PDF Utilities -----------------
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# ----------------- FAISS Vector Store -----------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    os.makedirs("faiss_store", exist_ok=True)
    vector_store.save_local("faiss_store")
    return vector_store

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists("faiss_store/index.faiss"):
        return FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
    return None

# ----------------- QA Chain -----------------
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

# ----------------- Caching -----------------
CACHE_FILE = "qa_cache.pkl"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

def user_input(user_question):
    cache = load_cache()
    if user_question in cache:
        st.write("💬 Reply (from cache):", cache[user_question])
        return

    vector_store = load_vector_store()
    if not vector_store:
        st.warning("No vector store found. Please upload PDFs first.")
        return

    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question})
    answer = response["output_text"]

    st.write("💬 Reply:", answer)

    cache[user_question] = answer
    save_cache(cache)

# ----------------- Streamlit App -----------------
def main():
    st.set_page_config(page_title="Chat with PDF using Google API & FAISS", layout="wide")
    st.header("📄 Chat with PDF using Google API & FAISS ✨")

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

