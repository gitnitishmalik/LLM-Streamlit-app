import os
import sys
import shutil
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import chromadb

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()

def get_api_key():
    """Get API key from Streamlit secrets (if available) or .env."""
    try:
        secrets_path = os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY")

google_api_key = get_api_key()

if not google_api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found. "
                     "Set it in .env (Local) or Streamlit secrets (Cloud)")

# Configure Gemini API
genai.configure(api_key=google_api_key)

# --------------------------
# SQLite fix for Chroma
# --------------------------
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

# --------------------------
# Utility Functions
# --------------------------
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Split text into chunks for embeddings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    ).split_text(text)

def get_vector_store(text_chunks):
    """Create and persist Chroma vector store safely."""
    # Clear old/corrupted DB
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    # Filter out invalid/empty chunks
    text_chunks = [chunk.strip() for chunk in text_chunks if isinstance(chunk, str) and chunk.strip()]

    if not text_chunks:
        st.error("❌ No valid text chunks to index. Please check your PDF content.")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma.from_texts(
            text_chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        vector_store.persist()
        st.success("✅ Vector store created successfully!")

    except chromadb.errors.InternalError as e:
        st.error(f"❌ Chroma internal error: {e}")
        st.warning("Clearing and retrying...")
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

def get_conversational_chain():
    """Load QA chain with Gemini model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "Answer is not available in the context" — don't make anything up.

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
    """Search in vector store and get response."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        results = db.similarity_search(user_question)

        clean_results = [doc for doc in results if isinstance(doc.page_content, str) and doc.page_content.strip()]

        if not clean_results:
            st.warning("⚠ No valid results found. Try re-uploading PDFs.")
            return

        chain = get_conversational_chain()
        response = chain({"input_documents": clean_results, "question": user_question})
        st.write("💬 Reply:", response["output_text"])

    except chromadb.errors.InternalError as e:
        st.error(f"❌ Chroma DB error: {e}")
        st.info("Rebuilding vector store...")
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")
        st.warning("⚠ Vector store cleared. Please re-upload PDFs.")

# --------------------------
# Streamlit App
# --------------------------
def main():
    st.set_page_config(page_title="Chat with PDF using Gemini")
    st.header("📄 Chat with PDF using Gemini ✨")

    # User Question
    question = st.text_input("Ask a question about your uploaded PDFs:")
    if question:
        user_input(question)

    # Sidebar for PDF Upload
    with st.sidebar:
        st.title("📚 Upload PDF Files")
        pdf_docs = st.file_uploader("Upload your PDF(s)", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(text)
                    get_vector_store(chunks)
            else:
                st.warning("⚠ Please upload at least one PDF.")

if __name__ == "__main__":
    main()
