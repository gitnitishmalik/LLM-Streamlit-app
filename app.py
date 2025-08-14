import os
import sys
import shutil
import errno
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

# =========================
# Config & environment
# =========================
load_dotenv()

def get_api_key():
    try:
        secrets_path = os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY")

GOOGLE_API_KEY = get_api_key()
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found. Set it in .env (local) or Streamlit secrets (cloud).")

genai.configure(api_key=GOOGLE_API_KEY)

# SQLite shim for some envs
try:
    import pysqlite3  # noqa
    sys.modules["sqlite3"] = __import__("pysqlite3")
except Exception:
    pass

# Always use a writeable directory on cloud
def pick_persist_dir() -> str:
    candidates = ["/tmp/chroma_db", "./chroma_db"]
    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            testfile = os.path.join(path, ".write_test")
            with open(testfile, "w") as f:
                f.write("ok")
            os.remove(testfile)
            return path
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EROFS):
                # if it's a different error, still skip to next candidate
                pass
    # Fallback to /tmp
    return "/tmp/chroma_db"

VECTOR_STORE_DIR = pick_persist_dir()

# =========================
# Helpers
# =========================
def get_pdf_text(pdf_docs):
    """Extract text, skipping pages that return None/empty."""
    parts = []
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                t = page.extract_text() or ""
                t = t.strip()
                if t:
                    parts.append(t)
        except Exception as e:
            st.warning(f"Couldn't read one PDF ({getattr(pdf, 'name', 'unknown')}): {e}")
    return "\n\n".join(parts)

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    return splitter.split_text(text)

def _clean_chunks(chunks):
    return [c.strip() for c in chunks if isinstance(c, str) and c and c.strip()]

def reset_vector_store():
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR, ignore_errors=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def build_vector_store(chunks):
    """Create & persist Chroma safely in a writable dir."""
    chunks = _clean_chunks(chunks)
    if not chunks:
        st.error("❌ No valid text to index from the uploaded PDFs.")
        return False
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vs = Chroma.from_texts(
            chunks,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_DIR,
        )
        vs.persist()
        return True
    except chromadb.errors.InternalError as e:
        st.error(f"❌ Chroma internal error: {e}")
        st.info("Clearing the vector store and retrying once...")
        reset_vector_store()
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vs = Chroma.from_texts(
                chunks,
                embedding=embeddings,
                persist_directory=VECTOR_STORE_DIR,
            )
            vs.persist()
            return True
        except Exception as e2:
            st.error(f"Failed again after reset: {e2}")
            return False
    except PermissionError:
        st.error("❌ Write permission denied for vector store directory.")
        return False

def get_chain():
    prompt_template = """
Answer the question as thoroughly as possible from the provided context.
If the answer is not in the context, say "Answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, prompt=prompt, chain_type="stuff")

def answer_question(q):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
        # Similarity search returns Document objects
        docs = db.similarity_search(q, k=4)
        docs = [d for d in docs if isinstance(d.page_content, str) and d.page_content.strip()]
        if not docs:
            st.warning("No valid results in the vector store. Try processing PDFs again.")
            return
        chain = get_chain()
        resp = chain({"input_documents": docs, "question": q})
        st.write("💬 Reply:", resp.get("output_text", "").strip() or "(empty)")
    except chromadb.errors.InternalError as e:
        st.error(f"Chroma error while querying: {e}")
        st.info("Resetting the vector store...")
        reset_vector_store()
        st.warning("Vector store cleared. Please re-upload & process PDFs.")

# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="Chat with PDF using Gemini")
    st.header("📄 Chat with PDF using Gemini ✨")

    # Ask
    q = st.text_input("Ask a question about your uploaded PDFs:")
    if q:
        answer_question(q)

    # Upload & index
    with st.sidebar:
        st.title("📚 Upload PDF Files")
        st.caption(f"Vector store path: `{VECTOR_STORE_DIR}`")
        pdfs = st.file_uploader("Upload your PDF(s)", type=["pdf"], accept_multiple_files=True)

        if st.button("Submit & Process"):
            if not pdfs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    reset_vector_store()  # ensure a clean, writable DB
                    text = get_pdf_text(pdfs)
                    chunks = get_text_chunks(text)
                    ok = build_vector_store(chunks)
                    if ok:
                        st.success("✅ PDFs processed and indexed!")

if __name__ == "__main__":
    main()
