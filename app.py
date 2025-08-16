import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found.")
    st.stop()

genai.configure(api_key=google_api_key)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    return vector_store

# Function to create a conversational chain
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

# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question})
    st.write("💬 Reply:", response["output_text"])

# Streamlit app
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
