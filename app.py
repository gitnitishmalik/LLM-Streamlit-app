import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyAKGY-pN5QMPRzgPxBsWFArQ2_0RLVSiic"  # You can remove this if using .env

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Make sure your .env has GOOGLE_API_KEY

# ----------------- Utility Functions ------------------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="chroma_db")
    vector_store.persist()
    return Chroma.from_texts(text_chunks, embedding=embeddings)

def get_conversational_chain():
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
    chain = load_qa_chain(llm=model, prompt=prompt, chain_type="stuff")
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question})
    st.write("Reply:", response["output_text"])  
    st.write(response)

# ----------------- Streamlit App ------------------

def main():
    st.set_page_config(page_title="Chat with PDF using Gemini")
    st.header("📄 Chat with PDF using Gemini ✨")

    user_question = st.text_input("Ask a question about your uploaded PDFs:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📚 Upload PDF Files")
        pdf_docs = st.file_uploader("Upload your PDF file(s)", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ PDFs processed and indexed!")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
