# commands
# pip install streamlit langchain langchain-community langchain-google-genai google-generativeai chromadb pypdf python-dotenv
# streamlit run streamlit_app.py   OR   python -m streamlit run streamlit_app.py

import os
import streamlit as st
from dotenv import load_dotenv

# LangChain + loaders + Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================
# 1) Load API key from .env
# ==============================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found. Add it in .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ==============================
# 2) Streamlit page setup
# ==============================
st.set_page_config(page_title="AI-Powered HR Assistant (Gemini + HF Embeddings)", layout="wide")
st.title("🤖 AI-Powered HR Assistant")
st.caption("Upload an HR policy PDF and ask questions. Powered by Gemini for answers, HuggingFace for embeddings.")

# ==============================
# 3) Upload PDF
# ==============================
uploaded_file = st.file_uploader("📄 Upload HR Policy PDF", type=["pdf"])

if uploaded_file:
    pdf_path = "uploaded_hr_policy.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ==============================
    # 4) Load + Split PDF
    # ==============================
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # ==============================
    # 5) Embeddings (Hugging Face local) + Vector DB
    # ==============================
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)

    # ==============================
    # 6) Gemini LLM + Retrieval QA
    # ==============================
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # ==============================
    # 7) Q&A UI
    # ==============================
    query = st.text_input("💬 Ask your HR question:")
    if st.button("Get Answer") and query:
        with st.spinner("Thinking..."):
            try:
                answer = qa_chain.run(query)
                st.success(answer)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload a PDF to get started.")