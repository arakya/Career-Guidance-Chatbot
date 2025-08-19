import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Directories
DB_DIR = "career_vectordb"
DATA_DIR = "data"

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_index():
    """Build FAISS index from PDFs in data/"""
    docs = []
    if not os.path.exists(DATA_DIR):
        return None
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            docs.extend(loader.load())

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(texts, embeddings)
    os.makedirs(DB_DIR, exist_ok=True)
    vectordb.save_local(DB_DIR)
    return vectordb

def load_chain(k: int):
    """Load FAISS index, or rebuild if not found"""
    try:
        vectordb = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        vectordb = build_index()
        if vectordb is None:
            raise RuntimeError("No FAISS index and no PDFs in data/ to build from.")

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ.get("GOOGLE_API_KEY"))
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Career Guidance Chatbot", page_icon="🎓")

st.title("🎓 Career Guidance Chatbot")

k = st.sidebar.slider("Retriever depth (k)", 1, 10, 5)
if st.sidebar.button("Refresh index"):
    build_index()
    st.session_state.pop("chain", None)
    st.success("Index refreshed!")

if "chain" not in st.session_state:
    st.session_state["chain"] = load_chain(k)

query = st.text_input("Ask me anything about your career 👇")
if query:
    try:
        answer = st.session_state["chain"].run(query)
        st.markdown(f"**Answer:** {answer}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
