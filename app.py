import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------
# Settings
# ----------------------------
DB_DIR = "career_vectordb"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="Career Guidance Chatbot", layout="wide")
st.title("📚 Career Guidance Chatbot")

# ----------------------------
# Load Vector DB + Chain
# ----------------------------
@st.cache_resource
def load_chain(k: int = 4):
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectordb = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "score_threshold": 0.1}
    )

    # Gemini 1.5 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain

# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Retriever depth (k)", min_value=2, max_value=10, value=5, step=1)

    if st.button("🔄 Refresh index"):
        st.cache_resource.clear()
        st.success("Index cache cleared. Ask again!")

    st.markdown("---")
    st.markdown("**Tips:**\n- Increase retriever depth if answers seem incomplete.\n- Use Refresh if you’ve added new PDFs.")

# ----------------------------
# Session State
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "chain" not in st.session_state or st.session_state.get("last_k") != k:
    st.session_state["chain"] = load_chain(k)
    st.session_state["last_k"] = k

# ----------------------------
# Chat Interface
# ----------------------------
query = st.chat_input("Ask me anything about your documents...")
if query:
    chain = st.session_state["chain"]
    result = chain({"question": query, "chat_history": st.session_state["chat_history"]})
    st.session_state["chat_history"].append((query, result["answer"], result.get("source_documents", [])))

# ----------------------------
# Display Conversation
# ----------------------------
for i, (q, a, sources) in enumerate(st.session_state["chat_history"]):
    st.markdown(f"### 🧑 You\n{q}")
    st.markdown(f"### 🤖 Bot\n{a}")

    if sources:
        with st.expander(f"📂 Sources for Q{i+1}"):
            for doc in sources:
                meta = doc.metadata
                src = meta.get("source") or meta.get("file_path") or "Unknown"
                st.markdown(f"**{os.path.basename(src)}**")
                st.text(doc.page_content[:250] + "...")
                with st.expander("Show full chunk"):
                    st.text(doc.page_content)
    st.markdown("---")
