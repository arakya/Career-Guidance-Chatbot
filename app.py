import streamlit as st
import os
import difflib
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("Career Advisor Chatbot")

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("career_vectordb", embedding_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        max_tokens=512,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful career advisor. Use this context to answer: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

qa_chain = load_chain()

def is_similar(snippet, seen, threshold=0.97):
    for s in seen:
        if difflib.SequenceMatcher(None, snippet, s).ratio() > threshold:
            return True
    return False

greetings = {"hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"}

question = st.text_input("Ask a career question:")

if st.button("Ask") and question:
    q_lower = question.lower().strip()
    if q_lower in greetings:
        st.markdown("Hello! I am your career guidance chatbot. How may I help you today?")
    else:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": question})
            answer = response["result"]

            fallback_phrases = [
                "i can't answer", "without knowing", "provide me with", "content of page", "page 14"
            ]
            if any(phrase in answer.lower() for phrase in fallback_phrases):
                st.markdown("I'm here to help with career guidance. Please ask me about careers, internships, exams, or professional development.")
            else:
                st.markdown("**Answer:**")
                st.write(answer)

            st.markdown("---")
            st.markdown("**Sources:**")

            seen_snippets = []
            source_count = 1
            for doc in response["source_documents"]:
                snippet = doc.page_content[:200].strip()
                if not is_similar(snippet, seen_snippets):
                    st.write(f"Source {source_count} snippet: {snippet}...")
                    seen_snippets.append(snippet)
                    source_count += 1
                if source_count > 2:
                    break
