# 🎓 Career Guidance Chatbot

A chatbot that helps students explore and prepare for suitable career paths using Retrieval-Augmented Generation (RAG).  
It provides personalized responses and career suggestions using advanced NLP models (Google Gemini 1.5 Flash) and vector search (FAISS).

---

## 🚀 Features
- Ingests PDF documents (career guides, resumes, interview prep, etc.) and creates a searchable knowledge base.  
- Uses **FAISS** for fast similarity search over document chunks.  
- Powered by **Google Gemini 1.5 Flash** via LangChain.  
- Streamlit app with an intuitive chat interface.  
- Displays sources for each answer (with expanders to inspect full chunks).  
- Easily deployable on **Streamlit Cloud**.

---

## 📂 Project Structure
```
cleaned_repo/
│── app.py                   # Streamlit chatbot app
│── ingest.py                # Script to ingest PDFs into FAISS
│── loader.ipynb             # Jupyter notebook for testing ingestion logic
│── query.ipynb              # Notebook for testing queries
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
│── data/                    # Place your PDFs here
│── .streamlit/
│   └── secrets.toml.example # Template for API keys
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/arakya/Career-Guidance-Chatbot.git
cd Career-Guidance-Chatbot
```

### 2. Create a virtual environment & install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate   # On macOS/Linux
.venv\Scripts\activate    # On Windows

pip install -r requirements.txt
```

### 3. Add your API key
Create a `.streamlit/secrets.toml` file with:
```toml
GOOGLE_API_KEY = "your_api_key_here"
```

---

## 📥 Ingest Documents
Put your PDFs inside the `data/` folder, then run:
```bash
python ingest.py
```
This will create/update the FAISS index in `career_vectordb/`.

---

## 💬 Run the Chatbot
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub.  
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) → New App.  
3. Select your repo and set `app.py` as the entrypoint.  
4. In **App Settings → Secrets**, add your key:
   ```toml
   GOOGLE_API_KEY="your_api_key_here"
   ```
5. Deploy! Your chatbot will be live.

---

## 🧠 How It Works
1. **Ingestion**: PDFs → text chunks → embeddings → FAISS index.  
2. **Retrieval**: User query → similarity search → top-k chunks returned.  
3. **Generation**: Gemini 1.5 Flash answers using retrieved context.  
4. **Response**: Streamlit displays the answer + sources.

---

## 📜 License
MIT License. Free to use and modify.
