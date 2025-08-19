import os
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, PDFPlumberLoader, PDFMinerLoader

# Paths
DATA_DIR = "data"
DB_DIR = "career_vectordb"
META_FILE = "vector_db_metadata.json"

# Load / init metadata tracker
if os.path.exists(META_FILE):
    with open(META_FILE, "r") as f:
        processed_files = set(json.load(f))
else:
    processed_files = set()

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing FAISS if any
if os.path.exists(DB_DIR):
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    db = None

# Chunker
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

def load_pdf(file_path):
    """Try multiple loaders until one works."""
    loaders = [PyMuPDFLoader, PDFPlumberLoader, PDFMinerLoader]
    for loader_cls in loaders:
        try:
            return loader_cls(file_path).load()
        except Exception:
            continue
    return None

new_files = []
for file in os.listdir(DATA_DIR):
    if not file.endswith(".pdf"):
        continue
    if file in processed_files:
        continue

    path = os.path.join(DATA_DIR, file)
    docs = load_pdf(path)

    if docs is None:
        print(f"[FAILED] {file}")
        continue

    print(f"[OK] {file} → {len(docs)} pages")
    # Split into chunks
    chunks = splitter.split_documents(docs)

    if db is None:
        db = FAISS.from_documents(chunks, embeddings)
    else:
        db.add_documents(chunks)

    processed_files.add(file)
    new_files.append(file)

# Save FAISS
if db is not None:
    db.save_local(DB_DIR)

# Update metadata
with open(META_FILE, "w") as f:
    json.dump(list(processed_files), f, indent=2)

print("\nIngestion complete.")
if new_files:
    print("Newly added files:", new_files)
else:
    print("No new files found.")
