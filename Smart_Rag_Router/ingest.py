from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
load_dotenv()

DATA_PATH=r"D:\GenerativeAI\Rag_Project\data"
DB_PATH="chroma_db"

# -------------------------
# LOAD DOCUMENTS
# -------------------------
documents=[]

for file in os.listdir(DATA_PATH):
    file_path =os.path.join(DATA_PATH,file)

    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        continue

    documents.extend(loader.load())


print(f"Loaded {len(documents)} documents")

# CHUNKING
# -------------------------

splitter =RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

splits=splitter.split_documents(documents)

print(f"Created {len(splits)} chunks")

# -------------------------
# EMBEDDINGS (Gemini)
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------
# STORE IN CHROMA
# -------------------------
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    persist_directory=DB_PATH
)

vectorstore.persist()

print("✅ Data successfully stored in Chroma DB")