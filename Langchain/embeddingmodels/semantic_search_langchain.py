import os
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()


# ==============================
# 📚 DOCUMENTS
# ==============================

texts = [
    "I love machine learning and artificial intelligence",
    "Football is a popular sport worldwide",
    "Deep learning is a subset of machine learning",
    "Cooking requires patience and practice",
    "Natural language processing enables computers to understand text",
    "Basketball and football are outdoor sports",
    "AI is transforming the future of technology",
]

# Convert to LangChain Document format
documents=[Document(page_content=text) for text in texts ]


# ==============================
# 🔢 EMBEDDING MODEL
# ==============================
embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

# ==============================
# 🧠 CREATE VECTOR STORE (FAISS)
# ==============================

vectorstore=FAISS.from_documents(
    documents,
    embeddings
)

# Save locally

vectorstore.save_local("faiss_index")

print("Vector db is created and saved")


# ==============================
# 🔍 LOAD VECTOR STORE
# ==============================

vectorstore=FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# ==============================
# 🔎 SEARCH FUNCTION
# ==============================
def search(query,k=3):
    print("\n Query :{query}")

    results=vectorstore.similarity_search(query,k=k)

    print("\n Top Results")
    for i, doc in enumerate(results,1):
        print(f"{i}. {doc.page_content}")



# ==============================
# 🧪 MAIN LOOP
# ==============================

if __name__ == "__main__":
    while True:
        query = input("\nEnter query (or 'exit'): ")

        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        search(query)