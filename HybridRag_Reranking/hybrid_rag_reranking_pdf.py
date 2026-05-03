from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np
import os


# -----------------------------
# 🔹 1. LOAD PDF DOCUMENTS
# -----------------------------
# NEW CONCEPT: Document Loader
# Reads PDF and converts into raw text documents

PDF_PATH = r"D:\GenerativeAI\HybridRag_Reranking\Fruit_Maturity_Detection__Tomato___Dragon_Fruit_ (2).pdf"   # 🔥 change this
loader = PyPDFLoader(PDF_PATH)
raw_docs = loader.load()

# -----------------------------
# 🔹 2. TEXT SPLITTING
# -----------------------------
# NEW CONCEPT: Chunking
# Large documents → smaller chunks
# Improves retrieval accuracy

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = text_splitter.split_documents(raw_docs)


# -----------------------------
# 🔹 3. EMBEDDINGS + VECTOR DB
# -----------------------------
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



# NEW CONCEPT: Persistent Vector DB
# Stores embeddings for reuse

if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(documents, embedding)
    vectorstore.save_local("faiss_index")


# -----------------------------
# 🔹 4. BM25 SETUP (KEYWORD SEARCH)
# -----------------------------
# IMPORTANT: BM25 uses SAME chunks as vector DB

tokenized_corpus = [doc.page_content.split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)



# -----------------------------
# 🔹 5. RERANKER
# -----------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



# -----------------------------
# 🔹 6. LLM (MISTRAL)
# -----------------------------
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
)


parser = StrOutputParser()

# -----------------------------
# 🔹 7. MULTI-QUERY GENERATION
# -----------------------------
multi_query_prompt = ChatPromptTemplate.from_template("""
Generate 3 different search queries.
Return each query on a new line.

Question: {question}
""")

multi_query_chain = multi_query_prompt | llm | parser

def generate_queries(query):
    res = multi_query_chain.invoke({"question": query})
    return [q.strip() for q in res.split("\n") if q.strip()]



# -----------------------------
# 🔹 8. HYBRID RETRIEVAL
# -----------------------------
def hybrid_retrieval_fn(query):
    queries = generate_queries(query)
    all_docs = []

    for q in queries:
        # 🔸 Vector search
        vector_results = vectorstore.similarity_search(q, k=5)

        # 🔸 BM25 search
        scores = bm25.get_scores(q.split())
        top_indices = np.argsort(scores)[-5:]
        bm25_results = [documents[i] for i in top_indices]

        all_docs.extend(vector_results + bm25_results)

    # Deduplicate
    return list({doc.page_content: doc for doc in all_docs}.values())


hybrid_retriever = RunnableLambda(hybrid_retrieval_fn)


# -----------------------------
# 🔹 9. RERANKING
# -----------------------------
def rerank_fn(inputs):
    query = inputs["query"]
    docs = inputs["docs"]

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    # 🔥 Take best 3 chunks only
    return [doc for doc, _ in ranked[:3]]


# -----------------------------
# 🔹 10. FORMAT CONTEXT
# -----------------------------
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# -----------------------------
# 🔹 11. PROMPT
# -----------------------------
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from context. If not found, say 'I don't know'."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])


# -----------------------------
# 🔹 12. FULL RAG CHAIN
# -----------------------------
rag_chain = (
    {
        "docs": hybrid_retriever,
        "query": RunnablePassthrough()
    }
    | RunnableLambda(lambda x: {
        "query": x["query"],
        "docs": rerank_fn(x)
    })
    | RunnableLambda(lambda x: {
        "context": format_docs(x["docs"]),
        "question": x["query"]
    })
    | rag_prompt
    | llm
    | parser
)


# -----------------------------
# 🔥 TEST
# -----------------------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk: ")
        if query=="0":
          break
        print(rag_chain.invoke(query))
