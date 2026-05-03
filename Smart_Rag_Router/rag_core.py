# rag_core.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Setup
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever()

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
)

chat_history = []

# Prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Use context if available, else general knowledge."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

router_prompt = ChatPromptTemplate.from_template("""
Decide:
RAG or GENERAL

Question: {question}
""")

# Router
def route_query(query):
    res = llm.invoke(router_prompt.invoke({"question": query}))
    return "RAG" if "RAG" in res.content.upper() else "GENERAL"

# RAG
def rag_pipeline(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    res = llm.invoke(
        rag_prompt.invoke({"context": context, "question": query})
    )
    return res.content

# General
def general_pipeline(query):
    return llm.invoke(query).content

# MAIN FUNCTION
def process_query(query):
    decision = route_query(query)

    if decision == "RAG":
        answer = rag_pipeline(query)
    else:
        answer = general_pipeline(query)

    return answer, decision