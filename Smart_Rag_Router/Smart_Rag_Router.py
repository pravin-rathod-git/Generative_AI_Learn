import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler


load_dotenv()


# -------------------------
# EMBEDDINGS (LOCAL)
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# -------------------------
# VECTOR STORE
# -------------------------
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# -------------------------
# LLM (MISTRAL + STREAMING)
# -------------------------
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)


# -------------------------
# MEMORY
# -------------------------
chat_history = []

# -------------------------
# RAG PROMPT
# -------------------------
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful AI assistant.

If the answer is found in the context, use it.
If not, you may answer using general knowledge.

Always prefer context when available.
"""),
        ("human", """Chat History:
{history}

Context:
{context}

Question:
{question}
""")
    ]
)

# -------------------------
# ROUTER PROMPT
# -------------------------
router_prompt = ChatPromptTemplate.from_template("""
You are a routing assistant.

Decide whether the user question requires:
- "RAG" → if it depends on documents
- "GENERAL" → if it is general knowledge or casual

Only respond with ONE word: RAG or GENERAL

Question: {question}
""")


# -------------------------
# ROUTER FUNCTION
# -------------------------
def route_query(query):
    response = llm.invoke(
        router_prompt.invoke({"question": query})
    )
    decision = response.content.strip().upper()

    if "RAG" in decision:
        return "RAG"
    return "GENERAL"


# -------------------------
# RAG PIPELINE
# -------------------------
def rag_pipeline(query):
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    history_text = "\n".join(chat_history)

    final_prompt = rag_prompt.invoke({
        "history": history_text,
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)
    return response.content



# -------------------------
# GENERAL PIPELINE
# -------------------------
def general_pipeline(query):
    response = llm.invoke(query)
    return response.content



# -------------------------
# MAIN LOOP
# -------------------------
print("🚀 Smart RAG System Ready (Mistral)")
print("Type '0' to exit")

while True:
    query = input("\nYou: ")

    if query == "0":
        break

    decision = route_query(query)

    print(f"\n[Router Decision]: {decision}")
    print("\nAI:", end=" ")

    if decision == "RAG":
        answer = rag_pipeline(query)
    else:
        answer = general_pipeline(query)

    # Save memory
    chat_history.append(f"You: {query}")
    chat_history.append(f"AI: {answer}")