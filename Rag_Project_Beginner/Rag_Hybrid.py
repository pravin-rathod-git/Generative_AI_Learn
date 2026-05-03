import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler

load_dotenv()

# -------------------------
# EMBEDDINGS
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
# PROMPT
# -------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful AI assistant.
Use ONLY the provided context.
If not found, say "I could not find the answer."
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

print("✅ RAG system created (Mistral + Streaming + Memory)")
print("Press 0 to exit")

# -------------------------
# QUERY LOOP
# -------------------------
while True:
    query = input("\nYou: ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    history_text = "\n".join(chat_history)

    final_prompt = prompt.invoke({
        "history": history_text,
        "context": context,
        "question": query
    })

    print("\nAI:", end=" ")

    response = llm.invoke(final_prompt)

    answer = response.content

    # Save memory
    chat_history.append(f"You: {query}")
    chat_history.append(f"AI: {answer}")