from dotenv import load_dotenv
import os

# 🔹 LangChain Core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

# 🔹 LLM
from langchain_mistralai import ChatMistralAI

# 🔹 Memory
from langchain_community.memory import ConversationBufferWindowMemory


# 🔹 Pydantic
from pydantic import BaseModel, Field
from typing import List, Optional


load_dotenv()

# ==============================
# 🚀 1. LLM (Fundamentals)
# ==============================
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3  # controls randomness
)


# ==============================
# 🚀 2. Output Schema
# ==============================

class Movie(BaseModel):
    title:str =Field(description="movie name")
    release_year:Optional[int]
    genre:List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser =PydanticOutputParser(pydantic_object=Movie)

# ==============================
# 🚀 3. Memory (Window)
# ==============================
memory=ConversationBufferWindowMemory(
    k=3,#last 3 interaction
    return_message=True
)


# ==============================
# 🚀 4. Prompt Engineering
# ==============================
prompt=ChatPromptTemplate.from_messages([
    #System Prompt
    (
        "system","""You are an expert movie data extraction AI.
        
        Rules:
        - Always return valid JSON
- Follow schema strictly
- Missing values → null
- No extra explanation

{format_intructions}"""
    ),

    #few-shot example
     ("human", "Inception is a 2010 sci-fi film directed by Christopher Nolan."),
    ("ai", """{
"title": "Inception",
"release_year": 2010,
"genre": ["sci-fi"],
"director": "Christopher Nolan",
"cast": [],
"rating": null,
"summary": "A film about dreams."
}"""),

    # Memory placeholder
    MessagesPlaceholder(variable_name="history"),

    # Actual user input
    ("human", "{input}")
])


# ==============================
# 🚀 5. LCEL Chain
# ==============================
chain = prompt | model

# ==============================
# 🚀 6. Chat Loop
# ==============================

print("🎬 Movie Extractor Chat (type 'exit' to quit)\n")

while True:
    user_input=input("You : ")

    if user_input.lower()=="exit":
        break

    #Load memory

    history=memory.load_memory_variables({})["history"]

    #Prepare input
    response=chain.invoke({
        "input":user_input,
        "history":history,
        "format_instructions":parser.get_format_instructions()
    })

    print("\n--- RAW OUTPUT ---")
    print(response.content)


    try:
        structured=parser.parse(response.content)

        print("\n----STRUCTURED OUTPUT---")
        print(structured)

        #save to memory 
        memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )

    except Exception as e:
        print("\n❌ Parsing Error:", e)

