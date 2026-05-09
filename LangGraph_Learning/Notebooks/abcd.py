from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import os

load_dotenv()

# 1. Initialize Model with Retries
# This helps handle the 429 error by waiting and trying again automatically
llm = ChatMistralAI(
    model="mistral-small-latest",
    
)

class GraphSchema(TypedDict):
    name: str
    message: str

# 2. Refined Node Function
def welcome(state: GraphSchema):
    # Get values from state
    curr_name = state.get("name", "User")
    curr_message = state.get("message", "")

    # Invoke LLM
    response = llm.invoke(f"My name is {curr_name}. {curr_message}")

    # Return only the updates (LangGraph merges this into the state)
    return {
        "message": f"AI Response: {response.content}"
    }

# 3. Build Graph
workflow = StateGraph(GraphSchema)
workflow.add_node("welcome", welcome)
workflow.add_edge(START, "welcome")
workflow.add_edge("welcome", END)

# Compile
app = workflow.compile()

# 4. Run
try:
    result = app.invoke({
        "name": "Pravin",
        "message": "Explain LangGraph in one sentence."
    })
    print(f"Name: {result['name']}")
    print(f"Result: {result['message']}")
except Exception as e:
    print(f"An error occurred: {e}")