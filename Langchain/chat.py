import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

# Load env
load_dotenv()

# Initialize model
llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

print("Chat with Mistral (LangChain) - type 'exit' to quit\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = llm.invoke([
        HumanMessage(content=user_input)
    ])

    print("Bot:", response.content, "\n")