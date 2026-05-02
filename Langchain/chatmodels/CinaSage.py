from dotenv import load_dotenv
import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field
from typing import List, Optional

# 🔹 Load env
load_dotenv()

# 🔹 Model
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)

# 🔹 Define Schema
class Movie(BaseModel):
    title: str = Field(description="Name of the movie")
    release_year: Optional[int] = Field(description="Year of release")
    genre: List[str] = Field(description="List of genres")
    director: Optional[str] = Field(description="Director name")
    cast: List[str] = Field(description="List of actors")
    rating: Optional[float] = Field(description="Movie rating")
    summary: str = Field(description="Short summary")

# 🔹 Parser
parser = PydanticOutputParser(pydantic_object=Movie)

# 🔹 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert AI that extracts structured movie data.

Return ONLY valid JSON.

{format_instructions}
"""),
    ("human", "{input}")
])

# 🔹 Input
user_input = input("Enter movie paragraph: ")

# 🔹 Create final prompt
final_prompt = prompt.invoke({
    "input": user_input,
    "format_instructions": parser.get_format_instructions()
})

# 🔹 Call model
response = model.invoke(final_prompt)

print("\n--- RAW OUTPUT ---\n")
print(response.content)

# 🔹 Parse output
try:
    result = parser.parse(response.content)

    print("\n--- STRUCTURED OUTPUT ---\n")
    print(result)

except Exception as e:
    print("\n❌ Parsing Failed:", e)

