# backend.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from rag_core import process_query

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Backend running"}


@app.post("/chat")
def chat(req: QueryRequest):
    answer, route = process_query(req.query)

    return {
        "answer": answer,
        "route": route
    }
