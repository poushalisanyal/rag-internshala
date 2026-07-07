from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from retriever import Retriever


# ---------------------------------------
# Initialize FastAPI
# ---------------------------------------

app = FastAPI(
    title="RAG Question Answering API",
    description="Retrieval Augmented Generation API using FAISS and Gemini",
    version="1.0"
)


# ---------------------------------------
# Initialize Retriever
# ---------------------------------------

retriever = Retriever()


# ---------------------------------------
# Request Schema
# ---------------------------------------

class QueryRequest(BaseModel):

    question: str



# ---------------------------------------
# Health Check API
# ---------------------------------------

@app.get("/")
def home():

    return {
        "message": "RAG API is running successfully"
    }



# ---------------------------------------
# Ask Question API
# ---------------------------------------

@app.post("/ask")
def ask_question(request: QueryRequest):

    try:

        result = retriever.ask(
            request.question
        )

        return {

            "question": request.question,

            "answer": result["answer"],

            "sources": result["sources"],

            "latency": result["latency"],

            "token_usage": result["token_usage"]

        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )