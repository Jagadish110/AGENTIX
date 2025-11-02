# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from news import run_agent

app = FastAPI(title="Agentix Backend")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    user_id: str = "default_user"

@app.get("/")
def root():
    return {"message": "Agentix Backend is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_agent(request: QueryRequest):
    result = run_agent(request.question, request.user_id)
    return result

