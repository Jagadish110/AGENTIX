# main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from news import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from news import run_agent
    logger.info("✅ Successfully imported run_agent")
except Exception as e:
    logger.error(f"❌ Failed to import run_agent: {str(e)}")
    raise


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
def ask_agent_endpoint(request: QueryRequest):
    return run_agent(request.question, request.user_id)
