# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from news import run_agent

# -------------------- FastAPI app setup --------------------
app = FastAPI(title="Agentix Backend")

# Enable CORS for all origins (adjust in production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Request/Response model --------------------
class QueryRequest(BaseModel):
    question: str
    user_id: str = "default_user"

# -------------------- Routes --------------------
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

# -------------------- Run with Uvicorn --------------------
# You can run this with:
# uvicorn backend:app --reload
