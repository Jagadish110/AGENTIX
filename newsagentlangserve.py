import streamlit as st
import os
import re
from dotenv import load_dotenv
from typing import TypedDict, Dict, Any
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import QdrantClient
from mem0 import Memory
import datetime

# Load API keys
load_dotenv()
os.environ["USER_AGENT"] = "Jagad-NewsAgent"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Warn if keys missing
if not GROQ_API_KEY:
    st.error("🚨 GROQ_API_KEY missing! Add to .env or Render env vars.")
if not TAVILY_API_KEY:
    st.warning("⚠️ TAVILY_API_KEY missing—news search will fail.")

# State structure
class ExpertState(TypedDict):
    question: str
    answer: str
    route: str
    source_tool: str
    final_summary: str
    mem0_user_id: str

# Qdrant initialize (with error handling)
try:
    qdrant_client = QdrantClient(
        url="https://dfd68a97-3764-4e1e-8b1e-d0ee4a395ba3.eu-west-2-0.aws.cloud.qdrant.io:6333",
        api_key=qdrant_api_key,
    )
except Exception as e:
    st.error(f"Qdrant init failed: {e}")
    qdrant_client = None

# LLM init (safe)
main_llm = None
if GROQ_API_KEY:
    main_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=GROQ_API_KEY,
        streaming=True
    )

# Mem0 config & init (safe)
mem0_memory = None
if main_llm and qdrant_client:
    mem0_config = {
        "llm": {
            "provider": "groq",
            "config": {"model": "llama-3.1-8b-instant", "temperature": 0.2, "max_tokens": 2000, "api_key": GROQ_API_KEY}
        },
        "embedder": {"provider": "huggingface", "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}},
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": "https://dfd68a97-3764-4e1e-8b1e-d0ee4a395ba3.eu-west-2-0.aws.cloud.qdrant.io:6333",
                "api_key": qdrant_api_key,
                "collection_name": "mem0_store",
                "embedding_model_dims": 768,
                "client": qdrant_client
            }
        }
    }
    try:
        mem0_memory = Memory.from_config(mem0_config)
    except Exception as e:
        st.error(f"Mem0 init failed: {e}")
else:
    st.warning("Mem0 skipped—missing LLM or Qdrant.")

# Retrieve relevant context from Mem0
def get_relevant_context(question: str, user_id: str) -> str:
    if not mem0_memory:
        return "Mem0 unavailable."
    try:
        past_results = mem0_memory.search(query=question, user_id=user_id, limit=2)
        if isinstance(past_results, dict):
            past_results = past_results.get('results', [])
        elif not isinstance(past_results, list):
            past_results = []
        if not past_results:
            return "No relevant context found."
        retrieved_texts = []
        for item in past_results:
            content = (item.get("content") or item.get("text") or item.get("memory") or str(item)).strip()
            if content:
                retrieved_texts.append(content)
        return "\n\n".join(retrieved_texts[:3]) if retrieved_texts else "No relevant context found."
    except Exception as e:
        print(f"Mem0 retrieval error: {e}")
        return "No relevant context due to retrieval error."

# Keywords (unchanged)
tavily_keywords = [
    "news", "latest news", "breaking news", "headlins", "updates", "current events",
    # ... (all your lists here - omitted for brevity)
]
youtube_keywords = [
    "youtube", "video", "watch video", "youtube link", "tutorial", "youtube tutorial",
    "video link", "watch on youtube"
]
code_keyword = [
    'python', 'javascript', 'java', 'c', 'c++', 'c#', 'PHP', 'R', 'SQL',
    'TypeScript', 'HTML', 'CSS', "Go", 'Rust', 'Swift', 'Ruby', 'Objective-C'
]

# Router (fixed code matching)
def expert_router(state: ExpertState) -> ExpertState:
    original_question = state.get("question", "")
    question = original_question.lower()
    if not question:
        return {**state, "route": "llm"}
    if "http" in question or "www" in question:
        return {**state, "route": "rag"}
    elif any(word in question for word in youtube_keywords):
        return {**state, "route": "youtube"}
    elif any(word in question for word in tavily_keywords):
        return {**state, "route": "tavily"}
    elif any(lang in question.split() for lang in code_keyword):  # Fixed: lang in split
        return {**state, "route": "code"}
    else:
        return {**state, "route": "llm"}

# Nodes (with fixes)
def codeagent(state: ExpertState) -> Dict[str, Any]:
    if not main_llm:
        return {"answer": "❌ LLM unavailable.", "source_tool": "Code"}
    try:
        question = state.get("question", "")
        user_id = state.get("mem0_user_id", "default_user")
        retrieved_context = get_relevant_context(question, user_id)
        system_prompt = '''
You are CodeGrok... (your full prompt - omitted for brevity)
User Query: {question}
Past context: {retrieved_context}
'''
        system_msg = SystemMessage(content=system_prompt.format(question=question, retrieved_context=retrieved_context))
        human_msg = HumanMessage(content="Provide the code solution based on the system instructions.")
        response = main_llm.invoke([system_msg, human_msg])
        ans = str(response.content).strip() or "No code output generated."
        if mem0_memory:
            mem0_memory.add(messages=[{"role": "user", "content": question}, {"role": "assistant", "content": ans}], user_id=user_id)
        return {"answer": ans, "source_tool": "Code"}
    except Exception as e:
        return {"answer": f"❌ Code Error: {str(e)}", "source_tool": "Code"}

# ... (Similar safe wraps for youtube_node, tavily_node, rag_node, llm_node, summarize_node, reflector_node - omitted for space; apply same pattern: check main_llm/TAVILY_API_KEY, use get_relevant_context if mem0)

# Example for tavily_node (abridged)
def tavily_node(state: ExpertState) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        return {"answer": "❌ Tavily API key missing.", "source_tool": "Tavily"}
    try:
        # ... (your code, with main_llm check)
        # Use main_llm.invoke only if main_llm
        if not main_llm:
            return {"answer": "❌ LLM unavailable for summary.", "source_tool": "Tavily"}
        # ... rest unchanged
    except Exception as e:
        return {"answer": f"❌ Tavily Error: {str(e)}", "source_tool": "Tavily"}

# (Apply to all nodes similarly)

# Graph build (unchanged, but safe)
graph = StateGraph(ExpertState)
# ... (all adds/edges same)
checkpointer = MemorySaver()
if "app" not in st.session_state:
    st.session_state.app = graph.compile(checkpointer=checkpointer)
app = st.session_state.app

# Streamlit UI (minor fixes)
st.set_page_config(page_title="News Agent", page_icon="📰", layout="wide")
# ... (CSS same)

# Header same

# Sidebar (fixed var: use prompt, but it's conditional)
with st.sidebar:
    st.title("Chat Options")
    if st.button("Clear Memory & New Chat"):
        current_user_id = st.session_state.get("thread_id", "default_user")
        if mem0_memory:
            try:
                all_memories = mem0_memory.get_all(user_id=current_user_id)
                for mem in all_memories:
                    if isinstance(mem, dict) and 'id' in mem:
                        mem0_memory.delete(mem['id'])
            except Exception as clear_e:
                print(f"Error clearing memory: {clear_e}")
        st.session_state.messages = []
        st.session_state.thread_id = f"user_session_{int(datetime.datetime.now().timestamp())}"
        st.session_state.last_history = []
        st.rerun()

# Chat history init same

# Display messages same

# Input & run (fixed Mem0 add: use 'prompt' not 'prompt' - was already correct, but ensured)
if prompt := st.chat_input("💬 Ask about latest news, a URL, or anything else..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking... ⚙️"):
            thread = {"configurable": {"thread_id": st.session_state.thread_id}}
            input_state = {
                "question": prompt,
                "mem0_user_id": st.session_state.thread_id,
                "answer": "",
                "final_summary": "",
                "route": "",
                "source_tool": ""
            }
            try:
                events = list(app.stream(input_state, config=thread, stream_mode="values"))
                final_state = events[-1] if events else None
                if not isinstance(final_state, dict):
                    final_state = {"answer": "❌ Invalid state received from graph.", "source_tool": "Error"}
            except Exception as e:
                final_state = {"answer": f"❌ Agent Error: {str(e)}", "source_tool": "Error"}
            response = final_state.get("final_summary", final_state.get("answer", "❌ No answer found.")).strip()
            source_tool = final_state.get("source_tool", "Unknown")
            st.markdown(f"**🛠 Source Tool:** {source_tool}")
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": f"[{source_tool}] {response}"})
            st.session_state.last_history.append({"question": prompt, "answer": response})
            # Mem0 persist (safe)
            if mem0_memory:
                try:
                    mem0_memory.add(
                        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                        user_id=st.session_state.thread_id
                    )
                except Exception as mem_e:
                    print(f"Mem0 add error: {mem_e}")
