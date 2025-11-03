import os, re
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Annotated
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Changed to uppercase
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# State structure
class ExpertState(TypedDict):
    question: str
    answer: str
    route: str
    source_tool: str
    final_summary: str
    mem0_user_id: str

# Qdrant initialize
qdrant_client = QdrantClient(
    url="https://dfd68a97-3764-4e1e-8b1e-d0ee4a395ba3.eu-west-2-0.aws.cloud.qdrant.io:6333",
    api_key=qdrant_api_key,
)

# Main LLM
main_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    groq_api_key=GROQ_API_KEY,
    streaming=True
)

# mem0 configuration
mmem0_config = {
    "llm": {
        "provider": "groq",
        "config": {
            "model": "llama-3.1-8b-instant",
            "temperature": 0.2,
            "max_tokens": 2000,
            "api_key": GROQ_API_KEY
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
    },
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
mem0_memory = Memory.from_config(mmem0_config)

def get_relevant_context(question, user_id):
    """Retrieve and combine relevant past memory for a user."""
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
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or item.get("memory") or str(item)
            else:
                content = str(item)
            if content.strip():
                retrieved_texts.append(content.strip())
        
        if not retrieved_texts:
            return "No relevant context found."
        
        return "\n\n".join(retrieved_texts[:3])
    
    except Exception as e:
        print(f"Mem0 retrieval error: {e}")
        return "No relevant context due to retrieval error."

tavily_keywords = [
    "news", "latest news", "breaking news", "headlines", "updates", "current events",
    "recent news", "today's news", "daily news", "trending news", "startup news",
    "tech news", "AI news", "cryptocurrency news", "crypto news", "stock market news",
    "sports news", "political news", "entertainment news", "announcement", "press release",
    "today", "yesterday", "this week", "trending", "viral news"
]

youtube_keywords = [
    "youtube", "video", "watch video", "youtube link", "tutorial", "youtube tutorial"
]

code_keyword = [
    'python', 'javascript', 'java', 'c++', 'c#', 'PHP', 'SQL', 'TypeScript', 
    'HTML', 'CSS', 'Go', 'Rust', 'Swift', 'Ruby'
]

def expert_router(state: ExpertState) -> ExpertState:
    original_question = state.get("question", "")
    question = original_question.lower()
    if not question:
        return {**state, "route": "llm"}
    
    if "http" in question or "www" in question:
        route_label = "rag"
    elif any(word in question for word in youtube_keywords):
        route_label = "youtube"
    elif any(word in question for word in tavily_keywords):
        route_label = "tavily"
    elif any(word in question.split() for word in code_keyword):
        route_label = 'code'
    else:
        route_label = "llm"
    return {**state, "route": route_label}

def codeagent(state: ExpertState) -> Dict:
    try:
        question = state.get("question", "")
        user_id = state.get("mem0_user_id", "default_user")
        retrieved_context = get_relevant_context(question, user_id)
        
        system_prompt = '''You are CodeGrok, a senior software engineer. Provide clean, efficient code solutions.

User Query: {question}
Past context: {retrieved_context}'''
        
        system_msg = SystemMessage(content=system_prompt.format(
            question=question, 
            retrieved_context=retrieved_context
        ))
        human_msg = HumanMessage(content="Provide the code solution.")
        
        response = main_llm.invoke([system_msg, human_msg])
        ans = str(response.content).strip() or "No code output generated."
        
        mem0_memory.add(
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": ans}
            ],
            user_id=user_id
        )
        return {"answer": ans, "source_tool": "Code"}
    except Exception as e:
        return {"answer": f"❌ Code Error: {str(e)}", "source_tool": "Code"}

def youtube_node(state: ExpertState) -> Dict:
    try:
        question = state.get("question", "")
        user_id = state.get("mem0_user_id", "default_user")
        
        tavi = TavilySearch(max_results=10, tavily_api_key=TAVILY_API_KEY)
        search_results = tavi.invoke(question)
        
        results = search_results.get("results", []) if isinstance(search_results, dict) else []
        yt_links = [
            f"[{r.get('title', 'No Title')}]({r.get('url', '')})"
            for r in results
            if isinstance(r, dict) and "youtube.com" in r.get("url", "")
        ]
        ans = "\n\n".join(yt_links) if yt_links else "No relevant YouTube videos found."
        
        mem0_memory.add(
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": ans}
            ],
            user_id=user_id
        )
        return {"answer": ans, "source_tool": "YouTube"}
    except Exception as e:
        return {"answer": f"❌ YouTube Error: {str(e)}", "source_tool": "YouTube"}

def tavily_node(state: ExpertState) -> Dict:
    try:
        question = state.get("question", "")
        user_id = state.get("mem0_user_id", "default_user")
        retrieved_context = get_relevant_context(question, user_id)
        
        tavi = TavilySearch(max_results=10, tavily_api_key=TAVILY_API_KEY)
        search_results = tavi.invoke(question)
        results = search_results.get("results", []) if isinstance(search_results, dict) else []
        
        formatted_links = []
        for i, r in enumerate(results[:5], 1):
            if isinstance(r, dict):
                title = r.get("title", "No Title")
                url = r.get("url", "")
                snippet = r.get("content", "")[:150]
                formatted_links.append(f"{i}. [{title}]({url})\n\n> {snippet}...\n")
        
        if not formatted_links:
            formatted_links = ["No news sources found."]
        
        raw_summary = "\n".join(formatted_links)
        
        system_msg = SystemMessage(
            content="You are a news summarizer. Create a concise summary with citations [1], [2], etc."
        )
        human_msg = HumanMessage(
            content=f"Past context:\n{retrieved_context}\n\nNews results:\n{raw_summary}\n\nUser query: {question}"
        )
        
        summary_response = main_llm.invoke([system_msg, human_msg])
        summary_text = summary_response.content.strip() or "No summary generated."
        
        final_answer = f"**🧭 Summary:**\n{summary_text}\n\n**📚 Sources:**\n{raw_summary}"
        
        mem0_memory.add(
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": final_answer}
            ],
            user_id=user_id
        )
        
        return {"answer": final_answer, "source_tool": "Tavily"}
    
    except Exception as e:
        return {"answer": f"❌ Tavily Error: {str(e)}", "source_tool": "Tavily"}

def rag_node(state: ExpertState) -> Dict:
    try:
        question = state.get("question", "")
        user_id = state.get("mem0_user_id", "default_user")
        retrieved_context = get_relevant_context(question, user_id)
        
        url_match = re.search(r'https?://(?:www\.)?[^ \t\n\r\f\v]+', question)
        if not url_match:
            ans = "No valid URL found. Please provide a full URL for analysis."
            mem0_memory.add(
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ans}
                ],
                user_id=user_id
            )
            return {"answer": ans, "source_tool": "RAG"}
        
        url = url_match.group(0)
        loader = WebBaseLoader(url)
        docs_raw = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = splitter.split_documents(docs_raw)
        limited_docs = docs[:10]
        
        # Create context from documents
        context_text = "\n\n".join([doc.page_content for doc in limited_docs])
        
        # Simple prompt without complex chains
        system_msg = SystemMessage(
            content="Use the context to answer. If you don't know, say so. Don't make up answers."
        )
        human_msg = HumanMessage(
            content=f"Memory context: {retrieved_context}\n\nWeb content: {context_text}\n\nQuestion: {question}"
        )
        
        response = main_llm.invoke([system_msg, human_msg])
        ans = str(response.content).strip() or "Summary could not be generated."
        
        mem0_memory.add(
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": ans}
            ],
            user_id=user_id
        )
        return {"answer": ans, "source_tool": "RAG"}
    except Exception as e:
        return {"answer": f"❌ RAG Error: {str(e)}", "source_tool": "RAG"}

def llm_node(state: ExpertState) -> Dict:
    try:
        question = state.get("question", "")
        user_id = state.get("mem0_user_id", "default_user")
        retrieved_context = get_relevant_context(question, user_id)
        
        system_msg = SystemMessage(
            content="You are a helpful AI assistant. Use context only if directly relevant."
        )
        human_msg = HumanMessage(
            content=f"Context: {retrieved_context}\n\nQuestion: {question}"
        )
        
        response = main_llm.invoke([system_msg, human_msg])
        ans = str(response.content).strip() or "No answer generated."
        
        existing = mem0_memory.search(query=question, user_id=user_id, limit=1)
        if isinstance(existing, dict):
            existing = existing.get('results', [])
        if not existing:
            mem0_memory.add(
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ans}
                ],
                user_id=user_id
            )
        
        return {"answer": ans, "source_tool": "AI"}
    except Exception as e:
        return {"answer": f"❌ LLM Error: {str(e)}", "source_tool": "AI"}

def summarize_node(state: ExpertState) -> Dict:
    try:
        answer_text = state.get("answer", "").strip()
        question = state.get("question", "")
        if not answer_text:
            return {"final_summary": "No content to summarize.", "source_tool": "Summarizer"}
        
        system_msg = SystemMessage(
            content="Create a clear, concise summary retaining key points."
        )
        human_msg = HumanMessage(
            content=f"Question: {question}\n\nContent: {answer_text}"
        )
        
        summary = main_llm.invoke([system_msg, human_msg])
        final_summary = str(summary.content).strip()
        return {"final_summary": final_summary, "source_tool": "Summarizer"}
    except Exception as e:
        return {"final_summary": f"❌ Summarizer Error: {str(e)}", "source_tool": "Summarizer"}

def reflector_node(state: ExpertState) -> Dict:
    try:
        current_summary = state.get("final_summary", "")
        question = state.get("question", "")
        user_id = state.get("mem0_user_id", "default_user")
        if not current_summary:
            return {"final_summary": "❌ No content to reflect on.", "source_tool": "Self-Improver"}
        
        system_msg = SystemMessage(
            content="Review and improve the summary for accuracy and clarity."
        )
        human_msg = HumanMessage(
            content=f"Question: {question}\n\nSummary: {current_summary}\n\nProvide improved version."
        )
        
        reflection = main_llm.invoke([system_msg, human_msg])
        improved_summary = str(reflection.content).strip()
        
        mem0_memory.add(
            messages=[
                {"role": "user", "content": f"Improve: {question}"},
                {"role": "assistant", "content": improved_summary}
            ],
            user_id=user_id
        )
        return {"final_summary": improved_summary, "source_tool": "Self-Improver"}
    except Exception as e:
        return {"final_summary": f"❌ Reflector Error: {str(e)}", "source_tool": "Self-Improver"}

# Build the graph
graph = StateGraph(ExpertState)
graph.add_node("Router", expert_router)
graph.add_node("Tavily", tavily_node)
graph.add_node("RAG", rag_node)
graph.add_node("LLM", llm_node)
graph.add_node("Summarizer", summarize_node)
graph.add_node('Reflector', reflector_node)
graph.add_node("YouTube", youtube_node)
graph.add_node('Procoder', codeagent)

graph.set_entry_point("Router")
graph.add_conditional_edges(
    "Router",
    lambda s: s["route"],
    {
        "tavily": "Tavily",
        "llm": "LLM",
        "rag": "RAG",
        "youtube": "YouTube",
        'code': 'Procoder'
    }
)
graph.add_edge("Tavily", END)
graph.add_edge("LLM", END)
graph.add_edge("Procoder", END)
graph.add_edge("RAG", "Summarizer")
graph.add_edge("YouTube", END)
graph.add_edge("Summarizer", "Reflector")
graph.add_edge('Reflector', END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

def run_agent(question: str, user_id: str = "default_user"):
    input_state = {
        "question": question,
        "mem0_user_id": user_id,
        "answer": "",
        "final_summary": "",
        "route": "",
        "source_tool": ""
    }
    
    config = {"configurable": {"thread_id": f"thread_{user_id}"}}
    
    final_state = input_state.copy()
    for chunk in app.stream(input_state, config):
        for node_output in chunk.values():
            if isinstance(node_output, dict):
                final_state.update(node_output)
    
    response = final_state.get("final_summary") or final_state.get("answer", "")
    source = final_state.get("source_tool", "unknown")
    
    return {"response": response.strip(), "source_tool": source}
