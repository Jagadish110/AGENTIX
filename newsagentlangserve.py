import streamlit as st
import os, re
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Annotated
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, AIMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain.callbacks.base import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.memory import MemorySaver
from letta_client import Letta
from langchain_community.chat_models import ChatOllama
from qdrant_client import QdrantClient
from mem0 import Memory
import datetime
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
# Gemini 2.5 Flash (commented if not used)

# Load API keys
load_dotenv()
os.environ["USER_AGENT"] = "Jagad-NewsAgent"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", os.getenv("groq_api_key", "")).strip()  # Unified key handling
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# State structure with short-term memory (history)
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

# Access safely
main_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    groq_api_key=GROQ_API_KEY,
    streaming=True
)

# Mem0 configuration for long term memory
mmem0_config = {
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
mem0_memory = Memory.from_config(mmem0_config)

# Retrieve relevant context from Mem0 before answering
def get_relevant_context(question, user_id):
    """Retrieve and combine relevant past memory for a user."""
    try:
        past_results = mem0_memory.search(query=question, user_id=user_id, limit=2)

        # Normalize return format: Mem0 search may return dict {'results': [...]} or direct list
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
            if content.strip():  # Avoid empty strings
                retrieved_texts.append(content.strip())

        if not retrieved_texts:
            return "No relevant context found."

        return "\n\n".join(retrieved_texts[:3])  # Limit to top 3 for brevity

    except Exception as e:
        print(f"Mem0 retrieval error: {e}")
        return "No relevant context due to retrieval error."

tavily_keywords = [
    # General news
    "news", "latest news", "breaking news", "headlins", "updates", "current events",
    "recent news", "today's news", "daily news", "news feed", "trending news",
    
    # Specific topics / industries
    "startup news", "tech news", "AI news", "artificial intelligence news",
    "cryptocurrency news", "crypto news", "stock market news", "finance news",
    "sports news", "political news", "entertainment news", "movie updates",
    "cinema news", "gaming news", "technology news", "business news", "economy news",
    
    # Event-based
    "announcement", "press release", "launch", "update", "report", "statement",
    "alert", "bulletin", "breaking", "urgent news", "exclusive", "developing news",
    
    # Geographical / regional
    "USA news", "India news", "Japan news", "local news", "regional updates",
    "global news", "world news", "international news", "Europe news", "Asia news",
    
    # Time-based
    "today", "yesterday", "this week", "last 24 hours", "latest update",
    "recently", "today's headlines", "daily update", "breaking today",
    
    # Trend / social media driven
    "trending", "viral news", "hot topic", "buzz", "top stories", "must-know news",
    "popular news", "breaking trends", "social buzz", "current trends",
    
    # Example phrases users ask
    "show me the latest news", "what's new today", "recent updates", 
    "top news today", "today's breaking news", "latest headlines", 
    "recent stories", "news updates", "current news events", "latest info",
    
    # Tech/Startup focused
    "funding news", "startup funding", "product launch news", "technology updates",
    "AI development", "machine learning news", "tech trends", "innovation news",
    
    # Finance / Crypto focused
    "bitcoin news", "Ethereum news", "crypto updates", "stock updates", 
    "market news", "economic updates", "financial headlines",
    
    # Sports / Entertainment
    "football news", "cricket news", "soccer news", "basketball news",
    "movies news", "hollywood news", "celebrity news", "entertainment updates"
]

youtube_keywords = [
    "youtube", "video", "watch video", "youtube link", "tutorial", "youtube tutorial", 
    "video link", "watch on youtube"
]
code_keyword = [
    'python','javascript','java','c','c++','c#','PHP','R','SQL',
    'TypeScript','HTML','CSS',"Go",'Rust','Swift','Ruby','Objective-C'
]

# Router function (initialize history if not present)
def expert_router(state: ExpertState) -> ExpertState:
    original_question = state.get("question", "")
    question = original_question.lower()
    if not question:
        return {**state, "route": "llm"}

    # URLs → RAG
    if "http" in question or "www" in question:
        route_label = "rag"
    # YouTube keywords
    elif any(word in question for word in youtube_keywords):
        route_label = "youtube"
    # News/Tavily keywords
    elif any(word in question for word in tavily_keywords):
        route_label = "tavily"
    # Programming languages → Code
    elif any(word in question.split() for word in code_keyword):  # split ensures whole-word match
        route_label = 'code'
    else:
        route_label = "llm"
    return {**state, "route": route_label}

# code execution
def codeagent(state: ExpertState) -> Dict:
    try:
        question = state.get("question","")
        user_id = state.get("mem0_user_id","default_user")

        # 🧠 Retrieve context
        retrieved_context = get_relevant_context(question, user_id)
        system_prompt = '''
You are CodeGrok, a senior software engineer with 15+ years of experience in all types of programming languages and system design. You specialize in clean, efficient, and maintainable code. Your goal is to assist users by generating, debugging, refactoring, or explaining code based on their requests.

Follow these rules strictly:
1. **Understand the Task**: Analyze the user's query, specified language, and any constraints (e.g., libraries, performance needs).
2. **Reason Step-by-Step**: Break down the problem: (a) Restate the goal, (b) Outline the approach, (c) Identify potential edge cases or errors, (d) Write the code.
3. **Write High-Quality Code**: Use best practices (e.g., PEP 8 for Python, meaningful variable names, comments for complex logic). Avoid unnecessary complexity.
4. **Test the Code**: Include a simple test case or example usage to verify it works.
5. **Handle Errors**: If the request is ambiguous, ask for clarification. If impossible, explain why and suggest alternatives.
6. **Output Format**: Always structure your response as:
   - **Explanation**: Brief overview of the solution.
   - **Code**: Full, executable code in a fenced block (e.g., ```python).
   - **Test/Usage**: How to run it and expected output.
   - **Improvements**: Optional suggestions for optimization or extensions.

User Query: {question}
Past context: {retrieved_context}
'''
        system_msg = SystemMessage(content=system_prompt.format(question=question, retrieved_context=retrieved_context))
        human_msg = HumanMessage(content="Provide the code solution based on the system instructions.")
        
        response = main_llm.invoke([system_msg, human_msg])
        ans = str(response.content).strip() or "No code output generated."

        # Always add to memory, Mem0 handles relevance
        mem0_memory.add(messages=[{"role":"user","content":question},{"role":"assistant","content":ans}], user_id=user_id)
        return {"answer": ans, "source_tool":"Code"}
    except Exception as e:
        error_msg = f"❌ Code Error: {str(e)}"
        return {"answer": error_msg, "source_tool":"Code"}

# YouTube node (update history)
def youtube_node(state: ExpertState) -> Dict:
    try:
        question = state.get("question","")
        user_id = state.get("mem0_user_id","default_user")

        # Search without context to keep query clean
        tavi = TavilySearch(max_results=10, tavily_api_key=TAVILY_API_KEY)
        search_results = tavi.invoke(question)

        results = search_results.get("results",[]) if isinstance(search_results, dict) else []
        yt_links = [f"[{r.get('title','No Title')}] ({r.get('url','')})" for r in results if isinstance(r, dict) and "youtube.com" in r.get("url","")]
        ans = "\n\n".join(yt_links) if yt_links else "No relevant YouTube videos found for your query."

        # Add to memory
        mem0_memory.add(messages=[{"role":"user","content":question},{"role":"assistant","content":ans}], user_id=user_id)
        return {"answer": ans, "source_tool":"YouTube"}
    except Exception as e:
        error_msg = f"❌ YouTube Error: {str(e)}"
        return {"answer": error_msg, "source_tool":"YouTube"}

# Tavily node (update history)
def tavily_node(state: ExpertState) -> Dict:
    try:
        question = state.get("question", "")
        user_id = state.get("mem0_user_id", "default_user")

        # Retrieve memory context
        retrieved_context = get_relevant_context(question, user_id)

        # Initialize TavilySearch
        tavi = TavilySearch(
            max_results=10,
            tavily_api_key=TAVILY_API_KEY,
        )

        # Perform the search
        search_results = tavi.invoke(question)

        # Tavily returns { "answer": "...", "results": [...] }
        results = search_results.get("results", []) if isinstance(search_results, dict) else []

        # Build numbered clickable link list
        formatted_links = []
        for i, r in enumerate(results[:5], 1):  # Limit to top 5
            if isinstance(r, dict):
                title = r.get("title", "No Title")
                url = r.get("url", "")
                snippet = r.get("content", "")[:150]  # short preview
                formatted_links.append(f"{i}. [{title}]({url})\n\n> {snippet}...\n")

        if not formatted_links:
            formatted_links = ["No news sources found."]

        raw_summary = "\n".join(formatted_links)

        # Ask LLM to summarize with citations
        system_msg = SystemMessage(
            content="You are a news summarizer. Create a concise, engaging summary from the provided search results, incorporating any relevant past context if it adds value. Number the sources as [1], [2], etc. in your summary for references."
        )
        human_msg = HumanMessage(
            content=f"Past context:\n{retrieved_context}\n\nNews results:\n{raw_summary}\n\nUser query: {question}"
        )

        summary_response = main_llm.invoke([system_msg, human_msg])
        summary_text = summary_response.content.strip() or "No summary generated."

        # Combine summary and sources
        final_answer = f"**🧭 Summary:**\n{summary_text}\n\n**📚 Sources:**\n" + raw_summary

        # Save to memory
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

# RAG Node (update history)
def rag_node(state: ExpertState) -> Dict:
    try:
        rag_prompt = ChatPromptTemplate.from_template(
    """Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
Memory context: {memory_context}

Web context:
{context}

Question: {input}

Helpful Answer:"""
)

        # Stuff chain for RAG
        stuff_chain = create_stuff_documents_chain(main_llm, rag_prompt)

        question = state.get("question", "")
        user_id = state.get("mem0_user_id", "default_user")

        # 🧠 Retrieve context
        retrieved_context = get_relevant_context(question, user_id)

        url_match = re.search(r'https?://(?:www\.)?[^ \t\n\r\f\v]+', question)
        if not url_match:
            ans = "No valid URL found in the query. Please provide a full URL for analysis."
            mem0_memory.add(messages=[{"role":"user","content":question},{"role":"assistant","content":ans}], user_id=user_id)
            return {"answer": ans, "source_tool":"RAG"}

        url = url_match.group(0)
        loader = WebBaseLoader(url)
        docs_raw = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = splitter.split_documents(docs_raw)
        limited_docs = docs[:10]

        # Invoke with memory_context
        resp = stuff_chain.invoke({
            "memory_context": retrieved_context,
            "context": limited_docs, 
            "input": question
        })

        ans = resp  # Chain outputs dict with 'text' or AIMessage, but for simplicity
        if isinstance(ans, dict):
            ans = ans.get("text", ans.get("output_text", str(ans)))
        else:
            ans = str(ans)
        ans = ans.strip() or "Summary could not be generated from the provided URL."

        mem0_memory.add(messages=[{"role":"user","content":question},{"role":"assistant","content":ans}], user_id=user_id)
        return {"answer": ans, "source_tool":"RAG"}
    except Exception as e:
        error_msg = f"❌ RAG Error: {str(e)}"
        return {"answer": error_msg, "source_tool":"RAG"}

# LLM Node (use history for context)
def llm_node(state: ExpertState) -> Dict:
    try:
        question = state.get("question", "")
        user_id = state.get("mem0_user_id", "default_user")

        # 🧠 Retrieve relevant context
        retrieved_context = get_relevant_context(question, user_id)

        system_msg = SystemMessage(content="You are a helpful, knowledgeable AI assistant. Provide concise, accurate answers. Use the provided context ONLY if it is directly relevant to the current question; otherwise, ignore it and rely on your general knowledge. For personal questions like names or greetings, draw from relevant memory if available.")
        human_msg = HumanMessage(content=f"Context (use ONLY if directly relevant): {retrieved_context}\n\nQuestion: {question}")

        response = main_llm.invoke([system_msg, human_msg])
        ans = str(response.content).strip() or "No answer generated."

        # Check for duplicate before adding (semantic search)
        existing = mem0_memory.search(query=question, user_id=user_id, limit=1)
        if isinstance(existing, dict):
            existing = existing.get('results', [])
        if not existing:
            mem0_memory.add(messages=[{"role": "user", "content": question}, {"role": "assistant", "content": ans}], user_id=user_id)

        return {"answer": ans, "source_tool": "AI"}
    except Exception as e:
        error_msg = f"❌ LLM Error: {str(e)}"
        return {"answer": error_msg, "source_tool": "AI"}

# Summarizer Node (use history)
def summarize_node(state: ExpertState) -> Dict:
    try:
        answer_text = state.get("answer","").strip()
        question = state.get("question","")
        if not answer_text:
            return {"final_summary":"No content to summarize.", "source_tool":"Summarizer"}
        
        system_msg = SystemMessage(content="You are a skilled summarizer. Create a clear, concise summary of the provided content while retaining key points relevant to the original question.")
        human_msg = HumanMessage(content=f"Question: {question}\n\nContent to summarize: {answer_text}")
        
        summary = main_llm.invoke([system_msg, human_msg])
        final_summary = str(summary.content).strip()
        return {"final_summary": final_summary, "source_tool":"Summarizer"}
    except Exception as e:
        error_msg = f"❌ Summarizer Error: {str(e)}"
        return {"final_summary": error_msg, "source_tool":"Summarizer"}

# -------------------- Reflector Node --------------------
def reflector_node(state: ExpertState) -> Dict:
    try:
        current_summary = state.get("final_summary","")
        question = state.get("question","")
        user_id = state.get("mem0_user_id","default_user")
        if not current_summary:
            return {"final_summary":"❌ No content to reflect on.", "source_tool":"Self-Improver"}
        
        system_msg = SystemMessage(content="You are a self-improving AI. Review the given summary for accuracy, completeness, and clarity. Suggest improvements and provide a refined version if needed.")
        human_msg = HumanMessage(content=f"Original question: {question}\n\nCurrent summary: {current_summary}\n\nProvide an improved version.")
        
        reflection = main_llm.invoke([system_msg, human_msg])
        improved_summary = str(reflection.content).strip()
        
        # Add reflection to memory
        mem0_memory.add(messages=[{"role":"user","content":f"Improve/Reflect on: {question}"},{"role":"assistant","content":improved_summary}], user_id=user_id)
        return {"final_summary": improved_summary, "source_tool":"Self-Improver"}
    except Exception as e:
        error_msg = f"❌ Reflector Error: {str(e)}"
        return {"final_summary": error_msg, "source_tool":"Self-Improver"}


# Build the LangGraph
graph = StateGraph(ExpertState)
graph.add_node("Router", expert_router)
graph.add_node("Tavily", tavily_node)
graph.add_node("RAG", rag_node)
graph.add_node("LLM", llm_node)
graph.add_node("Summarizer", summarize_node)
graph.add_node('Reflector', reflector_node)
graph.add_node("YouTube", youtube_node)
graph.add_node('Procoder',codeagent)

graph.set_entry_point("Router")
graph.add_conditional_edges("Router", lambda s: s["route"],
    {"tavily": "Tavily", "llm": "LLM","rag":"RAG","youtube":"YouTube",'code':'Procoder'})
graph.add_edge("Tavily", END)
graph.add_edge("LLM", END)
graph.add_edge("Procoder",END)
graph.add_edge("RAG", "Summarizer")# Optional, only if you want RAG
graph.add_edge("YouTube",END)
graph.add_edge("Summarizer", "Reflector")
graph.add_edge('Reflector',END)

# Checkpointer: Use Redis for persistence on Render; fallback to MemorySaver
redis_url = os.getenv("REDIS_URL")
if redis_url:
    checkpointer = RedisSaver.from_conn_string(redis_url)
else:
    st.warning("No REDIS_URL set; using in-memory saver (state lost on restarts).")
    checkpointer = MemorySaver()

# Persist the compiled app in session_state for short-term memory across reruns
if "app" not in st.session_state:
    st.session_state.app = graph.compile(checkpointer=checkpointer)

app = st.session_state.app

# Streamlit UI
st.set_page_config(page_title="News Agent", page_icon="📰", layout="wide")

# Custom CSS for ChatGPT-like styling and font
st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap">
    <style>
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stChatMessage.user {
            background-color: #f1f3f4;
        }
        .stChatMessage.assistant {
            background-color: #f7f7f8;
        }
        .stChatInput input {
            border-radius: 1rem !important;
            border: 1px solid #e0e0e0 !important;
            padding: 0.75rem 1rem !important;
            font-family: 'Inter', sans-serif !important;
        }
        .stChatInput input:focus {
            border-color: #10a37f !important;
            box-shadow: 0 0 0 1px #10a37f !important;
        }
        h1 {
            font-weight: 600;
            font-size: 2.5rem;
            color: #202123;
        }
        .stMarkdown {
            font-size: 1rem;
            line-height: 1.5;
        }
    </style>
""", unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>AGENTIX</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Your AI-powered assistant for the latest news, web analysis, and general knowledge</h4>", unsafe_allow_html=True)

# Sidebar for chat options
with st.sidebar:
    st.title("Chat Options")
    if st.button("Clear Memory & New Chat"):
        current_user_id = st.session_state.get("thread_id", "default_user")
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_session_1"

# Initialize last_history for manual persistence
if "last_history" not in st.session_state:
    st.session_state.last_history = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input (ChatGPT-style)
if prompt := st.chat_input("💬 Ask about latest news, a URL, or anything else..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking... ⚙️"):
            thread = {"configurable": {"thread_id": st.session_state.thread_id}}
            final_state = None

            # Prepare input state - RESET ALL FIELDS TO PREVENT CARRYOVER FROM PREVIOUS QUERIES
            input_state = {
                "question": prompt,
                "mem0_user_id": st.session_state.thread_id,
                "answer": "",
                "final_summary": "",
                "route": "",
                "source_tool": ""
            }

            # Run LangGraph and collect final state safely
            try:
                events = list(app.stream(input_state, config=thread, stream_mode="values"))
                final_state = events[-1] if events else None
                # Ensure final_state is dict
                if not isinstance(final_state, dict):
                    final_state = {"answer": "❌ Invalid state received from graph.", "source_tool": "Error"}
            except Exception as e:
                final_state = {"answer": f"❌ Agent Error: {str(e)}", "source_tool": "Error"}

            # Extract final response safely - Prioritize final_summary if set, else answer
            if isinstance(final_state, dict):
                response = final_state.get("final_summary", "").strip()
                if not response:
                    response = final_state.get("answer", "").strip()
                if not response:
                    response = "❌ No answer found."
                source_tool = final_state.get("source_tool", "unknown")
            else:
                response = "❌ No answer found. Try rephrasing your question!"
                source_tool = "Unknown"

            # Persist to Mem0 (redundant but ensures)
            try:
                mem0_memory.add(
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ],
                    user_id=st.session_state.thread_id
                )
            except Exception as mem_e:
                print(f"Mem0 add error: {mem_e}")

            # Display response
            st.markdown(f"**🛠 Source Tool:** {source_tool}")
            st.markdown(response)

            # Update Streamlit chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"[{source_tool}] {response}"
            })

            # Update last_history for short-term memory
            st.session_state.last_history.append({
                "question": prompt,
                "answer": response
            })
