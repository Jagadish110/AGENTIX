# Agentix 📰

**Agentix** is an AI-powered personal assistant that provides the latest news, web analysis, and general knowledge. It leverages **LangGraph**, **Tavily Search**, and **Mem0 long-term memory** to give accurate, context-aware responses while retaining conversational history. The application is built with **Streamlit** for an interactive web interface.

---

## **Features**

- **News Summarization**: Retrieves the latest news using Tavily and generates concise summaries.
- **RAG (Retrieval-Augmented Generation)**: Analyze content from any URL and provide structured summaries.
- **YouTube Video Search**: Fetch relevant YouTube videos based on queries.
- **Code Assistant**: Generate, debug, and explain code in multiple programming languages.
- **Contextual Memory**: Uses **Mem0** with Qdrant to store and retrieve long-term conversation memory.
- **Self-Improving Summarizer**: Refines responses through a reflection node for higher-quality answers.
- **Multi-Modal Routing**: Automatically routes queries based on type (news, code, YouTube, URL, general AI questions).

---

## **Tech Stack**

| Component                  | Technology / Library |
|----------------------------|-------------------|
| Frontend                   | Streamlit         |
| LLM & Agent                 | LangGraph, LangChain |
| Search & Retrieval          | Tavily, WebBaseLoader |
| Memory & Embeddings         | Mem0, HuggingFace, Qdrant |
| Chat Models                 | ChatGroq, ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic |
| Code Assistance             | Python, LangChain |
| Deployment                  | Vercel / Render / Cloud (supports serverless Python apps) |

---

## **Installation**

1. **Clone the repository**:

```bash
git clone https://github.com/Jagadish110/Agentix.git
cd Agentix
2.#Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
3.#Install Dependincies
pip install -r requirements.txt
4.Set up Environment Variables:
GROQ_API_KEY=YOUR KEY
TAVILY_API_KEY=your_tavily_key_here
QDRANT_API_KEY=your_qdrant_key_here
5.RUN THE AGENT
streamlit run streamlit_app.py
