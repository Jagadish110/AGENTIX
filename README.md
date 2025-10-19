# Agentix 📰✨

**Agentix** is an AI-powered personal assistant that delivers the **latest news**, **web analysis**, and **general knowledge** with a dash of magic! 🚀 It harnesses **LangGraph**, **Tavily Search**, and **Mem0 long-term memory** to provide accurate, context-aware responses while remembering your entire conversation history. Built with **Streamlit** for a sleek, interactive web interface. 🌐

---

## 🚀 **Features**

- **📰 News Summarization**: Pulls the hottest headlines via Tavily and crafts bite-sized, crystal-clear summaries. 📝
- **🔍 RAG (Retrieval-Augmented Generation)**: Dive into any URL, extract insights, and serve up structured summaries that shine. 📊
- **🎥 YouTube Video Search**: Hunt down spot-on videos tailored to your query—entertainment, tutorials, or deep dives! 🔎
- **💻 Code Assistant**: Whip up, debug, and demystify code in Python, JavaScript, and beyond. Your coding sidekick! 🐛
- **🧠 Contextual Memory**: Powered by **Mem0** + Qdrant, it stores and recalls long-term chat history like an elephant with a PhD. 🐘
- **🔄 Self-Improving Summarizer**: A clever reflection node polishes responses for top-tier quality every time. 💎
- **🛤️ Multi-Modal Routing**: Smartly steers queries to the right path—news, code, YouTube, URLs, or pure AI wisdom! 🧭

---

## 🛠️ **Tech Stack**

| Component                  | Technology / Library          | 🎯 Why It Rocks |
|----------------------------|-------------------------------|-----------------|
| **Frontend**               | Streamlit                     | Effortless, interactive UIs in minutes! 🎨 |
| **LLM & Agent**            | LangGraph, LangChain          | Builds robust AI workflows like a boss. 🤖 |
| **Search & Retrieval**     | Tavily, WebBaseLoader         | Lightning-fast, reliable web intel. ⚡ |
| **Memory & Embeddings**    | Mem0, HuggingFace, Qdrant     | Keeps convos fresh and unforgettable. 💾 |
| **Chat Models**            | ChatGroq, ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic | Power-packed LLMs for every vibe. 🗣️ |
| **Code Assistance**        | Python, LangChain             | Code gen that's smart and sassy. 🧑‍💻 |
| **Deployment**             | Vercel / Render / Cloud       | Serverless magic for seamless scaling. ☁️ |

---

## 📦 **Installation** (Quick & Easy! ⏱️)

1. **Clone the Repository**:
   bash
   git clone https://github.com/Jagadish110/Agentix.git
   cd Agentix

2. **Create a Virtual Environment**:
  bash
  python -m venv venv
  source venv/bin/activate  # Linux/Mac 🐧
  venv\Scripts\activate     # Windows 💻

3.**Install Dependencies (from requirements.txt)**:
  bash
  pip install -r requirements.txt

4.**Set Up Environment Variables (Your Secret Sauce! 🔑)**:
  textGROQ_API_KEY=YOUR_GROQ_KEY_HERE
  TAVILY_API_KEY=your_tavily_key_here
  QDRANT_API_KEY=your_qdrant_key_here

5.**Run the Agent and Watch the Magic Unfold! ✨**:
  bashstreamlit run streamlit_app.py
