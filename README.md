Agentix 📰✨
Agentix is an AI-powered personal assistant that delivers the latest news, web analysis, and general knowledge with a dash of magic! 🚀 It harnesses LangGraph, Tavily Search, and Mem0 long-term memory to provide accurate, context-aware responses while remembering your entire conversation history. Built with Streamlit for a sleek, interactive web interface. 🌐

🚀 Features

📰 News Summarization: Pulls the hottest headlines via Tavily and crafts bite-sized, crystal-clear summaries. 📝
🔍 RAG (Retrieval-Augmented Generation): Dive into any URL, extract insights, and serve up structured summaries that shine. 📊
🎥 YouTube Video Search: Hunt down spot-on videos tailored to your query—entertainment, tutorials, or deep dives! 🔎
💻 Code Assistant: Whip up, debug, and demystify code in Python, JavaScript, and beyond. Your coding sidekick! 🐛
🧠 Contextual Memory: Powered by Mem0 + Qdrant, it stores and recalls long-term chat history like an elephant with a PhD. 🐘
🔄 Self-Improving Summarizer: A clever reflection node polishes responses for top-tier quality every time. 💎
🛤️ Multi-Modal Routing: Smartly steers queries to the right path—news, code, YouTube, URLs, or pure AI wisdom! 🧭


🛠️ Tech Stack













































ComponentTechnology / Library🎯 Why It RocksFrontendStreamlitEffortless, interactive UIs in minutes! 🎨LLM & AgentLangGraph, LangChainBuilds robust AI workflows like a boss. 🤖Search & RetrievalTavily, WebBaseLoaderLightning-fast, reliable web intel. ⚡Memory & EmbeddingsMem0, HuggingFace, QdrantKeeps convos fresh and unforgettable. 💾Chat ModelsChatGroq, ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropicPower-packed LLMs for every vibe. 🗣️Code AssistancePython, LangChainCode gen that's smart and sassy. 🧑‍💻DeploymentVercel / Render / CloudServerless magic for seamless scaling. ☁️

📦 Installation (Quick & Easy! ⏱️)

Clone the Repository:
bashgit clone https://github.com/Jagadish110/Agentix.git
cd Agentix

Create a Virtual Environment:
bashpython -m venv venv
source venv/bin/activate  # Linux/Mac 🐧
venv\Scripts\activate     # Windows 💻

Install Dependencies:
bashpip install -r requirements.txt

Set Up Environment Variables (Your Secret Sauce! 🔑):
textGROQ_API_KEY=YOUR_GROQ_KEY_HERE
TAVILY_API_KEY=your_tavily_key_here
QDRANT_API_KEY=your_qdrant_key_here

Run the Agent and Watch the Magic Unfold! ✨:
bashstreamlit run streamlit_app.py
