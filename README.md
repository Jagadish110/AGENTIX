# Agentix 📰✨

[![Project](https://img.shields.io/badge/project-Agentix-6CCFF6?style=flat&logo=azurepipelines)](https://github.com/Jagadish110/AGENTIX)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streams](https://img.shields.io/badge/frontend-Streamlit-orange.svg)](https://streamlit.io)

> A colourful, friendly AI assistant that serves up the latest news, web analysis and code help — fast, concise and delightful.

---

🌟 Quick links
- Live demo: (add your deployment URL)
- Docs: (add docs link)
- Issues: https://github.com/Jagadish110/AGENTIX/issues

---

## ✨ What is Agentix?

Agentix is an AI-powered personal assistant that blends Retrieval-Augmented Generation (RAG), news summarization, deep web analysis, and code assistance into one friendly agent. It routes queries smartly (news, YouTube, URLs, coding, or general knowledge) and remembers past conversations with robust memory.

---

## 🚀 Features

- 📰 News Summarization — Fetches top headlines (Tavily) and returns crisp summaries.
- 🔍 RAG (Retrieval-Augmented Generation) — Crawl a URL, extract structured insights, and summarize key points.
- 🎥 YouTube Search — Find relevant videos (tutorials, deep dives) for a query.
- 💻 Code Assistant — Generate, explain and debug code in Python, JavaScript and more.
- 🧠 Contextual Memory — Mem0 + Qdrant powered long-term memory for consistent multi-turn chats.
- 🔄 Self-Improving Summaries — Reflection nodes polish outputs for improved quality.
- 🛤️ Multi-Modal Routing — Routes requests to the best pipeline automatically.

---

## 🧩 Tech Stack

| Component           | Technology / Library                                           | Why it rocks |
|--------------------:|----------------------------------------------------------------|--------------|
| Frontend            | Streamlit                                                      | Rapid, interactive UI |
| LLM & Orchestration | LangGraph, LangChain                                            | Composable AI workflows |
| Search & Retrieval  | Tavily, WebBaseLoader                                           | Fast web crawling & search |
| Memory & Embeddings | Mem0, HuggingFace, Qdrant                                       | Reliable vector memory |
| Chat Models         | ChatGroq, OpenAI, Google GAI, Anthropic                         | Variety of model styles |
| Code Assistant      | Python, LangChain                                               | Strong code generation & tooling |
| Deployment          | Vercel / Render / Cloud                                         | Serverless & scalable |

---

## 🎨 Preview

(Add screenshots or GIFs here — e.g. /docs/screenshot.png)

---

## 📦 Installation (Clean & Easy)

1. Clone the repository
```bash
git clone https://github.com/Jagadish110/AGENTIX.git
cd AGENTIX
```

2. Create and activate a virtual environment
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Environment variables — copy & fill your keys
```bash
# macOS / Linux (bash)
export GROQ_API_KEY="YOUR_GROQ_KEY_HERE"
export TAVILY_API_KEY="YOUR_TAVILY_KEY_HERE"
export QDRANT_API_KEY="YOUR_QDRANT_KEY_HERE"

# Windows (PowerShell)
$env:GROQ_API_KEY="YOUR_GROQ_KEY_HERE"
$env:TAVILY_API_KEY="YOUR_TAVILY_KEY_HERE"
$env:QDRANT_API_KEY="YOUR_QDRANT_KEY_HERE"
```

5. Run the app (Streamlit)
```bash
streamlit run streamlit_app.py
```

---

## 🧪 Usage examples

- Summarize a news page: paste a news URL in the "URL Analysis" tab and hit Summarize.
- Ask for code help: start a conversation and attach a snippet — Agentix will explain, refactor or debug.
- Search YouTube: choose "YouTube" mode and enter a topic to receive curated video results.

---

## 🛠️ Development Tips

- Add new models or connectors in the `agents/` or `nodes/` folder (where orchestration nodes live).
- Use the included `.env.example` as a template for required keys.
- Run tests (if present) via:
```bash
pytest
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Open a PR with a clear description

See CONTRIBUTING.md for more details (add this file if you want a contributor guide).

---

## 🔗 Links & Contact

- Repository: https://github.com/Jagadish110/AGENTIX
- Report issues: https://github.com/Jagadish110/AGENTIX/issues
- Author: Jagadish110 — feel free to open PRs or contact via GitHub

---

Thanks for using Agentix — let me know if you'd like me to:
1. Add real screenshots or a demo GIF,
2. Generate a CONTRIBUTING.md and CODE_OF_CONDUCT,
3. Include CI/Badges (build/test coverage),
or I can push these changes directly as a PR.
