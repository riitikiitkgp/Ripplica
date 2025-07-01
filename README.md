# 🔍 Gemini-Powered Web Query Agent with Caching & Scraping

An intelligent web agent that classifies user queries, retrieves similar past responses using embeddings, performs real-time web search and scraping (via Playwright), summarizes using **Gemini 1.5 Flash**, and caches results for future reuse.

---

## 🚀 Features

* ✅ **Query Classification** — Detects if the input is informational using Gemini LLM.
* 🧠 **Semantic Caching** — Uses vector embeddings + cosine similarity to retrieve similar past queries.
* 🌐 **Live Web Search** — Searches via Bing and Google using Playwright.
* 🕸 **Concurrent Scraping** — Extracts main content from top N search result pages.
* ✨ **LLM Summarization** — Summarizes results using Google Gemini (Generative Model).
* 💾 **Persistent Caching** — Stores results with embeddings in a local JSON file.

---

## 📦 Dependencies

Install all required packages:

```bash
pip install google-generativeai playwright numpy
playwright install chromium
pip install -r requirements.txt
```

---

## 🔐 API Key Setup

Set your **Google Gemini API key** in the code:

```python
GEMINI_API_KEY = "your-api-key-here"
```

Get the API key from: [https://console.cloud.google.com]

---

## 🛠️ How to Run

```bash
python3 main.py
```

Then enter your query when prompted:

```
🔍 Enter your query (or 'quit'):
```

---

## 🧠 What is an Embedding?

Embeddings are numerical vector representations of text. They allow semantically similar queries to be identified even when the wording is different. The agent uses **cosine similarity** between these vectors to decide if a new query is similar to a cached one.

---

## 📁 Cache Format

Results are stored in `query_cache.json` with:

```json
{
  "Best books on AI": {
    "response": "Gemini summary...",
    "timestamp": "2025-07-01T14:33:20.123456",
    "sources": [
      {"title": "...", "url": "..."}
    ],
    "embedding": [0.12, -0.44, ...]
  }
}
```

---

## ⚠️ Notes

* ❗ Only **informational queries** (like "how does inflation work?") are processed.
* ❌ Queries like “Walk my pet”, “Add apples to grocery” are rejected.

---

## 📚 Example Queries

* `Best places to visit in Jaipur`
* `How does Bitcoin work?`
* `Top books for learning AI`
