
# 🎵 Music Album Review RAG + MCP (SSE)

**A domain-focused Retrieval-Augmented Generation (RAG) system for music album reviews integrated with a Model Context Protocol (MCP) Server over Server-Sent Events (SSE).**  
This project combines *retrieval intelligence* for music-specific queries with *agentic routing* to decide when to use RAG or an LLM-powered MCP tool for general knowledge responses.

---

##  Overview

This system demonstrates a **Music Album Review Retrieval-Augmented Generation pipeline** that uses:
- **RAG** for music-intent queries with evidence and summarized context.
- **LLM (Groq/OpenAI)** for general knowledge questions with dynamic topic tagging.
- **MCP Server (SSE protocol)** to expose and manage tools discoverable by the Streamlit client.
- **Agent-based routing** to intelligently decide between RAG and LLM paths.
- **feedback and email sentiment mapping.**


##  Key Features

- **RAG-based answers** with context evidence preview and a short summary.
- **Context-guard agent**: detects music intent and routes queries accordingly.
- **Dynamic topic tags** for general knowledge (non-music) questions.
- **Feedback mechanism** with star ratings and email sentiment (optional).
- **Clean Docker Compose deployment** (Streamlit + MCP server).



##  System Architecture

### Components

| Component | Description |
|------------|-------------|
| **Streamlit Client** | Front-end UI; performs routing, displays answers and tags. |
| **MCP Server (SSE)** | Hosts and exposes tools: `query_rag`, `summarize_context`, `general_knowledge_query`. |
| **RAG Index** | Local document store with embeddings for music albums. |
| **LLM Provider** | Groq/OpenAI API for general queries and dynamic tag generation. |


###  Data Flow

1. **User Query → Streamlit Client**
2. **Client → MCP Server (SSE)**  
   Performs: `initialize` → `list_tools`
3. **If Music Intent:**  
   → Call `query_rag` → `summarize_context` → Render Summary
4. **If Non-Music Intent:**  
   → Call `general_knowledge_query` → Render Tags
5. **Answer Rendered on UI**



###  Architecture Diagram (ASCII)



[User]
|
[Streamlit UI]
|  (SSE: initialize → list_tools → call_tool)
v
[MCP Server]
|-- query_rag  ─┐
|               ├─> RAG Index (documents, embeddings)
|-- summarize_context ┘
|
|-- general_knowledge_query (LLM: Groq/OpenAI; dynamic tags)
|
[Answer Renderer]
|__ shows Summary for RAG or Tags for GK



##  MCP Server Tools

| Tool | Function |
|------|-----------|
| **`query_rag(question, top_k)`** | Retrieves evidence-based answer from local RAG index. |
| **`summarize_context(context, max_sentences)`** | Produces a 2–3 sentence summary of top retrieved context. |
| **`general_knowledge_query(query)`** | Uses Groq/OpenAI LLM to answer general queries and auto-generate dynamic tags. |


##  Agent Algorithm

| Step | Function |
|------|-----------|
| **Intent Detection** | Identifies music vs non-music query based on keywords. |
| **Confidence Estimation (optional)** | Uses token recall to assess relevance. |
| **Decision Logic** | Routes to `query_rag` if music intent; else calls `general_knowledge_query`. |
| **Output Rendering** | Displays RAG summary or GK dynamic tags on Streamlit UI. |


##  Quick Start (Docker)

### Prerequisites
- Docker Desktop (≥ 4 GB RAM)
- Internet for LLM API calls (Groq/OpenAI)

### Run Commands
At the project root:
```bash
docker compose up --build
````

Then open:

* App → [http://localhost:8501](http://localhost:8501)
* MCP SSE → [http://localhost:8001/sse](http://localhost:8001/sse)



##  Environment Variables

| Variable                                | Purpose                                                                |
| --------------------------------------- | ---------------------------------------------------------------------- |
| `GROQ_API_KEY`                          | Groq via OpenAI-compatible endpoint  |
| `OPENAI_API_KEY`                        | Fallback if Groq not set                                               |
| `SMTP_USER`, `SMTP_PASS`, `FEEDBACK_TO` | Optional for feedback email                                            |



##  Example Prompts

| Query Type                         | Example                                                                                |
| ---------------------------------- | -------------------------------------------------------------------------------------- |
|  **Music (RAG path)**            | “When was the album *Happier Than Ever* by Billie Eilish released?”                    |
|  **Music (Style)**               | “What is the musical style of *Happier Than Ever*?”                                    |
|  **General Knowledge (GK path)** | “When is Diwali celebrated?” or “Who won the Grammy for Best Pop Vocal Album in 2024?” |



##  Evaluation & Results

* **Metrics**: Precision, Recall, and F1 on a test subset of music album questions.
* **Visualization**: Streamlit dashboard displaying performance metrics.


##  Deployment Summary

| Step | Description                                                                     |
| ---- | ------------------------------------------------------------------------------- |
| 1️⃣  | Save `Dockerfile.app`, `Dockerfile.mcp`, and `docker-compose.yaml` at repo root |
| 2️⃣  | Create `.env` file with API keys                                                |
| 3️⃣  | Run `docker compose up --build`                                                 |
| 4️⃣  | Access app at `http://localhost:8501`                                           |
| 5️⃣  | Test both RAG and GK paths with example prompts                                 |


##  Notes

* The MCP client calls only tools **advertised** by the server.
  Restart the server after any tool modifications so `initialize` and `list_tools` reflect changes.
* General knowledge answers **do not** claim RAG sources; tags are **LLM-generated**.
* The architecture is modular — MCP tools can be extended or replaced easily.


##  References

* **Model Context Protocol (MCP)** – OpenAI Tools API Specification
* **Groq API Documentation** – OpenAI-compatible endpoint
* **Docker Compose Documentation** – Multi-container application deployment
* **Streamlit Framework** – Rapid LLM dashboarding



> **Author:** Swathee M
> **Project Title:** *Music Album Review RAG with MCP Agentic Routing*
> **Tech Stack:** Python, Streamlit, FastAPI (Uvicorn), Groq/OpenAI, Docker Compose
> **Year:** 2025

