# mcp_server.py — RAG + summarize + general_knowledge_query (LLM with dynamic tags)
# Groq or OpenAI via OpenAI-compatible Chat Completions; memory-light and SSE-ready.

import os, re, json
from typing import List, Dict
from fastmcp import FastMCP
import uvicorn
import requests

HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("MCP_PORT", "8001")))

# LLM provider: prefer Groq if GROQ_API_KEY is set (OpenAI-compatible endpoint), else OpenAI
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_BASE  = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")  # Groq OpenAI-compatible base
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _chat_json(prompt: str, max_tokens: int = 220, want_json: bool = True) -> str:
    if GROQ_KEY:
        url = f"{GROQ_BASE}/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
        body = {"model": GROQ_MODEL, "messages":[{"role":"user","content":prompt}], "max_tokens": max_tokens}
    else:
        url = "https://api.groq.com/openai/v1"
        headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
        body = {"model": OPENAI_MODEL, "messages":[{"role":"user","content":prompt}], "max_tokens": max_tokens}
    if want_json:
        body["response_format"] = {"type":"json_object"}
    r = requests.post(url, headers=headers, json=body, timeout=12)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

mcp = FastMCP("music-rag-sse")

@mcp.tool()
def query_rag(question: str, top_k: int = 12) -> dict:
    from app.rag import rag_answer  # lazy import to minimize boot RAM
    try:
        ans, ctx = rag_answer(question, return_context=True)
        return {"answer": ans, "context": [str(c) for c in (ctx or [])][:top_k]}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def summarize_context(context: List[str], max_sentences: int = 3) -> Dict[str, str]:
    text = " ".join((context or [])[:3])
    sents = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
    return {"summary": ". ".join(sents[:max_sentences])}

@mcp.tool()
def general_knowledge_query(query: str, max_tokens: int = 220) -> Dict[str, object]:
    """
    LLM-only answerer for non-RAG questions; returns dynamic tags from the LLM.
    Never mixes RAG context to avoid implied citations.
    """
    # If no provider keys, refuse gracefully
    if not (GROQ_KEY or OPENAI_KEY):
        return {"answer": "General knowledge answering is unavailable (no LLM key configured).", "tags": []}
    try:
        prompt = (
            "You are a knowledge agent. Answer the question clearly and concisely, then output 2–4 short topic tags. "
            "Respond strictly as JSON like {\"answer\":\"...\",\"tags\":[\"tag1\",\"tag2\"]}.\n\n"
            f"Question: {query}"
        )
        txt = _chat_json(prompt, max_tokens=max_tokens, want_json=True)
        js = json.loads(txt)
        ans = (js.get("answer") or "").strip()
        tags = js.get("tags") or []
        # Sanity bounds
        if not isinstance(tags, list):
            tags = [str(tags)]
        return {"query": query, "answer": ans, "tags": [str(t)[:40] for t in tags][:4]}
    except Exception as e:
        return {"query": query, "answer": f"LLM answering failed: {e}", "tags": []}

# Build ASGI app after registering tools
app = mcp.http_app(transport="sse")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
