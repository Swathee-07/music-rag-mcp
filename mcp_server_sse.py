# mcp_server_sse.py
# SSE/HTTP MCP server that exposes your local RAG as one tool.

import os
import json
from typing import List
from dotenv import load_dotenv

from fastmcp import FastMCP  # built-in HTTP/SSE app
import uvicorn

# Your local RAG (must return (answer, context_list))
from app.rag import rag_answer

load_dotenv()

# Render dynamically assigns PORT — fallback to 8001 locally
HOST = os.getenv("MCP_HOST", "0.0.0.0")  
PORT = int(os.getenv("PORT", os.getenv("MCP_PORT", "8001")))

mcp = FastMCP("music-rag-sse")

@mcp.tool()
def query_rag(question: str, top_k: int = 12) -> dict:
    """
    Wrap local RAG and return JSON payload with answer + context.
    """
    try:
        ans, ctx = rag_answer(question, return_context=True)
        ctx_text: List[str] = [str(c) for c in (ctx or [])][:top_k]
        return {"answer": ans, "context": ctx_text}
    except Exception as e:
        return {"error": str(e)}

from typing import List, Dict
import re

@mcp.tool()
def summarize_context(context: List[str], max_sentences: int = 3) -> Dict[str, str]:
    """
    Summarizes the retrieved context into a short paragraph.
    """
    text = " ".join((context or [])[:3])
    # very light heuristic summarizer
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
    summary = ". ".join(sentences[:max_sentences])
    return {"summary": summary}

@mcp.tool()
def generate_tags(answer: str) -> Dict[str, List[str]]:
    """
    Extract 2-3 tags (artist, mood, year-like tokens) from the answer.
    """
    a = (answer or "").lower()
    tags = []
    year = re.search(r"\b(19|20)\d{2}\b", a)
    if year:
        tags.append(year.group(0))
    for m in ["happy", "melancholic", "nostalgic", "energetic", "sad", "romantic"]:
        if m in a:
            tags.append(m)
    for g in ["pop", "rock", "jazz", "hip-hop", "r&b", "electropop", "folk", "country"]:
        if g in a and g not in tags:
            tags.append(g)
    return {"tags": tags[:3]}


# Create a Starlette ASGI app with SSE transport correctly mounted.
app = mcp.http_app(transport="sse")

if __name__ == "__main__":
    print(f"✅ Starting MCP server on http://{HOST}:{PORT}")
    print(f"SSE endpoint: /sse")
    print(f"Messages endpoint: /messages/")
    uvicorn.run(app, host=HOST, port=PORT)
