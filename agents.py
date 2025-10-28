# agents.py — ephemeral SSE per call (safe across Streamlit reruns)

from typing import Any, Dict, List, Tuple, Optional
import json
from app.rag import rag_answer

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except Exception:
    ClientSession = None  # type: ignore

class MCPClientSSE:
    # Light stub that just stores the URL; no long‑lived session
    def __init__(self, server_url: str):
        self.server_url = server_url
    async def connect(self):  # no-op (validated elsewhere)
        return True
    async def disconnect(self):  # no-op safe exit
        return True
    @property
    def session(self):
        return None

def _parse(mcp_result) -> Dict[str, Any]:
    try:
        if hasattr(mcp_result, "content") and mcp_result.content:
            item = mcp_result.content[0]
            s = getattr(item, "text", "") or getattr(item, "data", "")
            s = s if isinstance(s, str) else json.dumps(s)
            return json.loads(s) if s.strip().startswith("{") else {"answer": s, "context": []}
    except Exception:
        pass
    return {"answer": "No response", "context": []}

async def answer_with_mcp(client: MCPClientSSE, question: str):
    async with sse_client(client.server_url) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            res = await sess.call_tool("query_rag", {"question": question, "top_k": 12})
            parsed = _parse(res)
            ans = parsed.get("answer") or "No response."
            ctx = parsed.get("context") or []
            used = [("query_rag", "RAG via MCP")]

            # summarize_context
            try:
                sum_res = await sess.call_tool("summarize_context", {"context": ctx[:3], "max_sentences": 3})
                s = _parse(sum_res).get("summary") or ""
                if s:
                    ans = f"{ans}\n\nSummary: {s}"
                    used.append(("summarize_context", "Summarized top context"))
            except Exception:
                pass

            # generate_tags
            try:
                tag_res = await sess.call_tool("generate_tags", {"answer": ans})
                tags = _parse(tag_res).get("tags") or []
                if tags:
                    ans = f"{ans}\n\nTags: {', '.join(tags)}"
                    used.append(("generate_tags", "Extracted tags"))
            except Exception:
                pass

            return ans, [str(c) for c in ctx][:12], used

def answer_locally(question: str) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    ans, ctx = rag_answer(question, return_context=True)
    return ans, [str(c) for c in (ctx or [])][:12], []

import re

def classify_sentiment(text: str):
    print("🚀 classify_sentiment() from agents.py is running with text:", text)
    lo = (text or "").lower()
    lo = re.sub(r"[^\w\s]", "", lo)  # remove punctuation
    pos_words = ["good","great","nice","love","best","amazing","helpful","clear","thanks","excellent","perfect"]
    neg_words = ["bad","wrong","terrible","hate","worst","awful","confusing","bug","error","issue","poor"]
    pos = sum(word in lo.split() for word in pos_words)
    neg = sum(word in lo.split() for word in neg_words)

    intensity = 1 if any(w in lo for w in ["very","really","absolutely","so","super"]) else 0

    if pos > neg:
        score = min(1.0, 0.6 + 0.1 * (pos + intensity))
        print("✅ Positive score:", score)
        return "positive", round(score, 2)
    elif neg > pos:
        score = min(1.0, 0.6 + 0.1 * (neg + intensity))
        print("❌ Negative score:", score)
        return "negative", round(score, 2)
    else:
        print("😐 Neutral feedback")
        return "neutral", 0.5



