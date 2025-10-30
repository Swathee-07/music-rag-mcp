# agents.py — RAG-first; GK only for non-music or when RAG unavailable

from typing import Any, Dict, List, Tuple
import json, re
from app.rag import rag_answer

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except Exception:
    ClientSession = None  # type: ignore

class MCPClientSSE:
    def __init__(self, server_url: str): self.server_url = server_url
    async def connect(self): return True
    async def disconnect(self): return True
    @property
    def session(self): return None

def _parse(mcp_result) -> Dict[str, Any]:
    try:
        if hasattr(mcp_result, "content") and mcp_result.content:
            item = mcp_result.content[0]
            s = getattr(item, "text", "") or getattr(item, "data", "")
            s = s if isinstance(s, str) else json.dumps(s)
            return json.loads(s) if (isinstance(s, str) and s.strip().startswith("{")) else {"answer": s, "context": []}
    except Exception:
        pass
    return {"answer": "No response", "context": []}

# Intent — expanded so style/genre/music terms go to the music path
_MUSIC_HINTS = ["album","song","track","artist","band","lyrics","genre","style","musical",
                "music","review","release","single","ep","lp","chart"]

def _intent_is_music(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in _MUSIC_HINTS)

# Lighter confidence: fraction of question content words present in context
_STOP = set("""a an the is are was were be been being of for to in on at as by and or if then with from about than into after before during over under between up down out off near when who what which whose how why""".split())
_WORD = re.compile(r"[a-z0-9]+")

def _content_tokens(s: str) -> List[str]:
    return [t for t in _WORD.findall((s or "").lower()) if len(t) > 2 and t not in _STOP]

def _recall_confidence(question: str, ctx_list: List[str]) -> float:
    q = _content_tokens(question)
    if not q or not ctx_list:
        return 0.0
    ctx = " ".join((c or "") for c in ctx_list[:3])
    ctx_set = set(_content_tokens(ctx))
    hits = sum(1 for t in set(q) if t in ctx_set)
    return hits / max(1, len(set(q)))

async def answer_with_mcp(client: MCPClientSSE, question: str) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    if ClientSession is None:
        ans, ctx = rag_answer(question, return_context=True)
        return ans, [str(c) for c in (ctx or [])][:12], []

    used: List[Tuple[str, str]] = []
    async with sse_client(client.server_url) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            listed = await sess.list_tools()
            tool_names = {t.name for t in listed.tools}

            music_intent = _intent_is_music(question)

            # Prefer RAG if available and intent is music
            if music_intent and "query_rag" in tool_names:
                try:
                    res = await sess.call_tool("query_rag", {"question": question, "top_k": 12})
                    p = _parse(res)
                    ans = p.get("answer") or "No response."
                    ctx = p.get("context") or []
                    used.append(("query_rag","RAG via MCP"))

                    # Summarize context if available
                    if "summarize_context" in tool_names and ctx:
                        try:
                            sres = await sess.call_tool("summarize_context", {"context": ctx[:3], "max_sentences": 3})
                            summ = _parse(sres).get("summary") or ""
                            if summ:
                                ans = f"{ans}\n\nSummary: {summ}"
                                used.append(("summarize_context","Summarized context"))
                        except Exception:
                            pass

                    # Optional: if confidence is extremely low and you want automatic fallback, uncomment:
                    # conf = _recall_confidence(question, ctx)
                    # if conf < 0.15 and "general_knowledge_query" in tool_names:
                    #     g = await sess.call_tool("general_knowledge_query", {"query": question, "max_tokens": 180})
                    #     gp = _parse(g); gans = gp.get("answer") or "No response."
                    #     tags = gp.get("tags") or []
                    #     if tags: gans = f"{gans}\n\nTags: {', '.join(tags[:4])}"
                    #     used.append(("general_knowledge_query","LLM GK with dynamic tags"))
                    #     return gans, [], used

                    return ans, [str(c) for c in ctx][:12], used
                except Exception:
                    # Fall through to GK path below if RAG call fails
                    pass

            # Non-music or RAG unavailable → GK tool if present
            if "general_knowledge_query" in tool_names:
                try:
                    g = await sess.call_tool("general_knowledge_query", {"query": question, "max_tokens": 180})
                    gp = _parse(g)
                    gans = gp.get("answer") or "No response."
                    tags = gp.get("tags") or []
                    if tags:
                        gans = f"{gans}\n\nTags: {', '.join(tags[:4])}"
                    used.append(("general_knowledge_query","LLM GK with dynamic tags"))
                    return gans, [], used
                except Exception:
                    pass

            # Last fallback: local RAG
            lans, lctx = rag_answer(question, return_context=True)
            return lans, [str(c) for c in (lctx or [])][:12], used

def answer_locally(question: str) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    ans, ctx = rag_answer(question, return_context=True)
    return ans, [str(c) for c in (ctx or [])][:12], []

def classify_sentiment(text: str):
    lo = (text or "").lower()
    lo = re.sub(r"[^\w\s]", "", lo)
    pos_words = ["good","great","nice","love","best","amazing","helpful","clear","thanks","excellent","perfect"]
    neg_words = ["bad","wrong","terrible","hate","worst","awful","confusing","bug","error","issue","poor"]
    pos = sum(w in lo.split() for w in pos_words); neg = sum(w in lo.split() for w in neg_words)
    intensity = 1 if any(w in lo for w in ["very","really","absolutely","so","super"]) else 0
    if pos > neg:  return ("positive", round(min(1.0, 0.6 + 0.1*(pos+intensity)), 2))
    if neg > pos:  return ("negative", round(min(1.0, 0.6 + 0.1*(neg+intensity)), 2))
    return ("neutral", 0.5)
