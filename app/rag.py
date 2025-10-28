# app/rag.py
# RAG core with safe MCP stubs, memory, retrieval, re-ranking, and Groq chat completion.

import os
import time
import random
import re
import math
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer, CrossEncoder
from .chroma_db import query_db

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
LC_HISTORY_KEY = "langchain_messages"

# ---------- Meta intent helpers: previous question ----------
def _is_prev_question_query(q: str) -> bool:
    q = (q or "").lower().strip()
    pat = r"(what\s+(is|was)\s+the\s+previous\s+question)|(tell\s+me\s+(the\s+)?previous\s+question)|(\bprevious\s+question\b)|(\blast\s+question\b)"
    return re.search(pat, q) is not None

def _last_non_meta_question() -> str | None:
    for e in reversed(st.session_state.get("mem_events", [])):
        q = (e or {}).get("q", "")
        if q and not _is_prev_question_query(q):
            return q
    return None

# ---------- Environment / Models ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.1-8b-instant"

@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    print("Loading embedding model ...")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_reranker() -> CrossEncoder:
    print("Loading reranker ...")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------- Memory ----------
def _ensure_memory():
    st.session_state.setdefault("mem_events", [])
    st.session_state.setdefault("mem_summary", "")
    st.session_state.setdefault("mem_profile", {})

def _remember_turn(user_q: str, model_a: str, max_events: int = 6):
    _ensure_memory()
    st.session_state.mem_events.append({"q": user_q.strip(), "a": model_a.strip()})
    if len(st.session_state.mem_events) > max_events:
        st.session_state.mem_events = st.session_state.mem_events[-max_events:]

def _previous_question():
    _ensure_memory()
    if len(st.session_state.mem_events) >= 1:
        return st.session_state.mem_events[-1]["q"]
    return None

def _summarize_memory_if_needed(threshold_chars: int = 1200):
    _ensure_memory()
    serialized = "\n".join([f"U: {e['q']}\nA: {e['a']}" for e in st.session_state.mem_events])
    if len(serialized) < threshold_chars:
        return
    prompt = f"""
Summarize the following Q/A chat history into 3-5 bullet points capturing user intent, preferences, and unresolved details.
Be concise; do not include model-specific wording.

CHAT:
{serialized}

SUMMARY:
""".strip()
    if not GROQ_API_KEY:
        return
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 220, "temperature": 0.1},
            timeout=30,
        )
        resp.raise_for_status()
        st.session_state.mem_summary = resp.json()["choices"][0]["message"]["content"].strip()
        st.session_state.mem_events = st.session_state.mem_events[-4:]
    except Exception:
        pass

# ---------- Optional MCP “awareness” ----------
_MCP_TOOLS = [
    {"name": "album_facts", "desc": "Concise album facts from a local/remote knowledge store."},
    {"name": "critic_styles", "desc": "Common critic adjectives and genre descriptors in reviews."},
    {"name": "chart_trivia", "desc": "Chart performance tidbits like peaks and certifications."},
]

def _cos(a, b):
    da = math.sqrt(sum(x*x for x in a)) or 1.0
    db = math.sqrt(sum(x*x for x in b)) or 1.0
    return sum(x*y for x, y in zip(a, b)) / (da * db)

def _select_mcp_tools(embedder, query: str, top_k: int = 2):
    qv = embedder.encode([query])[0].tolist()
    scored = []
    for t in _MCP_TOOLS:
        tv = embedder.encode([t["desc"]])[0].tolist()
        scored.append((_cos(qv, tv), t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:max(1, min(top_k, len(scored)))]]

def _build_mcp_tools_block(embedder, query: str, top_k: int = 2) -> str:
    picks = _select_mcp_tools(embedder, query, top_k=top_k)
    if not picks:
        return ""
    lines = ["MCP-TOOLS:"]
    for t in picks:
        lines.append(f"- {t['name']}: {t['desc']}")
    return "\n".join(lines)

def _mcp_enabled() -> bool:
    return st.session_state.get("mcp_status") == "connected"

# Safe stubs so undefined helpers never crash the pipeline
if "mcp_document_search" not in globals():
    def mcp_document_search(*args, **kwargs):
        return []
if "mcp_album_facts" not in globals():
    def mcp_album_facts(*args, **kwargs):
        return ""

def _maybe_call_mcp_tool(user_query: str) -> str:
    if not _mcp_enabled():
        return ""
    q = user_query.lower()
    if "album" in q or "release" in q or "award" in q:
        m = re.search(r"'([^']+)'|\"([^\"]+)\"", user_query)
        album = (m.group(1) or m.group(2)) if m else ""
        facts = mcp_album_facts(album) if album else ""
        if facts and facts.strip() and "No facts found" not in facts:
            return f"[TOOL album_facts] {facts.strip()}"
    return ""

# ---------- Prompt builder ----------
def _build_mcp_prompt(user_query: str, context_chunks: List[str], mode: str):
    _ensure_memory()
    _summarize_memory_if_needed()

    system_msg = (
        "Return exactly one full sentence copied from TEXT that answers the question. "
        "Only if no such sentence exists in TEXT, reply exactly: No answer found in dataset."
    )
    profile = ""
    if st.session_state.mem_profile:
        profile = "Profile: " + "; ".join(f"{k}: {v}" for k, v in st.session_state.mem_profile.items())

    last_events = st.session_state.mem_events[-2:]
    convo_lines = []
    for e in last_events:
        if e.get("q"): convo_lines.append(f"U: {e['q']}")
        if e.get("a"): convo_lines.append(f"A: {e['a']}")
    convo_block = "\n".join(convo_lines).strip()

    text_block = "\n\n".join(context_chunks)
    mode_line = "You are a music expert." if mode == "Role-Based Answering (Advanced)" else "Direct answering."

    embedder = get_embedding_model()
    if _mcp_enabled():
        tools_block = _build_mcp_tools_block(embedder, user_query, top_k=2)
        if tools_block:
            system_msg = f"{system_msg}\n\n{tools_block}"

    user_body = f"""
{mode_line}
{profile}

MEMORY_SUMMARY:
{st.session_state.mem_summary or 'N/A'}

RECENT_TURNS:
{convo_block or 'N/A'}

TEXT:
{text_block}

QUESTION: {user_query}

ANSWER (copy one sentence from TEXT or say 'No answer found in dataset.'):
""".strip()

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_body},
    ]

# ---------- RAG main ----------
def rag_answer(user_query: str, return_context: bool = False,
               prompt_mode: str = "Direct Answering (Standard RAG)"):
    context_chunks: List[str] = []
    answer = "Error: Could not process query."
    lc_history = StreamlitChatMessageHistory(key=LC_HISTORY_KEY)

    try:
        # Meta intent: previous question (skip meta queries)
        q_norm = user_query.strip()
        if _is_prev_question_query(q_norm):
            prev_q = _last_non_meta_question()
            return ((prev_q or "No previous question found."), []) if return_context else (prev_q or "No previous question found.")

        # Pacing
        if hasattr(st.session_state, "last_api_call"):
            wait = 1.5 - (time.time() - st.session_state.last_api_call)
            if wait > 0:
                time.sleep(wait)

        # Embedding
        embedder = get_embedding_model()
        q_vec = embedder.encode([user_query])[0].tolist()

        # Retrieval via MCP (stub) then fallback to local Chroma
        raw_chunks: List[Any] = []
        if _mcp_enabled():
            try:
                mcp_res = mcp_document_search(q_vec, top_k=12) or []
                docs = mcp_res.get("documents", []) if isinstance(mcp_res, dict) else mcp_res
                for chunk in docs:
                    if isinstance(chunk, dict):
                        txt = chunk.get("text", "")
                    else:
                        txt = str(chunk)
                    if txt:
                        raw_chunks.append({"text": txt})
            except Exception:
                raw_chunks = []

        if not raw_chunks:
            raw_chunks = query_db(q_vec, top_k=25) or []

        # Normalize
        norm_chunks: List[Dict[str, str]] = []
        for ch in raw_chunks:
            if isinstance(ch, dict):
                norm_chunks.append({"text": str(ch.get("text", ch))})
            else:
                norm_chunks.append({"text": str(ch)})

        # Re-rank
        reranker = get_reranker()
        pairs = [[user_query, ch["text"]] for ch in norm_chunks]
        scores = reranker.predict(pairs)
        ranked = [ch for _, ch in sorted(zip(scores, norm_chunks), key=lambda z: z[0], reverse=True)]
        top_contexts = ranked[:12]

        # Optional tool augmentation
        tool_snippet = _maybe_call_mcp_tool(user_query)
        if tool_snippet:
            context_chunks = [tool_snippet] + [c["text"] for c in top_contexts]
        else:
            context_chunks = [c["text"] for c in top_contexts]

        # Prompt + LLM
        messages = _build_mcp_prompt(user_query, context_chunks, prompt_mode)
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME, "messages": messages, "max_tokens": 120, "temperature": 0.0}

        try:
            lc_history.add_user_message(user_query)
        except Exception:
            pass

        for attempt in range(5):
            try:
                if attempt:
                    delay = min(1 * 2**attempt, 8) + random.uniform(0.1, 0.4)
                    print(f"Retrying after {delay:.1f}s (attempt {attempt+1})")
                    time.sleep(delay)

                resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if data.get("choices"):
                    raw = data["choices"][0]["message"]["content"].strip()
                    answer = clean_answer(raw)
                    if answer == "No answer found in dataset.":
                        for ch in context_chunks:
                            if any(word.lower() in ch.lower() for word in user_query.split()):
                                candidate = ch.strip()
                                if candidate:
                                    answer = candidate
                                    break
                else:
                    answer = "Error: Invalid response from API."
                break

            except requests.exceptions.HTTPError:
                status = getattr(resp, "status_code", 0)
                print(f"HTTP {status} attempt {attempt+1}")
                print("BODY:", getattr(resp, "text", ""))
                if status == 500 and attempt < 4:
                    continue
                answer = ("Error: Bad request – check model/prompt."
                          if status == 400 else f"Error: API returned {status}")
                break
            except requests.exceptions.RequestException as e:
                print(f"Network error attempt {attempt+1}: {e}")
                if attempt == 4:
                    answer = "Error: Network problem – API unreachable."

        st.session_state.last_api_call = time.time()

        _remember_turn(user_query, answer)
        try:
            lc_history.add_ai_message(answer)
        except Exception:
            pass

    except Exception as err:
        print("rag_answer exception:", err)
        answer = f"Error: {err}"

    return (answer, context_chunks) if return_context else answer

# ---------- Utils ----------
def clean_answer(text: str) -> str:
    fillers = [
        "Return the sentence that directly answers the question.",
        "If absent, reply exactly:",
        "Answer with the exact sentence from the text below that answers the question.",
        "If it is missing, say:",
        "TEXT:", "QUESTION:", "EXACT SENTENCE:",
        "Q:", "A:", "ANSWER:"
    ]
    for f in fillers:
        text = text.replace(f, "")
    text = text.strip().strip('"').strip("'")
    return re.sub(r"\s+", " ", text)
