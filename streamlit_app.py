# streamlit_app.py
# Music Album Review RAG with MCP-first routing (SSE) and feedback email.

import os
import io
import json
import contextlib
import statistics as stats
import pandas as pd
import streamlit as st

from app.rag import rag_answer
from app.evaluation import all_metrics
from app.chroma_db import initialize_db

import asyncio, smtplib, ssl
from email.mime.text import MIMEText
from email.utils import formataddr
import re  # for first_float
from collections import Counter
import math

MCP_SERVER_URL = st.secrets.get("MCP_SERVER_URL", "http://127.0.0.1:8001/sse")


# ---------- Agent layer (SSE) ----------
# Guarded import prevents hard failures if agents.py is missing some symbol.
try:
    # IMPORTANT: agents.py should implement the "ephemeral per call" SSE pattern:
    # async with sse_client(url) as (read, write): async with ClientSession(read, write) as sess: ...
    from agents import MCPClientSSE as MCPClient, answer_with_mcp, answer_locally, classify_sentiment
except Exception:
    MCPClient = None
    def answer_locally(question: str):
        ans, ctx = rag_answer(question, return_context=True)
        return ans, [str(c) for c in (ctx or [])][:12], []
    def classify_sentiment(text: str):
        lo = (text or "").lower()
        pos = sum(w in lo for w in ["good","great","nice","love","best","amazing","helpful","clear","thanks","excellent","perfect"])
        neg = sum(w in lo for w in ["bad","wrong","terrible","hate","worst","awful","confusing","bug","error","issue","poor"])
        if pos > neg: return ("positive", 0.7)
        if neg > pos: return ("negative", 0.7)
        return ("neutral", 0.5)

def classify_sentiment_from_rating_and_comment(rating: int | None, text: str):
    # Text score from agents.classify_sentiment (e.g., 0.6..1.0)
    label_t, score_t = classify_sentiment(text or "")

    if isinstance(rating, int):
        # map stars to a rating label + confidence
        if rating <= 2:
            label_r, score_r = "negative", {1:0.20,2:0.40}[rating]
        elif rating == 3:
            label_r, score_r = "neutral", 0.60
        else:
            label_r, score_r = "positive", {4:0.80,5:1.00}[rating]

        # If labels agree, take the max confidence; if they conflict, average for caution
        if label_r == label_t:
            return label_r, round(max(score_r, score_t), 2)
        else:
            return (label_r if score_r >= score_t else label_t), round((score_r + score_t)/2, 2)

    # No rating → fall back to text classifier
    return label_t, round(score_t, 2)


# ---------------- Page config ----------------
st.set_page_config(page_title="Music Album Review RAG", page_icon="🎵", layout="centered")

# ---------------- Session State defaults ----------------
THEMES = {
    "light": {"primary": "#4A0D66", "bg": "#FFFFFF", "text": "#262730", "sidebar_bg": "#F8F8F8", "box_bg": "#f7efff"},
    "dark":  {"primary": "#C39BD3", "bg": "#0E1117", "text": "#FAFAFA", "sidebar_bg": "#171420", "box_bg": "#321352"},
}
if "theme" not in st.session_state: st.session_state.theme = "dark"
if "prompt_mode" not in st.session_state: st.session_state.prompt_mode = "Direct Answering (Standard RAG)"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "user_input" not in st.session_state: st.session_state.user_input = ""
if "sidebar_open" not in st.session_state: st.session_state.sidebar_open = True

# MCP + SMTP state
if "mcp_status" not in st.session_state: st.session_state.mcp_status = "disconnected"  # connected/disconnected/error
if "mcp_server_path" not in st.session_state: st.session_state.mcp_server_path = ""    # retained for compatibility
if "mcp_client" not in st.session_state: st.session_state.mcp_client = None           # store only URL wrapper, not a live session
if "mcp_server_url" not in st.session_state: st.session_state.mcp_server_url = st.secrets.get("MCP_SERVER_URL", "http://127.0.0.1:8001/sse")
if "feedback_recipient" not in st.session_state: st.session_state.feedback_recipient = st.secrets.get("FEEDBACK_TO", "")
if "smtp_user" not in st.session_state: st.session_state.smtp_user = st.secrets.get("SMTP_USER", "")
if "smtp_pass" not in st.session_state: st.session_state.smtp_pass = st.secrets.get("SMTP_PASS", "")
if "mcp_tools" not in st.session_state: st.session_state.mcp_tools = []
if "mcp_autotry_hash" not in st.session_state: st.session_state.mcp_autotry_hash = ""

# Memory used by rag.py
if "mem_events" not in st.session_state: st.session_state.mem_events = []
if "mem_summary" not in st.session_state: st.session_state.mem_summary = ""
if "mem_profile" not in st.session_state:
    st.session_state.mem_profile = {"domain": "music album QA", "answer_style": "copy exact sentence from context"}

# Ground truth init
if "ground_truth_mapping" not in st.session_state: st.session_state.ground_truth_mapping = {}
if "ground_truth_raw" not in st.session_state: st.session_state.ground_truth_raw = []
if "ground_truth_error" not in st.session_state: st.session_state.ground_truth_error = None

# Multi-chat persistence
if "conversations" not in st.session_state: st.session_state.conversations = {}
if "active_chat_id" not in st.session_state: st.session_state.active_chat_id = None

def _new_chat_id():
    import time, uuid
    return f"{int(time.time())}-{uuid.uuid4().hex[:6]}"

def _ensure_active_chat():
    if not st.session_state.active_chat_id:
        cid = _new_chat_id()
        st.session_state.active_chat_id = cid
        st.session_state.conversations[cid] = {"title": "New chat", "turns": [], "created": int(__import__('time').time())}

def _set_chat_title_if_empty(cid, first_question):
    conv = st.session_state.conversations.get(cid)
    if conv and (conv.get("title") in (None, "", "New chat")) and first_question:
        st.session_state.conversations[cid]["title"] = (first_question[:28] + "…") if len(first_question) > 28 else first_question

_ensure_active_chat()

# ---------------- Theme ----------------------
def apply_theme():
    t = THEMES[st.session_state.theme]
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {t['bg']}; color: {t['text']}; }}
        section[data-testid="stSidebar"] {{ background-color: {t['sidebar_bg']}; width: 260px !important; }}
        .stButton>button {{ color: {t['text']}; border: 1px solid {t['primary']}; background: transparent; }}
        a, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: {t['primary']} !important; }}

        .answer-container {{
            background-color: {t['box_bg']}; border-left: 5px solid {t['primary']};
            border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1rem; color: {t['text']};
        }}
        [data-testid="collapsedControl"] {{ background-color: {t['primary']} !important; }}

        .scrollable-chat {{ max-height: 55vh; overflow-y: auto; padding-bottom: 1rem; margin-bottom: 2rem; }}
        .centered {{ width:min(920px,96%); margin:0 auto; }}
        .input-wrap {{ position: sticky; bottom: 0; left: 0; right: 0; }}

        .sb-top {{ display:flex; align-items:center; justify-content:flex-start; height:36px; padding:6px; border-bottom:1px solid {t['primary']}; }}
        .sb-arrow {{
            width:28px; height:28px; border-radius:6px; cursor:pointer; user-select:none;
            border:2px solid {t['primary']}; background: transparent; color:{t['text']};
            font-weight:800; line-height:22px; display:inline-flex; align-items:center; justify-content:center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
apply_theme()

# ---------------- Optional secrets ----------------
if hasattr(st, "secrets"):
    try:
        for k in ["OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY", "SMTP_USER", "SMTP_PASS", "FEEDBACK_TO"]:
            v = st.secrets.get(k, "")
            if v: os.environ[k] = v
    except Exception:
        pass

# ---------------- Paths/DB --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def asset_path(*parts): return os.path.join(BASE_DIR, *parts)
if "db" not in st.session_state:
    st.session_state.db = initialize_db()

# ---------------- Utilities -------------------
def normalize_question(q: str) -> str:
    if not q: return ""
    q = q.strip().lower()
    q = q.replace("'", "'").replace("“", '"').replace("”", '"')
    q = " ".join(q.split())
    return q

def load_ground_truth_robust():
    try:
        possible_paths = [
            os.path.join(BASE_DIR, "evaluation", "queries.json"),
            os.path.join(BASE_DIR, "queries.json"),
            os.path.join(os.getcwd(), "evaluation", "queries.json"),
            os.path.join(os.getcwd(), "queries.json"),
            "evaluation/queries.json",
            "queries.json",
        ]
        data = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            data = json.loads(content)
                            break
            except Exception:
                continue
        if not data:
            return {}, [], "No valid queries.json file found"
        mapping, valid_items = {}, []
        for item in data:
            if not isinstance(item, dict):
                continue
            question = (item.get("question") or item.get("query") or item.get("q") or "").strip()
            answer = (item.get("answer") or item.get("ground_truth") or item.get("expected_answer") or item.get("a") or "").strip()
            if question and answer:
                normalized_q = normalize_question(question)
                mapping[normalized_q] = {
                    "original_question": question,
                    "expected_answer": answer,
                    "raw_item": item
                }
                valid_items.append(item)
        return mapping, valid_items, None
    except Exception as e:
        return {}, [], f"Error loading ground truth: {str(e)}"

def refresh_ground_truth():
    mapping, raw, err = load_ground_truth_robust()
    st.session_state.ground_truth_mapping = mapping
    st.session_state.ground_truth_raw = raw
    st.session_state.ground_truth_error = err
    return len(mapping)

if "ground_truth_mapping" in st.session_state and not st.session_state.ground_truth_mapping:
    refresh_ground_truth()

def find_ground_truth_answer(question: str):
    if not st.session_state.ground_truth_mapping:
        return None
    normalized_q = normalize_question(question)
    if normalized_q in st.session_state.ground_truth_mapping:
        return st.session_state.ground_truth_mapping[normalized_q]["expected_answer"]
    question_words = set(normalized_q.split())
    best_match, best_score = None, 0
    for gt_q, gt_data in st.session_state.ground_truth_mapping.items():
        gt_words = set(gt_q.split())
        if not gt_words:
            continue
        overlap = len(question_words.intersection(gt_words))
        union = len(question_words.union(gt_words))
        if union > 0:
            score = overlap / union
            if score > best_score and score > 0.4:
                best_score = score
                best_match = gt_data["expected_answer"]
    return best_match

# --------- Metrics helpers (robust parsing and fallback) ----------
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")  # robust float extractor [web:387]

def first_float(text, default=0.0) -> float:
    s = "" if text is None else str(text)
    m = NUM_RE.search(s)
    try:
        return float(m.group()) if m else float(default)
    except Exception:
        return float(default)

def _to_float_safe(val) -> float:
    try:
        if isinstance(val, str):
            val = val.strip().split()[0]
        x = float(val)
        if x != x: x = 0.0
        return max(0.0, min(1.0, x))
    except:
        return 0.0

def _tok(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower())]

def _vec(tokens: list[str]) -> Counter:
    return Counter(tokens)

def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b: return 0.0
    dot = sum(a[t]*b[t] for t in a)
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return 0.0 if na==0 or nb==0 else dot/(na*nb)

def simple_metrics(question: str, answer: str):
    q = _tok(question)
    a = _tok(answer)
    if not q or not a:
        return {"f1":0.0,"precision":0.0,"recall":0.0,"cosine":0.0,"f1_llm_combined":0.0,"rougeL":0.0}
    qs, as_ = set(q), set(a)
    inter = len(qs & as_)
    prec = inter / max(1, len(as_))
    rec  = inter / max(1, len(qs))
    f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    cos  = _cosine(_vec(q), _vec(a))
    return {"f1":f1,"precision":prec,"recall":rec,"cosine":cos,"f1_llm_combined":f1,"rougeL":f1}

def safe_all_metrics(question: str, answer: str) -> dict:
    # Capture noisy prints from judge code and sanitize numeric fields. [web:387]
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            res = all_metrics(question, answer)
    except Exception:
        return {}
    finally:
        _ = buf_out.getvalue(); _ = buf_err.getvalue()
    if not isinstance(res, dict): return {}
    clean = {}
    for key in ["f1","precision","recall","cosine","f1_llm_combined","rougeL"]:
        clean[key] = _to_float_safe(first_float(res.get(key, 0.0)))
    return clean

DISABLE_LLM_JUDGE = os.getenv("DISABLE_LLM_JUDGE", "1") == "1"  # default off to avoid noisy judge [web:387]

def cloud_safe_metrics(question: str, answer: str, expected_answer: str = None):
    if DISABLE_LLM_JUDGE:
        mets = simple_metrics(question, answer)
        return {k: _to_float_safe(first_float(v)) for k, v in mets.items()}, True
    try:
        metrics = safe_all_metrics(question, answer)
        if metrics and any(v > 0.0 for v in metrics.values()):
            return metrics, True
    except Exception:
        pass
    mets = simple_metrics(question, answer)
    return {k: _to_float_safe(first_float(v)) for k, v in mets.items()}, True

def compute_metrics_with_fallback(q_raw: str, ans: str):
    expected = find_ground_truth_answer(q_raw)
    has_gt = expected is not None
    metrics, _ = cloud_safe_metrics(q_raw, ans, expected)
    return metrics, has_gt

# ---------- Email ----------
def send_email(recipient: str, subject: str, body: str):
    user = st.session_state.smtp_user or st.secrets.get("SMTP_USER", "")
    pwd  = st.session_state.smtp_pass or st.secrets.get("SMTP_PASS", "")
    if not (user and pwd and recipient):
        return False, "Missing SMTP credentials or recipient"
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr(("Music RAG", user))
    msg["To"] = recipient
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.ehlo(); s.starttls(context=ssl.create_default_context()); s.login(user, pwd)
            s.sendmail(user, [recipient], msg.as_string())
        return True, "sent"
    except Exception as e:
        return False, str(e)

# ---------------- Sidebar ---------------------
def sidebar_body():
    # Avoid os shadowing by using a local alias; do NOT "import os" again later.
    import os as _os
    st.markdown("### Music RAG")
    logo = asset_path("logo2.jpg")
    if _os.path.exists(logo):
        st.image(logo, width=80)

    st.markdown("#### Sample Questions")
    if st.session_state.ground_truth_raw and len(st.session_state.ground_truth_raw) > 0:
        samples = []
        for item in st.session_state.ground_truth_raw[:4]:
            if isinstance(item, dict):
                q = item.get("question") or item.get("query") or item.get("q", "")
                if q: samples.append(q)
        while len(samples) < 4:
            defaults = [
                "When was the album Happier Than Ever by Billie Eilish released?",
                "What major British award did the song win in 2012?",
                "When was the song 'Hello' by Adele released?",
                "What musical styles does 'Dynamite' incorporate?",
            ]
            samples.extend(defaults[len(samples):4])
    else:
        samples = [
            "When was the album Happier Than Ever by Billie Eilish released?",
            "What major British award did the song win in 2012?",
            "When was the song 'Hello' by Adele released?",
            "What musical styles does 'Dynamite' incorporate?",
        ]
    for i, q in enumerate(samples):
        display_q = q[:65] + "..." if len(q) > 65 else q
        if st.button(display_q, key=f"sample_{i}", use_container_width=True):
            st.session_state.user_input = q

    st.divider()
    st.markdown("#### Settings")
    mode = st.radio(
        "Select Prompting Technique",
        ["Direct Answering (Standard RAG)", "Role-Based Answering (Advanced)"],
        index=0 if st.session_state.prompt_mode == "Direct Answering (Standard RAG)" else 1,
        key="prompt_selector",
    )
    if mode != st.session_state.prompt_mode:
        st.session_state.prompt_mode = mode; st.rerun()

    theme_choice = st.radio("Theme", ["dark", "light"], index=0 if st.session_state.theme=="dark" else 1, horizontal=True)
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice; st.rerun()

    st.divider()
    st.markdown("#### Chat History")

    conv_items = sorted(st.session_state.conversations.items(), key=lambda kv: kv[1].get("created", 0), reverse=True)
    for cid, meta in conv_items[:20]:
        label = meta.get("title") or "Untitled"
        is_active = (cid == st.session_state.active_chat_id)
        btn_label = ("• " if is_active else "") + (label[:40] + ("…" if len(label) > 40 else ""))
        if st.button(btn_label, key=f"chatpick_{cid}", use_container_width=True):
            st.session_state.active_chat_id = cid
            st.session_state.chat_history = [{"question": t["q"], "answer": t["a"], "context": t.get("ctx", [])} for t in meta.get("turns", [])]
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            cid = _new_chat_id()
            st.session_state.conversations[cid] = {"title": "New chat", "turns": [], "created": int(__import__('time').time())}
            st.session_state.active_chat_id = cid
            st.session_state.chat_history = []
            st.rerun()
    with c2:
        if st.button("🗑️ Delete", use_container_width=True, key="del_chat_btn"):
            ac = st.session_state.active_chat_id
            if ac and ac in st.session_state.conversations:
                del st.session_state.conversations[ac]
            st.session_state.active_chat_id = None
            st.session_state.chat_history = []
            _ensure_active_chat()
            st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear Current Chat", use_container_width=True, key="clear_curr_chat_btn"):
            ac = st.session_state.active_chat_id
            if ac in st.session_state.conversations:
                st.session_state.conversations[ac]["turns"] = []
            st.session_state.chat_history = []
            st.rerun()

    # Removed the Ground Truth block from the sidebar to simplify the UI. [web:395]

    st.divider()
    st.markdown("#### MCP")
    st.text_input("MCP Server URL", key="mcp_server_url",value=st.secrets.get("MCP_SERVER_URL", "http://127.0.0.1:8001/sse"))

    # Auto-connect silently and show read-only pill (no buttons). [web:395]
    cur_hash = f"{st.session_state.mcp_server_url}"
    def _autoconnect(url: str):
        try:
            if os.name == "nt":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            async def _connect_and_verify(u: str):
                from mcp import ClientSession
                from mcp.client.sse import sse_client
                async with sse_client(u) as (read, write):
                    async with ClientSession(read, write) as sess:
                        await sess.initialize()
                        tools = await sess.list_tools()
                        return [t.name for t in tools.tools]
            tool_names = asyncio.run(_connect_and_verify(url))
            if MCPClient is not None:
                st.session_state.mcp_client = MCPClient(url)  # URL wrapper; no long-lived session
                st.session_state.mcp_status = "connected"
                st.session_state.mcp_tools = tool_names
            else:
                st.session_state.mcp_status = "error"
        except Exception:
            st.session_state.mcp_client = None
            st.session_state.mcp_status = "disconnected"

    if (st.session_state.mcp_status != "connected") or (st.session_state.mcp_autotry_hash != cur_hash):
        _autoconnect(st.session_state.mcp_server_url)
        st.session_state.mcp_autotry_hash = cur_hash

    pill = "🟢 Connected" if st.session_state.mcp_status=="connected" else ("🔴 Disconnected" if st.session_state.mcp_status=="disconnected" else "🟠 Error")
    st.caption(f"MCP Server: {pill}")

    st.divider()
    st.markdown("#### Feedback")
    st.session_state.feedback_recipient = st.text_input("Recipient email", st.session_state.feedback_recipient, placeholder="pm@example.com")

with st.sidebar:
    st.markdown(
        f"""
        <div class="sb-top">
            <button id="sb-arrow" class="sb-arrow" title="Collapse/Expand">{'◀' if st.session_state.sidebar_open else '▶'}</button>
        </div>
        <script>
        (function(){{
            const btn = window.parent.document.getElementById("sb-arrow");
            if (btn && !btn._bound) {{
                btn._bound = true;
                btn.addEventListener("click", () => {{
                    const native = window.parent.document.querySelector('[data-testid="collapsedControl"]');
                    if (native) native.click();
                }});
            }}
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.sidebar_open:
        sidebar_body()

# ---------------- Title ----------------
st.markdown("""
<h1 style='color: #C39BD3; font-weight: 800; font-size: 2.7rem; margin-bottom:6px;margin-top:0' class='centered'>
<span style="font-size:2.2rem;vertical-align:middle;">🎵</span> 
<span style='color:#C39BD3'>Music Album</span> <span style="color:#9b59b6">Review <span style="color:#4A0D66">RAG</span></span>
</h1>
""", unsafe_allow_html=True)

# ---------------- Tabs -----------------------
tab_ask, tab_eval = st.tabs(["Ask AI", "Evaluation Dashboard"])

# ---------------- Ask AI ---------------------
with tab_ask:
    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.info(f"Currently using: {st.session_state.prompt_mode}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.markdown("<div class='scrollable-chat'>", unsafe_allow_html=True)
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(f"<div class='answer-container'>{chat['answer']}</div>", unsafe_allow_html=True)
            with st.expander("Show Evidence"):
                top3 = [str(ev) for ev in (chat.get("context") or [])[:3]]
                if top3:
                    preview = "\n\n".join(ev[:200] + "..." if len(ev) > 200 else ev for ev in top3)
                    st.info(preview)
                else:
                    st.write("No evidence available.")
            # Feedback under the last assistant turn so it survives reruns
            if i == len(st.session_state.chat_history) - 1:
                with st.form(f"fb_hist_{i}", clear_on_submit=True):
                    # Star rating (0-based index) → convert to 1..5
                    stars_idx = st.feedback("stars", key=f"stars_hist_{i}")  # built-in star picker [web:291]
                    rating_h = (stars_idx + 1) if stars_idx is not None else None  # 1..5

                    # Comment below the stars (better UX)
                    comment_h = st.text_area(
                        "Feedback",
                        placeholder="Tell us what you think",
                        key=f"comment_hist_{i}",
                        height=80,
                    )

                    send_btn_h = st.form_submit_button("Send feedback")
                if send_btn_h:
                    recipient = st.session_state.feedback_recipient or os.getenv("FEEDBACK_TO","")
                    if not recipient:
                        st.warning("Please enter a recipient email in the sidebar.")
                    else:
                        # Rating-first sentiment
                        label, score = classify_sentiment_from_rating_and_comment(rating_h, comment_h or chat["answer"])
                        subj = f"RAG Feedback - Rating: {rating_h or 0}/5"
                        body = (
                            f"Time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                            f"Question: {chat['question']}\n\n"
                            f"Answer: {chat['answer']}\n\n"
                            f"Rating: {rating_h or 0}\n"
                            f"Sentiment: {label} ({score:.2f})\n"
                            f"Comment: {comment_h or '(none)'}\n"
                        )
                        ok, msg = send_email(recipient, subj, body)
                        st.toast("Feedback emailed ✅" if ok else f"Email failed: {msg}", icon="📧" if ok else "⚠️")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-wrap centered'>", unsafe_allow_html=True)
    prompt = st.chat_input("Ask about an album review...", key="chat_widget")
    st.markdown("</div>", unsafe_allow_html=True)

    if prompt:
        st.session_state.user_input = prompt

    if st.session_state.user_input:
        q_raw = st.session_state.user_input
        st.session_state.user_input = ""
        with st.chat_message("user"): st.write(q_raw)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # MCP-first routing; use URL-only client when connected (no long-lived session object).
                try:
                    if st.session_state.mcp_status == "connected" and st.session_state.mcp_client is not None:
                        ans, ctx, used_tools = asyncio.run(answer_with_mcp(st.session_state.mcp_client, q_raw)); source_label = "MCP"
                    else:
                        ans, ctx, used_tools = answer_locally(q_raw); source_label = "Local RAG"
                except Exception:
                    ans, ctx, used_tools = answer_locally(q_raw); source_label = "Local RAG"

                mets, has_gt = compute_metrics_with_fallback(q_raw, ans)

        # Provenance badge
        st.caption(f"Source: {source_label}")  # will show MCP when routed via server [web:395]

        # Answer
        st.markdown(f"<div class='answer-container'>{ans}</div>", unsafe_allow_html=True)

        # Evidence
        if ctx:
            with st.expander("Show Evidence"):
                top3 = [str(ev) for ev in (ctx or [])[:3]]
                preview = "\n\n".join(ev[:200] + "..." if len(ev) > 200 else ev for ev in top3) if top3 else "No evidence available."
                st.info(preview)

        # Tools used (optional)
        with st.expander("Tools used"):
            if used_tools:
                for name, desc in used_tools: st.write(f"- {name}: {desc}")
            else:
                st.write("- Local RAG only")

        # Persist and rerun
        _ensure_active_chat()
        _set_chat_title_if_empty(st.session_state.active_chat_id, q_raw)
        ac = st.session_state.active_chat_id
        st.session_state.conversations[ac]["turns"].append({"q": q_raw, "a": ans, "ctx": ctx})
        st.session_state.chat_history.append(
            {"question": q_raw, "answer": ans, "context": ctx, "metrics": mets, "prompt_mode_used": st.session_state.prompt_mode, "has_ground_truth": has_gt, "source": source_label}
        )
        st.rerun()

# ---------------- Evaluation -----------------
with tab_eval:
    if not st.session_state.chat_history:
        st.info("Ask questions in the Ask AI tab to see the evaluation here.")
        err = st.session_state.get("ground_truth_error")
        if err:
            st.error(f"❌ {err}")
        elif len(st.session_state.ground_truth_mapping) > 0:
            st.success(f"✅ {len(st.session_state.ground_truth_mapping)} ground truth questions loaded")
        else:
            st.info("💡 Upload evaluation/queries.json for ground truth evaluation")
    else:
        eval_data = [c for c in st.session_state.chat_history if isinstance(c.get("metrics"), dict)]
        if not eval_data:
            st.warning("No questions have been evaluated yet.")
        else:
            st.markdown("#### 📊 Evaluation Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Questions", len(st.session_state.chat_history))
            with col2: st.metric("Evaluated", len(eval_data))
            with col3:
                gt_count = sum(1 for c in eval_data if c.get("has_ground_truth", False))
                st.metric("With Ground Truth", gt_count)
            with col4: st.metric("GT Available", len(st.session_state.ground_truth_mapping))

            keys = ["f1", "precision", "recall", "cosine", "f1_llm_combined", "rougeL"]
            agg = {}
            for k in keys:
                vals = []
                for c in eval_data:
                    try:
                        val = float(c["metrics"].get(k, 0.0))
                        if val > 0: vals.append(val)
                    except: continue
                agg[k] = round(stats.mean(vals), 3) if vals else 0.0

            st.markdown("#### 🎯 Overall Performance")
            c1, c2, c3 = st.columns(3); d1, d2, d3 = st.columns(3)
            c1.metric("F1 Score", f"{agg['f1']:.3f}")
            c2.metric("Precision", f"{agg['precision']:.3f}")
            c3.metric("Recall", f"{agg['recall']:.3f}")
            d1.metric("Cosine Similarity", f"{agg['cosine']:.3f}")
            d2.metric("LLM F1", f"{agg['f1_llm_combined']:.3f}")
            d3.metric("ROUGE-L", f"{agg['rougeL']:.3f}")

            st.markdown("---")
            st.markdown("#### 📈 Performance Visualization")
            bar_df = pd.DataFrame({"Metric": ["F1","Precision","Recall","Cosine","LLM+F1","ROUGE-L"],
                                   "Score": [agg['f1'],agg['precision'],agg['recall'],agg['cosine'],agg['f1_llm_combined'],agg['rougeL']]})
            st.bar_chart(bar_df.set_index("Metric"))

            st.markdown("#### 📚 Available Ground Truth Questions")
            if st.session_state.ground_truth_raw:
                for i, item in enumerate(st.session_state.ground_truth_raw[:10]):
                    if isinstance(item, dict):
                        q = item.get("question") or item.get("query") or ""
                        a = item.get("answer") or item.get("ground_truth") or ""
                        if q and a:
                            with st.expander(f"Q{i+1}: {q[:80]}{'...' if len(q) > 80 else ''}"):
                                st.markdown("*Question*"); st.write(q)
                                st.markdown("*Answer*");   st.write(a)
            else:
                st.caption("evaluation/queries.json not found (optional).")

            if st.button("Download Evaluation Results"):
                df = pd.DataFrame([{
                    "Question": c["question"], "Answer": c.get("answer",""),
                    "Prompt_Mode": c.get("prompt_mode_used","N/A"),
                    "Has_Ground_Truth": c.get("has_ground_truth", False),
                    **c.get("metrics", {})
                } for c in eval_data])
                st.download_button("Download CSV", df.to_csv(index=False).encode(),
                                   "evaluation_results.csv", "text/csv")
            st.caption(f"📊 Evaluated: {len(eval_data)} | With Ground Truth: {sum(1 for c in eval_data if c.get('has_ground_truth', False))}")
