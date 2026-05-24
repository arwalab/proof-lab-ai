import os
import json
import hashlib
import tempfile
import base64
import numpy as np
import streamlit as st
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
from supabase import create_client, Client

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# ─────────────────────────────────────────────
# Supabase client
# ─────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://sylodrwofujroxlbygdw.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

@st.cache_resource
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────
# Supabase Vector Store
# ─────────────────────────────────────────────

def vector_store_add(chunk_id: str, embedding: list, document: str, metadata: dict) -> bool:
    """Insert a chunk into Supabase vector_store. Returns True if inserted, False if duplicate."""
    sb = get_supabase()
    # Check for duplicate
    existing = sb.table("vector_store").select("id").eq("id", chunk_id).execute()
    if existing.data:
        return False
    sb.table("vector_store").insert({
        "id": chunk_id,
        "embedding": embedding,
        "document": document,
        "document_title": metadata.get("document_title"),
        "page_number": str(metadata.get("page", "")),
        "chunk_index": metadata.get("chunk_number", 0),
        "extraction_method": metadata.get("extraction_method", "pypdf"),
    }).execute()
    return True

def vector_store_query(query_embedding: list, n_results: int = 5):
    """Query Supabase for top-n similar chunks using pgvector cosine similarity."""
    sb = get_supabase()
    try:
        result = sb.rpc("match_vectors", {
            "query_embedding": query_embedding,
            "match_count": n_results
        }).execute()
        docs = [r["document"] for r in result.data]
        metas = [{
            "document_title": r.get("document_title", "Unknown"),
            "page": r.get("page_number", "N/A"),
            "chunk_number": r.get("chunk_index", "N/A"),
            "extraction_method": r.get("extraction_method", "unknown"),
            "similarity": r.get("similarity", 0)
        } for r in result.data]
        return docs, metas
    except Exception as e:
        st.error(f"Vector query error: {e}")
        return [], []

def vector_store_get_all():
    """Get all document metadata from Supabase for the document library."""
    sb = get_supabase()
    try:
        result = sb.table("vector_store").select("id, document_title, page_number, chunk_index, created_at").execute()
        docs = {}
        for row in result.data:
            title = row.get("document_title", "Unknown")
            if title not in docs:
                docs[title] = {"document": title, "pages": set(), "chunks": 0, "uploaded_at": str(row.get("created_at", ""))[:16]}
            docs[title]["chunks"] += 1
            page = row.get("page_number")
            if page and page not in ["", "N/A", "None"]:
                docs[title]["pages"].add(str(page))
        return [{"Document": info["document"], "Pages": len(info["pages"]), "Chunks": info["chunks"], "Uploaded": info["uploaded_at"]} for info in docs.values()]
    except Exception:
        return []

def vector_store_delete(document_title: str) -> int:
    """Delete all chunks for a given document from Supabase."""
    sb = get_supabase()
    existing = sb.table("vector_store").select("id").eq("document_title", document_title).execute()
    count = len(existing.data)
    sb.table("vector_store").delete().eq("document_title", document_title).execute()
    return count

def vector_store_count() -> int:
    """Get total chunk count."""
    sb = get_supabase()
    try:
        result = sb.table("vector_store").select("id", count="exact").execute()
        return result.count or 0
    except Exception:
        return 0

# ─────────────────────────────────────────────
# Supabase History Tables
# ─────────────────────────────────────────────

def load_history(table: str, columns: list) -> pd.DataFrame:
    sb = get_supabase()
    try:
        result = sb.table(table).select("*").order("id", desc=True).execute()
        if result.data:
            df = pd.DataFrame(result.data)
            # Keep only known columns + drop internal id
            keep = [c for c in columns if c in df.columns]
            return df[keep] if keep else df
        return pd.DataFrame(columns=columns)
    except Exception:
        return pd.DataFrame(columns=columns)

def append_history(table: str, row: dict):
    sb = get_supabase()
    try:
        sb.table(table).insert(row).execute()
    except Exception as e:
        st.warning(f"Could not save to history: {e}")

def load_batches() -> pd.DataFrame:
    return load_history("batch_tracker", BATCH_COLUMNS)

def save_batch(row: dict):
    append_history("batch_tracker", row)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
MODE_ICONS = {
    "Home":                  "",
    "Upload Knowledge Base": "",
    "Ask Knowledge Base":    "",
    "SOP Creator":           "",
    "Batch Tracker":         "",
    "Vision Analyzer":       "",
    "Recipe R&D Generator":  "",
    "Recipe Evaluator":      "",
    "Data Migration":        "",
}

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Home"

st.set_page_config(
    page_title=f"Proof Lab AI — {st.session_state.active_mode}",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --bg-main:     #1a0f20;
    --bg-card:     #2C1332;
    --bg-sidebar:  #200d28;
    --accent:      #FEF7CF;
    --accent-light:#ffffff;
    --accent-dim:  rgba(254,247,207,0.10);
    --text-primary:#FEF7CF;
    --text-muted:  #9BB7D4;
    --text-faint:  #757577;
    --border:      rgba(155,183,212,0.22);
    --border-solid:#3a2545;
    --radius:      12px;
    --shadow:      0 4px 24px rgba(0,0,0,0.5);
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"] {
    background: var(--bg-main) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stToolbar"] { background: var(--bg-main) !important; }
[data-testid="stDecoration"] { background: var(--bg-main) !important; display: none !important; }
[data-testid="stAppViewBlockContainer"] { padding-top: 2rem !important; }
.stApp > header { background-color: var(--bg-main) !important; }
.stApp { background-color: var(--bg-main) !important; }

[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.pl-logo-svg { width: 52px; height: 52px; flex-shrink: 0; filter: drop-shadow(0 0 12px rgba(155,183,212,0.5)); }

.proof-header {
    display: flex; align-items: center; gap: 16px;
    padding: 20px 0 8px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
}
.proof-header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important; font-weight: 700 !important;
    color: var(--accent) !important;
    margin: 0 !important; padding: 0 !important;
    letter-spacing: 0.5px;
}
.proof-header .tagline { font-size: 0.72rem; color: var(--text-muted); letter-spacing: 2px; text-transform: uppercase; margin-top: 3px; }

.mode-hero {
    margin: 0 0 28px 0; padding: 22px 28px;
    border-radius: 0 0 var(--radius) var(--radius);
    border: 1px solid var(--border); border-top: none;
    display: flex; align-items: center; gap: 16px;
    position: relative; overflow: hidden;
}
.mode-hero::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(155,183,212,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.mode-hero .hero-icon { font-size: 2.4rem; flex-shrink: 0; filter: drop-shadow(0 0 8px rgba(155,183,212,0.4)); }
.mode-hero .hero-title { font-family: 'Playfair Display', serif; font-size: 1.35rem; font-weight: 700; color: var(--accent); margin: 0; }
.mode-hero .hero-desc { font-size: 0.8rem; color: var(--text-muted); margin-top: 3px; line-height: 1.5; }
.mode-hero .hero-bg-text {
    position: absolute; right: 24px; top: 50%; transform: translateY(-50%);
    font-size: 5rem; opacity: 0.04; font-family: 'Playfair Display', serif;
    font-weight: 700; color: #9BB7D4; pointer-events: none; user-select: none; white-space: nowrap;
}

[data-testid="stSidebar"] .stButton > button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: transparent !important; font-size: 0 !important; padding: 0 !important;
    margin: -36px 0 0 0 !important; height: 36px !important; width: 100% !important;
    position: relative !important; z-index: 2 !important; cursor: pointer !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background: transparent !important; transform: none !important; box-shadow: none !important; }
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] .stButton:last-of-type > button {
    background: transparent !important; border: 1px solid rgba(155,183,212,0.22) !important;
    color: #9BB7D4 !important; font-size: 0.78rem !important; padding: 6px 12px !important;
    margin: 0 !important; height: auto !important; border-radius: 6px !important;
}

.pl-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; margin-bottom: 18px; box-shadow: var(--shadow); }
.pl-card-title { font-size: 0.68rem; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; color: var(--accent); margin-bottom: 14px; }

.metric-row { display: flex; gap: 12px; margin-bottom: 20px; }
.metric-card { flex: 1; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 18px; box-shadow: var(--shadow); }
.metric-card .metric-value { font-size: 1.8rem; font-weight: 700; color: var(--accent); line-height: 1; }
.metric-card .metric-label { font-size: 0.65rem; color: var(--text-faint); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 5px; }

.pl-divider { display: flex; align-items: center; gap: 12px; margin: 28px 0 20px 0; }
.pl-divider::before, .pl-divider::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.pl-divider span { font-size: 0.62rem; color: var(--text-faint); text-transform: uppercase; letter-spacing: 2px; white-space: nowrap; }

.empty-state { background: var(--bg-card); border: 1.5px dashed var(--border); border-radius: var(--radius); padding: 52px 40px; text-align: center; }
.empty-state .empty-icon { font-size: 2.8rem; margin-bottom: 14px; }
.empty-state .empty-title { font-size: 1rem; color: var(--text-muted); font-weight: 500; margin-bottom: 6px; }
.empty-state .empty-sub { font-size: 0.78rem; color: var(--text-faint); }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #2a1a35 !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #9BB7D4 !important; box-shadow: 0 0 0 2px rgba(254,247,207,0.15) !important; outline: none !important;
}
.input-hint { font-size: 0.68rem; color: var(--text-faint); text-align: right; margin-top: -10px; margin-bottom: 8px; }

.stSelectbox > div > div { background-color: #2a1a35 !important; border: 1px solid var(--border) !important; border-radius: 8px !important; color: var(--text-primary) !important; }

.stButton > button {
    background: linear-gradient(135deg, #9BB7D4, #5a7fa8) !important;
    color: #1a0f20 !important; font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important; border: none !important;
    border-radius: 8px !important; padding: 10px 24px !important;
    font-size: 0.88rem !important; letter-spacing: 0.4px !important;
    transition: all 0.2s ease !important; box-shadow: 0 2px 14px rgba(155,183,212,0.2) !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 5px 22px rgba(155,183,212,0.38) !important; }
.stDownloadButton > button { background: transparent !important; color: #9BB7D4 !important; border: 1px solid var(--border) !important; border-radius: 8px !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }
.stDownloadButton > button:hover { background: var(--accent-dim) !important; border-color: #9BB7D4 !important; }
[data-testid="stFormSubmitButton"] > button { background: linear-gradient(135deg, #9BB7D4, #5a7fa8) !important; color: #1a0f20 !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; width: 100% !important; padding: 12px !important; font-size: 0.92rem !important; }

[data-testid="stChatMessage"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; margin-bottom: 10px !important; }
[data-testid="stChatInput"] > div { background: #2a1a35 !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
[data-testid="stChatInput"] textarea { color: var(--text-primary) !important; }

.streamlit-expanderHeader { background: #261030 !important; border: 1px solid var(--border) !important; border-radius: 8px !important; color: var(--text-muted) !important; font-size: 0.82rem !important; }
.streamlit-expanderContent { background: #220e2e !important; border: 1px solid var(--border) !important; border-top: none !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; overflow: hidden !important; }
[data-testid="stDataFrame"] table { background-color: var(--bg-card) !important; }
[data-testid="stDataFrame"] thead tr th { background-color: #2e1540 !important; color: #9BB7D4 !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 1px !important; border-bottom: 1px solid var(--border) !important; }
[data-testid="stDataFrame"] tbody tr td { background-color: var(--bg-card) !important; color: var(--text-primary) !important; font-size: 0.82rem !important; border-bottom: 1px solid #2e1540 !important; }
[data-testid="stDataFrame"] tbody tr:hover td { background-color: #2e1540 !important; }

[data-testid="stFileUploader"] { background: #261030 !important; border: 1.5px dashed var(--border) !important; border-radius: var(--radius) !important; }
[data-testid="stFileUploader"]:hover { border-color: #9BB7D4 !important; }

[data-testid="stAlert"] { border-radius: 8px !important; border-left: 3px solid var(--accent) !important; background: #261030 !important; }

label, .stTextInput label, .stTextArea label, .stSelectbox label, .stDateInput label, .stCheckbox label {
    color: var(--text-muted) !important; font-size: 0.8rem !important; font-weight: 500 !important; letter-spacing: 0.3px !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: #3a2545; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9BB7D4; }

h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--text-primary) !important; }
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def pl_divider(label=""):
    st.markdown(f"<div class='pl-divider'><span>{label}</span></div>", unsafe_allow_html=True)

def show_history(title, table, columns, download_name):
    pl_divider(title)
    df = load_history(table, columns)
    if df.empty:
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-icon">📭</div>
            <div class="empty-title">No {title.lower()} yet</div>
            <div class="empty-sub">Records will appear here after your first submission.</div>
        </div>""", unsafe_allow_html=True)
    else:
        page_size = 10
        total = len(df)
        page_key = f"page_{download_name}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 0
        max_page = max(0, (total - 1) // page_size)
        start = st.session_state[page_key] * page_size
        end = min(start + page_size, total)
        st.dataframe(df.iloc[start:end], use_container_width=True)
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_a:
            if st.button("← Prev", key=f"prev_{download_name}", disabled=st.session_state[page_key] == 0):
                st.session_state[page_key] -= 1
                st.rerun()
        with col_b:
            st.markdown(f"<div style='text-align:center;font-size:0.75rem;color:var(--text-faint);padding-top:8px;'>Page {st.session_state[page_key]+1} of {max_page+1} · {total} records</div>", unsafe_allow_html=True)
        with col_c:
            if st.button("Next →", key=f"next_{download_name}", disabled=st.session_state[page_key] >= max_page):
                st.session_state[page_key] += 1
                st.rerun()
        st.download_button(
            label=f"⬇ Download {title} CSV",
            data=df.to_csv(index=False),
            file_name=download_name,
            mime="text/csv"
        )

def get_document_library():
    return vector_store_get_all()

def delete_document(document_title):
    return vector_store_delete(document_title)

def extract_text_with_pypdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    pages = []
    for page_index, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": page_index, "text": text})
    return pages, len(pdf.pages)

def extract_text_with_llamaparse(file_path):
    parser = LlamaParse(api_key=llama_api_key, result_type="markdown")
    documents = parser.load_data(file_path)
    combined_text = "\n\n".join(doc.text for doc in documents if doc.text)
    return [{"page": "OCR", "text": combined_text}]

def add_chunks_to_store(uploaded_file_name, extracted_pages, page_count, extraction_method):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    )
    added_chunks = 0
    skipped_chunks = 0
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    for page_data in extracted_pages:
        chunks = text_splitter.split_text(page_data["text"])
        for chunk_index, chunk in enumerate(chunks):
            if len(chunk.strip()) <= 100:
                continue
            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            chunk_id = f"{uploaded_file_name}_page_{page_data['page']}_{chunk_hash}"
            embedding = client.embeddings.create(model="text-embedding-3-small", input=chunk).data[0].embedding
            added = vector_store_add(chunk_id, embedding, chunk, {
                "document_title": uploaded_file_name,
                "page": page_data["page"],
                "chunk_number": chunk_index,
                "chunk_hash": chunk_hash,
                "extraction_method": extraction_method,
                "uploaded_at": upload_time,
                "page_count": page_count
            })
            if added:
                added_chunks += 1
            else:
                skipped_chunks += 1
    return added_chunks, skipped_chunks

def retrieve_context(question, n_results=5):
    total = vector_store_count()
    if total == 0:
        return "No documents in knowledge base yet.", []
    query_embedding = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
    documents, metadatas = vector_store_query(query_embedding, n_results=n_results)
    context_blocks = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source_label = (f"[Source {i}: {meta.get('document_title', 'Unknown')} | "
                        f"Page {meta.get('page', 'N/A')} | Chunk {meta.get('chunk_number', 'N/A')} | "
                        f"Method: {meta.get('extraction_method', 'unknown')}]")
        context_blocks.append(source_label + "\n" + doc)
    return "\n\n".join(context_blocks), metadatas

def format_sources(metadatas):
    return "\n".join(
        f"Source {i}: {meta.get('document_title', 'Unknown')} | Page {meta.get('page', 'N/A')} | "
        f"Chunk {meta.get('chunk_number', 'N/A')} | Extraction: {meta.get('extraction_method', 'unknown')}"
        for i, meta in enumerate(metadatas, start=1)
    )

def encode_image(uploaded_image):
    return base64.b64encode(uploaded_image.getvalue()).decode("utf-8")

def mode_hero(icon, title, description, bg_text=""):
    st.markdown(f"""
    <div class="mode-hero">
        <div class="hero-icon">{icon}</div>
        <div>
            <div class="hero-title">{title}</div>
            <div class="hero-desc">{description}</div>
        </div>
        <div class="hero-bg-text">{bg_text or title}</div>
    </div>""", unsafe_allow_html=True)

def empty_state(icon, title, subtitle):
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-icon">{icon}</div>
        <div class="empty-title">{title}</div>
        <div class="empty-sub">{subtitle}</div>
    </div>""", unsafe_allow_html=True)

BATCH_COLUMNS   = ["timestamp","batch_name","product","batch_date","formula_notes","process_notes",
                   "dough_temp","butter_temp","room_temp","proof_temp","proof_time","bake_temp","bake_time",
                   "result","issues","next_adjustment"]
ASK_COLUMNS     = ["timestamp","question","answer","sources"]
SOP_COLUMNS     = ["timestamp","product_name","user_notes","generated_sop","sources"]
VISION_COLUMNS  = ["timestamp","image_name","product_type","batch_notes","diagnosis","sources"]
RD_COLUMNS      = ["timestamp","product_type","flavor_direction","texture_goal","constraints","brand_mood","batch_size","generated_concept","sources"]

# ─────────────────────────────────────────────
# SVG Logo
# ─────────────────────────────────────────────
PROOF_LAB_LOGO_SVG = """
<svg width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="26" cy="26" r="26" fill="url(#grad)"/>
  <defs>
    <radialGradient id="grad" cx="35%" cy="30%" r="70%">
      <stop offset="0%" stop-color="#9BB7D4"/>
      <stop offset="100%" stop-color="#2C1332"/>
    </radialGradient>
  </defs>
  <path d="M20 14 L20 26 L14 38 Q13 40 15 41 L37 41 Q39 40 38 38 L32 26 L32 14 Z"
        fill="none" stroke="#1a0f20" stroke-width="2" stroke-linejoin="round"/>
  <rect x="19" y="12" width="14" height="3" rx="1.5" fill="#1a0f20"/>
  <path d="M16.5 35 Q18 30 26 30 Q34 30 35.5 35 L37 41 Q39 40 38 38 L32 26 L32 14 L20 14 L20 26 L14 38 Q13 40 15 41 Z"
        fill="rgba(0,0,0,0.35)"/>
  <circle cx="22" cy="36" r="1.5" fill="rgba(255,255,255,0.5)"/>
  <circle cx="28" cy="33" r="1" fill="rgba(255,255,255,0.4)"/>
  <circle cx="25" cy="38" r="1" fill="rgba(255,255,255,0.3)"/>
</svg>
"""

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 0 16px 0;border-bottom:1px solid rgba(254,247,207,0.18);margin-bottom:20px;
                display:flex;align-items:center;gap:12px;">
        <div class="pl-logo-svg">{PROOF_LAB_LOGO_SVG}</div>
        <div>
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:#FEF7CF;line-height:1.2;">Proof Lab AI</div>
            <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:2px;margin-top:3px;">Bakery Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;padding-left:4px;'>Navigation</div>", unsafe_allow_html=True)

    nav_items = [
        ("Home",                  "Home"),
        ("Upload Knowledge Base", "Upload Knowledge Base"),
        ("Ask Knowledge Base",    "Ask Knowledge Base"),
        ("SOP Creator",           "SOP Creator"),
        ("Batch Tracker",         "Batch Tracker"),
        ("Vision Analyzer",       "Vision Analyzer"),
        ("Recipe R&D Generator",  "Recipe R&D Generator"),
        ("Recipe Evaluator",      "Recipe Evaluator"),
        ("Data Migration",        "Data Migration"),
    ]
    for nav_key, nav_label in nav_items:
        is_active = st.session_state.active_mode == nav_key
        active_style = "color:#FEF7CF;font-weight:600;border-left:2px solid #9BB7D4;padding-left:10px;" if is_active else "color:#9BB7D4;font-weight:400;border-left:2px solid transparent;padding-left:10px;"
        st.markdown(
            f"""<div style='{active_style}font-size:0.88rem;padding-top:8px;padding-bottom:8px;
            cursor:pointer;transition:all 0.15s ease;letter-spacing:0.2px;'>
            {nav_label}</div>""",
            unsafe_allow_html=True
        )
        if st.button(nav_label, key=f"nav_{nav_key}", use_container_width=True):
            st.session_state.active_mode = nav_key
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear Chat History", key="clear_chat", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat history cleared.", icon="🗑")
        st.rerun()

    st.markdown("<div style='height:1px;background:rgba(254,247,207,0.12);margin:12px 0 16px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;padding-left:4px;'>Knowledge Base</div>", unsafe_allow_html=True)

    library_rows = get_document_library()

    if library_rows:
        total_chunks = sum(r["Chunks"] for r in library_rows)
        total_docs   = len(library_rows)
        st.markdown(f"""
        <div style="display:flex;gap:8px;margin-bottom:12px;">
            <div style="flex:1;background:#261030;border:1px solid rgba(254,247,207,0.18);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:1.3rem;font-weight:700;color:#FEF7CF;">{total_docs}</div>
                <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Docs</div>
            </div>
            <div style="flex:1;background:#261030;border:1px solid rgba(254,247,207,0.18);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:1.3rem;font-weight:700;color:#FEF7CF;">{total_chunks}</div>
                <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        library_df = pd.DataFrame(library_rows)
        st.dataframe(library_df, use_container_width=True, height=150)
        doc_names    = [row["Document"] for row in library_rows]
        selected_doc = st.selectbox("Select document to delete", doc_names, label_visibility="collapsed")
        if st.button("🗑  Delete Document", use_container_width=True):
            deleted_count = delete_document(selected_doc)
            st.toast(f"Deleted {deleted_count} chunks from '{selected_doc}'", icon="🗑")
            st.rerun()
    else:
        st.markdown("""
        <div style="background:#261030;border:1px dashed rgba(254,247,207,0.18);border-radius:8px;
                    padding:18px;text-align:center;">
            <div style="font-size:1.4rem;margin-bottom:6px;">📂</div>
            <div style="font-size:0.75rem;color:#555;">No documents yet.<br>Upload a PDF to get started.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.58rem;color:#3a2545;text-align:center;'>Proof Lab AI · v4.0 · Supabase</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
mode = st.session_state.active_mode

# ─────────────────────────────────────────────
# Mode: Home
# ─────────────────────────────────────────────
if mode == "Home":
    library_rows_home  = get_document_library()
    total_docs_home    = len(library_rows_home)
    total_chunks_home  = sum(r["Chunks"] for r in library_rows_home)
    batch_df_home      = load_batches()
    total_batches_home = len(batch_df_home)

    st.markdown(f"""
    <div style="padding:48px 0 32px 0;">
        <div class="pl-logo-svg" style="margin-bottom:20px;">{PROOF_LAB_LOGO_SVG}</div>
        <div style="font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:700;
                    color:#FEF7CF;line-height:1.15;margin-bottom:10px;">
            Proof Lab AI
        </div>
        <div style="font-size:0.78rem;color:#9BB7D4;text-transform:uppercase;
                    letter-spacing:3px;margin-bottom:32px;">
            Bakery Intelligence Platform
        </div>
        <div style="font-size:1rem;color:#9BB7D4;max-width:560px;line-height:1.8;margin-bottom:40px;">
            Your AI-powered R&D lab — built for The Proof Lab team to query knowledge,
            build SOPs, track batches, analyze product photos, and generate new concepts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-value">{total_docs_home}</div>
            <div class="metric-label">Documents</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_chunks_home}</div>
            <div class="metric-label">Knowledge Chunks</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_batches_home}</div>
            <div class="metric-label">Batches Logged</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">6</div>
            <div class="metric-label">AI Modules</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pl_divider("Modules")

    modules = [
        ("Ask Knowledge Base",   "Query your uploaded documents with AI-powered semantic search and retrieval."),
        ("SOP Creator",          "Generate professional, staff-ready Standard Operating Procedures from rough notes."),
        ("Batch Tracker",        "Log every production batch with full parameters and AI-powered analysis."),
        ("Vision Analyzer",      "Upload a product photo for AI visual diagnosis and corrective guidance."),
        ("Recipe R&D Generator", "Invent technically original bakery concepts based on flavor and texture goals."),
        ("Recipe Evaluator",     "Critically evaluate test recipes for formula balance and brand fit."),
    ]
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for i, (mod_name, mod_desc) in enumerate(modules):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="pl-card" style="cursor:pointer;">
                <div class="pl-card-title">{mod_name}</div>
                <div style="font-size:0.82rem;color:#9BB7D4;line-height:1.7;">{mod_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {mod_name}", key=f"home_btn_{mod_name}", use_container_width=True):
                st.session_state.active_mode = mod_name
                st.rerun()

    pl_divider("Getting Started")
    st.markdown("""
    <div class="pl-card">
        <div class="pl-card-title">Quick Start Guide</div>
        <div style="font-size:0.85rem;color:#9BB7D4;line-height:2.1;">
            <b style="color:#FEF7CF;">1. Upload your documents</b> &nbsp;—&nbsp;
                Use <i>Upload Knowledge Base</i> to add baking references, recipes, or SOPs.<br>
            <b style="color:#FEF7CF;">2. Ask questions</b> &nbsp;—&nbsp;
                Navigate to <i>Ask Knowledge Base</i> and ask anything about your uploaded documents.<br>
            <b style="color:#FEF7CF;">3. Create SOPs</b> &nbsp;—&nbsp;
                Paste rough notes into <i>SOP Creator</i> to generate a professional procedure.<br>
            <b style="color:#FEF7CF;">4. Log batches</b> &nbsp;—&nbsp;
                Use <i>Batch Tracker</i> to record every production run with AI analysis.<br>
            <b style="color:#FEF7CF;">5. Analyze photos</b> &nbsp;—&nbsp;
                Upload a product image in <i>Vision Analyzer</i> for instant visual diagnosis.<br>
            <b style="color:#FEF7CF;">6. Invent new concepts</b> &nbsp;—&nbsp;
                Use <i>Recipe R&D Generator</i> to create original, technically sound bakery concepts.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header (shown on non-Home modes)
# ─────────────────────────────────────────────
if mode not in ("Home", "Upload Knowledge Base"):
    st.markdown(f"""
    <div class="proof-header">
        <div class="pl-logo-svg">{PROOF_LAB_LOGO_SVG}</div>
        <div>
            <h1>Proof Lab AI</h1>
            <div class="tagline">Bakery Intelligence Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Mode: Upload Knowledge Base
# ─────────────────────────────────────────────
if mode == "Upload Knowledge Base":
    import time as _time
    mode_hero("📂", "Upload Knowledge Base",
              "Add PDF documents to your knowledge base for AI-powered search and retrieval.",
              "Upload")

    up_col, lib_col = st.columns([3, 2])

    with up_col:
        st.markdown("""<div class="pl-card"><div class="pl-card-title">Upload PDF Documents</div>""", unsafe_allow_html=True)

        force_ocr = st.checkbox("Force OCR with LlamaParse (for scanned / image-based PDFs)")
        uploaded_files = st.file_uploader(
            "Select one or more PDF files", type="pdf",
            accept_multiple_files=True, label_visibility="visible"
        )

        if uploaded_files:
            total_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
            est_chunks_total = 0
            for f in uploaded_files:
                try:
                    reader_est = PdfReader(f)
                    sample_text = ""
                    for pg in reader_est.pages[:min(3, len(reader_est.pages))]:
                        t = pg.extract_text()
                        if t:
                            sample_text += t
                    avg_chars = len(sample_text) / min(3, len(reader_est.pages)) if reader_est.pages else 0
                    est_chunks_total += max(1, int((avg_chars * len(reader_est.pages)) / 1000))
                    f.seek(0)
                except Exception:
                    est_chunks_total += 30
            est_seconds = max(15, int(est_chunks_total * 0.55))
            est_min = est_seconds // 60
            est_sec = est_seconds % 60
            est_label = f"{est_min}m {est_sec}s" if est_min > 0 else f"{est_sec}s"

            st.markdown(f"""
            <div style="background:#1a0f20;border:1px solid rgba(155,183,212,0.22);border-radius:10px;
                        padding:14px 18px;margin:10px 0 14px 0;display:flex;gap:24px;align-items:center;">
                <div style="text-align:center;">
                    <div style="font-size:1.4rem;font-weight:700;color:#FEF7CF;">{len(uploaded_files)}</div>
                    <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Files</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.4rem;font-weight:700;color:#FEF7CF;">{total_size_mb:.1f} MB</div>
                    <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Total Size</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.4rem;font-weight:700;color:#FEF7CF;">~{est_chunks_total}</div>
                    <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Est. Chunks</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.4rem;font-weight:700;color:#9BB7D4;">~{est_label}</div>
                    <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Est. Time</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Process {len(uploaded_files)} PDF(s)", key="bulk_upload_btn", use_container_width=True):
                start_time = _time.time()
                progress_bar = st.progress(0, text="Starting...")
                results = []

                for idx, uploaded_file in enumerate(uploaded_files):
                    progress_pct = int((idx / len(uploaded_files)) * 100)
                    elapsed = _time.time() - start_time
                    remaining = max(0, est_seconds - int(elapsed))
                    rem_min = remaining // 60
                    rem_sec = remaining % 60
                    rem_label = f"{rem_min}m {rem_sec}s remaining" if rem_min > 0 else f"{rem_sec}s remaining"
                    progress_bar.progress(progress_pct,
                        text=f"Processing {uploaded_file.name} ({idx+1}/{len(uploaded_files)}) — ~{rem_label}")

                    extracted_pages   = []
                    page_count        = 0
                    extraction_method = "pypdf"

                    if not force_ocr:
                        try:
                            extracted_pages, page_count = extract_text_with_pypdf(uploaded_file)
                        except Exception:
                            extracted_pages = []

                    total_text_length = sum(len(p["text"]) for p in extracted_pages)

                    if force_ocr or total_text_length < 500:
                        if not llama_api_key:
                            st.error("Missing LLAMA_CLOUD_API_KEY in your secrets.")
                            st.stop()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getbuffer())
                            temp_path = temp_file.name
                        extracted_pages   = extract_text_with_llamaparse(temp_path)
                        page_count        = "OCR"
                        extraction_method = "llamaparse_ocr"

                    added_chunks, skipped_chunks = add_chunks_to_store(
                        uploaded_file.name, extracted_pages, page_count, extraction_method
                    )
                    results.append({"File": uploaded_file.name, "Method": extraction_method,
                                    "Added": added_chunks, "Skipped": skipped_chunks})

                actual_time  = _time.time() - start_time
                actual_min   = int(actual_time) // 60
                actual_sec   = int(actual_time) % 60
                actual_label = f"{actual_min}m {actual_sec}s" if actual_min > 0 else f"{actual_sec}s"

                progress_bar.progress(100, text=f"✅ All {len(uploaded_files)} file(s) processed in {actual_label}!")

                results_df    = pd.DataFrame(results)
                total_added   = results_df["Added"].sum()
                total_skipped = results_df["Skipped"].sum()

                st.toast(f"✅ Upload complete — {total_added} chunks added!", icon="📂")

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(100,200,120,0.12),rgba(44,19,50,0.7));
                            border:2px solid rgba(100,200,120,0.5);border-radius:14px;
                            padding:24px 28px;margin:20px 0;box-shadow:0 0 24px rgba(100,200,120,0.15);">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                        <span style="font-size:2rem;">✅</span>
                        <div>
                            <div style="font-size:1.2rem;font-weight:800;color:#a8f0b8;">Upload Complete!</div>
                            <div style="font-size:0.82rem;color:#9BB7D4;margin-top:2px;">
                                Your documents are now saved to Supabase and permanently searchable.
                            </div>
                        </div>
                    </div>
                    <div style="display:flex;gap:24px;flex-wrap:wrap;margin:14px 0 10px 0;">
                        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px 18px;text-align:center;">
                            <div style="font-size:1.5rem;font-weight:800;color:#FEF7CF;">{len(uploaded_files)}</div>
                            <div style="font-size:0.75rem;color:#9BB7D4;">Files Processed</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px 18px;text-align:center;">
                            <div style="font-size:1.5rem;font-weight:800;color:#a8f0b8;">{total_added}</div>
                            <div style="font-size:0.75rem;color:#9BB7D4;">Chunks Added</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px 18px;text-align:center;">
                            <div style="font-size:1.5rem;font-weight:800;color:#757577;">{total_skipped}</div>
                            <div style="font-size:0.75rem;color:#9BB7D4;">Duplicates Skipped</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px 18px;text-align:center;">
                            <div style="font-size:1.5rem;font-weight:800;color:#9BB7D4;">{actual_label}</div>
                            <div style="font-size:0.75rem;color:#9BB7D4;">Time Taken</div>
                        </div>
                    </div>
                    <div style="font-size:0.8rem;color:#a8f0b8;margin-top:6px;">
                        → Go to <b>Ask Knowledge Base</b> in the sidebar to start querying your documents.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='font-size:0.82rem;color:#9BB7D4;margin:8px 0 4px 0;font-weight:600;letter-spacing:0.05em;'>FILE BREAKDOWN</div>", unsafe_allow_html=True)
                st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="pl-card" style="margin-top:18px;">
            <div class="pl-card-title">Tips for Best Results</div>
            <div style="font-size:0.83rem;color:#9BB7D4;line-height:2.0;">
                <b style="color:#FEF7CF;">Text-based PDFs</b> are processed automatically with PyPDF — fast and accurate.<br>
                <b style="color:#FEF7CF;">Scanned / image PDFs</b> require OCR — enable the checkbox above before uploading.<br>
                <b style="color:#FEF7CF;">Duplicate chunks</b> are automatically detected and skipped.<br>
                <b style="color:#FEF7CF;">All data is stored in Supabase</b> — persistent across all app reboots and redeployments.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with lib_col:
        st.markdown("""<div class="pl-card"><div class="pl-card-title">Document Library</div>""", unsafe_allow_html=True)

        library_rows_up = get_document_library()
        if library_rows_up:
            total_chunks_up = sum(r["Chunks"] for r in library_rows_up)
            total_docs_up   = len(library_rows_up)
            st.markdown(f"""
            <div style="display:flex;gap:8px;margin-bottom:14px;">
                <div style="flex:1;background:#1a0f20;border:1px solid rgba(155,183,212,0.22);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:700;color:#FEF7CF;">{total_docs_up}</div>
                    <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Documents</div>
                </div>
                <div style="flex:1;background:#1a0f20;border:1px solid rgba(155,183,212,0.22);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:700;color:#FEF7CF;">{total_chunks_up}</div>
                    <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Chunks</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            library_df_up = pd.DataFrame(library_rows_up)
            st.dataframe(library_df_up, use_container_width=True, height=200)

            pl_divider("Delete a Document")
            doc_names_up    = [row["Document"] for row in library_rows_up]
            selected_doc_up = st.selectbox("Select document to remove", doc_names_up)
            if st.button("Delete Selected Document", key="delete_doc_up", use_container_width=True):
                deleted_count = delete_document(selected_doc_up)
                st.toast(f"Deleted {deleted_count} chunks from '{selected_doc_up}'", icon="🗑")
                st.rerun()
        else:
            st.markdown("""
            <div style="text-align:center;padding:32px 0;">
                <div style="font-size:2.5rem;margin-bottom:12px;">📂</div>
                <div style="font-size:0.85rem;color:#9BB7D4;">No documents yet.</div>
                <div style="font-size:0.78rem;color:#757577;margin-top:4px;">Upload a PDF to get started.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Mode: Ask Knowledge Base
# ─────────────────────────────────────────────
if mode == "Ask Knowledge Base":
    mode_hero("💬", "Ask Knowledge Base",
              "Query your uploaded documents using AI-powered semantic search and retrieval.",
              "Ask")

    chat_col, info_col = st.columns([3, 1])

    with info_col:
        st.markdown("""
        <div class="pl-card">
            <div class="pl-card-title">How it works</div>
            <div style="font-size:0.82rem;color:#9BB7D4;line-height:1.8;">
                Your question is embedded and matched against uploaded documents using cosine similarity.
                The top chunks are sent to GPT-4.1 as context.
            </div>
            <div style="margin-top:16px;" class="pl-card-title">Tips</div>
            <div style="font-size:0.78rem;color:#9BB7D4;margin-top:6px;line-height:1.9;">
                • Be specific in your questions<br>
                • Reference product types<br>
                • Ask follow-up questions<br>
                • Check sources for citations
            </div>
            <div style="margin-top:16px;font-size:0.68rem;color:#757577;border-top:1px solid rgba(254,247,207,0.1);padding-top:12px;">
                ⌘ Enter &nbsp;·&nbsp; Submit message
            </div>
        </div>
        """, unsafe_allow_html=True)

    with chat_col:
        if not st.session_state.messages:
            empty_state("💬", "No conversation yet",
                        "Ask a question about baking science, fermentation, or pastry techniques.")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        question = st.chat_input("Ask about baking, fermentation, pastry science…  (⌘ Enter to send)")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            context, metadatas = retrieve_context(question, n_results=5)
            recent_chat_history = st.session_state.messages[-6:]

            messages_payload = [{"role": "system", "content": """
You are a professional baking and pastry science assistant.
Use the provided context to answer the user's question.
You may use recent chat history to understand follow-up questions.
If the answer is not supported by the provided context, say: 'I could not find enough information in the knowledge base.'
At the end of every answer, include a section called "Sources used".
List the source number, document title, page number, chunk number, and extraction method.
"""}]
            for msg in recent_chat_history:
                messages_payload.append(msg)
            messages_payload.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"})

            with st.spinner("Retrieving context and generating answer..."):
                response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages_payload)

            answer = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": answer})
            sources_text = format_sources(metadatas)

            append_history("ask_history", {
                "timestamp": datetime.now().isoformat(),
                "question": question, "answer": answer, "sources": sources_text
            })

            with st.chat_message("assistant"):
                st.write(answer)
                with st.expander("📎 Retrieved sources"):
                    st.text(sources_text)

            st.toast("Answer generated.", icon="💬")

    show_history("Ask History", "ask_history", ASK_COLUMNS, "ask_history.csv")


# ─────────────────────────────────────────────
# Mode: SOP Creator
# ─────────────────────────────────────────────
if mode == "SOP Creator":
    mode_hero("📋", "SOP Creator",
              "Generate professional, staff-ready Standard Operating Procedures from rough notes.",
              "SOP")

    left, right = st.columns([2, 1])

    with right:
        st.markdown("""
        <div class="pl-card">
            <div class="pl-card-title">SOP Structure</div>
            <div style="font-size:0.78rem;color:#9BB7D4;line-height:2.1;">
                1. SOP Title<br>2. Purpose<br>3. Product Overview<br>
                4. Required Equipment<br>5. Ingredients / Components<br>
                6. Mise en Place<br>7. Production Procedure<br>
                8. Proofing / Baking Parameters<br>9. Quality Control<br>
                10. Common Defects<br>11. Storage / Shelf Life<br>
                12. Food Safety Notes<br>13. Staff Training Notes<br>
                14. Final Approval Checklist
            </div>
        </div>
        """, unsafe_allow_html=True)

    with left:
        product_name = st.text_input("Product / SOP Name", placeholder="Example: The BUN — Cardamom")
        sop_notes    = st.text_area("Paste rough notes, recipe, process, or staff instructions", height=220,
                                    placeholder="Example: Mix dough 8 min, bulk 45 min, shape, proof until puffy, bake at 180°C…")
        st.markdown("<div class='input-hint'>⌘ Enter &nbsp;·&nbsp; New line &nbsp;·&nbsp; Shift+Enter for paragraph break</div>", unsafe_allow_html=True)
        use_knowledge_base = st.checkbox("Use knowledge base for technical support", value=True)

        if st.button("📋 Generate SOP", use_container_width=True):
            if not sop_notes.strip():
                st.toast("Please enter rough notes first.", icon="⚠️")
            else:
                context   = ""
                metadatas = []
                if use_knowledge_base:
                    context, metadatas = retrieve_context(
                        f"Technical support for SOP: {product_name}. {sop_notes}", n_results=5)

                with st.spinner("Generating professional SOP…"):
                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": """
You are The Proof Lab's senior bakery operations writer.
Create a professional, staff-ready bakery SOP.
Structure: 1. SOP Title, 2. Purpose, 3. Product Overview, 4. Required Equipment, 5. Ingredients / Components,
6. Mise en Place, 7. Production Procedure, 8. Proofing / Resting / Baking Parameters, 9. Quality Control Checkpoints,
10. Common Defects + Corrections, 11. Storage / Holding / Shelf Life, 12. Food Safety Notes, 13. Staff Training Notes, 14. Final Approval Checklist.
Tone: professional, precise, clear for staff, aligned with The Proof Lab: experimental, process-driven, premium, but practical.
If information is missing, add reasonable placeholders marked as [TO CONFIRM].
"""},
                            {"role": "user", "content": f"Product / SOP Name:\n{product_name}\n\nUser Notes:\n{sop_notes}\n\nKnowledge Base Context:\n{context}"}
                        ]
                    )

                sop_output   = response.choices[0].message.content
                sources_text = format_sources(metadatas)
                append_history("sop_history", {
                    "timestamp": datetime.now().isoformat(),
                    "product_name": product_name, "user_notes": sop_notes,
                    "generated_sop": sop_output, "sources": sources_text
                })

                pl_divider("Generated SOP")
                st.markdown(sop_output)
                st.download_button(label="⬇ Download SOP as Markdown",
                                   data=sop_output,
                                   file_name=f"{product_name or 'proof_lab_sop'}.md",
                                   mime="text/markdown")
                if use_knowledge_base and metadatas:
                    with st.expander("📎 Knowledge sources used"):
                        st.text(sources_text)
                st.toast("SOP generated successfully.", icon="📋")

    show_history("SOP History", "sop_history", SOP_COLUMNS, "sop_history.csv")


# ─────────────────────────────────────────────
# Mode: Batch Tracker
# ─────────────────────────────────────────────
if mode == "Batch Tracker":
    mode_hero("📊", "Batch Tracker",
              "Log every production batch with full process parameters and AI-powered analysis.",
              "Batch")

    batch_df_all    = load_batches()
    total_batches   = len(batch_df_all)
    last_product    = batch_df_all["product"].iloc[-1] if total_batches > 0 else "—"
    last_bake_temp  = batch_df_all["bake_temp"].iloc[-1] if total_batches > 0 else "—"

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-value">{total_batches}</div>
            <div class="metric-label">Total Batches</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{last_product}</div>
            <div class="metric-label">Last Product</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{last_bake_temp}</div>
            <div class="metric-label">Last Bake Temp</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pl_divider("Log a New Batch")

    with st.form("batch_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div style='font-size:0.68rem;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Batch Info</div>", unsafe_allow_html=True)
            batch_name = st.text_input("Batch Name", placeholder="Croissant Batch 6")
            product    = st.text_input("Product", placeholder="Croissant / Bun / Brownie")
            batch_date = st.date_input("Batch Date")

        with col2:
            st.markdown("<div style='font-size:0.68rem;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Temperature Log</div>", unsafe_allow_html=True)
            dough_temp  = st.text_input("Dough Temp",  placeholder="8°C")
            butter_temp = st.text_input("Butter Temp", placeholder="13°C")
            room_temp   = st.text_input("Room Temp",   placeholder="21°C")

        with col3:
            st.markdown("<div style='font-size:0.68rem;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Process Parameters</div>", unsafe_allow_html=True)
            proof_temp = st.text_input("Proof Temp", placeholder="26°C")
            proof_time = st.text_input("Proof Time", placeholder="2.5 hr")
            bake_temp  = st.text_input("Bake Temp",  placeholder="180°C")
            bake_time  = st.text_input("Bake Time",  placeholder="18 min")

        st.markdown("<div style='height:1px;background:rgba(254,247,207,0.1);margin:12px 0;'></div>", unsafe_allow_html=True)
        note_col1, note_col2 = st.columns(2)
        with note_col1:
            formula_notes = st.text_area("Formula Notes",  height=100)
            process_notes = st.text_area("Process Notes",  height=100)
        with note_col2:
            result = st.text_area("Result",           height=100)
            issues = st.text_area("Issues / Defects", height=100)

        next_adjustment = st.text_area("Next Adjustment", height=80)
        submitted = st.form_submit_button("💾 Save Batch", use_container_width=True)

        if submitted:
            save_batch({
                "timestamp": datetime.now().isoformat(),
                "batch_name": batch_name, "product": product, "batch_date": str(batch_date),
                "formula_notes": formula_notes, "process_notes": process_notes,
                "dough_temp": dough_temp, "butter_temp": butter_temp, "room_temp": room_temp,
                "proof_temp": proof_temp, "proof_time": proof_time,
                "bake_temp": bake_temp, "bake_time": bake_time,
                "result": result, "issues": issues, "next_adjustment": next_adjustment
            })
            st.toast(f"Batch '{batch_name}' saved successfully.", icon="💾")
            st.rerun()

    pl_divider("Saved Batches")
    batch_df = load_batches()

    if batch_df.empty:
        empty_state("📊", "No batches logged yet", "Fill in the form above to log your first production batch.")
    else:
        st.dataframe(batch_df, use_container_width=True)
        st.download_button(label="⬇ Download Batch Tracker CSV",
                           data=batch_df.to_csv(index=False),
                           file_name="batch_tracker.csv", mime="text/csv")

        pl_divider("AI Batch Analysis")
        analysis_question = st.text_input("Ask about your saved batches",
                                          placeholder="Example: Why did butter leak in Batch 6?")
        st.markdown("<div class='input-hint'>Ask any process or quality question about your logged batches</div>", unsafe_allow_html=True)

        if st.button("🔍 Analyze Batch Data"):
            if not analysis_question.strip():
                st.toast("Please enter a batch analysis question.", icon="⚠️")
            else:
                with st.spinner("Analyzing batch data with knowledge base support…"):
                    batch_context         = batch_df.to_string(index=False)
                    kb_context, metadatas = retrieve_context(
                        f"Technical baking support for: {analysis_question}", n_results=5)
                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": """
You are The Proof Lab's R&D bakery analyst.
Analyze the saved batch tracker data. Use the knowledge base context for technical support.
Your response should include: 1. Direct answer, 2. Likely causes, 3. Pattern across batches if visible,
4. Recommended next adjustment, 5. What to track in the next batch.
Be practical and specific. If the data is insufficient, say what is missing.
"""},
                            {"role": "user", "content": f"Batch Tracker Data:\n{batch_context}\n\nKnowledge Base Context:\n{kb_context}\n\nQuestion:\n{analysis_question}"}
                        ]
                    )
                pl_divider("Batch Analysis")
                st.markdown(response.choices[0].message.content)
                st.toast("Batch analysis complete.", icon="📊")


# ─────────────────────────────────────────────
# Mode: Vision Analyzer
# ─────────────────────────────────────────────
if mode == "Vision Analyzer":
    mode_hero("🔬", "Vision Analyzer",
              "Upload a photo of your baked product for AI-powered visual diagnosis and corrective guidance.",
              "Vision")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("<div class='input-hint'>Supports PNG, JPG, JPEG</div>", unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload image", type=["png","jpg","jpeg"], label_visibility="collapsed")
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

        product_type = st.selectbox("Product type",
            ["Croissant","Laminated dough","Bread crumb","Bun","Brownie","Cookie","Other pastry"])
        notes = st.text_area("Optional batch notes", height=120,
                             placeholder="Example: Hydration 48%, butter temp 13°C, proofed at 26°C for 2.5 hours…")
        use_knowledge_base_for_vision = st.checkbox("Use knowledge base for technical support", value=True)
        analyze_btn = st.button("🔬 Analyze Image", use_container_width=True, disabled=not uploaded_image)

    with right_col:
        if not uploaded_image:
            st.markdown("""
            <div class="empty-state" style="min-height:380px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;">
                <div class="empty-icon">🔬</div>
                <div class="empty-title">Upload an image to analyze</div>
                <div class="empty-sub">Supports PNG, JPG, JPEG · Max 200 MB</div>
            </div>
            """, unsafe_allow_html=True)
        elif analyze_btn:
            base64_image = encode_image(uploaded_image)
            kb_context   = ""
            metadatas    = []

            if use_knowledge_base_for_vision:
                kb_context, metadatas = retrieve_context(
                    f"Technical support for visual diagnosis of {product_type}. Notes: {notes}. "
                    "Analyze possible defects such as underproofing, overproofing, weak gluten, butter leakage, "
                    "lamination breakage, shaping issues, dense crumb, tunneling, poor oven spring.",
                    n_results=5
                )

            with st.spinner("Analyzing image — this may take a moment…"):
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": """
You are The Proof Lab's bakery visual diagnostic assistant.
Analyze the uploaded image carefully.
Focus on: crumb structure, fermentation, proofing, gluten strength, lamination quality, butter distribution,
shaping, bake color, oven spring, signs of underproofing or overproofing, likely process causes, next batch adjustments.
Be practical, specific, and honest. If the image is unclear, say what cannot be determined visually.
Use knowledge base context only to support technical reasoning.
"""},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"Product type:\n{product_type}\n\nBatch notes:\n{notes}\n\nKnowledge base context:\n{kb_context}\n\nPlease provide:\n1. Visual observations\n2. Likely diagnosis\n3. Most probable causes\n4. Corrective actions\n5. What to track in the next batch\n6. Confidence level"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}
                    ]
                )

            diagnosis    = response.choices[0].message.content
            sources_text = format_sources(metadatas)
            append_history("vision_history", {
                "timestamp": datetime.now().isoformat(),
                "image_name": uploaded_image.name, "product_type": product_type,
                "batch_notes": notes, "diagnosis": diagnosis, "sources": sources_text
            })

            pl_divider("Vision Diagnosis")
            st.markdown(diagnosis)
            if use_knowledge_base_for_vision and metadatas:
                with st.expander("📎 Knowledge sources used"):
                    st.text(sources_text)
            st.toast("Vision analysis complete.", icon="🔬")

    show_history("Vision History", "vision_history", VISION_COLUMNS, "vision_history.csv")


# ─────────────────────────────────────────────
# Mode: Recipe R&D Generator
# ─────────────────────────────────────────────
if mode == "Recipe R&D Generator":
    mode_hero("⚗️", "Recipe R&D Generator",
              "Invent technically original bakery concepts based on flavor direction, texture goals, and brand mood.",
              "R&D")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='pl-card-title'>Product Parameters</div>", unsafe_allow_html=True)
        product_type     = st.selectbox("Product Type",
            ["Croissant","Cookie","Brownie","Bun","Sourdough","Dessert","Other"])
        flavor_direction = st.text_input("Flavor Direction",
            placeholder="Example: mango + tajin + brown butter")
        texture_goal     = st.text_input("Texture Goal",
            placeholder="Example: crispy shell with soft center")
        batch_size       = st.selectbox("Test Batch Size",
            ["6 pieces","12 pieces","24 pieces","1 tray","Small R&D batch"])

    with col2:
        st.markdown("<div class='pl-card-title'>Creative Direction</div>", unsafe_allow_html=True)
        constraints = st.text_area("Constraints", height=100,
            placeholder="Example: delivery stable, freezer stable, no wet toppings, same-day bake")
        brand_mood  = st.text_input("Brand Mood / Feeling",
            placeholder="Example: unexpected luxury convenience store")
        use_kb      = st.checkbox("Use knowledge base for technical support", value=True)

    if st.button("⚗️ Generate R&D Concept", use_container_width=True):
        if not flavor_direction.strip():
            st.toast("Please enter a flavor direction.", icon="⚠️")
        else:
            kb_context = ""
            metadatas  = []
            if use_kb:
                kb_context, metadatas = retrieve_context(
                    f"Generate a technically strong {product_type} concept. Flavor: {flavor_direction}. "
                    f"Texture: {texture_goal}. Constraints: {constraints}. Brand: {brand_mood}.",
                    n_results=5
                )

            with st.spinner("Generating R&D concept — thinking deeply…"):
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": """
You are The Proof Lab's senior culinary R&D strategist.
Your job is to help invent highly original bakery concepts. Do NOT generate generic recipes.
Focus on: flavor architecture, sensory contrast, texture engineering, technical feasibility, bakery science, operational practicality, Proof Lab brand alignment.
The concept should feel: experimental, premium, memorable, technically intentional.
Structure the response as:
1. Concept Name, 2. Concept Summary, 3. Why It Fits The Proof Lab, 4. Flavor Architecture, 5. Texture Architecture,
6. Draft Test Formula, 7. Suggested Components, 8. Technical Risks, 9. Stability Concerns, 10. First Test Plan,
11. Variables To Monitor, 12. Suggested Next Iteration,
13. Confidence Scoring (Flavor potential /10, Texture feasibility /10, Technical difficulty /10, Delivery stability /10, Repeatability /10, Brand fit /10, Overall R&D confidence /10).
Be highly specific. Avoid generic pastry ideas.
"""},
                        {"role": "user", "content": f"Product Type:\n{product_type}\n\nFlavor Direction:\n{flavor_direction}\n\nTexture Goal:\n{texture_goal}\n\nConstraints:\n{constraints}\n\nBrand Mood:\n{brand_mood}\n\nTest Batch Size:\n{batch_size}\n\nKnowledge Base Context:\n{kb_context}"}
                    ]
                )

            output       = response.choices[0].message.content
            sources_text = format_sources(metadatas)
            append_history("rd_history", {
                "timestamp": datetime.now().isoformat(),
                "product_type": product_type, "flavor_direction": flavor_direction,
                "texture_goal": texture_goal, "constraints": constraints,
                "brand_mood": brand_mood, "batch_size": batch_size,
                "generated_concept": output, "sources": sources_text
            })

            pl_divider("Generated R&D Concept")
            st.markdown(output)
            st.download_button(label="⬇ Download R&D Concept",
                               data=output, file_name="proof_lab_rd_concept.md", mime="text/markdown")
            if use_kb and metadatas:
                with st.expander("📎 Knowledge sources used"):
                    st.text(sources_text)
            st.toast("R&D concept generated.", icon="⚗️")

    show_history("R&D History", "rd_history", RD_COLUMNS, "rd_history.csv")


# ─────────────────────────────────────────────
# Mode: Recipe Evaluator
# ─────────────────────────────────────────────
if mode == "Recipe Evaluator":
    mode_hero("✅", "Recipe Evaluator",
              "Critically evaluate test recipes for formula balance, texture feasibility, and Proof Lab brand fit.",
              "Evaluate")

    col1, col2 = st.columns(2)
    with col1:
        recipe_name  = st.text_input("Recipe Name", placeholder="Example: Mango Tango Cookie V1")
        product_type = st.selectbox("Product Type",
            ["Croissant","Cookie","Brownie","Bun","Sourdough","Dessert","Other"])
    with col2:
        target_outcome = st.text_area("Target Outcome", height=108,
            placeholder="Example: chewy center, crisp edge, stable for delivery, strong mango aroma")

    recipe_text = st.text_area(
        "Paste Recipe / Formula / Process Notes", height=280,
        placeholder="Example:\nButter 120g\nBrown sugar 80g\nCaster sugar 40g\nEgg 50g\nFlour 180g\nBake 180°C for 12 minutes…"
    )
    st.markdown("<div class='input-hint'>Paste your full formula including weights, process steps, and baking parameters</div>", unsafe_allow_html=True)
    use_kb_for_evaluation = st.checkbox("Use knowledge base for evaluation", value=True)

    if st.button("✅ Evaluate Recipe", use_container_width=True):
        if not recipe_text.strip():
            st.toast("Please paste a recipe first.", icon="⚠️")
        else:
            kb_context = ""
            metadatas  = []
            if use_kb_for_evaluation:
                kb_context, metadatas = retrieve_context(
                    f"Evaluate this {product_type} recipe. Name: {recipe_name}. "
                    f"Target: {target_outcome}. Recipe: {recipe_text}.",
                    n_results=5
                )

            with st.spinner("Evaluating recipe — running technical analysis…"):
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": """
You are The Proof Lab's senior recipe evaluator and bakery R&D analyst.
Your job is to evaluate test recipes critically and constructively.
Do NOT simply compliment the recipe. Do NOT rewrite it as a final SOP unless asked.
Your role is to diagnose whether the recipe is technically sound, interesting, testable, and aligned with The Proof Lab.
Use the user's recipe as the main source. Use the knowledge base context to support technical reasoning.
Structure the response as:
1. Executive Verdict (Is this recipe worth testing? Yes / No / Yes, but revise first)
2. Recipe Summary
3. Formula Balance Review (flour/starch, hydration, fat, sugar, egg/liquid, leavening, salt, inclusions)
4. Process Review (mixing, resting/chilling/proofing, shaping, baking, cooling/storage)
5. Texture Feasibility
6. Flavor Logic (sweetness, acidity, salt, fat, bitterness, aroma, contrast)
7. Stability & Delivery Risk
8. Technical Risks
9. Proof Lab Brand Fit
10. Scorecard (/10 for: Flavor potential, Texture feasibility, Formula balance, Process clarity, Stability, Production practicality, Brand fit, Overall R&D potential)
11. Recommended Adjustments
12. First Test Plan
13. Variables To Track
14. Final Recommendation (test as is / revise then test / reject for now / split into two experiments)
Be direct, practical, and technically specific. If the recipe is missing key data, clearly list what is missing.
"""},
                        {"role": "user", "content": f"Recipe Name:\n{recipe_name}\n\nProduct Type:\n{product_type}\n\nTarget Outcome:\n{target_outcome}\n\nRecipe / Formula / Process:\n{recipe_text}\n\nKnowledge Base Context:\n{kb_context}"}
                    ]
                )

            evaluation   = response.choices[0].message.content
            sources_text = format_sources(metadatas)

            pl_divider("Recipe Evaluation")
            st.markdown(evaluation)
            st.download_button(label="⬇ Download Recipe Evaluation",
                               data=evaluation,
                               file_name=f"{recipe_name or 'recipe_evaluation'}.md",
                               mime="text/markdown")
            if use_kb_for_evaluation and metadatas:
                with st.expander("📎 Knowledge sources used"):
                    st.text(sources_text)
            st.toast("Recipe evaluation complete.", icon="✅")


# ─────────────────────────────────────────────
# Mode: Data Migration
# ─────────────────────────────────────────────
if mode == "Data Migration":
    import time as _mig_time

    mode_hero("🚚", "Data Migration",
              "Restore all your history data and rebuild the knowledge base from your local files in one go.",
              "Migrate")

    st.markdown("""
    <div class="pl-card" style="margin-bottom:18px;">
        <div class="pl-card-title">How Migration Works</div>
        <div style="font-size:0.83rem;color:#9BB7D4;line-height:2.0;">
            <b style="color:#FEF7CF;">Step 1 — History CSVs</b> &nbsp;—&nbsp;
                Upload your 4 history CSV files. Records are merged into Supabase instantly.<br>
            <b style="color:#FEF7CF;">Step 2 — Knowledge Base PDFs</b> &nbsp;—&nbsp;
                Upload all your PDFs at once. Each is processed, chunked, embedded, and stored in Supabase.<br>
            <b style="color:#FEF7CF;">Persistent storage</b> &nbsp;—&nbsp;
                All data is stored in Supabase — it will never be lost on app reboot or redeployment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: History CSVs ──
    pl_divider("Step 1 — Restore History Data")

    st.markdown("""
    <div style="font-size:0.82rem;color:#9BB7D4;margin-bottom:12px;">
        Upload your CSV history files. Expected files:
        <code style="color:#FEF7CF;">ask_history.csv</code>,
        <code style="color:#FEF7CF;">sop_history.csv</code>,
        <code style="color:#FEF7CF;">rd_history.csv</code>,
        <code style="color:#FEF7CF;">vision_history.csv</code>
    </div>
    """, unsafe_allow_html=True)

    csv_files = st.file_uploader("Upload CSV history files", type="csv",
                                  accept_multiple_files=True, key="migration_csv")

    CSV_TABLE_MAP = {
        "ask_history.csv":    ("ask_history",    ASK_COLUMNS),
        "sop_history.csv":    ("sop_history",    SOP_COLUMNS),
        "rd_history.csv":     ("rd_history",     RD_COLUMNS),
        "vision_history.csv": ("vision_history", VISION_COLUMNS),
    }

    if csv_files:
        if st.button("📥 Restore History Data", use_container_width=True):
            sb = get_supabase()
            restore_results = []
            for csv_file in csv_files:
                fname = csv_file.name
                if fname in CSV_TABLE_MAP:
                    table_name, columns = CSV_TABLE_MAP[fname]
                    try:
                        df = pd.read_csv(csv_file)
                        # Keep only valid columns
                        valid_cols = [c for c in columns if c in df.columns]
                        df = df[valid_cols].fillna("")
                        records = df.to_dict(orient="records")
                        # Insert in batches of 100
                        inserted = 0
                        for i in range(0, len(records), 100):
                            batch = records[i:i+100]
                            sb.table(table_name).insert(batch).execute()
                            inserted += len(batch)
                        restore_results.append({"File": fname, "Table": table_name, "Records": inserted, "Status": "✅ Done"})
                    except Exception as e:
                        restore_results.append({"File": fname, "Table": table_name, "Records": 0, "Status": f"❌ {str(e)[:60]}"})
                else:
                    restore_results.append({"File": fname, "Table": "—", "Records": 0, "Status": "⚠️ Unknown file"})

            total_restored = sum(r["Records"] for r in restore_results)
            st.toast(f"✅ History restored — {total_restored} records imported!", icon="📥")
            st.dataframe(pd.DataFrame(restore_results), use_container_width=True, hide_index=True)

    # ── Step 2: PDF Knowledge Base ──
    pl_divider("Step 2 — Rebuild Knowledge Base")

    st.markdown("""
    <div style="font-size:0.82rem;color:#9BB7D4;margin-bottom:12px;">
        Upload all your PDF documents. They will be processed, chunked, embedded, and stored permanently in Supabase.
    </div>
    """, unsafe_allow_html=True)

    mig_force_ocr  = st.checkbox("Force OCR for all files (for scanned PDFs)", key="mig_ocr")
    mig_pdf_files  = st.file_uploader("Upload PDF documents", type="pdf",
                                       accept_multiple_files=True, key="migration_pdf")

    if mig_pdf_files:
        total_size_mb = sum(f.size for f in mig_pdf_files) / (1024 * 1024)
        est_chunks_total = 0
        for f in mig_pdf_files:
            try:
                reader_est = PdfReader(f)
                sample_text = ""
                for pg in reader_est.pages[:min(3, len(reader_est.pages))]:
                    t = pg.extract_text()
                    if t:
                        sample_text += t
                avg_chars = len(sample_text) / min(3, len(reader_est.pages)) if reader_est.pages else 0
                est_chunks_total += max(1, int((avg_chars * len(reader_est.pages)) / 1000))
                f.seek(0)
            except Exception:
                est_chunks_total += 30
        est_seconds = max(15, int(est_chunks_total * 0.55))
        est_min = est_seconds // 60
        est_sec = est_seconds % 60
        est_label = f"{est_min}m {est_sec}s" if est_min > 0 else f"{est_sec}s"

        st.markdown(f"""
        <div style="background:#1a0f20;border:1px solid rgba(155,183,212,0.22);border-radius:10px;
                    padding:14px 18px;margin:10px 0 14px 0;display:flex;gap:24px;align-items:center;">
            <div style="text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#FEF7CF;">{len(mig_pdf_files)}</div>
                <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Files</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#FEF7CF;">{total_size_mb:.1f} MB</div>
                <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Total Size</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#FEF7CF;">~{est_chunks_total}</div>
                <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Est. Chunks</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#9BB7D4;">~{est_label}</div>
                <div style="font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1px;">Est. Time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"🚀 Start Migration ({len(mig_pdf_files)} PDFs)", use_container_width=True):
            start_time   = _mig_time.time()
            progress_bar = st.progress(0, text="Starting migration…")
            mig_results  = []

            for idx, pdf_file in enumerate(mig_pdf_files):
                progress_pct = int((idx / len(mig_pdf_files)) * 100)
                elapsed      = _mig_time.time() - start_time
                remaining    = max(0, est_seconds - int(elapsed))
                rem_min      = remaining // 60
                rem_sec      = remaining % 60
                rem_label    = f"{rem_min}m {rem_sec}s remaining" if rem_min > 0 else f"{rem_sec}s remaining"
                progress_bar.progress(progress_pct,
                    text=f"Migrating {pdf_file.name} ({idx+1}/{len(mig_pdf_files)}) — ~{rem_label}")

                extracted_pages   = []
                page_count        = 0
                extraction_method = "pypdf"

                if not mig_force_ocr:
                    try:
                        extracted_pages, page_count = extract_text_with_pypdf(pdf_file)
                    except Exception:
                        extracted_pages = []

                total_text_length = sum(len(p["text"]) for p in extracted_pages)

                if mig_force_ocr or total_text_length < 500:
                    if not llama_api_key:
                        st.error("Missing LLAMA_CLOUD_API_KEY in your secrets.")
                        st.stop()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(pdf_file.getbuffer())
                        temp_path = temp_file.name
                    extracted_pages   = extract_text_with_llamaparse(temp_path)
                    page_count        = "OCR"
                    extraction_method = "llamaparse_ocr"

                added_chunks, skipped_chunks = add_chunks_to_store(
                    pdf_file.name, extracted_pages, page_count, extraction_method
                )
                mig_results.append({"File": pdf_file.name, "Method": extraction_method,
                                     "Added": added_chunks, "Skipped": skipped_chunks})

            actual_time  = _mig_time.time() - start_time
            actual_min   = int(actual_time) // 60
            actual_sec   = int(actual_time) % 60
            actual_label = f"{actual_min}m {actual_sec}s" if actual_min > 0 else f"{actual_sec}s"

            mig_df      = pd.DataFrame(mig_results)
            total_added = mig_df["Added"].sum()
            total_skip  = mig_df["Skipped"].sum()

            progress_bar.progress(100, text=f"✅ Migration complete in {actual_label}!")
            st.toast(f"✅ Migration done — {total_added} chunks stored in Supabase!", icon="🚀")

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(100,200,120,0.12),rgba(44,19,50,0.7));
                        border:2px solid rgba(100,200,120,0.5);border-radius:14px;
                        padding:24px 28px;margin:20px 0;">
                <div style="font-size:1.2rem;font-weight:800;color:#a8f0b8;margin-bottom:10px;">🎉 Migration Complete!</div>
                <div style="display:flex;gap:24px;flex-wrap:wrap;margin:14px 0;">
                    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px 18px;text-align:center;">
                        <div style="font-size:1.5rem;font-weight:800;color:#FEF7CF;">{len(mig_pdf_files)}</div>
                        <div style="font-size:0.75rem;color:#9BB7D4;">Files Migrated</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px 18px;text-align:center;">
                        <div style="font-size:1.5rem;font-weight:800;color:#a8f0b8;">{total_added}</div>
                        <div style="font-size:0.75rem;color:#9BB7D4;">Chunks in Supabase</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px 18px;text-align:center;">
                        <div style="font-size:1.5rem;font-weight:800;color:#757577;">{total_skip}</div>
                        <div style="font-size:0.75rem;color:#9BB7D4;">Duplicates Skipped</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px 18px;text-align:center;">
                        <div style="font-size:1.5rem;font-weight:800;color:#9BB7D4;">{actual_label}</div>
                        <div style="font-size:0.75rem;color:#9BB7D4;">Total Time</div>
                    </div>
                </div>
                <div style="font-size:0.8rem;color:#a8f0b8;">
                    All data is now permanently stored in Supabase. Your knowledge base will persist across all app reboots.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(mig_df, use_container_width=True, hide_index=True)
