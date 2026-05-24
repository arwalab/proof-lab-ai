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

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# ─────────────────────────────────────────────
# Pure-Python Vector Store (replaces chromadb)
# ─────────────────────────────────────────────

VECTOR_DB_FILE = Path("vector_store.json")

def load_vector_store():
    if VECTOR_DB_FILE.exists():
        with open(VECTOR_DB_FILE, "r") as f:
            return json.load(f)
    return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

def save_vector_store(store):
    with open(VECTOR_DB_FILE, "w") as f:
        json.dump(store, f)

def vector_store_add(store, chunk_id, embedding, document, metadata):
    if chunk_id in store["ids"]:
        return False
    store["ids"].append(chunk_id)
    store["embeddings"].append(embedding)
    store["documents"].append(document)
    store["metadatas"].append(metadata)
    save_vector_store(store)
    return True

def vector_store_query(store, query_embedding, n_results=5):
    if not store["embeddings"]:
        return [], []
    embeddings = np.array(store["embeddings"], dtype=np.float32)
    query = np.array(query_embedding, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query)
    norms = np.where(norms == 0, 1e-10, norms)
    similarities = np.dot(embeddings, query) / norms
    top_indices = np.argsort(similarities)[::-1][:n_results]
    docs = [store["documents"][i] for i in top_indices]
    metas = [store["metadatas"][i] for i in top_indices]
    return docs, metas

def vector_store_get_all(store):
    return store["metadatas"]

def vector_store_delete(store, document_title):
    indices_to_keep = [i for i, m in enumerate(store["metadatas"]) if m.get("document_title") != document_title]
    deleted = len(store["ids"]) - len(indices_to_keep)
    store["ids"] = [store["ids"][i] for i in indices_to_keep]
    store["embeddings"] = [store["embeddings"][i] for i in indices_to_keep]
    store["documents"] = [store["documents"][i] for i in indices_to_keep]
    store["metadatas"] = [store["metadatas"][i] for i in indices_to_keep]
    save_vector_store(store)
    return deleted

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()

ASK_HISTORY_FILE  = Path("ask_history.csv")
SOP_HISTORY_FILE  = Path("sop_history.csv")
BATCH_FILE        = Path("batch_tracker.csv")
VISION_HISTORY_FILE = Path("vision_history.csv")
RD_HISTORY_FILE   = Path("rd_history.csv")

# ─────────────────────────────────────────────
# Page config — dynamic title per mode
# ─────────────────────────────────────────────

MODE_ICONS = {
    "Ask Knowledge Base":   "💬",
    "SOP Creator":          "📋",
    "Batch Tracker":        "📊",
    "Vision Analyzer":      "🔬",
    "Recipe R&D Generator": "⚗️",
    "Recipe Evaluator":     "✅",
}

# Read mode from session state early so page_config can use it
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Ask Knowledge Base"

st.set_page_config(
    page_title=f"Proof Lab AI — {st.session_state.active_mode}",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Global CSS — all improvements bundled
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

/* ── Design tokens ── */
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

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Custom logo SVG ── */
.pl-logo-svg {
    width: 52px; height: 52px; flex-shrink: 0;
    filter: drop-shadow(0 0 12px rgba(155,183,212,0.5));
}

/* ── Header ── */
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
.proof-header .tagline {
    font-size: 0.72rem; color: var(--text-muted);
    letter-spacing: 2px; text-transform: uppercase; margin-top: 3px;
}

/* ── Mode hero banner ── */
.mode-hero {
    margin: 0 0 28px 0;
    padding: 22px 28px;
    border-radius: 0 0 var(--radius) var(--radius);
    border: 1px solid var(--border);
    border-top: none;
    display: flex; align-items: center; gap: 16px;
    position: relative; overflow: hidden;
}
.mode-hero::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(155,183,212,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.mode-hero .hero-icon {
    font-size: 2.4rem; flex-shrink: 0;
    filter: drop-shadow(0 0 8px rgba(155,183,212,0.4));
}
.mode-hero .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem; font-weight: 700;
    color: var(--accent); margin: 0;
}
.mode-hero .hero-desc {
    font-size: 0.8rem; color: var(--text-muted);
    margin-top: 3px; line-height: 1.5;
}
.mode-hero .hero-bg-text {
    position: absolute; right: 24px; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem; opacity: 0.04;
    font-family: 'Playfair Display', serif;
    font-weight: 700; color: #9BB7D4;
    pointer-events: none; user-select: none;
    white-space: nowrap;
}

/* ── Sidebar nav buttons ── */
.nav-btn {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 8px;
    cursor: pointer; margin-bottom: 4px;
    font-size: 0.85rem; font-weight: 500;
    color: var(--text-muted);
    border: 1px solid transparent;
    transition: all 0.15s ease;
    text-decoration: none;
}
.nav-btn:hover {
    background: var(--accent-dim);
    color: var(--accent-light);
    border-color: var(--border);
}
.nav-btn.active {
    background: var(--accent-dim);
    color: var(--accent);
    border-color: var(--border);
    font-weight: 600;
}
.nav-btn .nav-icon { font-size: 1.1rem; width: 22px; text-align: center; }

/* ── Cards ── */
.pl-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 22px;
    margin-bottom: 18px; box-shadow: var(--shadow);
}
.pl-card-title {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: var(--accent); margin-bottom: 14px;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 20px; }
.metric-card {
    flex: 1; background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px 18px;
    box-shadow: var(--shadow);
}
.metric-card .metric-value {
    font-size: 1.8rem; font-weight: 700;
    color: var(--accent); line-height: 1;
}
.metric-card .metric-label {
    font-size: 0.65rem; color: var(--text-faint);
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-top: 5px;
}
.metric-card .metric-delta {
    font-size: 0.72rem; color: #5cb85c;
    margin-top: 3px;
}

/* ── Section divider ── */
.pl-divider {
    display: flex; align-items: center; gap: 12px;
    margin: 28px 0 20px 0;
}
.pl-divider::before, .pl-divider::after {
    content: ''; flex: 1;
    height: 1px; background: var(--border);
}
.pl-divider span {
    font-size: 0.62rem; color: var(--text-faint);
    text-transform: uppercase; letter-spacing: 2px;
    white-space: nowrap;
}

/* ── Empty state ── */
.empty-state {
    background: var(--bg-card);
    border: 1.5px dashed var(--border);
    border-radius: var(--radius);
    padding: 52px 40px;
    text-align: center;
}
.empty-state .empty-icon { font-size: 2.8rem; margin-bottom: 14px; }
.empty-state .empty-title {
    font-size: 1rem; color: var(--text-muted);
    font-weight: 500; margin-bottom: 6px;
}
.empty-state .empty-sub { font-size: 0.78rem; color: var(--text-faint); }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #2a1a35 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #9BB7D4 !important;
    box-shadow: 0 0 0 2px rgba(254,247,207,0.15) !important;
    outline: none !important;
}
.input-hint {
    font-size: 0.68rem; color: var(--text-faint);
    text-align: right; margin-top: -10px; margin-bottom: 8px;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background-color: #2a1a35 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #9BB7D4, #5a7fa8) !important;
    color: #1a0f20 !important; font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important; border-radius: 8px !important;
    padding: 10px 24px !important; font-size: 0.88rem !important;
    letter-spacing: 0.4px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 14px rgba(155,183,212,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 22px rgba(155,183,212,0.38) !important;
}
.stDownloadButton > button {
    background: transparent !important; color: #9BB7D4 !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background: var(--accent-dim) !important; border-color: #9BB7D4 !important;
}
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #9BB7D4, #5a7fa8) !important;
    color: #1a0f20 !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
    width: 100% !important; padding: 12px !important;
    font-size: 0.92rem !important;
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 10px !important;
}
[data-testid="stChatInput"] > div {
    background: #2a1a35 !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea { color: var(--text-primary) !important; }

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: #261030 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
}
.streamlit-expanderContent {
    background: #220e2e !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── DataFrames — dark override ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table {
    background-color: var(--bg-card) !important;
}
[data-testid="stDataFrame"] thead tr th {
    background-color: #2e1540 !important;
    color: #9BB7D4 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] tbody tr td {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-size: 0.82rem !important;
    border-bottom: 1px solid #2e1540 !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background-color: #2e1540 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #261030 !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"]:hover { border-color: #9BB7D4 !important; }

/* ── Alerts — styled ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left: 3px solid var(--accent) !important;
    background: #261030 !important;
}

/* ── Labels ── */
label, .stTextInput label, .stTextArea label,
.stSelectbox label, .stDateInput label, .stCheckbox label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important; font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: #3a2545; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9BB7D4; }

/* ── Headings ── */
h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--text-primary) !important; }

/* ── Spinner override ── */
[data-testid="stSpinner"] > div {
    border-top-color: var(--accent) !important;
}
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

def load_csv(file_path, columns):
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame(columns=columns)

def append_csv(file_path, row, columns):
    df = load_csv(file_path, columns)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(file_path, index=False)

def pl_divider(label=""):
    st.markdown(f"<div class='pl-divider'><span>{label}</span></div>", unsafe_allow_html=True)

def show_history(title, file_path, columns, download_name):
    pl_divider(title)
    df = load_csv(file_path, columns)
    if df.empty:
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-icon">📭</div>
            <div class="empty-title">No {title.lower()} yet</div>
            <div class="empty-sub">Records will appear here after your first submission.</div>
        </div>""", unsafe_allow_html=True)
    else:
        # Pagination
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
    store = st.session_state.vector_store
    metadatas = vector_store_get_all(store)
    docs = {}
    for meta in metadatas:
        doc = meta.get("document_title", "Unknown document")
        if doc not in docs:
            docs[doc] = {"document": doc, "pages": set(), "chunks": 0, "uploaded_at": meta.get("uploaded_at", "Unknown")}
        docs[doc]["chunks"] += 1
        page = meta.get("page")
        if page not in [None, "N/A", ""]:
            docs[doc]["pages"].add(str(page))
    return [{"Document": info["document"], "Pages": len(info["pages"]), "Chunks": info["chunks"], "Uploaded": info["uploaded_at"]} for info in docs.values()]

def delete_document(document_title):
    store = st.session_state.vector_store
    return vector_store_delete(store, document_title)

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
    store = st.session_state.vector_store
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
            added = vector_store_add(store, chunk_id, embedding, chunk, {
                "document_title": uploaded_file_name,
                "page": page_data["page"],
                "chunk_number": chunk_index,
                "chunk_hash": chunk_hash,
                "chunking_method": "recursive_semantic",
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
    store = st.session_state.vector_store
    if not store["embeddings"]:
        return "No documents in knowledge base yet.", []
    query_embedding = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
    documents, metadatas = vector_store_query(store, query_embedding, n_results=n_results)
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

def load_batches(): return load_csv(BATCH_FILE, BATCH_COLUMNS)
def save_batch(row): append_csv(BATCH_FILE, row, BATCH_COLUMNS)


# ─────────────────────────────────────────────
# Sidebar — vertical nav menu
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
  <!-- Flask body -->
  <path d="M20 14 L20 26 L14 38 Q13 40 15 41 L37 41 Q39 40 38 38 L32 26 L32 14 Z"
        fill="none" stroke="#1a0f20" stroke-width="2" stroke-linejoin="round"/>
  <!-- Flask neck top -->
  <rect x="19" y="12" width="14" height="3" rx="1.5" fill="#1a0f20"/>
  <!-- Liquid inside -->
  <path d="M16.5 35 Q18 30 26 30 Q34 30 35.5 35 L37 41 Q39 40 38 38 L32 26 L32 14 L20 14 L20 26 L14 38 Q13 40 15 41 Z"
        fill="rgba(0,0,0,0.35)"/>
  <!-- Bubbles -->
  <circle cx="22" cy="36" r="1.5" fill="rgba(255,255,255,0.5)"/>
  <circle cx="28" cy="33" r="1" fill="rgba(255,255,255,0.4)"/>
  <circle cx="25" cy="38" r="1" fill="rgba(255,255,255,0.3)"/>
</svg>
"""

with st.sidebar:
    # Logo + brand
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

    st.markdown("<div style='font-size:0.6rem;color:#757577;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;padding-left:4px;'>Navigation</div>", unsafe_allow_html=True)

    # Vertical nav buttons
    for m, icon in MODE_ICONS.items():
        is_active = st.session_state.active_mode == m
        active_class = "active" if is_active else ""
        if st.button(f"{icon}  {m}", key=f"nav_{m}", use_container_width=True):
            st.session_state.active_mode = m
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑  Clear Chat History", use_container_width=True):
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
    st.markdown("<div style='font-size:0.58rem;color:#3a2545;text-align:center;'>Proof Lab AI · v3.0</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main header with custom SVG logo
# ─────────────────────────────────────────────

mode = st.session_state.active_mode

st.markdown(f"""
<div class="proof-header">
    <div class="pl-logo-svg">{PROOF_LAB_LOGO_SVG}</div>
    <div>
        <h1>Proof Lab AI</h1>
        <div class="tagline">Bakery Intelligence Platform</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Upload PDF
# ─────────────────────────────────────────────

with st.expander("📄 Upload PDF to Knowledge Base", expanded=False):
    force_ocr   = st.checkbox("Force OCR with LlamaParse (for scanned PDFs)")
    st.markdown("<div class='input-hint'>⌘ Drag & drop or click to browse</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("Processing PDF — extracting and embedding chunks..."):
            extracted_pages  = []
            page_count       = 0
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
                st.warning("Switching to LlamaParse OCR for scanned content...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name
                extracted_pages   = extract_text_with_llamaparse(temp_path)
                page_count        = "OCR"
                extraction_method = "llamaparse_ocr"

            added_chunks, skipped_chunks = add_chunks_to_store(
                uploaded_file.name, extracted_pages, page_count, extraction_method
            )
        st.toast(f"✅ Added {added_chunks} chunks from '{uploaded_file.name}'", icon="📄")
        st.success(f"Processed with **{extraction_method}** — Added **{added_chunks}** chunks, skipped **{skipped_chunks}** duplicates.")

st.markdown("<br>", unsafe_allow_html=True)


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

            append_csv(ASK_HISTORY_FILE, {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                       "question": question, "answer": answer, "sources": sources_text}, ASK_COLUMNS)

            with st.chat_message("assistant"):
                st.write(answer)
                with st.expander("📎 Retrieved sources"):
                    st.text(sources_text)

            st.toast("Answer generated.", icon="💬")

    show_history("Ask History", ASK_HISTORY_FILE, ASK_COLUMNS, "ask_history.csv")


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
                context  = ""
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
                append_csv(SOP_HISTORY_FILE, {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                           "product_name": product_name, "user_notes": sop_notes,
                           "generated_sop": sop_output, "sources": sources_text}, SOP_COLUMNS)

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

    show_history("SOP History", SOP_HISTORY_FILE, SOP_COLUMNS, "sop_history.csv")


# ─────────────────────────────────────────────
# Mode: Batch Tracker
# ─────────────────────────────────────────────

if mode == "Batch Tracker":
    mode_hero("📊", "Batch Tracker",
              "Log every production batch with full process parameters and AI-powered analysis.",
              "Batch")

    # Metric cards at top
    batch_df_all = load_batches()
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
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
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
                    batch_context  = batch_df.to_string(index=False)
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
            append_csv(VISION_HISTORY_FILE, {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                       "image_name": uploaded_image.name, "product_type": product_type,
                       "batch_notes": notes, "diagnosis": diagnosis, "sources": sources_text}, VISION_COLUMNS)

            pl_divider("Vision Diagnosis")
            st.markdown(diagnosis)
            if use_knowledge_base_for_vision and metadatas:
                with st.expander("📎 Knowledge sources used"):
                    st.text(sources_text)
            st.toast("Vision analysis complete.", icon="🔬")

    show_history("Vision History", VISION_HISTORY_FILE, VISION_COLUMNS, "vision_history.csv")


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
            append_csv(RD_HISTORY_FILE, {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                       "product_type": product_type, "flavor_direction": flavor_direction,
                       "texture_goal": texture_goal, "constraints": constraints,
                       "brand_mood": brand_mood, "batch_size": batch_size,
                       "generated_concept": output, "sources": sources_text}, RD_COLUMNS)

            pl_divider("Generated R&D Concept")
            st.markdown(output)
            st.download_button(label="⬇ Download R&D Concept",
                               data=output, file_name="proof_lab_rd_concept.md", mime="text/markdown")
            if use_kb and metadatas:
                with st.expander("📎 Knowledge sources used"):
                    st.text(sources_text)
            st.toast("R&D concept generated.", icon="⚗️")

    show_history("R&D History", RD_HISTORY_FILE, RD_COLUMNS, "rd_history.csv")


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
