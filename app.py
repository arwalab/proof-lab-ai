import streamlit as st
import os
import hashlib
import tempfile
import base64
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="proof_lab_knowledge")

ASK_HISTORY_FILE = Path("ask_history.csv")
SOP_HISTORY_FILE = Path("sop_history.csv")
BATCH_FILE = Path("batch_tracker.csv")
VISION_HISTORY_FILE = Path("vision_history.csv")
RD_HISTORY_FILE = Path("rd_history.csv")

# ─────────────────────────────────────────────
# Page config & global CSS
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Proof Lab AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --bg-main: #0d0d0d;
    --bg-card: #161616;
    --bg-sidebar: #111111;
    --accent: #c9a84c;
    --accent-light: #e8c97a;
    --accent-dim: rgba(201,168,76,0.12);
    --text-primary: #f0ece4;
    --text-secondary: #9e9a93;
    --border: rgba(201,168,76,0.18);
    --radius: 12px;
    --shadow: 0 4px 24px rgba(0,0,0,0.4);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.proof-header {
    display: flex; align-items: center; gap: 16px;
    padding: 20px 0 8px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.proof-header .logo-circle {
    width: 52px; height: 52px; border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), #7a5c10);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; flex-shrink: 0;
    box-shadow: 0 0 24px rgba(201,168,76,0.35);
}
.proof-header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important; font-weight: 700 !important;
    color: var(--accent) !important;
    margin: 0 !important; padding: 0 !important;
    letter-spacing: 0.5px;
}
.proof-header .tagline {
    font-size: 0.72rem; color: var(--text-secondary);
    letter-spacing: 2px; text-transform: uppercase; margin-top: 3px;
}

.mode-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--accent-dim);
    border: 1px solid var(--border);
    border-radius: 999px; padding: 6px 18px;
    font-size: 0.82rem; font-weight: 600;
    color: var(--accent-light); letter-spacing: 0.5px;
    margin-bottom: 24px;
}

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

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #1c1c1c !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
    outline: none !important;
}

.stSelectbox > div > div {
    background-color: #1c1c1c !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #8b6914) !important;
    color: #0d0d0d !important; font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important; border-radius: 8px !important;
    padding: 10px 24px !important; font-size: 0.88rem !important;
    letter-spacing: 0.4px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 14px rgba(201,168,76,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 22px rgba(201,168,76,0.38) !important;
}

.stDownloadButton > button {
    background: transparent !important; color: var(--accent) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background: var(--accent-dim) !important; border-color: var(--accent) !important;
}

[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, var(--accent), #8b6914) !important;
    color: #0d0d0d !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
    width: 100% !important; padding: 12px !important;
    font-size: 0.92rem !important;
}

[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 10px !important;
}
[data-testid="stChatInput"] > div {
    background: #1c1c1c !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea { color: var(--text-primary) !important; }

.streamlit-expanderHeader {
    background: #1a1a1a !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}
.streamlit-expanderContent {
    background: #181818 !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

[data-testid="stFileUploader"] {
    background: #1a1a1a !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

hr { border-color: var(--border) !important; margin: 22px 0 !important; }

label, .stTextInput label, .stTextArea label,
.stSelectbox label, .stDateInput label, .stCheckbox label {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important; font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--text-primary) !important; }
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


def show_history(title, file_path, columns, download_name):
    st.divider()
    st.markdown(f"<div class='pl-card-title'>{title}</div>", unsafe_allow_html=True)
    df = load_csv(file_path, columns)
    if df.empty:
        st.info("No history saved yet.")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            label=f"⬇ Download {title}",
            data=df.to_csv(index=False),
            file_name=download_name,
            mime="text/csv"
        )


def get_document_library():
    all_items = collection.get(include=["metadatas"])
    metadatas = all_items.get("metadatas", [])
    docs = {}
    for meta in metadatas:
        doc = meta.get("document_title", "Unknown document")
        if doc not in docs:
            docs[doc] = {"document": doc, "pages": set(), "chunks": 0, "uploaded_at": meta.get("uploaded_at", "Unknown")}
        docs[doc]["chunks"] += 1
        page = meta.get("page")
        if page not in [None, "N/A", ""]:
            docs[doc]["pages"].add(page)
    return [{"Document": info["document"], "Pages": len(info["pages"]), "Chunks": info["chunks"], "Uploaded": info["uploaded_at"]} for info in docs.values()]


def delete_document(document_title):
    all_items = collection.get(where={"document_title": document_title}, include=["metadatas"])
    ids = all_items.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


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


def add_chunks_to_chroma(uploaded_file_name, extracted_pages, page_count, extraction_method):
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
            existing = collection.get(ids=[chunk_id])
            if len(existing["ids"]) == 0:
                embedding = client.embeddings.create(model="text-embedding-3-small", input=chunk).data[0].embedding
                collection.add(
                    ids=[chunk_id], embeddings=[embedding], documents=[chunk],
                    metadatas=[{"document_title": uploaded_file_name, "page": page_data["page"],
                                "chunk_number": chunk_index, "chunk_hash": chunk_hash,
                                "chunking_method": "recursive_semantic", "extraction_method": extraction_method,
                                "uploaded_at": upload_time, "page_count": page_count}]
                )
                added_chunks += 1
            else:
                skipped_chunks += 1
    return added_chunks, skipped_chunks


def retrieve_context(question, n_results=5):
    query_embedding = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
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


BATCH_COLUMNS = ["timestamp", "batch_name", "product", "batch_date", "formula_notes", "process_notes",
                 "dough_temp", "butter_temp", "room_temp", "proof_temp", "proof_time", "bake_temp", "bake_time",
                 "result", "issues", "next_adjustment"]
ASK_COLUMNS = ["timestamp", "question", "answer", "sources"]
SOP_COLUMNS = ["timestamp", "product_name", "user_notes", "generated_sop", "sources"]
VISION_COLUMNS = ["timestamp", "image_name", "product_type", "batch_notes", "diagnosis", "sources"]
RD_COLUMNS = ["timestamp", "product_type", "flavor_direction", "texture_goal", "constraints", "brand_mood", "batch_size", "generated_concept", "sources"]

def load_batches(): return load_csv(BATCH_FILE, BATCH_COLUMNS)
def save_batch(row): append_csv(BATCH_FILE, row, BATCH_COLUMNS)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

MODE_ICONS = {
    "Ask Knowledge Base": "💬",
    "SOP Creator": "📋",
    "Batch Tracker": "📊",
    "Vision Analyzer": "🔬",
    "Recipe R&D Generator": "⚗️",
    "Recipe Evaluator": "✅"
}

with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 16px 0;border-bottom:1px solid rgba(201,168,76,0.18);margin-bottom:20px;">
        <div style="font-family:'Playfair Display',serif;font-size:1.25rem;font-weight:700;color:#c9a84c;">🧪 Proof Lab AI</div>
        <div style="font-size:0.65rem;color:#555;text-transform:uppercase;letter-spacing:2px;margin-top:4px;">Bakery Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.65rem;color:#555;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)
    mode = st.selectbox("Mode", list(MODE_ICONS.keys()), label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("<div style='font-size:0.65rem;color:#555;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;'>Knowledge Base</div>", unsafe_allow_html=True)

    library_rows = get_document_library()

    if library_rows:
        total_chunks = sum(r["Chunks"] for r in library_rows)
        total_docs = len(library_rows)
        st.markdown(f"""
        <div style="display:flex;gap:8px;margin-bottom:12px;">
            <div style="flex:1;background:#1a1a1a;border:1px solid rgba(201,168,76,0.18);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:1.3rem;font-weight:700;color:#c9a84c;">{total_docs}</div>
                <div style="font-size:0.62rem;color:#555;text-transform:uppercase;letter-spacing:1px;">Docs</div>
            </div>
            <div style="flex:1;background:#1a1a1a;border:1px solid rgba(201,168,76,0.18);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:1.3rem;font-weight:700;color:#c9a84c;">{total_chunks}</div>
                <div style="font-size:0.62rem;color:#555;text-transform:uppercase;letter-spacing:1px;">Chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        library_df = pd.DataFrame(library_rows)
        st.dataframe(library_df, use_container_width=True, height=150)
        doc_names = [row["Document"] for row in library_rows]
        selected_doc = st.selectbox("Select document to delete", doc_names, label_visibility="collapsed")
        if st.button("🗑 Delete Document", use_container_width=True):
            deleted_count = delete_document(selected_doc)
            st.success(f"Deleted {deleted_count} chunks from '{selected_doc}'")
            st.rerun()
    else:
        st.markdown("""
        <div style="background:#1a1a1a;border:1px dashed rgba(201,168,76,0.18);border-radius:8px;padding:18px;text-align:center;">
            <div style="font-size:1.4rem;margin-bottom:6px;">📂</div>
            <div style="font-size:0.75rem;color:#555;">No documents yet.<br>Upload a PDF to get started.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.6rem;color:#333;text-align:center;'>Proof Lab AI · v2.0</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────

st.markdown(f"""
<div class="proof-header">
    <div class="logo-circle">🧪</div>
    <div>
        <h1>Proof Lab AI</h1>
        <div class="tagline">Bakery Intelligence Platform</div>
    </div>
</div>
<div class="mode-badge">{MODE_ICONS[mode]} &nbsp; {mode}</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Upload PDF
# ─────────────────────────────────────────────

with st.expander("📄 Upload PDF to Knowledge Base", expanded=False):
    force_ocr = st.checkbox("Force OCR with LlamaParse (for scanned PDFs)")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            extracted_pages = []
            page_count = 0
            extraction_method = "pypdf"

            if not force_ocr:
                try:
                    extracted_pages, page_count = extract_text_with_pypdf(uploaded_file)
                except Exception:
                    extracted_pages = []

            total_text_length = sum(len(p["text"]) for p in extracted_pages)

            if force_ocr or total_text_length < 500:
                if not llama_api_key:
                    st.error("Missing LLAMA_CLOUD_API_KEY in your .env file.")
                    st.stop()
                st.warning("Using LlamaParse OCR...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name
                extracted_pages = extract_text_with_llamaparse(temp_path)
                page_count = "OCR"
                extraction_method = "llamaparse_ocr"

            added_chunks, skipped_chunks = add_chunks_to_chroma(
                uploaded_file.name, extracted_pages, page_count, extraction_method
            )
        st.success(f"✅ Processed with **{extraction_method}** — Added **{added_chunks}** chunks, skipped **{skipped_chunks}** duplicates.")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Mode: Ask Knowledge Base
# ─────────────────────────────────────────────

if mode == "Ask Knowledge Base":
    chat_col, info_col = st.columns([3, 1])

    with info_col:
        st.markdown("""
        <div class="pl-card">
            <div class="pl-card-title">About this mode</div>
            <div style="font-size:0.82rem;color:#9e9a93;line-height:1.7;">
                Ask questions about baking science, fermentation, pastry techniques, and more — powered by your uploaded knowledge base.
            </div>
            <div style="margin-top:16px;font-size:0.65rem;color:#555;text-transform:uppercase;letter-spacing:1px;">Tips</div>
            <div style="font-size:0.78rem;color:#9e9a93;margin-top:6px;line-height:1.8;">
                • Be specific in your questions<br>
                • Reference product types<br>
                • Ask follow-up questions<br>
                • Check sources for citations
            </div>
        </div>
        """, unsafe_allow_html=True)

    with chat_col:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input("Ask about baking, fermentation, pastry science...")

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

            with st.spinner("Thinking..."):
                response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages_payload)

            answer = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": answer})
            sources_text = format_sources(metadatas)

            append_csv(ASK_HISTORY_FILE, {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "question": question, "answer": answer, "sources": sources_text}, ASK_COLUMNS)

            with st.chat_message("assistant"):
                st.write(answer)
                with st.expander("📎 Retrieved sources"):
                    st.text(sources_text)

    show_history("Ask History", ASK_HISTORY_FILE, ASK_COLUMNS, "ask_history.csv")


# ─────────────────────────────────────────────
# Mode: SOP Creator
# ─────────────────────────────────────────────

if mode == "SOP Creator":
    left, right = st.columns([2, 1])

    with right:
        st.markdown("""
        <div class="pl-card">
            <div class="pl-card-title">SOP Structure</div>
            <div style="font-size:0.78rem;color:#9e9a93;line-height:2.0;">
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
        sop_notes = st.text_area("Paste rough notes, recipe, process, or staff instructions", height=220,
                                  placeholder="Example: Mix dough 8 min, bulk 45 min, shape, proof until puffy, bake at 180°C...")
        use_knowledge_base = st.checkbox("Use knowledge base for technical support", value=True)

        if st.button("📋 Generate SOP", use_container_width=True):
            if not sop_notes.strip():
                st.warning("Please enter rough notes first.")
            else:
                context = ""
                metadatas = []
                if use_knowledge_base:
                    context, metadatas = retrieve_context(f"Technical support for SOP: {product_name}. {sop_notes}", n_results=5)

                with st.spinner("Generating SOP..."):
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

                sop_output = response.choices[0].message.content
                sources_text = format_sources(metadatas)
                append_csv(SOP_HISTORY_FILE, {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "product_name": product_name, "user_notes": sop_notes, "generated_sop": sop_output, "sources": sources_text}, SOP_COLUMNS)

                st.divider()
                st.markdown("<div class='pl-card-title'>Generated SOP</div>", unsafe_allow_html=True)
                st.markdown(sop_output)
                st.download_button(label="⬇ Download SOP as Markdown", data=sop_output, file_name=f"{product_name or 'proof_lab_sop'}.md", mime="text/markdown")
                if use_knowledge_base and metadatas:
                    with st.expander("📎 Knowledge sources used"):
                        st.text(sources_text)

    show_history("SOP History", SOP_HISTORY_FILE, SOP_COLUMNS, "sop_history.csv")


# ─────────────────────────────────────────────
# Mode: Batch Tracker
# ─────────────────────────────────────────────

if mode == "Batch Tracker":
    st.markdown("<div class='pl-card-title'>Log a New Batch</div>", unsafe_allow_html=True)

    with st.form("batch_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div style='font-size:0.68rem;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Batch Info</div>", unsafe_allow_html=True)
            batch_name = st.text_input("Batch Name", placeholder="Croissant Batch 6")
            product = st.text_input("Product", placeholder="Croissant / Bun / Brownie")
            batch_date = st.date_input("Batch Date")

        with col2:
            st.markdown("<div style='font-size:0.68rem;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Temperature Log</div>", unsafe_allow_html=True)
            dough_temp = st.text_input("Dough Temp", placeholder="8°C")
            butter_temp = st.text_input("Butter Temp", placeholder="13°C")
            room_temp = st.text_input("Room Temp", placeholder="21°C")

        with col3:
            st.markdown("<div style='font-size:0.68rem;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Process Parameters</div>", unsafe_allow_html=True)
            proof_temp = st.text_input("Proof Temp", placeholder="26°C")
            proof_time = st.text_input("Proof Time", placeholder="2.5 hr")
            bake_temp = st.text_input("Bake Temp", placeholder="180°C")
            bake_time = st.text_input("Bake Time", placeholder="18 min")

        st.divider()
        note_col1, note_col2 = st.columns(2)
        with note_col1:
            formula_notes = st.text_area("Formula Notes", height=100)
            process_notes = st.text_area("Process Notes", height=100)
        with note_col2:
            result = st.text_area("Result", height=100)
            issues = st.text_area("Issues / Defects", height=100)

        next_adjustment = st.text_area("Next Adjustment", height=80)
        submitted = st.form_submit_button("💾 Save Batch", use_container_width=True)

        if submitted:
            save_batch({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "batch_name": batch_name,
                "product": product, "batch_date": str(batch_date), "formula_notes": formula_notes,
                "process_notes": process_notes, "dough_temp": dough_temp, "butter_temp": butter_temp,
                "room_temp": room_temp, "proof_temp": proof_temp, "proof_time": proof_time,
                "bake_temp": bake_temp, "bake_time": bake_time, "result": result,
                "issues": issues, "next_adjustment": next_adjustment
            })
            st.success("✅ Batch saved successfully.")

    st.divider()
    st.markdown("<div class='pl-card-title'>Saved Batches</div>", unsafe_allow_html=True)
    batch_df = load_batches()

    if batch_df.empty:
        st.info("No batches saved yet.")
    else:
        st.dataframe(batch_df, use_container_width=True)
        st.download_button(label="⬇ Download Batch Tracker CSV", data=batch_df.to_csv(index=False), file_name="batch_tracker.csv", mime="text/csv")

        st.divider()
        st.markdown("<div class='pl-card-title'>AI Batch Analysis</div>", unsafe_allow_html=True)
        analysis_question = st.text_input("Ask about your saved batches", placeholder="Example: Why did butter leak in Batch 6?")

        if st.button("Analyze Batch Data"):
            if not analysis_question.strip():
                st.warning("Please enter a batch analysis question.")
            else:
                with st.spinner("Analyzing batches..."):
                    batch_context = batch_df.to_string(index=False)
                    kb_context, metadatas = retrieve_context(f"Technical baking support for: {analysis_question}", n_results=5)
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
                st.markdown("<div class='pl-card-title'>Batch Analysis</div>", unsafe_allow_html=True)
                st.markdown(response.choices[0].message.content)


# ─────────────────────────────────────────────
# Mode: Vision Analyzer
# ─────────────────────────────────────────────

if mode == "Vision Analyzer":
    left_col, right_col = st.columns([1, 2])

    with left_col:
        uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

        product_type = st.selectbox("Product type", ["Croissant", "Laminated dough", "Bread crumb", "Bun", "Brownie", "Cookie", "Other pastry"])
        notes = st.text_area("Optional batch notes", height=120, placeholder="Example: Hydration 48%, butter temp 13°C, proofed at 26°C for 2.5 hours...")
        use_knowledge_base_for_vision = st.checkbox("Use knowledge base for technical support", value=True)
        analyze_btn = st.button("🔬 Analyze Image", use_container_width=True, disabled=not uploaded_image)

    with right_col:
        if not uploaded_image:
            st.markdown("""
            <div style="background:#161616;border:1.5px dashed rgba(201,168,76,0.18);border-radius:12px;
                        padding:60px 40px;text-align:center;min-height:380px;
                        display:flex;flex-direction:column;align-items:center;justify-content:center;">
                <div style="font-size:3rem;margin-bottom:16px;">🔬</div>
                <div style="font-size:1rem;color:#9e9a93;font-weight:500;">Upload an image to analyze</div>
                <div style="font-size:0.78rem;color:#444;margin-top:8px;">Supports PNG, JPG, JPEG</div>
            </div>
            """, unsafe_allow_html=True)
        elif analyze_btn:
            base64_image = encode_image(uploaded_image)
            kb_context = ""
            metadatas = []

            if use_knowledge_base_for_vision:
                kb_context, metadatas = retrieve_context(
                    f"Technical support for visual diagnosis of {product_type}. Notes: {notes}. "
                    "Analyze possible defects such as underproofing, overproofing, weak gluten, butter leakage, lamination breakage, shaping issues, dense crumb, tunneling, poor oven spring.",
                    n_results=5
                )

            with st.spinner("Analyzing image..."):
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

            diagnosis = response.choices[0].message.content
            sources_text = format_sources(metadatas)
            append_csv(VISION_HISTORY_FILE, {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "image_name": uploaded_image.name, "product_type": product_type, "batch_notes": notes, "diagnosis": diagnosis, "sources": sources_text}, VISION_COLUMNS)

            st.markdown("<div class='pl-card-title'>Vision Diagnosis</div>", unsafe_allow_html=True)
            st.markdown(diagnosis)
            if use_knowledge_base_for_vision and metadatas:
                with st.expander("📎 Knowledge sources used"):
                    st.text(sources_text)

    show_history("Vision History", VISION_HISTORY_FILE, VISION_COLUMNS, "vision_history.csv")


# ─────────────────────────────────────────────
# Mode: Recipe R&D Generator
# ─────────────────────────────────────────────

if mode == "Recipe R&D Generator":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='pl-card-title'>Product Parameters</div>", unsafe_allow_html=True)
        product_type = st.selectbox("Product Type", ["Croissant", "Cookie", "Brownie", "Bun", "Sourdough", "Dessert", "Other"])
        flavor_direction = st.text_input("Flavor Direction", placeholder="Example: mango + tajin + brown butter")
        texture_goal = st.text_input("Texture Goal", placeholder="Example: crispy shell with soft center")
        batch_size = st.selectbox("Test Batch Size", ["6 pieces", "12 pieces", "24 pieces", "1 tray", "Small R&D batch"])

    with col2:
        st.markdown("<div class='pl-card-title'>Creative Direction</div>", unsafe_allow_html=True)
        constraints = st.text_area("Constraints", height=100, placeholder="Example: delivery stable, freezer stable, no wet toppings, same-day bake")
        brand_mood = st.text_input("Brand Mood / Feeling", placeholder="Example: unexpected luxury convenience store")
        use_kb = st.checkbox("Use knowledge base for technical support", value=True)

    if st.button("⚗️ Generate R&D Concept", use_container_width=True):
        if not flavor_direction.strip():
            st.warning("Please enter a flavor direction.")
        else:
            kb_context = ""
            metadatas = []
            if use_kb:
                kb_context, metadatas = retrieve_context(
                    f"Generate a technically strong {product_type} concept. Flavor: {flavor_direction}. Texture: {texture_goal}. Constraints: {constraints}. Brand: {brand_mood}.",
                    n_results=5
                )

            with st.spinner("Generating R&D concept..."):
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

            output = response.choices[0].message.content
            sources_text = format_sources(metadatas)
            append_csv(RD_HISTORY_FILE, {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "product_type": product_type, "flavor_direction": flavor_direction, "texture_goal": texture_goal, "constraints": constraints, "brand_mood": brand_mood, "batch_size": batch_size, "generated_concept": output, "sources": sources_text}, RD_COLUMNS)

            st.divider()
            st.markdown("<div class='pl-card-title'>Generated R&D Concept</div>", unsafe_allow_html=True)
            st.markdown(output)
            st.download_button(label="⬇ Download R&D Concept", data=output, file_name="proof_lab_rd_concept.md", mime="text/markdown")
            if use_kb and metadatas:
                with st.expander("📎 Knowledge sources used"):
                    st.text(sources_text)

    show_history("R&D History", RD_HISTORY_FILE, RD_COLUMNS, "rd_history.csv")


# ─────────────────────────────────────────────
# Mode: Recipe Evaluator
# ─────────────────────────────────────────────

if mode == "Recipe Evaluator":
    st.markdown("""
    <div style="font-size:0.85rem;color:#9e9a93;margin-bottom:20px;line-height:1.7;">
        Paste a test recipe and the tool will evaluate it critically using your knowledge base — covering formula balance,
        texture feasibility, flavor logic, stability, and Proof Lab brand fit.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        recipe_name = st.text_input("Recipe Name", placeholder="Example: Mango Tango Cookie V1")
        product_type = st.selectbox("Product Type", ["Croissant", "Cookie", "Brownie", "Bun", "Sourdough", "Dessert", "Other"])
    with col2:
        target_outcome = st.text_area("Target Outcome", height=108, placeholder="Example: chewy center, crisp edge, stable for delivery, strong mango aroma")

    recipe_text = st.text_area(
        "Paste Recipe / Formula / Process Notes", height=280,
        placeholder="Example:\nButter 120g\nBrown sugar 80g\nCaster sugar 40g\nEgg 50g\nFlour 180g\nBake 180°C for 12 minutes..."
    )
    use_kb_for_evaluation = st.checkbox("Use knowledge base for evaluation", value=True)

    if st.button("✅ Evaluate Recipe", use_container_width=True):
        if not recipe_text.strip():
            st.warning("Please paste a recipe first.")
        else:
            kb_context = ""
            metadatas = []
            if use_kb_for_evaluation:
                kb_context, metadatas = retrieve_context(
                    f"Evaluate this {product_type} recipe. Name: {recipe_name}. Target: {target_outcome}. Recipe: {recipe_text}.",
                    n_results=5
                )

            with st.spinner("Evaluating recipe..."):
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

            evaluation = response.choices[0].message.content
            sources_text = format_sources(metadatas)

            st.divider()
            st.markdown("<div class='pl-card-title'>Recipe Evaluation</div>", unsafe_allow_html=True)
            st.markdown(evaluation)
            st.download_button(label="⬇ Download Recipe Evaluation", data=evaluation, file_name=f"{recipe_name or 'recipe_evaluation'}.md", mime="text/markdown")
            if use_kb_for_evaluation and metadatas:
                with st.expander("📎 Knowledge sources used"):
                    st.text(sources_text)
