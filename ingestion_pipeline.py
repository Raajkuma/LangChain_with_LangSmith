import os
import json
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------------------------
# Environment (.env) + LangSmith tracing
# ----------------------------------------------------------------------------
load_dotenv()

# LangSmith: accept new (LANGSMITH_*) or old (LANGCHAIN_*) variable names
_ls_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_ENABLED = bool(_ls_key)
if LANGSMITH_ENABLED:
    os.environ["LANGSMITH_API_KEY"] = _ls_key
    os.environ["LANGCHAIN_API_KEY"] = _ls_key
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "document-qa-rag")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])

GEMINI_ENABLED = bool(os.getenv("GEMINI_API_KEY"))

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
INITIAL_DATA_FILE = "data.txt"
INDEX_DIR = "faiss_index"
REGISTRY_FILE = os.path.join(INDEX_DIR, "registry.json")  # {file_hash: file_name}
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

NO_ANSWER = "No any data available regarding the question"

PROMPT_TEMPLATE = """You are a strict question-answering assistant.
Answer the question using ONLY the context provided below.

Rules:
1. Use only information that is explicitly stated in the context.
2. Do NOT use outside knowledge and do NOT guess or infer beyond the context.
3. If the answer is not explicitly found in the context, reply with exactly this sentence and nothing else:
{no_answer}
4. Keep the answer clear and concise.

Context:
{context}

Question: {question}

Answer:"""


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def get_llm(model_name: str, temperature: float):
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


# ----------------------------------------------------------------------------
# Registry helpers (tracks which files are already ingested)
# ----------------------------------------------------------------------------
def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def bytes_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ----------------------------------------------------------------------------
# Ingestion: load -> chunk -> embed -> store
# ----------------------------------------------------------------------------
def load_documents(file_name: str, data: bytes):
    """Write bytes to a temp file, load with the right loader, fix metadata."""
    suffix = os.path.splitext(file_name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        docs = TextLoader(tmp_path, encoding="utf-8").load()  # .txt, .md
    finally:
        os.remove(tmp_path)

    for d in docs:
        d.metadata["source"] = file_name
    return docs


def chunk_documents(documents):
    embeddings = get_embeddings()

    # Stage 1: split by meaning
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    semantic_chunks = semantic_splitter.split_documents(documents)

    # Stage 2: cap size and add overlap
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return size_splitter.split_documents(semantic_chunks)


def load_vectorstore():
    """Load the saved FAISS index from disk, or return None if it doesn't exist."""
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True,  # safe: index was created by this app
        )
    return None


def chunk_id(chunk) -> str:
    """Deterministic ID from chunk text: the same text always gets the same ID."""
    return hashlib.md5(chunk.page_content.strip().encode("utf-8")).hexdigest()


def existing_contents(vectorstore) -> set:
    """Hashes of every chunk text already stored in the index."""
    if vectorstore is None:
        return set()
    return {
        hashlib.md5(d.page_content.strip().encode("utf-8")).hexdigest()
        for d in vectorstore.docstore._dict.values()
    }


def indexed_sources(vectorstore) -> list:
    """File names present in the index (read from the index itself)."""
    if vectorstore is None:
        return []
    return sorted({d.metadata.get("source", "unknown") for d in vectorstore.docstore._dict.values()})


def remove_duplicates(vectorstore) -> int:
    """Delete chunks whose text appears more than once. Returns how many were removed."""
    if vectorstore is None:
        return 0
    seen, to_delete = set(), []
    for doc_id, d in vectorstore.docstore._dict.items():
        h = hashlib.md5(d.page_content.strip().encode("utf-8")).hexdigest()
        if h in seen:
            to_delete.append(doc_id)
        else:
            seen.add(h)
    if to_delete:
        vectorstore.delete(to_delete)
        vectorstore.save_local(INDEX_DIR)
    return len(to_delete)


def ingest_files(files, vectorstore):
    """
    files: list of (file_name, bytes).
    Skips files already ingested (same content hash).
    Returns (vectorstore, report) where report is a list of (file_name, status).
    """
    registry = load_registry()
    report, new_chunks, new_entries = [], [], {}
    known = existing_contents(vectorstore)

    for name, data in files:
        h = bytes_hash(data)
        if h in registry or h in new_entries:
            report.append((name, "skipped (already ingested)"))
            continue
        try:
            docs = load_documents(name, data)
            chunks = chunk_documents(docs)

            # Chunk-level duplicate check (against the index and this batch)
            fresh = []
            for c in chunks:
                cid = chunk_id(c)
                if cid in known:
                    continue
                known.add(cid)
                fresh.append(c)

            new_entries[h] = name  # remember the file even if all chunks were duplicates
            if not chunks:
                report.append((name, "skipped (no text found)"))
            elif not fresh:
                report.append((name, "skipped (content already in index)"))
            else:
                new_chunks.extend(fresh)
                dup = len(chunks) - len(fresh)
                note = f", {dup} duplicate chunks skipped" if dup else ""
                report.append((name, f"ingested ({len(fresh)} chunks{note})"))
        except Exception as e:
            report.append((name, f"failed: {e}"))

    if new_chunks:
        if vectorstore is None:
            vectorstore = FAISS.from_documents(new_chunks, get_embeddings())
        else:
            vectorstore.add_documents(new_chunks)
        os.makedirs(INDEX_DIR, exist_ok=True)
        vectorstore.save_local(INDEX_DIR)
    if new_entries:
        registry.update(new_entries)
        save_registry(registry)

    return vectorstore, report


def reset_index():
    if os.path.isdir(INDEX_DIR):
        for f in os.listdir(INDEX_DIR):
            os.remove(os.path.join(INDEX_DIR, f))


# ----------------------------------------------------------------------------
# RAG chain
# ----------------------------------------------------------------------------
def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


def build_rag_chain(vectorstore, llm, k: int):
    """
    question (str) -> {"question", "docs", "context", "answer"}
    Keeps the retrieved docs in the output so the UI can show them.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).partial(no_answer=NO_ANSWER)

    chain = (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RunnablePassthrough.assign(answer=prompt | llm | StrOutputParser())
    )
    return chain


def generate_response(question: str, vectorstore, llm, k: int):
    """Run the chain and enforce the strict 'no data' rule. Returns (answer, docs)."""
    if vectorstore is None:
        return NO_ANSWER, []

    result = build_rag_chain(vectorstore, llm, k).invoke(
        question,
        config={
            "run_name": "rag_query",
            "tags": ["rag", "faiss", "gemini"],
            "metadata": {"k": k},
        },
    )
    answer = result["answer"].strip()

    # Strict rule: if the model signals "not found", return the exact sentence only.
    if NO_ANSWER.lower() in answer.lower() or not result["docs"]:
        return NO_ANSWER, result["docs"]
    return answer, result["docs"]


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")
st.title("📄 Simple RAG App")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initial_checked" not in st.session_state:
    st.session_state.initial_checked = False

# First-ever run only: if NO index exists yet, build it from the initial data.txt.
# Once an index exists, nothing is ingested automatically; only the Upload button changes it.
if not st.session_state.initial_checked:
    st.session_state.initial_checked = True
    if st.session_state.vectorstore is None and os.path.exists(INITIAL_DATA_FILE):
        with open(INITIAL_DATA_FILE, "rb") as f:
            initial_bytes = f.read()
        with st.spinner(f"First run: building index from {INITIAL_DATA_FILE}..."):
            st.session_state.vectorstore, _ = ingest_files(
                [(INITIAL_DATA_FILE, initial_bytes)], None
            )

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.text_input("Gemini model", value="gemini-3.8-flash")
    st.caption(
        f"{'✅' if GEMINI_ENABLED else '❌'} Gemini key   "
        f"{'✅' if LANGSMITH_ENABLED else '⚪'} LangSmith tracing"
        + (f" ({os.getenv('LANGSMITH_PROJECT')})" if LANGSMITH_ENABLED else "")
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 4)

    st.divider()
    st.header("📤 Upload documents")
    uploaded = st.file_uploader(
        "Add .txt or .md files",
        type=["txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Process documents", type="primary", disabled=not uploaded):
        files = [(u.name, u.getvalue()) for u in uploaded]
        with st.spinner("Chunking, embedding and storing..."):
            st.session_state.vectorstore, report = ingest_files(
                files, st.session_state.vectorstore
            )
        for name, status in report:
            if status.startswith("ingested"):
                st.success(f"{name}: {status}")
            elif status.startswith("failed"):
                st.error(f"{name}: {status}")
            else:
                st.info(f"{name}: {status}")

    st.divider()
    st.header("🗂️ Indexed documents")
    sources = indexed_sources(st.session_state.vectorstore)
    if sources:
        for name in sources:
            st.write(f"• {name}")
        st.caption(f"{len(st.session_state.vectorstore.docstore._dict)} chunks in index")
    else:
        st.caption("Nothing indexed yet.")

    if st.button("Remove duplicate chunks", disabled=not sources):
        removed = remove_duplicates(st.session_state.vectorstore)
        st.success(f"Removed {removed} duplicate chunks.") if removed else st.info("No duplicates found.")

    if st.button("Reset index"):
        reset_index()
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def render_sources(docs):
    with st.expander(f"📚 Retrieved chunks ({len(docs)})"):
        if not docs:
            st.caption("No chunks retrieved.")
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "unknown")
            st.markdown(f"**Chunk {i}** — `{src}`")
            st.text(d.page_content)
            if i < len(docs):
                st.divider()


# ---- Chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "docs" in msg:
            render_sources(msg["docs"])

# ---- Chat input ----
if st.session_state.vectorstore is None:
    st.info("No documents indexed yet. Upload files from the sidebar to get started.")

if not GEMINI_ENABLED:
    st.error("GOOGLE_API_KEY (or GEMINI_API_KEY) not found. Add it to your .env file and restart.")

question = st.chat_input(
    "Ask a question about your documents...", disabled=not GEMINI_ENABLED
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and generating answer..."):
                llm = get_llm(model_name, temperature)
                answer, docs = generate_response(
                    question, st.session_state.vectorstore, llm, top_k
                )
            st.markdown(answer)
            render_sources(docs)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "docs": docs}
            )
        except Exception as e:
            st.error(
                f"Could not generate a response: {e}\n\n"
                f"Check your GEMINI_API_KEY and that the model name "
                f"`{model_name}` is valid."
            )