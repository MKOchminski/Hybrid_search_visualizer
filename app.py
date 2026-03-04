import streamlit as st
import json
import numpy as np
import re
import hashlib

# -----------------------------
# Optional imports (safe fallback)
# -----------------------------
try:
    # ✅ langchain-openai integration (required for semantic / hybrid)
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

try:
    import umap
except ImportError:
    umap = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

try:
    import plotly.express as px
except ImportError:
    px = None


# -----------------------------
# Helpers
# -----------------------------
def highlight_text(text: str, query: str) -> str:
    """Highlight query tokens in text using <mark> (HTML)."""
    if not text or not query:
        return text or ""

    tokens = re.findall(r"[\w\-]{2,}", query, flags=re.UNICODE)
    if not tokens:
        return text

    tokens = sorted(set(tokens), key=len, reverse=True)
    pattern = re.compile(r"(" + "|".join(map(re.escape, tokens)) + r")", flags=re.IGNORECASE)

    highlighted = pattern.sub(r"<mark>\1</mark>", text)
    highlighted = highlighted.replace("\n", "<br>")
    return highlighted


def cosine_similarity_docs_query(doc_embeds: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cosine similarity between doc_embeds [N,D] and q [D]."""
    eps = 1e-10
    doc_norms = np.linalg.norm(doc_embeds, axis=1, keepdims=True) + eps
    q_norm = np.linalg.norm(q) + eps
    return (doc_embeds / doc_norms) @ (q / q_norm)


def minmax_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def validate_embeddings_matrix(emb: np.ndarray, expected_rows: int) -> None:
    """Raise ValueError if embeddings matrix is not a finite 2D array with expected rows."""
    if not isinstance(emb, np.ndarray):
        raise ValueError("Embeddings must be a numpy array.")
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D [N, D], got shape={emb.shape} (ndim={emb.ndim}).")
    if emb.shape[0] != expected_rows:
        raise ValueError(f"Embeddings rows mismatch: {emb.shape[0]} vs chunks {expected_rows}.")
    if emb.size == 0:
        raise ValueError("Embeddings are empty.")
    if not np.isfinite(emb).all():
        raise ValueError("Embeddings contain NaN/Inf.")


def validate_query_embedding(vec: np.ndarray) -> None:
    if not isinstance(vec, np.ndarray):
        raise ValueError("Query embedding must be a numpy array.")
    if vec.ndim != 1:
        raise ValueError(f"Query embedding must be 1D [D], got shape={vec.shape} (ndim={vec.ndim}).")
    if vec.size == 0:
        raise ValueError("Query embedding is empty.")
    if not np.isfinite(vec).all():
        raise ValueError("Query embedding contains NaN/Inf.")


def parse_chunks_from_doc_data(doc_data: dict) -> list:
    """Parse your JSON structure into a flat list of chunks with page metadata."""
    chunks = []
    for page_key, page_val in doc_data.items():
        page_num = int(page_key) if str(page_key).isdigit() else None

        page_list_key = None
        for k in page_val.keys():
            if k.startswith("page_") and isinstance(page_val[k], list):
                page_list_key = k
                break
        if page_list_key is None:
            continue

        page_chunks = page_val[page_list_key]
        for chunk in page_chunks:
            if page_num is not None:
                chunk["page"] = page_num
            else:
                cid = chunk.get("id")
                if isinstance(cid, str) and cid.startswith("p"):
                    try:
                        chunk["page"] = int(cid.split("_")[0].lstrip("p"))
                    except Exception:
                        chunk["page"] = None
            chunks.append(chunk)

    chunks.sort(key=lambda c: (c.get("page", 0) or 0, c.get("id", "")))
    return chunks


def rebuild_lexical_indexes():
    """Build TF-IDF and BM25 structures for current chunk_texts."""
    st.session_state.tfidf_vectorizer = None
    st.session_state.tfidf_matrix = None
    st.session_state.bm25 = None

    texts = st.session_state.chunk_texts
    if not texts:
        return

    if TfidfVectorizer is not None:
        vectorizer = TfidfVectorizer().fit(texts)
        tfidf_matrix = vectorizer.transform(texts)
        st.session_state.tfidf_vectorizer = vectorizer
        st.session_state.tfidf_matrix = tfidf_matrix

    if BM25Okapi is not None:
        tokenized_corpus = [t.lower().split() for t in texts]
        st.session_state.bm25 = BM25Okapi(tokenized_corpus)


def render_last_results():
    """Render results from session_state so they survive Streamlit reruns (e.g., toggling 2D/3D)."""
    idxs = st.session_state.get("last_results_idx", [])
    if not idxs:
        return

    st.subheader("Search Results")

    meta_bits = []
    if st.session_state.get("last_mode"):
        meta_bits.append(f"Mode: **{st.session_state['last_mode']}**")
    if st.session_state.get("last_mode") in ("Lexical", "Hybrid"):
        meta_bits.append(f"Lexical: **{st.session_state.get('last_lexical')}**")
    if st.session_state.get("last_mode") == "Hybrid":
        a = st.session_state.get("last_alpha")
        if a is not None:
            meta_bits.append(f"α: **{a:.2f}** (0→lexical/sparse, 1→semantic/dense)")
    if st.session_state.get("last_query"):
        meta_bits.append(f"Query: `{st.session_state['last_query']}`")

    if meta_bits:
        st.caption(" | ".join(meta_bits))

    scores = st.session_state.get("last_scores", None)
    q = st.session_state.get("last_query", "")

    for rank, idx in enumerate(idxs, start=1):
        chunk = st.session_state.chunks[idx]
        text = chunk.get("text", "")
        page = chunk.get("page", "?")
        ctype = chunk.get("chunk_type", "chunk")
        cid = chunk.get("id", "")

        score_info = ""
        if isinstance(scores, np.ndarray):
            try:
                score_info = f" — Score: {float(scores[idx]):.4f}"
            except Exception:
                score_info = ""

        st.markdown(f"**{rank}.** *Page {page}, {ctype}, ID {cid}{score_info}*")
        st.markdown(highlight_text(text, q), unsafe_allow_html=True)
        st.divider()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Document Chunk Search", layout="wide")
st.title("Document Chunk Search and Comparison App")


# -----------------------------
# Session state initialization
# -----------------------------
if "doc_hash" not in st.session_state:
    # Document persistence
    st.session_state.doc_hash = None
    st.session_state.doc_name = None
    st.session_state.doc_data = None

    # Parsed chunks
    st.session_state.chunks = []
    st.session_state.chunk_texts = []
    st.session_state.doc_fingerprint = None

    # Dense
    st.session_state.embeddings = None
    st.session_state.embeddings_model = None
    st.session_state.last_query_embedding = None

    # Lexical
    st.session_state.tfidf_vectorizer = None
    st.session_state.tfidf_matrix = None
    st.session_state.bm25 = None

    # Viz caches
    st.session_state.umap_coords = None
    st.session_state.pca_coords = None

    # Results persistence across reruns
    st.session_state.last_query = None
    st.session_state.last_results_idx = []
    st.session_state.last_scores = None
    st.session_state.last_mode = None
    st.session_state.last_lexical = None
    st.session_state.last_alpha = None

    # ✅ NEW: increments on every Search; used to reset camera to default on new search
    st.session_state.search_rev = 0


# -----------------------------
# Sidebar settings
# -----------------------------
st.sidebar.header("Settings")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Needed for Semantic & Hybrid search (OpenAI embeddings via langchain-openai).",
)

embedding_model = st.sidebar.selectbox(
    "Embedding model",
    ["text-embedding-3-small", "text-embedding-3-large"],
    index=0,
)

search_mode = st.sidebar.selectbox("Search mode", ["Semantic", "Lexical", "Hybrid"])
lexical_method = st.sidebar.selectbox("Lexical method", ["BM25", "TF-IDF", "SPLADE (placeholder)"])
num_results = st.sidebar.slider("Number of results to show", min_value=1, max_value=20, value=8)

DEFAULT_ALPHA = 0.6
if search_mode == "Hybrid":
    st.sidebar.subheader("Hybrid mixing (α)")
    hybrid_alpha = st.sidebar.slider(
        "α (semantic weight)",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_ALPHA,
        step=0.05,
        help=(
            "Hybrid score = α · semantic(dense) + (1−α) · lexical(sparse), after per-query normalization.\n\n"
            "α=0.0 → lexical-only (sparse TF-IDF/BM25)\n"
            "α=1.0 → semantic-only (dense OpenAI embeddings)"
        ),
    )
    st.sidebar.markdown(
        "**How to read α:**  \n"
        "- **α = 0.0 → 100% lexical / sparse** (TF-IDF or BM25)  \n"
        "- **α = 1.0 → 100% semantic / dense** (OpenAI embeddings)  \n"
        "- values in-between → weighted mix"
    )
else:
    hybrid_alpha = DEFAULT_ALPHA

stats_placeholder = st.sidebar.empty()

# Document controls
st.sidebar.subheader("Document")
if st.session_state.doc_name:
    st.sidebar.caption(f"Loaded: **{st.session_state.doc_name}**")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Clear document", use_container_width=True):
        # clear document + everything derived
        st.session_state.doc_hash = None
        st.session_state.doc_name = None
        st.session_state.doc_data = None

        st.session_state.chunks = []
        st.session_state.chunk_texts = []
        st.session_state.doc_fingerprint = None

        st.session_state.embeddings = None
        st.session_state.embeddings_model = None
        st.session_state.last_query_embedding = None

        st.session_state.tfidf_vectorizer = None
        st.session_state.tfidf_matrix = None
        st.session_state.bm25 = None

        st.session_state.umap_coords = None
        st.session_state.pca_coords = None

        st.session_state.last_query = None
        st.session_state.last_results_idx = []
        st.session_state.last_scores = None
        st.session_state.last_mode = None
        st.session_state.last_lexical = None
        st.session_state.last_alpha = None

        # ✅ reset camera revision counter
        st.session_state.search_rev = 0

        st.rerun()

with col_b:
    if st.button("Reset embeddings", use_container_width=True):
        st.session_state.embeddings = None
        st.session_state.embeddings_model = None
        st.session_state.last_query_embedding = None
        st.session_state.umap_coords = None
        st.session_state.pca_coords = None

        # ✅ reset camera revision counter
        st.session_state.search_rev = 0

        st.rerun()


# -----------------------------
# File loader (persisted)
# -----------------------------
uploaded_file = st.file_uploader("Upload chunk JSON file", type=["json"], key="chunk_json_uploader")

# ✅ Critical fix: use getvalue() and store in session_state
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # only parse if it's a new/different file
    if st.session_state.doc_hash != file_hash:
        try:
            doc_data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON: {e}")
            doc_data = None

        if doc_data is not None:
            st.session_state.doc_hash = file_hash
            st.session_state.doc_name = uploaded_file.name
            st.session_state.doc_data = doc_data

            # parse chunks once
            chunks = parse_chunks_from_doc_data(doc_data)
            st.session_state.chunks = chunks
            st.session_state.chunk_texts = [c.get("text", "") for c in chunks]

            # fingerprint + indexes
            st.session_state.doc_fingerprint = (
                len(chunks),
                chunks[0].get("id") if chunks else None,
                chunks[-1].get("id") if chunks else None,
            )
            rebuild_lexical_indexes()

            # reset dense/viz/results (doc changed)
            st.session_state.embeddings = None
            st.session_state.embeddings_model = None
            st.session_state.last_query_embedding = None
            st.session_state.umap_coords = None
            st.session_state.pca_coords = None

            st.session_state.last_results_idx = []
            st.session_state.last_query = None
            st.session_state.last_scores = None
            st.session_state.last_mode = None
            st.session_state.last_lexical = None
            st.session_state.last_alpha = None

            # ✅ reset camera revision counter (new doc)
            st.session_state.search_rev = 0

            st.success(f"Loaded document: {uploaded_file.name}")

# If no upload but doc already loaded -> keep it
doc_data = st.session_state.doc_data

if doc_data is None:
    st.info("Upload a chunked JSON file to start searching.")
    st.stop()


# -----------------------------
# Stats
# -----------------------------
chunks = st.session_state.chunks
texts = st.session_state.chunk_texts

total_chunks = len(chunks)
pages = {c.get("page") for c in chunks if c.get("page") is not None}
num_pages = len(pages)
lengths = [len(c.get("text", "")) for c in chunks]
avg_len_char = float(np.mean(lengths)) if lengths else 0.0
avg_len_words = float(np.mean([len(c.get("text", "").split()) for c in chunks])) if lengths else 0.0

type_counts = {}
for c in chunks:
    ctype = c.get("chunk_type", "unknown")
    type_counts[ctype] = type_counts.get(ctype, 0) + 1
type_summary = "; ".join(f"{t}: {n}" for t, n in type_counts.items())

stats_placeholder.markdown(
    f"**Document Stats:**  \n"
    f"- Pages: {num_pages}  \n"
    f"- Total chunks: {total_chunks}  \n"
    f"- Avg chunk length: {avg_len_words:.1f} words ({avg_len_char:.0f} chars)  \n"
    f"- Chunk types: {type_summary}"
)


# -----------------------------
# Query + Search
# -----------------------------
query = st.text_input("Enter your query:", key="query_input")

if st.button("Search", key="search_button") and query:
    query_embedding = None

    # --- Semantic embeddings via langchain-openai ---
    if search_mode in ["Semantic", "Hybrid"]:
        if not openai_api_key:
            st.error("OpenAI API key is required for semantic or hybrid search. Please enter it in the sidebar.")
        elif OpenAIEmbeddings is None:
            st.error("langchain-openai is not available. Install `langchain-openai`.")
        else:
            need_recompute = (
                st.session_state.embeddings is None
                or st.session_state.embeddings_model != embedding_model
            )

            embedder = OpenAIEmbeddings(
                model=embedding_model,
                api_key=openai_api_key,
            )

            if need_recompute:
                with st.spinner(f"Generating embeddings for all chunks using {embedding_model}..."):
                    try:
                        chunk_embeddings = embedder.embed_documents(texts)
                        emb = np.asarray(chunk_embeddings, dtype=np.float32)
                        validate_embeddings_matrix(emb, expected_rows=len(texts))

                        st.session_state.embeddings = emb
                        st.session_state.embeddings_model = embedding_model

                        # reset projections because embedding space changed
                        st.session_state.umap_coords = None
                        st.session_state.pca_coords = None

                        st.success("Document embeddings computed.")
                    except Exception as e:
                        st.error(f"Error during embedding generation: {e}")
                        st.session_state.embeddings = None
                        st.session_state.embeddings_model = None
                        st.session_state.last_query_embedding = None

            if st.session_state.embeddings is not None:
                try:
                    qv = np.asarray(embedder.embed_query(query), dtype=np.float32)
                    validate_query_embedding(qv)
                    query_embedding = qv
                    st.session_state.last_query_embedding = qv
                except Exception as e:
                    st.error(f"Error embedding query: {e}")
                    query_embedding = None
                    st.session_state.last_query_embedding = None

    # --- Lexical scoring ---
    lexical_scores = None
    if search_mode in ["Lexical", "Hybrid"]:
        if lexical_method == "TF-IDF":
            if st.session_state.tfidf_vectorizer is None or st.session_state.tfidf_matrix is None:
                st.error("TF-IDF not available (missing scikit-learn or empty corpus).")
            else:
                q_vec = st.session_state.tfidf_vectorizer.transform([query])
                scores = (st.session_state.tfidf_matrix @ q_vec.T).toarray().ravel()
                lexical_scores = scores.astype(np.float32)

        elif lexical_method == "BM25":
            if st.session_state.bm25 is None:
                st.error("BM25 not available (missing rank-bm25 or empty corpus).")
            else:
                q_tokens = query.lower().split()
                scores = st.session_state.bm25.get_scores(q_tokens)
                lexical_scores = np.asarray(scores, dtype=np.float32)

        else:  # SPLADE placeholder
            st.warning("SPLADE is a placeholder in this demo. Use TF-IDF or BM25.")
            lexical_scores = None

    # --- Rank results ---
    result_indices = []
    combined_scores = None

    if search_mode == "Semantic":
        if query_embedding is not None and st.session_state.embeddings is not None:
            sim_scores = cosine_similarity_docs_query(st.session_state.embeddings, query_embedding)
            combined_scores = sim_scores
            result_indices = np.argsort(sim_scores)[::-1][:num_results].tolist()

    elif search_mode == "Lexical":
        if lexical_scores is not None:
            combined_scores = lexical_scores
            result_indices = np.argsort(lexical_scores)[::-1][:num_results].tolist()

    elif search_mode == "Hybrid":
        if query_embedding is not None and st.session_state.embeddings is not None and lexical_scores is not None:
            sim_scores = cosine_similarity_docs_query(st.session_state.embeddings, query_embedding)
            sem_norm = minmax_01(sim_scores)
            lex_norm = minmax_01(lexical_scores)
            fused = (hybrid_alpha * sem_norm) + ((1.0 - hybrid_alpha) * lex_norm)

            combined_scores = fused
            result_indices = np.argsort(fused)[::-1][:num_results].tolist()
        else:
            st.error("Hybrid requires BOTH semantic embeddings and a lexical method that produces scores (TF-IDF or BM25).")

    # Persist results so they survive reruns
    st.session_state.last_query = query
    st.session_state.last_results_idx = result_indices
    st.session_state.last_scores = combined_scores
    st.session_state.last_mode = search_mode
    st.session_state.last_lexical = lexical_method if search_mode in ("Lexical", "Hybrid") else None
    st.session_state.last_alpha = hybrid_alpha if search_mode == "Hybrid" else None

    # ✅ NEW: bump revision so the next render resets camera to default
    st.session_state.search_rev += 1

# Always render last results
render_last_results()


# -----------------------------
# Embedding Space Visualization
# -----------------------------
st.subheader("Embedding Space Visualization")

if not openai_api_key:
    st.info("Enter OpenAI API key to enable embedding visualization (semantic embeddings).")
elif st.session_state.embeddings is None:
    st.info("Run a Semantic/Hybrid search to generate embeddings; then the visualization will appear.")
elif px is None:
    st.error("Plotly is not installed. Install `plotly` to see the embedding map.")
else:
    E = st.session_state.embeddings
    try:
        validate_embeddings_matrix(E, expected_rows=len(texts))
    except Exception:
        st.error(
            "Embeddings are invalid (not a finite 2D array). "
            "Click **Reset embeddings** and run a Semantic/Hybrid search again."
        )
        st.stop()

    dim_method = st.selectbox("Dimensionality reduction method", ["PCA", "UMAP"], key="dim_method")
    dim_num = st.radio("Dimensions", [2, 3], horizontal=True, key="dim_num")

    coords = None
    query_coords = None

    if dim_method == "PCA":
        if PCA is None:
            st.error("PCA (scikit-learn) not available.")
        else:
            if st.session_state.pca_coords is None:
                st.session_state.pca_coords = {}

            key = f"PCA_{dim_num}_{st.session_state.embeddings_model}_{st.session_state.doc_fingerprint}"
            if key not in st.session_state.pca_coords:
                pca_model = PCA(n_components=dim_num, random_state=42)
                coords_data = pca_model.fit_transform(E)

                qc = None
                if st.session_state.last_query_embedding is not None:
                    try:
                        qc = pca_model.transform(st.session_state.last_query_embedding.reshape(1, -1))
                    except Exception:
                        qc = None

                st.session_state.pca_coords[key] = (coords_data, qc)

            coords, query_coords = st.session_state.pca_coords[key]

    else:  # UMAP
        if umap is None:
            st.error("UMAP is not available. Install `umap-learn`.")
        else:
            if st.session_state.umap_coords is None:
                st.session_state.umap_coords = {}

            key = f"UMAP_{dim_num}_{st.session_state.embeddings_model}_{st.session_state.doc_fingerprint}"
            if key not in st.session_state.umap_coords:
                reducer = umap.UMAP(n_components=dim_num, random_state=42)
                coords_data = reducer.fit_transform(E)

                qc = None
                if st.session_state.last_query_embedding is not None:
                    try:
                        qc = reducer.transform(st.session_state.last_query_embedding.reshape(1, -1))
                    except Exception:
                        qc = None

                st.session_state.umap_coords[key] = (coords_data, qc)

            coords, query_coords = st.session_state.umap_coords[key]

    if coords is None:
        st.warning("Could not compute coordinates.")
        st.stop()

    import pandas as pd

    dim = coords.shape[1]
    df_data = {"x": coords[:, 0], "y": coords[:, 1]}
    if dim == 3:
        df_data["z"] = coords[:, 2]

    highlight_idx = set(st.session_state.last_results_idx) if st.session_state.last_results_idx else set()

    # "Other" should keep default color; "Result" should be green
    df_data["Type"] = ["Result" if i in highlight_idx else "Other" for i in range(coords.shape[0])]
    df_data["ChunkID"] = [c.get("id", str(i)) for i, c in enumerate(chunks)]
    df_data["Page"] = [c.get("page", None) for c in chunks]
    df_data["ChunkType"] = [c.get("chunk_type", "unknown") for c in chunks]
    df_data["Text"] = [
        (c.get("text", "")[:80].replace("\n", " ") + ("..." if len(c.get("text", "")) > 80 else ""))
        for c in chunks
    ]
    df_plot = pd.DataFrame(df_data)

    # Keep Other as default, force Result to green, ensure Other is first category
    color_map = {"Result": "green"}
    category_orders = {"Type": ["Other", "Result"]}

    if dim == 3:
        fig = px.scatter_3d(
            df_plot,
            x="x", y="y", z="z",
            color="Type",
            symbol="Type",
            color_discrete_map=color_map,
            category_orders=category_orders,
            hover_data={"ChunkID": True, "Page": True, "ChunkType": True, "Text": True},
            title="Document Chunks Embedding Space (3D)",
        )
    else:
        fig = px.scatter(
            df_plot,
            x="x", y="y",
            color="Type",
            symbol="Type",
            color_discrete_map=color_map,
            category_orders=category_orders,
            hover_data={"ChunkID": True, "Page": True, "ChunkType": True, "Text": True},
            title="Document Chunks Embedding Space (2D)",
        )

    # Query marker as RED "X"
    if query_coords is not None:
        if dim == 3:
            fig.add_scatter3d(
                x=[query_coords[0, 0]], y=[query_coords[0, 1]], z=[query_coords[0, 2]],
                mode="markers",
                marker=dict(symbol="x", size=10, color="red"),
                name="Query",
            )
        else:
            fig.add_scatter(
                x=[query_coords[0, 0]], y=[query_coords[0, 1]],
                mode="markers",
                marker=dict(symbol="x", size=14, color="red"),
                name="Query",
            )

    # ✅ NEW: preserve camera/zoom/pan between reruns (but reset on new Search)
    ui_rev = f"{st.session_state.doc_hash}|{st.session_state.embeddings_model}|{dim_method}|{dim_num}|{st.session_state.search_rev}"

    # Preserve camera/zoom/pan across reruns, reset only when search_rev changes
    fig.update_layout(uirevision=ui_rev)

    # IMPORTANT: update scenes safely (do NOT overwrite the whole scene dict)
    if dim == 3:
        fig.update_scenes(uirevision=ui_rev)

    # Optional but helpful for 3D UX
    plotly_cfg = {
        "displaylogo": False,
        "scrollZoom": True,  # zoom with scroll/trackpad
    }

    st.plotly_chart(fig, use_container_width=True, key="embedding_plot", config=plotly_cfg)