import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# -----------------------------
# Optional imports
# -----------------------------
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError:
    OpenAIEmbeddings = None
    ChatOpenAI = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

try:
    import umap
except ImportError:
    umap = None

try:
    import plotly.express as px
except ImportError:
    px = None

try:
    from pystempel import Stemmer
except ImportError:
    Stemmer = None


# -----------------------------
# Text utilities
# -----------------------------
POLISH_DIACRITICS_MAP = str.maketrans(
    "ąćęłńóśżźĄĆĘŁŃÓŚŻŹ",
    "acelnoszzACELNOSZZ",
)


def strip_polish_diacritics(text: str) -> str:
    return text.translate(POLISH_DIACRITICS_MAP)


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[\w\-]{2,}", text.lower(), flags=re.UNICODE)


def safe_json_loads(text: str) -> Optional[dict]:
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def highlight_text(text: str, query_terms: List[str]) -> str:
    if not text or not query_terms:
        return text or ""

    tokens = [t for t in query_terms if t and len(t) >= 2]
    if not tokens:
        return text

    tokens = sorted(set(tokens), key=len, reverse=True)
    pattern = re.compile(
        r"(" + "|".join(map(re.escape, tokens)) + r")",
        flags=re.IGNORECASE,
    )

    highlighted = pattern.sub(r"<mark>\1</mark>", text)
    return highlighted.replace("\n", "<br>")


# -----------------------------
# Search math
# -----------------------------
def cosine_similarity_docs_query(doc_embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    eps = 1e-10
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + eps
    q_norm = np.linalg.norm(query_embedding) + eps
    return (doc_embeddings / doc_norms) @ (query_embedding / q_norm)


def minmax_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def zscore_sigmoid_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        return np.zeros_like(x)
    z = (x - mean) / std
    return 1.0 / (1.0 + np.exp(-z))


def normalize_semantic_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)

    # Cosine is typically in [-1, 1]. Shift first, then normalize.
    shifted = (scores + 1.0) / 2.0

    if method == "zscore":
        return zscore_sigmoid_01(shifted)

    return minmax_01(shifted)


def normalize_lexical_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    scores = np.maximum(scores, 0.0)

    # BM25 often has a long tail. log1p makes hybrid scores more stable.
    transformed = np.log1p(scores)

    if method == "zscore":
        return zscore_sigmoid_01(transformed)

    return minmax_01(transformed)


def rank_based_unit_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores)[::-1]
    n = len(scores)

    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    if n == 1:
        out[order[0]] = 1.0
        return out

    for rank, idx in enumerate(order):
        out[idx] = 1.0 - (rank / (n - 1))
    return out


def reciprocal_rank_fusion_scores(
    score_lists: List[np.ndarray],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> np.ndarray:
    if not score_lists:
        return np.array([], dtype=np.float32)

    n = len(score_lists[0])
    fused = np.zeros(n, dtype=np.float32)

    if weights is None:
        weights = [1.0] * len(score_lists)

    for scores, weight in zip(score_lists, weights):
        order = np.argsort(scores)[::-1]
        for rank, idx in enumerate(order, start=1):
            fused[idx] += weight * (1.0 / (k + rank))

    return minmax_01(fused)


# -----------------------------
# Validation helpers
# -----------------------------
def validate_embeddings_matrix(embeddings: np.ndarray, expected_rows: int) -> None:
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embeddings must be a numpy array.")
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D [N, D], got shape={embeddings.shape}.")
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Embeddings row count mismatch: {embeddings.shape[0]} vs expected {expected_rows}."
        )
    if embeddings.size == 0:
        raise ValueError("Embeddings are empty.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or Inf.")


def validate_query_embedding(vec: np.ndarray) -> None:
    if not isinstance(vec, np.ndarray):
        raise ValueError("Query embedding must be a numpy array.")
    if vec.ndim != 1:
        raise ValueError(f"Query embedding must be 1D [D], got shape={vec.shape}.")
    if vec.size == 0:
        raise ValueError("Query embedding is empty.")
    if not np.isfinite(vec).all():
        raise ValueError("Query embedding contains NaN or Inf.")


def looks_structural(chunk: dict) -> bool:
    text = (chunk.get("text") or "").strip()
    chunk_type = str(chunk.get("chunk_type", "")).lower()

    if chunk_type in {"header", "footer", "page_number", "toc", "table_of_contents"}:
        return True
    if len(text) < 30:
        return True
    if len(tokenize_text(text)) < 4:
        return True

    return False


# -----------------------------
# Chunk parsing
# -----------------------------
def parse_chunks_from_doc_data(doc_data: dict) -> List[dict]:
    chunks = []

    for page_key, page_val in doc_data.items():
        page_num = int(page_key) if str(page_key).isdigit() else None

        # Case 1: {"1": {"page_1": [...]}}
        if isinstance(page_val, dict):
            page_list_key = None
            for key in page_val.keys():
                if key.startswith("page_") and isinstance(page_val[key], list):
                    page_list_key = key
                    break

            if page_list_key is None:
                continue

            page_chunks = page_val[page_list_key]

        # Case 2: {"page_1": [...]}
        elif isinstance(page_val, list):
            page_chunks = page_val
        else:
            continue

        for chunk in page_chunks:
            chunk = dict(chunk)

            if page_num is not None:
                chunk["page"] = page_num
            elif "page" not in chunk:
                chunk_id = chunk.get("id")
                if isinstance(chunk_id, str) and chunk_id.startswith("p"):
                    try:
                        chunk["page"] = int(chunk_id.split("_")[0].lstrip("p"))
                    except Exception:
                        chunk["page"] = None

            chunks.append(chunk)

    chunks.sort(key=lambda c: (c.get("page", 0) or 0, c.get("id", "")))
    return chunks


# -----------------------------
# Lexical preprocessing
# -----------------------------
def get_stemmer():
    if Stemmer is None:
        return None
    try:
        return Stemmer.polimorf()
    except Exception:
        return None


def stem_token(token: str, stemmer) -> str:
    if not stemmer:
        return token
    try:
        value = stemmer.stem(token)
        return value if value else token
    except Exception:
        return token


def preprocess_text_for_lexical_search(
    text: str,
    stemmer=None,
    include_ascii_variants: bool = True,
) -> str:
    tokens = tokenize_text(text)
    out = []

    for token in tokens:
        out.append(token)

        folded = strip_polish_diacritics(token)
        if include_ascii_variants and folded != token:
            out.append(folded)

        stemmed = stem_token(token, stemmer)
        if stemmed and stemmed != token:
            out.append(stemmed)

        stemmed_folded = strip_polish_diacritics(stemmed)
        if include_ascii_variants and stemmed_folded not in {token, folded, stemmed}:
            out.append(stemmed_folded)

    return " ".join(out)


def expand_query_for_lexical_search(
    query: str,
    llm: Optional[Any],
    enable_llm_expansion: bool = True,
) -> Dict[str, Any]:
    fallback = {
        "normalized_query": query,
        "keywords": tokenize_text(query),
        "variants": [],
        "query_for_search": query,
    }

    if not enable_llm_expansion or llm is None:
        return fallback

    prompt = f"""
You are improving lexical search over Polish documents.

Task:
Rewrite the user query so lexical search (BM25 / TF-IDF) works better on Polish inflection,
cases, number, close wording variants and synonyms.

Return STRICT JSON only with this schema:
{{
  "normalized_query": "short normalized query in Polish",
  "keywords": ["keyword1", "keyword2"],
  "variants": ["variant1", "variant2", "variant3"],
  "query_for_search": "single search string containing the best lexical variants"
}}

Rules:
- Keep the meaning identical.
- Do not answer the question.
- Do not add broad unrelated terms.
- Prefer base forms, close inflections, short synonym variants, abbreviations only if clearly relevant.
- Keep it concise and useful for keyword search.
- Output JSON only.

User query:
{query}
""".strip()

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        parsed = safe_json_loads(content)
        if not parsed:
            return fallback

        normalized_query = parsed.get("normalized_query") or query
        keywords = parsed.get("keywords") or tokenize_text(query)
        variants = parsed.get("variants") or []

        query_for_search = parsed.get("query_for_search")
        if not query_for_search:
            pieces = [normalized_query] + keywords + variants
            query_for_search = " ".join([p for p in pieces if p])

        return {
            "normalized_query": normalized_query,
            "keywords": keywords,
            "variants": variants,
            "query_for_search": query_for_search,
        }

    except Exception:
        return fallback


# -----------------------------
# LLM validator
# -----------------------------
def validate_candidates_with_llm(
    user_query: str,
    chunks: List[dict],
    candidate_indices: List[int],
    llm: Optional[Any],
    threshold: float = 0.55,
    debug: bool = False,
) -> Tuple[List[dict], List[dict]]:
    accepted = []
    trace = []

    if llm is None:
        for idx in candidate_indices:
            accepted.append(
                {
                    "index": idx,
                    "llm_score": 1.0,
                    "llm_label": "validator_disabled",
                    "llm_reason": "LLM validator disabled.",
                }
            )
        return accepted, trace

    for idx in candidate_indices:
        chunk = chunks[idx]
        chunk_text = chunk.get("text", "")
        structural_hint = looks_structural(chunk)

        prompt = f"""
You are a strict retrieval validator for RAG.

Question:
{user_query}

Chunk metadata:
- page: {chunk.get("page")}
- chunk_type: {chunk.get("chunk_type")}
- structural_hint: {structural_hint}

Chunk text:
\"\"\"
{chunk_text}
\"\"\"

Return STRICT JSON only:
{{
  "relevant": true,
  "score": 0.0,
  "label": "answer|supporting_context|structural_noise|irrelevant",
  "reason": "short explanation"
}}

Validation rules:
- Reject headers, footers, page numbers, isolated titles, and navigation-like fragments unless they directly answer the question.
- Reject chunks that only share topic words but do not contain useful information.
- Accept chunks that directly answer the question or provide concrete supporting information.
- Be strict.
""".strip()

        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            parsed = safe_json_loads(content) or {}

            llm_score = float(parsed.get("score", 0.0))
            llm_label = str(parsed.get("label", "unknown"))
            llm_reason = str(parsed.get("reason", ""))
            relevant = bool(parsed.get("relevant", False))

            decision = {
                "index": idx,
                "relevant": relevant,
                "llm_score": llm_score,
                "llm_label": llm_label,
                "llm_reason": llm_reason,
            }

            trace.append(decision)

            if relevant and llm_score >= threshold:
                accepted.append(decision)

        except Exception as exc:
            trace.append(
                {
                    "index": idx,
                    "relevant": False,
                    "llm_score": 0.0,
                    "llm_label": "validator_error",
                    "llm_reason": str(exc),
                }
            )

    if debug:
        st.session_state.last_validator_trace = trace

    return accepted, trace


# -----------------------------
# Index building
# -----------------------------
def rebuild_lexical_indexes():
    st.session_state.tfidf_vectorizer = None
    st.session_state.tfidf_matrix = None
    st.session_state.bm25 = None

    lexical_texts = st.session_state.lexical_texts
    if not lexical_texts:
        return

    if TfidfVectorizer is not None:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(lexical_texts)
        st.session_state.tfidf_vectorizer = vectorizer
        st.session_state.tfidf_matrix = tfidf_matrix

    if BM25Okapi is not None:
        tokenized_corpus = [text.split() for text in lexical_texts]
        st.session_state.bm25 = BM25Okapi(tokenized_corpus)


# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Advanced Document Search", layout="wide")
st.title("Advanced Document Search")

if "doc_hash" not in st.session_state:
    st.session_state.doc_hash = None
    st.session_state.doc_name = None
    st.session_state.doc_data = None

    st.session_state.chunks = []
    st.session_state.chunk_texts = []
    st.session_state.lexical_texts = []
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
    st.session_state.last_results = []
    st.session_state.last_mode = None
    st.session_state.last_lexical_method = None
    st.session_state.last_hybrid_method = None
    st.session_state.last_alpha = None
    st.session_state.last_query_expansion = None
    st.session_state.last_validator_trace = []

    st.session_state.search_rev = 0


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Required for semantic search, hybrid search, query rewriting, and LLM validation.",
)

embedding_model = st.sidebar.selectbox(
    "Embedding model",
    ["text-embedding-3-small", "text-embedding-3-large"],
    index=0,
)

chat_model = st.sidebar.selectbox(
    "Chat model",
    ["gpt-4o-mini", "gpt-4o"],
    index=0,
)

search_mode = st.sidebar.selectbox("Search mode", ["Semantic", "Lexical", "Hybrid"])
lexical_method = st.sidebar.selectbox("Lexical method", ["BM25", "TF-IDF"])

hybrid_method = st.sidebar.selectbox(
    "Hybrid fusion",
    ["RRF", "Linear", "Rank"],
    help=(
        "RRF is usually the safest default when lexical and semantic scores live on different scales.\n"
        "Linear mixes normalized scores.\n"
        "Rank mixes rank-based unit scores."
    ),
)

score_normalization = st.sidebar.selectbox(
    "Score normalization",
    ["minmax", "zscore"],
    help="Used by Linear fusion and normalized score display.",
)

num_results = st.sidebar.slider("Final results", 1, 20, 8)
candidate_pool = st.sidebar.slider(
    "Candidate pool before LLM validation",
    5,
    50,
    20,
    help="Top-N initial results passed to the LLM validator.",
)

alpha = st.sidebar.slider(
    "Semantic weight (alpha)",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
)

rrf_k = st.sidebar.slider(
    "RRF k",
    min_value=10,
    max_value=200,
    value=60,
    step=5,
)

enable_llm_query_expansion = st.sidebar.checkbox(
    "Use LLM query expansion for lexical search",
    value=True,
)

enable_stemming = st.sidebar.checkbox(
    "Use Polish stemming (pystempel)",
    value=True,
)

enable_llm_validator = st.sidebar.checkbox(
    "Use LLM validator for final filtering",
    value=True,
)

llm_threshold = st.sidebar.slider(
    "LLM relevance threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.55,
    step=0.05,
)

debug_mode = st.sidebar.checkbox("Debug mode", value=False)

stats_placeholder = st.sidebar.empty()

st.sidebar.subheader("Document")
if st.session_state.doc_name:
    st.sidebar.caption(f"Loaded: **{st.session_state.doc_name}**")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Clear document", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

with col_b:
    if st.button("Reset embeddings", use_container_width=True):
        st.session_state.embeddings = None
        st.session_state.embeddings_model = None
        st.session_state.last_query_embedding = None
        st.session_state.umap_coords = None
        st.session_state.pca_coords = None
        st.session_state.search_rev = 0
        st.rerun()


# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload chunk JSON file", type=["json"], key="chunk_json_uploader")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.doc_hash != file_hash:
        try:
            doc_data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError as exc:
            st.error(f"Failed to parse JSON: {exc}")
            doc_data = None

        if doc_data is not None:
            stemmer = get_stemmer() if enable_stemming else None

            chunks = parse_chunks_from_doc_data(doc_data)
            chunk_texts = [chunk.get("text", "") for chunk in chunks]
            lexical_texts = [
                preprocess_text_for_lexical_search(text, stemmer=stemmer)
                for text in chunk_texts
            ]

            st.session_state.doc_hash = file_hash
            st.session_state.doc_name = uploaded_file.name
            st.session_state.doc_data = doc_data
            st.session_state.chunks = chunks
            st.session_state.chunk_texts = chunk_texts
            st.session_state.lexical_texts = lexical_texts
            st.session_state.doc_fingerprint = (
                len(chunks),
                chunks[0].get("id") if chunks else None,
                chunks[-1].get("id") if chunks else None,
            )

            rebuild_lexical_indexes()

            st.session_state.embeddings = None
            st.session_state.embeddings_model = None
            st.session_state.last_query_embedding = None
            st.session_state.umap_coords = None
            st.session_state.pca_coords = None
            st.session_state.last_results = []
            st.session_state.last_query = None
            st.session_state.last_query_expansion = None
            st.session_state.last_validator_trace = []
            st.session_state.search_rev = 0

            st.success(f"Loaded document: {uploaded_file.name}")

doc_data = st.session_state.doc_data
if doc_data is None:
    st.info("Upload a chunked JSON file to begin.")
    st.stop()


# -----------------------------
# Stats
# -----------------------------
chunks = st.session_state.chunks
chunk_texts = st.session_state.chunk_texts
lexical_texts = st.session_state.lexical_texts

total_chunks = len(chunks)
pages = {chunk.get("page") for chunk in chunks if chunk.get("page") is not None}
num_pages = len(pages)
avg_words = float(np.mean([len(tokenize_text(text)) for text in chunk_texts])) if chunk_texts else 0.0

type_counts = {}
for chunk in chunks:
    ctype = chunk.get("chunk_type", "unknown")
    type_counts[ctype] = type_counts.get(ctype, 0) + 1

type_summary = "; ".join(f"{k}: {v}" for k, v in type_counts.items())

stats_placeholder.markdown(
    f"**Document stats**  \n"
    f"- Pages: {num_pages}  \n"
    f"- Total chunks: {total_chunks}  \n"
    f"- Average chunk length: {avg_words:.1f} words  \n"
    f"- Chunk types: {type_summary}"
)


# -----------------------------
# Query and search
# -----------------------------
user_query = st.text_input("Enter your query", key="query_input")

if st.button("Search", key="search_button") and user_query:
    st.session_state.last_validator_trace = []

    stemmer = get_stemmer() if enable_stemming else None

    llm = None
    if openai_api_key and ChatOpenAI is not None:
        llm = ChatOpenAI(
            model=chat_model,
            api_key=openai_api_key,
            temperature=0,
        )

    # Lexical query rewriting
    query_expansion = expand_query_for_lexical_search(
        user_query,
        llm=llm,
        enable_llm_expansion=enable_llm_query_expansion and search_mode in {"Lexical", "Hybrid"},
    )
    st.session_state.last_query_expansion = query_expansion

    lexical_query_for_search = query_expansion["query_for_search"]
    lexical_query_processed = preprocess_text_for_lexical_search(
        lexical_query_for_search,
        stemmer=stemmer,
    )

    semantic_scores = None
    semantic_norm = None
    query_embedding = None

    lexical_scores = None
    lexical_norm = None

    if search_mode in {"Semantic", "Hybrid"}:
        if not openai_api_key:
            st.error("OpenAI API key is required for semantic or hybrid search.")
            st.stop()

        if OpenAIEmbeddings is None:
            st.error("langchain-openai is not installed.")
            st.stop()

        embedder = OpenAIEmbeddings(model=embedding_model, api_key=openai_api_key)

        need_recompute = (
            st.session_state.embeddings is None
            or st.session_state.embeddings_model != embedding_model
        )

        if need_recompute:
            with st.spinner(f"Computing document embeddings with {embedding_model}..."):
                try:
                    emb = np.asarray(embedder.embed_documents(chunk_texts), dtype=np.float32)
                    validate_embeddings_matrix(emb, len(chunk_texts))
                    st.session_state.embeddings = emb
                    st.session_state.embeddings_model = embedding_model
                    st.session_state.umap_coords = None
                    st.session_state.pca_coords = None
                except Exception as exc:
                    st.error(f"Embedding generation failed: {exc}")
                    st.stop()

        try:
            query_embedding = np.asarray(embedder.embed_query(user_query), dtype=np.float32)
            validate_query_embedding(query_embedding)
            st.session_state.last_query_embedding = query_embedding
            semantic_scores = cosine_similarity_docs_query(st.session_state.embeddings, query_embedding)
            semantic_norm = normalize_semantic_scores(semantic_scores, method=score_normalization)
        except Exception as exc:
            st.error(f"Query embedding failed: {exc}")
            st.stop()

    if search_mode in {"Lexical", "Hybrid"}:
        if lexical_method == "TF-IDF":
            if st.session_state.tfidf_vectorizer is None or st.session_state.tfidf_matrix is None:
                st.error("TF-IDF index is not available.")
                st.stop()

            q_vec = st.session_state.tfidf_vectorizer.transform([lexical_query_processed])
            lexical_scores = (st.session_state.tfidf_matrix @ q_vec.T).toarray().ravel().astype(np.float32)

        elif lexical_method == "BM25":
            if st.session_state.bm25 is None:
                st.error("BM25 index is not available.")
                st.stop()

            q_tokens = lexical_query_processed.split()
            lexical_scores = np.asarray(st.session_state.bm25.get_scores(q_tokens), dtype=np.float32)

        lexical_norm = normalize_lexical_scores(lexical_scores, method=score_normalization)

    # Fusion
    if search_mode == "Semantic":
        fused_scores = semantic_norm
    elif search_mode == "Lexical":
        fused_scores = lexical_norm
    else:
        if hybrid_method == "Linear":
            fused_scores = (alpha * semantic_norm) + ((1.0 - alpha) * lexical_norm)
        elif hybrid_method == "Rank":
            sem_rank_scores = rank_based_unit_scores(semantic_scores)
            lex_rank_scores = rank_based_unit_scores(lexical_scores)
            fused_scores = (alpha * sem_rank_scores) + ((1.0 - alpha) * lex_rank_scores)
        else:
            fused_scores = reciprocal_rank_fusion_scores(
                [semantic_scores, lexical_scores],
                weights=[alpha, 1.0 - alpha],
                k=rrf_k,
            )

    initial_indices = np.argsort(fused_scores)[::-1][:candidate_pool].tolist()

    # LLM validator
    if enable_llm_validator:
        if llm is None:
            st.warning("LLM validator requested, but no chat model is available. Showing unvalidated results.")
            validator_accepted = [
                {"index": idx, "llm_score": 1.0, "llm_label": "validator_unavailable", "llm_reason": ""}
                for idx in initial_indices
            ]
            validator_trace = []
        else:
            validator_accepted, validator_trace = validate_candidates_with_llm(
                user_query=user_query,
                chunks=chunks,
                candidate_indices=initial_indices,
                llm=llm,
                threshold=llm_threshold,
                debug=debug_mode,
            )
    else:
        validator_accepted = [
            {"index": idx, "llm_score": 1.0, "llm_label": "disabled", "llm_reason": ""}
            for idx in initial_indices
        ]
        validator_trace = []

    accepted_map = {item["index"]: item for item in validator_accepted}

    if enable_llm_validator and validator_accepted:
        rescored = []
        for idx in initial_indices:
            if idx not in accepted_map:
                continue
            llm_score = accepted_map[idx]["llm_score"]
            final_score = (0.8 * float(fused_scores[idx])) + (0.2 * float(llm_score))
            rescored.append((idx, final_score))
        final_indices = [idx for idx, _ in sorted(rescored, key=lambda x: x[1], reverse=True)[:num_results]]
    else:
        final_indices = initial_indices[:num_results]

    results = []
    for rank, idx in enumerate(final_indices, start=1):
        validator_info = accepted_map.get(
            idx,
            {"llm_score": 0.0, "llm_label": "rejected", "llm_reason": "Rejected by LLM validator."},
        )

        results.append(
            {
                "rank": rank,
                "index": idx,
                "page": chunks[idx].get("page"),
                "chunk_type": chunks[idx].get("chunk_type", "chunk"),
                "chunk_id": chunks[idx].get("id", ""),
                "text": chunks[idx].get("text", ""),
                "semantic_raw": None if semantic_scores is None else float(semantic_scores[idx]),
                "semantic_norm": None if semantic_norm is None else float(semantic_norm[idx]),
                "lexical_raw": None if lexical_scores is None else float(lexical_scores[idx]),
                "lexical_norm": None if lexical_norm is None else float(lexical_norm[idx]),
                "fused_score": float(fused_scores[idx]),
                "llm_score": float(validator_info.get("llm_score", 0.0)),
                "llm_label": validator_info.get("llm_label", ""),
                "llm_reason": validator_info.get("llm_reason", ""),
            }
        )

    st.session_state.last_results = results
    st.session_state.last_query = user_query
    st.session_state.last_mode = search_mode
    st.session_state.last_lexical_method = lexical_method if search_mode in {"Lexical", "Hybrid"} else None
    st.session_state.last_hybrid_method = hybrid_method if search_mode == "Hybrid" else None
    st.session_state.last_alpha = alpha if search_mode == "Hybrid" else None
    st.session_state.search_rev += 1


# -----------------------------
# Results rendering
# -----------------------------
if st.session_state.last_results:
    st.subheader("Search Results")

    meta = []
    if st.session_state.last_mode:
        meta.append(f"Mode: **{st.session_state.last_mode}**")
    if st.session_state.last_lexical_method:
        meta.append(f"Lexical: **{st.session_state.last_lexical_method}**")
    if st.session_state.last_hybrid_method:
        meta.append(f"Fusion: **{st.session_state.last_hybrid_method}**")
    if st.session_state.last_alpha is not None:
        meta.append(f"Alpha: **{st.session_state.last_alpha:.2f}**")
    if st.session_state.last_query:
        meta.append(f"Query: `{st.session_state.last_query}`")

    st.caption(" | ".join(meta))

    query_terms_for_highlight = tokenize_text(st.session_state.last_query or "")
    expansion = st.session_state.last_query_expansion or {}
    query_terms_for_highlight += expansion.get("keywords", [])
    query_terms_for_highlight += expansion.get("variants", [])

    for item in st.session_state.last_results:
        st.markdown(
            f"**{item['rank']}.** "
            f"*Page {item['page']}, {item['chunk_type']}, ID {item['chunk_id']}*  "
            f"**Final score:** `{item['fused_score']:.4f}`"
        )

        score_parts = []
        if item["semantic_raw"] is not None:
            score_parts.append(
                f"semantic raw: `{item['semantic_raw']:.4f}` | semantic norm: `{item['semantic_norm']:.4f}`"
            )
        if item["lexical_raw"] is not None:
            score_parts.append(
                f"lexical raw: `{item['lexical_raw']:.4f}` | lexical norm: `{item['lexical_norm']:.4f}`"
            )
        if enable_llm_validator:
            score_parts.append(
                f"llm score: `{item['llm_score']:.4f}` | label: `{item['llm_label']}`"
            )

        if score_parts:
            st.caption(" | ".join(score_parts))

        st.markdown(highlight_text(item["text"], query_terms_for_highlight), unsafe_allow_html=True)

        if debug_mode:
            with st.expander(f"Debug for result #{item['rank']}"):
                st.write(
                    {
                        "chunk_id": item["chunk_id"],
                        "page": item["page"],
                        "chunk_type": item["chunk_type"],
                        "semantic_raw": item["semantic_raw"],
                        "semantic_norm": item["semantic_norm"],
                        "lexical_raw": item["lexical_raw"],
                        "lexical_norm": item["lexical_norm"],
                        "fused_score": item["fused_score"],
                        "llm_score": item["llm_score"],
                        "llm_label": item["llm_label"],
                        "llm_reason": item["llm_reason"],
                    }
                )

        st.divider()

    if debug_mode:
        st.subheader("Debug: Lexical Query Expansion")
        st.json(st.session_state.last_query_expansion or {})

        if st.session_state.last_validator_trace:
            st.subheader("Debug: LLM Validator Trace")
            st.json(st.session_state.last_validator_trace)


# -----------------------------
# Embedding visualization
# -----------------------------
st.subheader("Embedding Space Visualization")

if not openai_api_key:
    st.info("Provide an OpenAI API key to enable embedding visualization.")
elif st.session_state.embeddings is None:
    st.info("Run a Semantic or Hybrid search first to generate embeddings.")
elif px is None:
    st.error("Plotly is not installed.")
else:
    embeddings = st.session_state.embeddings

    try:
        validate_embeddings_matrix(embeddings, expected_rows=len(chunk_texts))
    except Exception as exc:
        st.error(f"Embeddings are invalid: {exc}")
        st.stop()

    dim_method = st.selectbox("Dimensionality reduction", ["PCA", "UMAP"], key="dim_method")
    dim_num = st.radio("Dimensions", [2, 3], horizontal=True, key="dim_num")

    coords = None
    query_coords = None

    if dim_method == "PCA":
        if PCA is None:
            st.error("PCA is not available.")
        else:
            if st.session_state.pca_coords is None:
                st.session_state.pca_coords = {}

            key = f"PCA_{dim_num}_{st.session_state.embeddings_model}_{st.session_state.doc_fingerprint}"
            if key not in st.session_state.pca_coords:
                pca_model = PCA(n_components=dim_num, random_state=42)
                coords_data = pca_model.fit_transform(embeddings)

                qc = None
                if st.session_state.last_query_embedding is not None:
                    try:
                        qc = pca_model.transform(st.session_state.last_query_embedding.reshape(1, -1))
                    except Exception:
                        qc = None

                st.session_state.pca_coords[key] = (coords_data, qc)

            coords, query_coords = st.session_state.pca_coords[key]

    else:
        if umap is None:
            st.error("UMAP is not installed.")
        else:
            if st.session_state.umap_coords is None:
                st.session_state.umap_coords = {}

            key = f"UMAP_{dim_num}_{st.session_state.embeddings_model}_{st.session_state.doc_fingerprint}"
            if key not in st.session_state.umap_coords:
                reducer = umap.UMAP(n_components=dim_num, random_state=42)
                coords_data = reducer.fit_transform(embeddings)

                qc = None
                if st.session_state.last_query_embedding is not None:
                    try:
                        qc = reducer.transform(st.session_state.last_query_embedding.reshape(1, -1))
                    except Exception:
                        qc = None

                st.session_state.umap_coords[key] = (coords_data, qc)

            coords, query_coords = st.session_state.umap_coords[key]

    if coords is not None:
        import pandas as pd

        dim = coords.shape[1]
        df = {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "Type": [
                "Result"
                if any(result["index"] == i for result in st.session_state.last_results)
                else "Other"
                for i in range(coords.shape[0])
            ],
            "ChunkID": [chunk.get("id", str(i)) for i, chunk in enumerate(chunks)],
            "Page": [chunk.get("page", None) for chunk in chunks],
            "ChunkType": [chunk.get("chunk_type", "unknown") for chunk in chunks],
            "Text": [
                (chunk.get("text", "")[:100].replace("\n", " ") + ("..." if len(chunk.get("text", "")) > 100 else ""))
                for chunk in chunks
            ],
        }

        if dim == 3:
            df["z"] = coords[:, 2]

        df_plot = pd.DataFrame(df)

        color_map = {"Result": "green"}
        category_orders = {"Type": ["Other", "Result"]}

        if dim == 3:
            fig = px.scatter_3d(
                df_plot,
                x="x",
                y="y",
                z="z",
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
                x="x",
                y="y",
                color="Type",
                symbol="Type",
                color_discrete_map=color_map,
                category_orders=category_orders,
                hover_data={"ChunkID": True, "Page": True, "ChunkType": True, "Text": True},
                title="Document Chunks Embedding Space (2D)",
            )

        if query_coords is not None:
            if dim == 3:
                fig.add_scatter3d(
                    x=[query_coords[0, 0]],
                    y=[query_coords[0, 1]],
                    z=[query_coords[0, 2]],
                    mode="markers",
                    marker=dict(symbol="x", size=10, color="red"),
                    name="Query",
                )
            else:
                fig.add_scatter(
                    x=[query_coords[0, 0]],
                    y=[query_coords[0, 1]],
                    mode="markers",
                    marker=dict(symbol="x", size=14, color="red"),
                    name="Query",
                )

        ui_revision = (
            f"{st.session_state.doc_hash}|"
            f"{st.session_state.embeddings_model}|"
            f"{dim_method}|{dim_num}|{st.session_state.search_rev}"
        )

        fig.update_layout(uirevision=ui_revision)
        if dim == 3:
            fig.update_scenes(uirevision=ui_revision)

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="embedding_plot",
            config={"displaylogo": False, "scrollZoom": True},
        )
