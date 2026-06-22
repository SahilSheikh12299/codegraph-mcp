import numpy as np
import networkx as nx
from typing import List, Dict, Any
import re

from codegraph_mcp.file_parsing.scanAST import build_semantic_skeleton

SEARCH_RESULT_LIMIT = 2
SEMANTIC_POOL_SIZE = 50
STAGE1_POOL_SIZE = 20
STAGE1_SEMANTIC_WEIGHT = 0.7
STAGE1_KEYWORD_WEIGHT = 0.3
NEIGHBOR_ID_LIMIT = 5
BUCKET_SEMANTIC_TOP_K = 2
GREP_FUZZY_TOP_K = 3
GREP_EXACT_TOP_K = 1
NEIGHBOR_RERANK_POOL = 20


def compute_keyword_score(data: Dict[str, Any], keywords: List[str]) -> float:
    """Substring presence score; prefer name/signature over body text."""
    if not keywords:
        return 0.0
    name = (data.get("name") or "").lower()
    signature = (data.get("signature") or "").lower()
    docstring = (data.get("docstring") or "").lower()
    chunk = (data.get("chunk_text") or "").lower()

    hits = 0.0
    for kw in keywords:
        if kw in name or kw in signature:
            hits += 3.0
        elif kw in docstring:
            hits += 2.0
        elif kw in chunk:
            hits += 1.0
    max_possible = len(keywords) * 3.0
    return hits / max_possible if max_possible else 0.0


def build_rerank_passage(data: Dict[str, Any]) -> str:
    """Compact embedding metadata + AST semantic skeleton for cross-encoder reranking."""
    meta = (data.get("embedding_text") or "").strip()
    skeleton = build_semantic_skeleton(
        data.get("chunk_text", ""),
        data.get("type", "FUNCTION"),
        data=data,
    )
    if meta and skeleton and skeleton != meta:
        return f"{meta}\n\n{skeleton}"
    return skeleton or meta


def build_grep_text(data: Dict[str, Any]) -> str:
    """Compact searchable text for literal grep-replica (no chunk_text)."""
    parts = [
        data.get("name") or "",
        data.get("signature") or "",
        data.get("file_path") or "",
        data.get("embedding_text") or "",
    ]
    return "\n".join(p for p in parts if p).strip()


def _parse_line_span(meta: Dict[str, Any]) -> tuple[int | None, int | None]:
    span = meta.get("line_span")
    if span and isinstance(span, (list, tuple)) and len(span) >= 2:
        return int(span[0]), int(span[1])
    return None, None


def format_grep_match(
    node_id: str,
    meta: Dict[str, Any],
    *,
    score: float | None = None,
    match_mode: str | None = None,
) -> Dict[str, Any]:
    """Minimal grep hit: identity + span only (no neighbors, no embedding_text)."""
    start_line, end_line = _parse_line_span(meta)
    out: Dict[str, Any] = {
        "node_id": node_id,
        "type": meta.get("type"),
        "file_path": meta.get("file_path"),
        "name": meta.get("name"),
        "signature": meta.get("signature", ""),
        "start_line": start_line,
        "end_line": end_line,
    }
    if score is not None:
        out["score"] = round(score, 4)
    if match_mode:
        out["match_mode"] = match_mode
    return out


def format_search_candidate(
    G: nx.DiGraph,
    node_id: str,
    meta: Dict[str, Any],
    *,
    score: float | None = None,
    match_mode: str | None = None,
) -> Dict[str, Any]:
    """Search hit: identity + span only (no embedding_text, no neighbor arrays)."""
    start_line, end_line = _parse_line_span(meta)
    out: Dict[str, Any] = {
        "node_id": node_id,
        "type": meta.get("type"),
        "file_path": meta.get("file_path"),
        "name": meta.get("name"),
        "signature": meta.get("signature", ""),
        "start_line": start_line,
        "end_line": end_line,
    }
    if score is not None:
        out["score"] = round(score, 4)
    if match_mode:
        out["match_mode"] = match_mode
    return out


def call_chain_entry(G: nx.DiGraph, node_id: str, role: str) -> Dict[str, Any]:
    meta = G.nodes.get(node_id, {})
    start_line, end_line = _parse_line_span(meta)
    return {
        "role": role,
        "node_id": node_id,
        "name": meta.get("name"),
        "file_path": meta.get("file_path"),
        "start_line": start_line,
        "end_line": end_line,
        "signature": meta.get("signature", ""),
    }


def build_query_ranked_call_chain(
    G: nx.DiGraph,
    anchor_id: str,
    queries: List[str],
    *,
    embedder: Any = None,
    reranker: Any = None,
    max_callers: int = 2,
    max_callees: int = 2,
    max_downstream_hops: int = 1,
) -> List[Dict[str, Any]]:
    """Ordered upstream callers → anchor → downstream callees (query-ranked)."""
    if not G.has_node(anchor_id):
        return [call_chain_entry(G, anchor_id, "anchor")]

    neighbors = rank_call_neighbors(
        G,
        anchor_id,
        queries,
        embedder=embedder,
        reranker=reranker,
        limit=max(max_callers, max_callees, 1),
    )
    callers = neighbors.get("callers", [])[:max_callers]
    callees = neighbors.get("callees", [])[:max_callees]

    chain: List[Dict[str, Any]] = []
    for nid in reversed(callers):
        chain.append(call_chain_entry(G, nid, "caller"))
    chain.append(call_chain_entry(G, anchor_id, "anchor"))
    for nid in callees:
        chain.append(call_chain_entry(G, nid, "callee"))

    if max_downstream_hops > 0 and callees:
        top_callee = callees[0]
        if G.has_node(top_callee):
            sub = rank_call_neighbors(
                G,
                top_callee,
                queries,
                embedder=embedder,
                reranker=reranker,
                limit=1,
            )
            seen = {e["node_id"] for e in chain}
            for nid in sub.get("callees", [])[:1]:
                if nid not in seen:
                    chain.append(call_chain_entry(G, nid, "downstream"))

    return chain


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _max_cosine_sims_per_node(
    query_vectors: np.ndarray,
    node_matrix: np.ndarray,
) -> np.ndarray:
    """Max cosine similarity across query rows for each node row. Returns shape (N,)."""
    if node_matrix.size == 0:
        return np.array([], dtype=np.float64)
    q_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    n_norms = np.linalg.norm(node_matrix, axis=1, keepdims=True)
    q_normalized = query_vectors / np.where(q_norms == 0, 1.0, q_norms)
    n_normalized = node_matrix / np.where(n_norms == 0, 1.0, n_norms)
    max_sim = (q_normalized @ n_normalized.T).max(axis=0)
    max_sim[n_norms.ravel() == 0] = 0.0
    return max_sim


def _all_call_neighbors(G: nx.DiGraph, node_id: str) -> tuple[list[str], list[str]]:
    """Return all 1-hop CALLS predecessors and successors (uncapped)."""
    if not G.has_node(node_id):
        return [], []
    callers = sorted(
        p for p in G.predecessors(node_id)
        if G.edges[p, node_id].get("relationship") == "CALLS"
    )
    callees = sorted(
        s for s in G.successors(node_id)
        if G.edges[node_id, s].get("relationship") == "CALLS"
    )
    return callers, callees


def rank_call_neighbors(
    G: nx.DiGraph,
    node_id: str,
    queries: List[str],
    *,
    embedder: Any = None,
    reranker: Any = None,
    limit: int = NEIGHBOR_ID_LIMIT,
) -> dict[str, list[str]]:
    """Rank 1-hop callers/callees by query relevance (embedding + optional cross-encoder)."""
    callers_all, callees_all = _all_call_neighbors(G, node_id)
    if not callers_all and not callees_all:
        return {"callers": [], "callees": []}

    cleaned_queries = [q.strip() for q in queries if q and q.strip()]
    query_vectors: list[list[float]] = []
    if embedder is not None and cleaned_queries:
        encoded = embedder.model.encode(cleaned_queries, convert_to_numpy=True)
        if encoded.ndim == 1:
            encoded = encoded.reshape(1, -1)
        query_vectors = [qv.tolist() for qv in encoded]

    def _embedding_score(nid: str) -> float:
        data = G.nodes.get(nid, {})
        emb = data.get("embedding")
        if not emb or not query_vectors:
            return float(data.get("call_centrality") or 0)
        sim = max(_cosine_similarity(qv, emb) for qv in query_vectors)
        centrality = float(data.get("call_centrality") or 0)
        return sim + 0.1 * centrality

    def _rank_side(ids: list[str]) -> list[str]:
        if not ids:
            return []
        scored = [(nid, _embedding_score(nid)) for nid in ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        pool = scored[:NEIGHBOR_RERANK_POOL]

        if reranker and cleaned_queries and pool:
            pairs: list[list[str]] = []
            pair_ids: list[str] = []
            for nid, _ in pool:
                passage = build_rerank_passage(G.nodes[nid])
                for query in cleaned_queries:
                    pairs.append([query, passage])
                    pair_ids.append(nid)
            ce_scores = reranker.predict(pairs, batch_size=16)
            node_max: dict[str, float] = {}
            for nid, score in zip(pair_ids, ce_scores):
                score_f = float(score)
                if nid not in node_max or score_f > node_max[nid]:
                    node_max[nid] = score_f
            pool = sorted(node_max.items(), key=lambda x: x[1], reverse=True)
            return [nid for nid, _ in pool[:limit]]

        return [nid for nid, _ in pool[:limit]]

    return {
        "callers": _rank_side(callers_all),
        "callees": _rank_side(callees_all),
    }


def _grep_literal_rank(
    term: str,
    node_id: str,
    data: Dict[str, Any],
    case_sensitive: bool,
    affinity_paths: set[str] | None = None,
) -> float:
    """Higher = better literal match (exact name preferred)."""
    name = data.get("name") or ""
    sig = data.get("signature") or ""
    t = term if case_sensitive else term.lower()
    n = name if case_sensitive else name.lower()
    s = sig if case_sensitive else sig.lower()
    centrality = float(data.get("call_centrality") or 0)

    if n == t:
        score = 100.0
    elif n.endswith(f".{t}") or f".{t}" in node_id:
        score = 90.0
    elif t in n:
        score = 70.0 + centrality
    elif t in s and t not in n:
        score = 35.0 + centrality
    else:
        score = 50.0 + centrality

    if len(t) <= 4 and n != t and t in s:
        score -= 25.0

    if affinity_paths and data.get("file_path") in affinity_paths:
        score += 15.0

    return score


def _grep_fuzzy_rank(term: str, node_id: str, data: Dict[str, Any], case_sensitive: bool) -> float:
    """Token overlap on name/signature when no literal grep_text hit."""
    t = term if case_sensitive else term.lower()
    name = (data.get("name") or "")
    sig = (data.get("signature") or "")
    hay = f"{name} {sig}" if case_sensitive else f"{name} {sig}".lower()
    if t in hay:
        return 80.0 + float(data.get("call_centrality") or 0)
    tokens = [tok for tok in re.findall(r"[a-zA-Z0-9_]+", t) if len(tok) >= 2]
    if not tokens:
        return 0.0
    hits = sum(1 for tok in tokens if tok in hay)
    return (hits / len(tokens)) * 50.0 + float(data.get("call_centrality") or 0)


def _expand_symbolic_grep_term(term: str) -> list[tuple[str, float]]:
    """Expand dotted/qualified symbol strings into weighted sub-terms.

    Goal: keep precision while improving recall for terms like `Session.send` where
    the code symbol is typically just `send`.
    """
    t = (term or "").strip()
    if not t:
        return []

    # Default: no expansion.
    expansions: list[tuple[str, float]] = [(t, 1.0)]

    # Handle dotted / qualified forms.
    if "." in t or "::" in t:
        # Normalize separators into dots for tokenization.
        normalized = t.replace("::", ".")
        parts = [p for p in normalized.split(".") if p]
        if len(parts) >= 2:
            cls, member = parts[-2], parts[-1]
            # Keep original (highest weight), then add member-only for recall.
            expansions = [
                (t, 1.0),
                (member, 0.85),
                (f"{cls} {member}", 0.8),
                (cls, 0.35),
            ]

    # Deduplicate while preserving best weight.
    best: dict[str, float] = {}
    for s, w in expansions:
        s2 = s.strip()
        if not s2:
            continue
        if s2 not in best or w > best[s2]:
            best[s2] = w
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def grep_search_nodes_symbol_aware(
    G: nx.DiGraph,
    term: str,
    *,
    case_sensitive: bool = False,
    embedder: Any = None,
    fuzzy_top_k: int = GREP_FUZZY_TOP_K,
    affinity_paths: set[str] | None = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Symbol-aware grep: expand qualified terms and fuse results.

    This avoids `Session.send` missing the real `send()` method and accidentally
    preferring `test_*` functions that contain both tokens.
    """
    expansions = _expand_symbolic_grep_term(term)
    if not expansions:
        return []

    fused: dict[str, Dict[str, Any]] = {}
    fused_scores: dict[str, float] = {}

    for subterm, weight in expansions:
        hits = grep_search_nodes(
            G,
            subterm,
            case_sensitive=case_sensitive,
            embedder=embedder,
            fuzzy_top_k=fuzzy_top_k,
            affinity_paths=affinity_paths,
        )
        for h in hits:
            nid = h.get("node_id")
            if not nid:
                continue
            # `grep_search_nodes` provides `score` for both exact and fuzzy.
            base = float(h.get("score") or 0.0)
            score = base * weight
            if nid not in fused_scores or score > fused_scores[nid]:
                fused_scores[nid] = score
                fused[nid] = h

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[: max(1, top_k)]
    out: list[Dict[str, Any]] = []
    for nid, score in ranked:
        h = dict(fused[nid])
        h["score"] = round(score, 4)
        out.append(h)
    return out


def grep_search_nodes(
    G: nx.DiGraph,
    term: str,
    *,
    case_sensitive: bool = False,
    embedder: Any = None,
    fuzzy_top_k: int = GREP_FUZZY_TOP_K,
    affinity_paths: set[str] | None = None,
) -> List[Dict[str, Any]]:
    """Literal grep on grep_text; exact mode returns top 1, fuzzy mode returns top 3."""
    term = (term or "").strip()
    if not term:
        return []

    literal_hits: list[tuple[str, Dict[str, Any], float]] = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") not in ("CLASS", "FUNCTION"):
            continue
        grep_text = data.get("grep_text") or build_grep_text(data)
        hay = grep_text if case_sensitive else grep_text.lower()
        needle = term if case_sensitive else term.lower()
        if needle in hay:
            score = _grep_literal_rank(
                term, node_id, data, case_sensitive, affinity_paths=affinity_paths
            )
            literal_hits.append((node_id, data, score))

    if literal_hits:
        literal_hits.sort(key=lambda x: x[2], reverse=True)
        top = literal_hits[:GREP_EXACT_TOP_K]
        return [
            format_grep_match(nid, meta, score=sc, match_mode="exact")
            for nid, meta, sc in top
        ]

    fuzzy_hits: list[tuple[str, Dict[str, Any], float]] = []
    term_vec = None
    if embedder is not None:
        term_vec = embedder.model.encode(term, convert_to_numpy=True).tolist()

    for node_id, data in G.nodes(data=True):
        if data.get("type") not in ("CLASS", "FUNCTION"):
            continue
        score = _grep_fuzzy_rank(term, node_id, data, case_sensitive)
        name_emb = data.get("name_embedding")
        if term_vec and name_emb:
            cos = float(np.dot(term_vec, name_emb) / (np.linalg.norm(term_vec) * np.linalg.norm(name_emb) + 1e-9))
            score = max(score, cos * 60.0)
        if score > 0:
            fuzzy_hits.append((node_id, data, score))

    fuzzy_hits.sort(key=lambda x: x[2], reverse=True)
    top = fuzzy_hits[:fuzzy_top_k]
    return [
        format_grep_match(nid, meta, score=sc, match_mode="fuzzy")
        for nid, meta, sc in top
    ]


def recompute_call_centrality(G: nx.DiGraph) -> None:
    """CALLS in-degree per CLASS/FUNCTION node, normalized to call_centrality in [0, 1]."""
    code_nodes = {
        nid for nid, d in G.nodes(data=True) if d.get("type") in ("CLASS", "FUNCTION")
    }
    in_degree = {nid: 0 for nid in code_nodes}
    for _, v, d in G.edges(data=True):
        if d.get("relationship") == "CALLS" and v in in_degree:
            in_degree[v] += 1
    max_deg = max(in_degree.values()) if in_degree else 0
    for nid, deg in in_degree.items():
        G.nodes[nid]["calls_in_degree"] = deg
        G.nodes[nid]["call_centrality"] = (deg / max_deg) if max_deg else 0.0
    G.graph["centrality_schema"] = 1


class AdvancedRetrievalEngine:
    """The analytical core of the Graph-RAG system. Handles dual-pronged retrieval,
    mathematical cliff-detection re-ranking, and recursive upstream blast-radius mappings.
    """

    def __init__(
        self,
        graph_instance: nx.DiGraph,
        embedder_instance: Any,
        reranker: Any = None,
    ):
        """Initializes the engine with a living instance of the codebase graph
        and an offline-locked embedding module from the server pool.
        """
        self.G = graph_instance
        self.embedder = embedder_instance
        self.reranker = reranker
    
    def run_hybrid_retrieval(
        self,
        search_queries: List[str],
        keywords: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Two-stage retrieval: semantic top 50 → keyword blend top 20 → cross-encoder top 2."""
        keyword_list = keywords or []
        queries = [q.strip() for q in search_queries if q and q.strip()]
        if not queries:
            return []

        query_vectors = self.embedder.model.encode(queries, convert_to_numpy=True)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        node_ids: list[str] = []
        node_data_list: list[Dict[str, Any]] = []
        node_rows: list[list[float]] = []
        for node_id, data in self.G.nodes(data=True):
            if data.get("type") not in ("CLASS", "FUNCTION"):
                continue
            node_vector = data.get("embedding")
            if not node_vector or len(node_vector) == 0:
                continue
            node_ids.append(node_id)
            node_data_list.append(data)
            node_rows.append(node_vector)

        scored_hits: list[tuple[str, Dict[str, Any], float]] = []
        if node_rows:
            node_matrix = np.asarray(node_rows, dtype=np.float64)
            max_sims = _max_cosine_sims_per_node(query_vectors, node_matrix)
            for node_id, data, max_sim in zip(node_ids, node_data_list, max_sims):
                max_sim_f = float(max_sim)
                if max_sim_f > 0.0:
                    scored_hits.append((node_id, data, max_sim_f))

        scored_hits.sort(key=lambda x: x[2], reverse=True)
        semantic_pool = scored_hits[:SEMANTIC_POOL_SIZE]

        stage1: list[tuple[str, Dict[str, Any], float, float, float]] = []
        for node_id, data, sim in semantic_pool:
            kw_score = compute_keyword_score(data, keyword_list)
            prelim = STAGE1_SEMANTIC_WEIGHT * sim + STAGE1_KEYWORD_WEIGHT * kw_score
            stage1.append((node_id, data, prelim, kw_score, sim))

        stage1.sort(key=lambda x: x[2], reverse=True)
        stage1_pool = stage1[:STAGE1_POOL_SIZE]

        if self.reranker and stage1_pool:
            pairs: list[list[str]] = []
            pair_node_ids: list[str] = []
            for node_id, data, _, _, _ in stage1_pool:
                passage = build_rerank_passage(data)
                for query in queries:
                    pairs.append([query, passage])
                    pair_node_ids.append(node_id)

            ce_scores = self.reranker.predict(pairs, batch_size=16)
            node_meta = {nid: (data, kw) for nid, data, _, kw, _ in stage1_pool}
            node_max_scores: dict[str, float] = {}
            for node_id, score in zip(pair_node_ids, ce_scores):
                score_f = float(score)
                if node_id not in node_max_scores or score_f > node_max_scores[node_id]:
                    node_max_scores[node_id] = score_f

            final_reranked = [
                (nid, node_meta[nid][0], node_max_scores[nid], node_meta[nid][1])
                for nid in node_max_scores
            ]
            final_reranked.sort(key=lambda x: x[2], reverse=True)
        else:
            final_reranked = [
                (node_id, data, prelim, kw_score)
                for node_id, data, prelim, kw_score, _ in stage1_pool
            ]

        final_selections = final_reranked[:SEARCH_RESULT_LIMIT]

        return [
            {
                "node_id": hit[0],
                "metadata": hit[1],
                "hybrid_score": round(hit[2], 4),
                "keyword_score": round(hit[3], 4),
                "callers": hit[1].get("calls_in_degree", 0),
            }
            for hit in final_selections
        ]

    def compile_bucketed_context_package(
        self,
        search_queries: List[str],
        grep_terms: List[str] | None = None,
    ) -> dict[str, Any]:
        """Bucketed retrieval.

        Returns:
        - semantic_by_query: top semantic candidates per individual query (preferred for term-scoped UX)
        - semantic_bucket: legacy pooled semantic candidates across all queries (kept for compatibility)
        - grep_buckets: per-grep-term candidates
        """
        grep_terms = [t.strip() for t in (grep_terms or []) if t and t.strip()]
        queries = [q.strip() for q in search_queries if q and q.strip()]

        semantic_by_query: list[dict[str, Any]] = []
        semantic_nodes: list[dict[str, Any]] = []  # legacy pooled bucket
        if queries:
            # Per-query semantic hits (top-K per query).
            for q in queries:
                q_hits = self.run_hybrid_retrieval([q], keywords=[])
                q_nodes: list[dict[str, Any]] = []
                for hit in q_hits[:BUCKET_SEMANTIC_TOP_K]:
                    q_nodes.append(
                        format_search_candidate(
                            self.G,
                            hit["node_id"],
                            hit["metadata"],
                            score=hit["hybrid_score"],
                        )
                    )
                semantic_by_query.append({"query": q, "nodes": q_nodes})

            # Legacy pooled semantic hits across the full query list.
            query_hits = self.run_hybrid_retrieval(queries, keywords=[])
            for hit in query_hits[:BUCKET_SEMANTIC_TOP_K]:
                semantic_nodes.append(
                    format_search_candidate(
                        self.G,
                        hit["node_id"],
                        hit["metadata"],
                        score=hit["hybrid_score"],
                    )
                )

        affinity_paths = {n["file_path"] for n in semantic_nodes if n.get("file_path")}

        grep_buckets: list[dict[str, Any]] = []
        for term in grep_terms:
            # Symbol-aware expansion helps with qualified terms like `Session.send`.
            nodes = grep_search_nodes_symbol_aware(
                self.G,
                term,
                embedder=self.embedder,
                affinity_paths=affinity_paths or None,
                top_k=max(GREP_FUZZY_TOP_K, 3),
            )
            grep_buckets.append({
                "term": term,
                "match_mode": nodes[0]["match_mode"] if nodes else "none",
                "nodes": nodes,
            })

        return {
            "rewritten_queries": queries,
            "grep_terms": grep_terms,
            "semantic_by_query": semantic_by_query,
            "grep_buckets": grep_buckets,
        }

