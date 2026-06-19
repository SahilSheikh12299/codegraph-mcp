# import numpy as np
# import networkx as nx
# from typing import List, Dict, Any, Tuple
# import ast

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
import json
import re

from fileParsing.scanAST import build_semantic_skeleton

SEARCH_RESULT_LIMIT = 6
SEMANTIC_POOL_SIZE = 50
STAGE1_POOL_SIZE = 20
STAGE1_SEMANTIC_WEIGHT = 0.7
STAGE1_KEYWORD_WEIGHT = 0.3
SOURCE_EXCERPT_MAX_LINES = 100
NEIGHBOR_ID_LIMIT = 50
BUCKET_SEMANTIC_TOP_K = 3
GREP_NEIGHBOR_LIMIT = 5
GREP_FUZZY_TOP_K = 3
GREP_EXACT_TOP_K = 1

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "how", "what", "where", "when",
    "does", "do", "did", "for", "and", "or", "to", "in", "on", "of", "with",
    "after", "before", "from", "by", "it", "this", "that", "can", "will", "not",
})


def extract_keywords_from_queries(search_queries: List[str]) -> List[str]:
    """Loose keyword tokens from all search queries (no exact-symbol mode)."""
    # TODO: Add more advanced keyword extraction logic here. Use spacy for better keyword extraction.
    keywords: set[str] = set()
    for query in search_queries:
        for token in re.findall(r"[a-zA-Z0-9_]+", query.lower()):
            if len(token) >= 3 and token not in _STOPWORDS:
                keywords.add(token)
    return sorted(keywords)


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


def extract_source_excerpt(
    chunk_text: str,
    node_type: str = "FUNCTION",
    max_lines: int = SOURCE_EXCERPT_MAX_LINES,
) -> str:
    """Capped source: first max_lines, plus return line(s) if truncated."""
    if not chunk_text:
        return ""

    lines = chunk_text.splitlines()
    if not lines:
        return ""

    if node_type == "CLASS":
        excerpt = lines[:max_lines]
        if len(lines) > max_lines:
            excerpt.append(f"# ... truncated ({len(lines)} lines total)")
        return "\n".join(excerpt)

    head = lines[:max_lines]
    tail_returns: list[str] = []
    for ln in lines[max_lines:]:
        stripped = ln.lstrip()
        if stripped.startswith("return ") or stripped == "return":
            tail_returns.append(ln)
            if len(tail_returns) >= 3:
                break

    parts = list(head)
    if len(lines) > max_lines:
        parts.append(f"# ... truncated ({len(lines)} lines total)")
    parts.extend(tail_returns)
    return "\n".join(parts)


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


def graph_coverage_fields(G: nx.DiGraph) -> dict[str, Any]:
    file_nodes = [d for _, d in G.nodes(data=True) if d.get("type") == "FILE"]
    return {
        "python_files_indexed": len(file_nodes),
        "last_indexed_at": G.graph.get("last_synced_at"),
        "missing_paths": [],
    }


def build_grep_text(data: Dict[str, Any]) -> str:
    """Compact searchable text for literal grep-replica (no chunk_text)."""
    parts = [
        data.get("name") or "",
        data.get("signature") or "",
        data.get("file_path") or "",
        data.get("embedding_text") or "",
    ]
    return "\n".join(p for p in parts if p).strip()


def format_compact_candidate(
    G: nx.DiGraph,
    node_id: str,
    meta: Dict[str, Any],
    *,
    score: float | None = None,
    neighbors_limit: int = GREP_NEIGHBOR_LIMIT,
    match_mode: str | None = None,
) -> Dict[str, Any]:
    """Token-tight node payload: embedding_text + neighbor IDs only."""
    callers, callees = get_call_neighbors(G, node_id, limit=neighbors_limit)
    start_line, end_line = None, None
    span = meta.get("line_span")
    if span and isinstance(span, (list, tuple)) and len(span) >= 2:
        start_line, end_line = int(span[0]), int(span[1])
    out: Dict[str, Any] = {
        "node_id": node_id,
        "type": meta.get("type"),
        "file_path": meta.get("file_path"),
        "name": meta.get("name"),
        "signature": meta.get("signature", ""),
        "start_line": start_line,
        "end_line": end_line,
        "embedding_text": (meta.get("embedding_text") or "").strip(),
        "neighbors": {"callers": callers, "callees": callees},
    }
    if score is not None:
        out["score"] = round(score, 4)
    if match_mode:
        out["match_mode"] = match_mode
    return out


def _grep_literal_rank(term: str, node_id: str, data: Dict[str, Any], case_sensitive: bool) -> float:
    """Higher = better literal match (exact name preferred)."""
    name = data.get("name") or ""
    t = term if case_sensitive else term.lower()
    n = name if case_sensitive else name.lower()
    if n == t:
        return 100.0
    if n.endswith(f".{t}") or f".{t}" in node_id:
        return 90.0
    if t in n:
        return 70.0 + float(data.get("call_centrality") or 0)
    return 50.0 + float(data.get("call_centrality") or 0)


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


def grep_search_nodes(
    G: nx.DiGraph,
    term: str,
    *,
    case_sensitive: bool = False,
    embedder: Any = None,
    fuzzy_top_k: int = GREP_FUZZY_TOP_K,
) -> List[Dict[str, Any]]:
    """Literal grep on grep_text; exact mode returns top 1, fuzzy mode returns top 3."""
    term = (term or "").strip()
    if not term:
        return []

    literal_hits: list[tuple[str, Dict[str, Any], float]] = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") not in ("CLASS", "FUNCTION", "CONSTANT"):
            continue
        grep_text = data.get("grep_text") or build_grep_text(data)
        hay = grep_text if case_sensitive else grep_text.lower()
        needle = term if case_sensitive else term.lower()
        if needle in hay:
            score = _grep_literal_rank(term, node_id, data, case_sensitive)
            literal_hits.append((node_id, data, score))

    if literal_hits:
        literal_hits.sort(key=lambda x: x[2], reverse=True)
        top = literal_hits[:GREP_EXACT_TOP_K]
        return [
            format_compact_candidate(G, nid, meta, score=sc, match_mode="exact")
            for nid, meta, sc in top
        ]

    fuzzy_hits: list[tuple[str, Dict[str, Any], float]] = []
    term_vec = None
    if embedder is not None:
        term_vec = embedder.model.encode(term, convert_to_numpy=True).tolist()

    for node_id, data in G.nodes(data=True):
        if data.get("type") not in ("CLASS", "FUNCTION", "CONSTANT"):
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
        format_compact_candidate(G, nid, meta, score=sc, match_mode="fuzzy")
        for nid, meta, sc in top
    ]


def get_call_neighbors(G: nx.DiGraph, node_id: str, limit: int = 5) -> Tuple[List[str], List[str]]:
    """Return 1-hop CALLS predecessors (callers) and successors (callees)."""
    if not G.has_node(node_id):
        return [], []
    callers = [
        p for p in G.predecessors(node_id)
        if G.edges[p, node_id].get("relationship") == "CALLS"
    ]
    callees = [
        s for s in G.successors(node_id)
        if G.edges[node_id, s].get("relationship") == "CALLS"
    ]
    return sorted(callers)[:limit], sorted(callees)[:limit]


def format_call_neighbors(G: nx.DiGraph, node_id: str, limit: int = 5) -> str:
    callers, callees = get_call_neighbors(G, node_id, limit)
    lines = []
    if callers:
        lines.append(f"- called_by: {', '.join(f'`{c}`' for c in callers)}")
    if callees:
        lines.append(f"- calls: {', '.join(f'`{c}`' for c in callees)}")
    return "\n".join(lines)


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

    def _extract_tiny_slice(self, chunk_text: str, max_lines: int = 8) -> str:
        """Returns signature + first lines of actual body logic (skips docstring/comments)."""
        if not chunk_text:
            return ""
        lines = [ln.rstrip() for ln in chunk_text.strip().splitlines() if ln.strip()]
        if not lines:
            return ""
        result = [lines[0]]
        in_docstring = False
        body_lines = 0
        for ln in lines[1:]:
            s = ln.strip()
            if s.startswith('"""') or s.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring or s.startswith("#"):
                continue
            result.append(ln)
            body_lines += 1
            if body_lines >= max_lines - 1:
                break
        return "\n".join(result)

    def _calculate_cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Computes the standard cosine similarity between two normalized vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        
        # Absolute safety check taken from your older vector calculation mechanics
        if a.shape[0] == 0 or b.shape[0] == 0:
            return 0.0
            
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
    
    def _execute_cliff_detection(self, scored_hits: List[Tuple[str, Dict[str, Any], float, str]], 
                                 min_k: int = 2, max_k: int = 5, sensitivity: float = 0.05) -> int:
        """Analyzes similarity drop-offs mathematically to identify either sharp cliffs
        or relative plateaus of relevancy, returning a dynamic cutoff index.
        """
        if len(scored_hits) <= min_k:
            return len(scored_hits)
            
        limit = min(len(scored_hits), max_k)
        
        for i in range(min_k - 1, limit - 1):
            current_score = scored_hits[i][2]
            next_score = scored_hits[i + 1][2]
            delta = current_score - next_score
            
            if delta >= sensitivity:
                return i + 1
                
        return limit

    def run_hybrid_retrieval(
        self,
        search_queries: List[str],
        keywords: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Two-stage retrieval: semantic top 50 → keyword blend top 20 → cross-encoder top 6."""
        keyword_list = keywords or []
        queries = [q.strip() for q in search_queries if q and q.strip()]
        if not queries:
            return []

        query_vectors = self.embedder.model.encode(queries, convert_to_numpy=True)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        scored_hits: list[tuple[str, Dict[str, Any], float]] = []
        for node_id, data in self.G.nodes(data=True):
            if "type" not in data:
                continue

            node_vector = data.get("embedding")
            if not node_vector or len(node_vector) == 0:
                continue

            max_sim = max(
                self._calculate_cosine_similarity(qv.tolist(), node_vector)
                for qv in query_vectors
            )
            if max_sim > 0.0:
                scored_hits.append((node_id, data, max_sim))

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

    def compute_upstream_blast_radius(self, target_nodes: List[str]) -> List[Dict[str, Any]]:
        """Traverses the directional edges of the network graph upward to trace
        every entity that will break if the target nodes are modified.
        """
        all_affected_nodes = set()
        
        for node_id in target_nodes:
            if self.G.has_node(node_id):
                upstream_ancestors = nx.ancestors(self.G, node_id)
                all_affected_nodes.update(upstream_ancestors)

        affected_profiles = []
        for node_id in all_affected_nodes:
            data = self.G.nodes[node_id]
            if "type" not in data:
                continue
            affected_profiles.append({
                "node_id": node_id,
                "type": data.get("type"),
                "name": data.get("name"),
                "file_path": data.get("file_path"),
                "signature": data.get("signature", "")
            })
            
        return affected_profiles

    def compile_context_package(
        self,
        search_queries: List[str],
        # targeted_symbols: List[str] = None,
        top_k: int | None = None,
        output_format: str = "json",
        include_next_action: bool = True,
    ) -> str:
        """Multi-query hybrid retrieval. Top 2 hits include capped source_excerpt + neighbor IDs."""
        keywords = extract_keywords_from_queries(search_queries)
        query_hits = self.run_hybrid_retrieval(search_queries, keywords=keywords)

        if not query_hits:
            if output_format == "json":
                return json.dumps(
                    {"queries": search_queries, "candidates": [], "message": "No matches found."},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            return "### [Graph-RAG System Message]\nNo relevant structural context matched this query matrix."

        limit = SEARCH_RESULT_LIMIT if top_k is None else max(1, int(top_k))
        final_seeds = query_hits[:limit]

        def _json_dumps(obj: Any) -> str:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)

        def _neighbor_ids(nid: str) -> dict[str, list[str]]:
            callers, callees = get_call_neighbors(self.G, nid, limit=NEIGHBOR_ID_LIMIT)
            return {"callers": callers, "callees": callees}

        if output_format == "json":
            candidates: list[dict[str, Any]] = []
            for idx, hit in enumerate(final_seeds):
                meta = hit["metadata"]
                node_id = hit["node_id"]
                candidate: dict[str, Any] = {
                    "node_id": node_id,
                    "type": meta.get("type"),
                    "file_path": meta.get("file_path"),
                    "name": meta.get("name"),
                    "signature": meta.get("signature", ""),
                    "score": hit["hybrid_score"],
                    "neighbors": _neighbor_ids(node_id),
                }
                if idx < 2 and meta.get("chunk_text"):
                    excerpt = extract_source_excerpt(
                        meta.get("chunk_text", ""),
                        meta.get("type", "FUNCTION"),
                    )
                    candidate["source_excerpt"] = excerpt
                    truncated = "# ... truncated" in excerpt
                    candidate["truncated"] = truncated
                    candidate["source_complete"] = not truncated
                    if truncated:
                        candidate["fetch_hint"] = "Call fetch_node_source for full source."
                candidates.append(candidate)

            best_id = candidates[0]["node_id"] if candidates else None
            has_hits = bool(candidates)
            out: dict[str, Any] = {
                "queries": search_queries,
                "keywords": keywords,
                "candidates": candidates,
                "graph_coverage": graph_coverage_fields(self.G),
                "sufficient_to_answer": has_hits,
                "do_not_use_native_read": has_hits,
            }
            if include_next_action and best_id:
                out["next_action"] = {
                    "tool": "fetch_node_source",
                    "args": {"node_ids": [best_id]},
                }
            return _json_dumps(out)

        markdown = "## Codebase Search Results\n\n"
        markdown += f"Queries: {', '.join(search_queries)}\n"
        markdown += (
            f"Showing top {len(final_seeds)} matches. "
            f"Stage 1: {STAGE1_SEMANTIC_WEIGHT}×semantic + {STAGE1_KEYWORD_WEIGHT}×keywords (top {STAGE1_POOL_SIZE}); "
            f"Stage 2: cross-encoder rerank.\n\n"
        )

        for idx, hit in enumerate(final_seeds, 1):
            meta = hit["metadata"]
            node_id = hit["node_id"]
            neighbors = _neighbor_ids(node_id)

            markdown += f"### Hit {idx}: `{node_id}`\n"
            markdown += f"- score: {hit['hybrid_score']:.4f} | callers: {hit['callers']}\n"
            markdown += f"- type: {meta.get('type')}\n"
            markdown += f"- signature: {meta.get('signature', '()')}\n"
            markdown += f"- neighbors: callers={len(neighbors['callers'])}, callees={len(neighbors['callees'])}\n"

            if idx <= 2 and meta.get("chunk_text"):
                excerpt = extract_source_excerpt(meta.get("chunk_text", ""), meta.get("type", "FUNCTION"))
                markdown += f"\n```python\n{excerpt}\n```\n"
            markdown += "\n"

        return markdown

    def compile_bucketed_context_package(
        self,
        search_queries: List[str],
        grep_terms: List[str] | None = None,
        *,
        include_next_action: bool = True,
    ) -> str:
        """Bucketed retrieval: semantic top-3 + per-grep-term top-(1|3), no source excerpts."""
        grep_terms = [t.strip() for t in (grep_terms or []) if t and t.strip()]
        queries = [q.strip() for q in search_queries if q and q.strip()]

        semantic_nodes: list[dict[str, Any]] = []
        if queries:
            query_hits = self.run_hybrid_retrieval(queries, keywords=[])
            for hit in query_hits[:BUCKET_SEMANTIC_TOP_K]:
                semantic_nodes.append(
                    format_compact_candidate(
                        self.G,
                        hit["node_id"],
                        hit["metadata"],
                        score=hit["hybrid_score"],
                        neighbors_limit=GREP_NEIGHBOR_LIMIT,
                    )
                )

        grep_buckets: list[dict[str, Any]] = []
        for term in grep_terms:
            nodes = grep_search_nodes(
                self.G,
                term,
                embedder=self.embedder,
            )
            grep_buckets.append({
                "term": term,
                "match_mode": nodes[0]["match_mode"] if nodes else "none",
                "nodes": nodes,
            })

        all_ids = [n["node_id"] for n in semantic_nodes]
        for bucket in grep_buckets:
            all_ids.extend(n["node_id"] for n in bucket.get("nodes", []))

        has_hits = bool(semantic_nodes) or any(b.get("nodes") for b in grep_buckets)
        out: dict[str, Any] = {
            "rewritten_queries": queries,
            "grep_terms": grep_terms,
            "semantic_bucket": {
                "nodes": semantic_nodes,
            },
            "grep_buckets": grep_buckets,
            "graph_coverage": graph_coverage_fields(self.G),
            "sufficient_to_answer": has_hits,
            "do_not_use_native_read": has_hits,
        }
        if include_next_action and all_ids:
            out["next_action"] = {
                "tool": "fetch_node_source",
                "args": {"node_ids": all_ids[:3]},
                "hint": "Use fetch_node_source with node_ids when you need full source code.",
            }
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str)

