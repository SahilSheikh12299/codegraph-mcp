# import numpy as np
# import networkx as nx
# from typing import List, Dict, Any, Tuple
# import ast

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
import ast
import json
import re

SEARCH_RESULT_LIMIT = 6
SEMANTIC_WEIGHT = 0.6
GRAPH_WEIGHT = 0.3
KEYWORD_WEIGHT = 0.1
SOURCE_EXCERPT_MAX_LINES = 100
NEIGHBOR_ID_LIMIT = 50

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "how", "what", "where", "when",
    "does", "do", "did", "for", "and", "or", "to", "in", "on", "of", "with",
    "after", "before", "from", "by", "it", "this", "that", "can", "will", "not",
})


def extract_keywords_from_queries(search_queries: List[str]) -> List[str]:
    """Loose keyword tokens from all search queries (no exact-symbol mode)."""
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

    def __init__(self, graph_instance: nx.DiGraph, embedder_instance: Any):
        """Initializes the engine with a living instance of the codebase graph
        and an offline-locked embedding module from the server pool.
        """
        self.G = graph_instance
        self.embedder = embedder_instance

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
        query_text: str,
        keywords: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Semantic vector pool (top 20) reranked with keyword presence + call centrality."""
        scored_hits = []
        keyword_list = keywords or []

        query_vector = self.embedder.model.encode(query_text, convert_to_numpy=True).tolist()

        for node_id, data in self.G.nodes(data=True):
            if "type" not in data:
                continue

            node_vector = data.get("embedding")
            if not node_vector or len(node_vector) == 0:
                confidence = 0.0
            else:
                confidence = self._calculate_cosine_similarity(query_vector, node_vector)

            if confidence > 0.0:
                scored_hits.append((node_id, data, confidence))

        scored_hits.sort(key=lambda x: x[2], reverse=True)
        candidates = [h for h in scored_hits if h[2] >= 0.35][:20]

        reranked = []
        for node_id, data, sim in candidates:
            kw_score = compute_keyword_score(data, keyword_list)
            final = (
                SEMANTIC_WEIGHT * sim
                + GRAPH_WEIGHT * data.get("call_centrality", 0.0)
                + KEYWORD_WEIGHT * kw_score
            )
            reranked.append((node_id, data, final, kw_score))

        reranked.sort(key=lambda x: x[2], reverse=True)
        final_selections = reranked[:SEARCH_RESULT_LIMIT]

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
        targeted_symbols: List[str] = None,
        top_k: int | None = None,
        output_format: str = "json",
        include_next_action: bool = True,
    ) -> str:
        """Multi-query hybrid retrieval. Top 2 hits include capped source_excerpt + neighbor IDs."""
        global_seeds_pool: Dict[str, Dict[str, Any]] = {}
        keywords = extract_keywords_from_queries(search_queries)

        for query in search_queries:
            query = query.strip()
            if not query:
                continue

            query_hits = self.run_hybrid_retrieval(query, keywords=keywords)
            if not query_hits:
                continue

            for hit in query_hits:
                node_id = hit["node_id"]
                if node_id in global_seeds_pool:
                    if hit["hybrid_score"] > global_seeds_pool[node_id]["hybrid_score"]:
                        global_seeds_pool[node_id] = hit
                else:
                    global_seeds_pool[node_id] = hit

        if not global_seeds_pool:
            if output_format == "json":
                return json.dumps(
                    {"queries": search_queries, "candidates": [], "message": "No matches found."},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            return "### [Graph-RAG System Message]\nNo relevant structural context matched this query matrix."

        unified_candidates = list(global_seeds_pool.values())
        unified_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        limit = SEARCH_RESULT_LIMIT if top_k is None else max(1, int(top_k))
        final_seeds = unified_candidates[:limit]

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
                    candidate["source_excerpt"] = extract_source_excerpt(
                        meta.get("chunk_text", ""),
                        meta.get("type", "FUNCTION"),
                    )
                candidates.append(candidate)

            best_id = candidates[0]["node_id"] if candidates else None
            out: dict[str, Any] = {
                "queries": search_queries,
                "keywords": keywords,
                "candidates": candidates,
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
            f"Score = {SEMANTIC_WEIGHT}×semantic + {GRAPH_WEIGHT}×centrality + {KEYWORD_WEIGHT}×keywords.\n\n"
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
        
    def _create_surgical_preview(self, chunk_text: str) -> str:
        """Surgically cuts operational body execution tasks out of the preview display."""
        if not chunk_text:
            return "# No implementation source found."
            
        try:
            tree = ast.parse(chunk_text.strip())
            if not tree.body:
                return chunk_text.strip()
                
            top_node = tree.body[0]

            if isinstance(top_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(top_node)
                skeletal_body = []
                if docstring:
                    skeletal_body.append(ast.Expr(value=ast.Constant(value=docstring)))
                placeholder_msg = "... [Execution Logic Truncated. Use 'fetch_node_source' tool to view full logic] ..."
                skeletal_body.append(ast.Expr(value=ast.Constant(value=placeholder_msg)))
                top_node.body = skeletal_body
                return ast.unparse(top_node)

            elif isinstance(top_node, ast.ClassDef):
                class_docstring = ast.get_docstring(top_node)
                skeletal_class_contents = []
                
                if class_docstring:
                    skeletal_class_contents.append(ast.Expr(value=ast.Constant(value=class_docstring)))
                    
                for sub_node in top_node.body:
                    if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_docstring = ast.get_docstring(sub_node)
                        skeletal_method_body = []
                        if method_docstring:
                            skeletal_method_body.append(ast.Expr(value=ast.Constant(value=method_docstring)))
                        skeletal_method_body.append(ast.Pass())
                        sub_node.body = skeletal_method_body
                        skeletal_class_contents.append(sub_node)
                    elif isinstance(sub_node, (ast.Assign, ast.AnnAssign)):
                        skeletal_class_contents.append(sub_node)
                        
                if not skeletal_class_contents:
                    skeletal_class_contents.append(ast.Pass())
                    
                top_node.body = skeletal_class_contents
                return ast.unparse(top_node)
                
        except Exception:
            lines = chunk_text.splitlines()
            if len(lines) <= 12:
                return chunk_text.strip()
            return "\n".join(lines[:12]) + "\n\n        ... [Fallback Truncation Active] ..."

