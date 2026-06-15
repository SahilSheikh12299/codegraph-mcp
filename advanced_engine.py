# import numpy as np
# import networkx as nx
# from typing import List, Dict, Any, Tuple
# import ast

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
import ast

SEARCH_RESULT_LIMIT = 6


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

    def run_hybrid_retrieval(self, query_text: str, targeted_symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Prong 1 & Prong 2 Core Matcher. Combines explicit structural keyword scans
        with dense semantic vector scoring, pulling vector-skipping mechanics from your old engine.
        """
        scored_hits = []
        symbols_set = set(targeted_symbols or [])
        
        # Generate semantic search vector for the incoming intent
        query_vector = self.embedder.model.encode(query_text, convert_to_numpy=True).tolist()

        for node_id, data in self.G.nodes(data=True):
            if "type" not in data:
                continue
                
            node_vector = data.get("embedding")
            match_mechanism = "SEMANTIC_VECTOR"
            
            # INTEGRATION POINT FROM YOUR OLD ENGINE:
            # If the node has no vector array (e.g., folder MODULES), bypass cosine calculations
            if not node_vector or len(node_vector) == 0:
                confidence = 0.0
            else:
                confidence = self._calculate_cosine_similarity(query_vector, node_vector)

            # Prong 1: Explicit Token Matching rule check
            is_exact_match = False
            node_name = data.get("name", "")
            
            if node_name in symbols_set or node_id in symbols_set:
                is_exact_match = True
            else:
                # Fallback matching checking for notebook style absolute scoped naming (path::symbol)
                for symbol in symbols_set:
                    clean_symbol = symbol.replace("()", "").strip()
                    if node_id.endswith(f"::{clean_symbol}") or node_name.lower() == clean_symbol.lower():
                        is_exact_match = True
                        break

            if is_exact_match:
                match_mechanism = "EXACT_SYMBOL_MATCH"
                confidence = max(confidence, 0.95)  # Force structural anchor prioritization

            if confidence > 0.0 or match_mechanism == "EXACT_SYMBOL_MATCH":
                scored_hits.append((node_id, data, confidence, match_mechanism))

        scored_hits.sort(key=lambda x: x[2], reverse=True)

        exact_hits = [h for h in scored_hits if h[3] == "EXACT_SYMBOL_MATCH"]
        vector_pool = [
            h for h in scored_hits
            if h[3] != "EXACT_SYMBOL_MATCH" and h[2] >= 0.35
        ][:20]
        seen = {h[0] for h in exact_hits}
        candidates = exact_hits + [h for h in vector_pool if h[0] not in seen]

        reranked = []
        for node_id, data, sim, mechanism in candidates:
            if mechanism == "EXACT_SYMBOL_MATCH":
                final = max(sim, 0.95)
            else:
                final = 0.7 * sim + 0.3 * data.get("call_centrality", 0.0)
            reranked.append((node_id, data, final, mechanism))

        reranked.sort(key=lambda x: x[2], reverse=True)
        final_selections = reranked[:SEARCH_RESULT_LIMIT]

        return [
            {
                "node_id": hit[0],
                "metadata": hit[1],
                "hybrid_score": round(hit[2], 4),
                "callers": hit[1].get("calls_in_degree", 0),
                "mechanism": hit[3],
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

    def compile_context_package(self, search_queries: List[str], targeted_symbols: List[str] = None) -> str:
        """Executes multi-query hybrid retrieval, unifies and deduplicates candidates."""
        global_seeds_pool: Dict[str, Dict[str, Any]] = {}

        for query in search_queries:
            query = query.strip()
            if not query:
                continue
                
            query_hits = self.run_hybrid_retrieval(query, targeted_symbols)
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
            return "### [Graph-RAG System Message]\nNo relevant structural context matched this query matrix."

        unified_candidates = list(global_seeds_pool.values())
        unified_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        final_seeds = unified_candidates[:SEARCH_RESULT_LIMIT]

        markdown = "## Codebase Search Results\n\n"
        markdown += f"Queries: {', '.join(search_queries)}\n"
        markdown += (
            f"Showing top {len(final_seeds)} matches — pick one or more `node_id`s to fetch.\n"
            "`hybrid_score` = 0.7 × semantic similarity + 0.3 × call centrality; "
            "`callers` = how many functions call this node (higher often means entry-point/orchestrator).\n\n"
        )

        for idx, hit in enumerate(final_seeds, 1):
            meta = hit["metadata"]
            node_id = hit["node_id"]

            markdown += f"### Hit {idx}: `{node_id}`\n"
            markdown += (
                f"- hybrid_score: {hit['hybrid_score']:.4f} | callers: {hit['callers']} "
                f"| match: {hit['mechanism']}\n"
            )
            markdown += f"- type: {meta.get('type')}\n"

            if meta.get("type") == "CLASS":
                if meta.get("docstring"):
                    summary = meta.get("docstring").strip().split("\n")[0]
                    markdown += f"- summary: {summary[:120]}\n"
                markdown += "- methods (use fetch_node_source on node_id):\n"
                method_count = 0
                for sub_id, sub_data in self.G.nodes(data=True):
                    if sub_data.get("belongs_to_class") == meta.get("name") and sub_data.get("file_path") == meta.get("file_path"):
                        markdown += f"  - `{sub_id}` def {sub_data.get('name')}{sub_data.get('signature', '()')}\n"
                        method_count += 1
                        if method_count >= 8:
                            markdown += "  - ... (more methods available via fetch_node_source)\n"
                            break
            else:
                markdown += f"- signature: {meta.get('signature', '()')}\n"
                if meta.get("docstring"):
                    summary = meta.get("docstring").strip()
                    markdown += f"- summary: {summary[:120]}{'...' if len(summary) > 120 else ''}\n"

            neighbors = format_call_neighbors(self.G, node_id)
            if neighbors:
                markdown += neighbors + "\n"
            markdown += f"- action: call fetch_node_source with node_id `{node_id}` to read implementation\n"
            if neighbors and "called_by:" in neighbors:
                markdown += "- if this looks like a helper, fetch a `called_by` node or use trace_callers\n"
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

