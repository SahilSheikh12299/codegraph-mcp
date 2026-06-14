# import numpy as np
# import networkx as nx
# from typing import List, Dict, Any, Tuple
# import ast

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
import ast

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

            # Only retain data slots providing score signal or explicit matches
            if confidence > 0.0 or match_mechanism == "EXACT_SYMBOL_MATCH":
                scored_hits.append((node_id, data, confidence, match_mechanism))

        # Sort elements by confidence alignment descending
        scored_hits.sort(key=lambda x: x[2], reverse=True)

        # Apply cliff-detection math to calculate dynamic slice boundary
        cutoff_index = self._execute_cliff_detection(scored_hits)
        final_selections = scored_hits[:cutoff_index]

        return [
            {
                "node_id": hit[0],
                "metadata": hit[1],
                "confidence": round(hit[2], 4),
                "mechanism": hit[3]
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
        """Executes multi-query hybrid retrieval, unifies and deduplicates candidates,
        applies global score-cliff detection, and runs a single recursive blast radius analysis.
        """
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
                    if hit["confidence"] > global_seeds_pool[node_id]["confidence"]:
                        global_seeds_pool[node_id] = hit
                else:
                    global_seeds_pool[node_id] = hit

        if not global_seeds_pool:
            return "### [Graph-RAG System Message]\nNo relevant structural context matched this query matrix."

        unified_candidates = list(global_seeds_pool.values())
        unified_candidates.sort(key=lambda x: x["confidence"], reverse=True)

        # INTEGRATION POINT FROM YOUR OLD ENGINE: 
        # Apply your notebook's sequential gaps math to detect the ultimate score drop-off boundary
        upper_bound = min(len(unified_candidates), 6)
        max_gap = -1.0
        dynamic_cutoff = min(len(unified_candidates), 2)
        
        for i in range(dynamic_cutoff - 1, upper_bound - 1):
            current_score = unified_candidates[i]["confidence"]
            next_score = unified_candidates[i + 1]["confidence"]
            gap = current_score - next_score
            
            if gap > max_gap:
                max_gap = gap
                dynamic_cutoff = i + 1

        final_seeds = unified_candidates[:dynamic_cutoff]
        target_ids = [hit["node_id"] for hit in final_seeds]
        
        blast_radius = self.compute_upstream_blast_radius(target_ids)

        markdown = f"========================================================\n"
        markdown += f"  INTELLIGENT KNOWLEDGE BLUEPRINT FOR WORKSPACE ASSISTANT\n"
        markdown += f"========================================================\n\n"
        markdown += f"**Processed Operational Goals:** {', '.join(search_queries)}\n\n"
        markdown += f"## SECTION 1: PRIMARY TARGET ENTITIES\n\n"

        for idx, hit in enumerate(final_seeds, 1):
            meta = hit["metadata"]
            markdown += f"### [Hit {idx}] `{hit['node_id']}`\n"
            markdown += f"  - Match Mechanics: {hit['mechanism']}\n"
            markdown += f"  - System Alignment Confidence: {hit['confidence']:.4f}\n\n"
            markdown += f"```text\n"
            markdown += f"Entity Type: {meta.get('type')}\n"
            
            if meta.get("type") == "CLASS":
                markdown += f"Class Ancestors: {meta.get('bases', [])}\n"
            else:
                markdown += f"Execution Signature: {meta.get('signature', '()')}\n"
                
            # preview_body = self._create_surgical_preview(meta.get('chunk_text', ''))
            # markdown += f"Functional Content Outline:\n{preview_body}\n"
            markdown += f"Implementation Logic: [Omitted to protect token window context. Use the 'fetch_node_source' tool with this node's exact ID to read its source code.]\n"
            markdown += f"```\n---\n"

        markdown += f"\n## SECTION 2: BARE-BONES RECURSIVE DEPENDENCY SKELETONS\n\n"
        if blast_radius:
            markdown += f"**CRITICAL WARNING:** Altering the entities in Section 1 causes a domino risk across **{len(blast_radius)}** upstream code structures. Ensure API signature stability:\n"
            
            file_groups = {}
            for item in blast_radius:
                display_group = item["file_path"] if item["file_path"] else "Project Core Module Setup"
                file_groups.setdefault(display_group, []).append(item)
            
            for file_path, items in file_groups.items():
                markdown += f"  -> [CONTEXT BLOCK] Location: `{file_path}`\n"
                for item in items:
                    if item["type"] == "FUNCTION":
                        markdown += f"     - **[FUNCTION]** `def {item['name']}{item['signature']}` | Node ID: `{item['node_id']}`\n"
                    elif item["type"] == "CLASS":
                        markdown += f"     - **[CLASS]** `class {item['name']}` | Node ID: `{item['node_id']}`\n"
                    elif item["type"] == "FILE":
                        markdown += f"     - **[FILE]** Node ID: `{item['node_id']}`\n"
                    elif item["type"] == "MODULE":
                        markdown += f"     - **[MODULE]** Node ID: `{item['node_id']}`\n"
                markdown += f"  --------------------------------------------------\n"
        else:
            markdown += f"*No upstream recursive dependencies found for these targets. Modifications pose localized risk only.*\n"
        print(f"THIS IS MARKDOWN ------------> {markdown}")
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

# class AdvancedRetrievalEngine:
#     """The analytical core of the Graph-RAG system. Handles dual-pronged retrieval,

#     mathematical cliff-detection re-ranking, and recursive upstream blast-radius mappings.
#     """

#     def __init__(self, graph_instance: nx.DiGraph, embedder_instance: Any):
#         """Initializes the engine with a living instance of the codebase graph

#         and an offline-locked embedding module.
#         """
#         self.G = graph_instance
#         self.embedder = embedder_instance

#     def _calculate_cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
#         """Computes the standard cosine similarity between two normalized vectors."""
#         a = np.array(vec_a)
#         b = np.array(vec_b)
#         dot_product = np.dot(a, b)
#         norm_a = np.linalg.norm(a)
#         norm_b = np.linalg.norm(b)
        
#         if norm_a == 0 or norm_b == 0:
#             return 0.0
#         return float(dot_product / (norm_a * norm_b))
    
#     def _detect_score_cliff(self, candidates: List[Dict[str, Any]], min_k: int = 2, max_k: int = 6) -> int:
#         """Finds the sharpest statistical drop-off (cliff) between consecutive 
#         confidence scores to dynamically truncate low-signal search hits.
        
#         Adapts gracefully to both 'confidence' and 'score' dictionary keys.
#         """
#         total_candidates = len(candidates)
#         if total_candidates <= min_k:
#             return total_candidates
            
#         # Robust fallback tracking for dictionary key variations
#         score_key = "confidence" if "confidence" in candidates[0] else "score"
        
#         max_search_bound = min(total_candidates, max_k)
#         biggest_drop = -1.0
#         cutoff_index = max_search_bound  # Default fallback if no sharp cliff is found
        
#         # Sweep through the sorted list to find the largest delta gap
#         for i in range(max_search_bound - 1):
#             current_score = candidates[i][score_key]
#             next_score = candidates[i + 1][score_key]
#             score_delta = current_score - next_score
            
#             # Evaluate cliffs starting from the minimum element requirement floor
#             if score_delta > biggest_drop and i >= (min_k - 1):
#                 biggest_drop = score_delta
#                 cutoff_index = i + 1
                
#         # Enforce strict boundary guards
#         return max(min_k, min(cutoff_index, max_k))

#     def _execute_cliff_detection(self, scored_hits: List[Tuple[str, Dict[str, Any], float]], 
#                                  min_k: int = 2, max_k: int = 5, sensitivity: float = 0.05) -> int:
#         """Analyzes similarity drop-offs mathematically to identify either sharp cliffs

#         or relative plateaus of relevancy, returning a dynamic cutoff index.
#         """
#         if len(scored_hits) <= min_k:
#             return len(scored_hits)
            
#         limit = min(len(scored_hits), max_k)
        
#         # Track relative drop-offs sequentially
#         for i in range(min_k - 1, limit - 1):
#             current_score = scored_hits[i][2]
#             next_score = scored_hits[i + 1][2]
#             delta = current_score - next_score
            
#             # If a sudden cliff is detected, draw the cutoff line here
#             if delta >= sensitivity:
#                 return i + 1
                
#         # Fallback: if it's a smooth plateau, keep up to the max limit
#         return limit

#     def run_hybrid_retrieval(self, query_text: str, targeted_symbols: List[str] = None) -> List[Dict[str, Any]]:
#         """Prong 1 & Prong 2 Core Matcher. Combines explicit structural keyword scans

#         with dense semantic vector scoring.
#         """
#         scored_hits = []
#         symbols_set = set(targeted_symbols or [])
        
#         # Generate semantic search vector for the incoming intent
#         query_vector = self.embedder.model.encode(query_text, convert_to_numpy=True).tolist()

#         for node_id, data in self.G.nodes(data=True):
#             # Guard against structural metadata nodes (e.g., timestamps ledger)
#             if "type" not in data:
#                 continue
                
#             match_mechanism = "SEMANTIC_VECTOR"
#             confidence = self._calculate_cosine_similarity(query_vector, data.get("embedding", []))

#             # Prong 1: Override confidence or boost if premium LLM flags an exact symbol match
#             if data.get("name") in symbols_set or node_id in symbols_set:
#                 match_mechanism = "EXACT_SYMBOL_MATCH"
#                 confidence = max(confidence, 0.95)  # Force structural anchor placement

#             scored_hits.append((node_id, data, confidence, match_mechanism))

#         # Sort elements by confidence alignment descending
#         scored_hits.sort(key=lambda x: x[2], reverse=True)

#         # Apply cliff-detection math to calculate dynamic slice boundary
#         cutoff_index = self._execute_cliff_detection(scored_hits)
#         final_selections = scored_hits[:cutoff_index]

#         return [
#             {
#                 "node_id": hit[0],
#                 "metadata": hit[1],
#                 "confidence": round(hit[2], 4),
#                 "mechanism": hit[3]
#             }
#             for hit in final_selections
#         ]

#     def compute_upstream_blast_radius(self, target_nodes: List[str]) -> List[Dict[str, Any]]:
#         """Traverses the directional edges of the network graph upward to trace

#         every entity that will break if the target nodes are modified.
#         """
#         all_affected_nodes = set()
        
#         for node_id in target_nodes:
#             if self.G.has_node(node_id):
#                 # NetworkX isolates all upward structural predecessors recursively
#                 upstream_ancestors = nx.ancestors(self.G, node_id)
#                 all_affected_nodes.update(upstream_ancestors)

#         affected_profiles = []
#         for node_id in all_affected_nodes:
#             data = self.G.nodes[node_id]
#             if "type" not in data:
#                 continue
#             affected_profiles.append({
#                 "node_id": node_id,
#                 "type": data.get("type"),
#                 "name": data.get("name"),
#                 "file_path": data.get("file_path"),
#                 "signature": data.get("signature", "")
#             })
            
#         return affected_profiles

#     def compile_context_package(self, search_queries: List[str], targeted_symbols: List[str] = None) -> str:
#         """Executes multi-query hybrid retrieval, unifies and deduplicates candidates,
#         applies global score-cliff detection, and runs a single recursive blast radius analysis.
        
#         Surgically outlines context blocks to maximize token efficiency.
#         """
#         global_seeds_pool: Dict[str, Dict[str, Any]] = {}

#         # =========================================================================
#         # 1. GLOBAL NODE ID DEDUPLICATION LAYER
#         # =========================================================================
#         for query in search_queries:
#             query = query.strip()
#             if not query:
#                 continue
                
#             # Execute hybrid retrieval layer for the current query slice
#             query_hits = self.run_hybrid_retrieval(query, targeted_symbols)
#             if not query_hits:
#                 continue

#             for hit in query_hits:
#                 node_id = hit["node_id"]
#                 # Collision Resolution: Preserve the highest confidence score across queries
#                 if node_id in global_seeds_pool:
#                     if hit["confidence"] > global_seeds_pool[node_id]["confidence"]:
#                         global_seeds_pool[node_id] = hit
#                 else:
#                     global_seeds_pool[node_id] = hit

#         if not global_seeds_pool:
#             return "### [Graph-RAG System Message]\nNo relevant structural context matched this query matrix."

#         # Convert back to list and sort globally by confidence before applying the cutoff
#         unified_candidates = list(global_seeds_pool.values())
#         unified_candidates.sort(key=lambda x: x["confidence"], reverse=True)

#         # =========================================================================
#         # 2. GLOBAL CLIFF-DETECTION EVALUATION
#         # =========================================================================
#         # Executes your exact Jupyter notebook math across the unified candidate pool
#         dynamic_cutoff = self._detect_score_cliff(unified_candidates, min_k=2, max_k=6)
#         final_seeds = unified_candidates[:dynamic_cutoff]

#         # Extract target node IDs for the remaining high-signal winners
#         target_ids = [hit["node_id"] for hit in final_seeds]
        
#         # Compute ONE single unified recursive blast radius pass for the final targets
#         blast_radius = self.compute_upstream_blast_radius(target_ids)

#         # =========================================================================
#         # 3. COMPILING THE HIGH-SIGNAL BLUEPRINT REPORT
#         # =========================================================================
#         markdown = f"========================================================\n"
#         markdown += f"  INTELLIGENT KNOWLEDGE BLUEPRINT FOR WORKSPACE ASSISTANT\n"
#         markdown += f"========================================================\n\n"
#         markdown += f"**Processed Operational Goals:** {', '.join(search_queries)}\n\n"
#         markdown += f"## SECTION 1: PRIMARY TARGET ENTITIES\n\n"

#         for idx, hit in enumerate(final_seeds, 1):
#             meta = hit["metadata"]
#             markdown += f"### [Hit {idx}] `{hit['node_id']}`\n"
#             markdown += f"  - Match Mechanics: {hit['mechanism']}\n"
#             markdown += f"  - System Alignment Confidence: {hit['confidence']:.4f}\n\n"
#             markdown += f"```text\n"
#             markdown += f"Entity Type: {meta.get('type')}\n"
            
#             if meta.get("type") == "CLASS":
#                 markdown += f"Class Ancestors: {meta.get('bases', [])}\n"
#             else:
#                 markdown += f"Execution Signature: {meta.get('signature', '()')}\n"
                
#             # Trigger Surgical Context Snippeting Layer
#             preview_body = self._create_surgical_preview(meta.get('chunk_text', ''))
#             markdown += f"Functional Content Outline:\n{preview_body}\n"
#             markdown += f"```\n---\n"

#         # =========================================================================
#         # 4. SECTION 2: BARE-BONES RECURSIVE DEPENDENCY SKELETONS
#         # =========================================================================
#         markdown += f"\n## SECTION 2: BARE-BONES RECURSIVE DEPENDENCY SKELETONS\n\n"
#         if blast_radius:
#             markdown += f"**CRITICAL WARNING:** Altering the entities in Section 1 causes a domino risk across **{len(blast_radius)}** upstream code structures. Ensure API signature stability:\n"
            
#             # Group by file path to keep output dense and highly organized
#             file_groups = {}
#             for item in blast_radius:
#                 file_groups.setdefault(item["file_path"], []).append(item)
                
#             for file_path, items in file_groups.items():
#                 markdown += f"  -> [AFFECTED FILE] {file_path}\n"
#                 for item in items:
#                     if item["type"] == "FUNCTION":
#                         markdown += f"     - Function Element: `def {item['name']}{item['signature']}`\n"
#                     else:
#                         markdown += f"     - Class Element: `class {item['name']}`\n"
#                 markdown += f"  --------------------------------------------------\n"
#         else:
#             markdown += f"*No upstream recursive dependencies found for these targets. Modifications pose localized risk only.*\n"

#         return markdown
        
#     def _create_surgical_preview(self, chunk_text: str) -> str:
#         """Surgically strips all operational logic from a python source chunk 
#         using AST parsing.
        
#         Preserves definitions, inheritance lineages, signatures, docstrings, 
#         and method headers while wiping away internal loops and calculations.
#         """
#         if not chunk_text:
#             return "# No implementation source found."
            
#         try:
#             # Parse the text fragment into an isolated Abstract Syntax Tree node
#             tree = ast.parse(chunk_text.strip())
#             if not tree.body:
#                 return chunk_text.strip()
                
#             top_node = tree.body[0]

#             # =================================================================
#             # CASE A: Processing a Standalone Function or Method
#             # =================================================================
#             if isinstance(top_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
#                 docstring = ast.get_docstring(top_node)
                
#                 # Rebuild a clean skeleton body
#                 skeletal_body = []
#                 if docstring:
#                     skeletal_body.append(ast.Expr(value=ast.Constant(value=docstring)))
                    
#                 # Append a standardized systemic placeholder string as the final execution node
#                 placeholder_msg = "... [Execution Logic Truncated. Use 'fetch_node_source' tool to view full logic] ..."
#                 skeletal_body.append(ast.Expr(value=ast.Constant(value=placeholder_msg)))
                
#                 top_node.body = skeletal_body
#                 return ast.unparse(top_node)

#             # =================================================================
#             # CASE B: Processing a Complete Class Definition
#             # =================================================================
#             elif isinstance(top_node, ast.ClassDef):
#                 class_docstring = ast.get_docstring(top_node)
#                 skeletal_class_contents = []
                
#                 if class_docstring:
#                     skeletal_class_contents.append(ast.Expr(value=ast.Constant(value=class_docstring)))
                    
#                 # Scan all nested elements within the class layout
#                 for sub_node in top_node.body:
#                     if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
#                         method_docstring = ast.get_docstring(sub_node)
#                         skeletal_method_body = []
                        
#                         if method_docstring:
#                             skeletal_method_body.append(ast.Expr(value=ast.Constant(value=method_docstring)))
                            
#                         # Use standard python pass to keep the method header valid
#                         skeletal_method_body.append(ast.Pass())
#                         sub_node.body = skeletal_method_body
#                         skeletal_class_contents.append(sub_node)
                        
#                     elif isinstance(sub_node, (ast.Assign, ast.AnnAssign)):
#                         # Preserve class-level constant field attributes/type hints
#                         skeletal_class_contents.append(sub_node)
                        
#                 if not skeletal_class_contents:
#                     skeletal_class_contents.append(ast.Pass())
                    
#                 top_node.body = skeletal_class_contents
#                 return ast.unparse(top_node)
                
#         except Exception:
#             # Fallback Guard: If the code chunk is a raw partial code fragment that 
#             # breaks the AST compiler, fall back to a safe line-slice baseline.
#             lines = chunk_text.splitlines()
#             if len(lines) <= 12:
#                 return chunk_text.strip()
#             return "\n".join(lines[:12]) + "\n\n        ... [Fallback Truncation Active] ..."