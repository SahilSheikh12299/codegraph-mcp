import os
import ast
import sys
import gc
import logging
import hashlib
import json
from pathlib import Path
import networkx as nx
from mcp.server.fastmcp import FastMCP
from filelock import FileLock
import re
import os
from typing import Dict, Any, Tuple, List
from fileParsing import extract_file_entities, WorkspaceScanner, ASTParser, ImportTracker

# Completely muzzle third-party library progress bars before they can print
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import your validated local analytical engine modules
from graph_io import GraphSerializer
from embeddingPipeline import EmbeddingModelLifecycleManager, LocalEmbeddingPipeline
from advanced_engine import (
    AdvancedRetrievalEngine,
    extract_source_excerpt,
    # format_call_neighbors,
    get_call_neighbors,
    recompute_call_centrality,
    NEIGHBOR_ID_LIMIT,
    SOURCE_EXCERPT_MAX_LINES,
)
from buildGraph import (
    wire_calls_for_file,
    build_func_registry_from_graph,
    strip_calls_edges,
    strip_calls_edges_for_file,
)

# Configure silent logging to avoid corrupting standard output
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

# =========================================================================
# 1. MASTER MODEL MCP INITIALIZATION & LIFECYCLE
# =========================================================================
mcp = FastMCP("Cursor-Graph-RAG-Engine")



model_manager = EmbeddingModelLifecycleManager()



def _is_stub_source(source: str) -> bool:
    s = source.strip()
    return s.endswith("...") or s.endswith(": ...")


def _find_impl_alternate(G: nx.DiGraph, node_id: str, node_data: dict) -> str | None:
    """If node is a Protocol stub, find a fuller same-name implementation in the same file."""
    name, file_path = node_data.get("name"), node_data.get("file_path")
    if not name or not file_path:
        return None
    best_id, best_len = None, 0
    for nid, d in G.nodes(data=True):
        if d.get("file_path") != file_path or d.get("name") != name or _is_stub_source(d.get("chunk_text", "")):
            continue
        chunk_len = len(d.get("chunk_text", ""))
        if chunk_len > best_len:
            best_len, best_id = chunk_len, nid
    return best_id if best_id and best_id != node_id else None


# def _graph_neighbors_footer(G: nx.DiGraph, node_id: str) -> str:
#     neighbors = format_call_neighbors(G, node_id)
#     if not neighbors:
#         return ""
#     return f"\n\n### Graph neighbors\n{neighbors}"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


# def _get_capped_call_neighbors(G: nx.DiGraph, node_id: str, neighbors_max: int) -> dict:
#     """Return callers/callees node_ids capped to neighbors_max with *_more counts."""
#     if not G.has_node(node_id):
#         return {"callers": [], "callees": [], "callers_more": 0, "callees_more": 0}

#     callers_all = sorted(
#         p for p in G.predecessors(node_id) if G.edges[p, node_id].get("relationship") == "CALLS"
#     )
#     callees_all = sorted(
#         s for s in G.successors(node_id) if G.edges[node_id, s].get("relationship") == "CALLS"
#     )

#     callers = callers_all[: max(0, neighbors_max)]
#     callees = callees_all[: max(0, neighbors_max)]
#     return {
#         "callers": callers,
#         "callees": callees,
#         "callers_more": max(0, len(callers_all) - len(callers)),
#         "callees_more": max(0, len(callees_all) - len(callees)),
#         "callers_count": len(callers_all),
#         "callees_count": len(callees_all),
#     }


def _render_neighbors_markdown(neighbors: dict) -> str:
    lines: list[str] = []
    callers = neighbors.get("callers") or []
    callees = neighbors.get("callees") or []
    if callers:
        lines.append(f"- called_by: {', '.join(f'`{c}`' for c in callers)}")
        more = int(neighbors.get("callers_more") or 0)
        if more:
            lines.append(f"  - ... and {more} more caller(s)")
    if callees:
        lines.append(f"- calls: {', '.join(f'`{c}`' for c in callees)}")
        more = int(neighbors.get("callees_more") or 0)
        if more:
            lines.append(f"  - ... and {more} more callee(s)")
    return "\n".join(lines)


# def _extract_docstring(source: str) -> str | None:
#     # Best-effort: first triple-quoted literal after signature.
#     m = re.search(r'^[ \t]*(?:"""|\'\'\')([\s\S]*?)(?:"""|\'\'\')', source, re.MULTILINE)
#     if not m:
#         return None
#     return m.group(1).strip()


# def _slice_source(source: str, slices: list[str]) -> dict:
#     lines = source.splitlines()
#     out: dict[str, Any] = {}

#     if "signature" in slices:
#         sig_lines: list[str] = []
#         for ln in lines[:25]:
#             if ln.lstrip().startswith(("def ", "async def ", "class ")):
#                 sig_lines.append(ln)
#                 break
#         out["signature_lines"] = sig_lines

#     if "docstring" in slices:
#         out["docstring"] = _extract_docstring(source)

#     if "args" in slices:
#         # Minimal: include the signature line only (arguments are inside it).
#         out["args_hint"] = "See signature_lines"

#     def _cap_matches(prefixes: tuple[str, ...], cap: int = 12) -> list[str]:
#         matches = []
#         for ln in lines:
#             s = ln.lstrip()
#             if s.startswith(prefixes):
#                 matches.append(ln)
#                 if len(matches) >= cap:
#                     break
#         return matches

#     if "returns" in slices:
#         out["return_lines"] = _cap_matches(("return ",))
#     if "raises" in slices:
#         out["raise_lines"] = _cap_matches(("raise ",))
#     if "ifs" in slices:
#         out["if_lines"] = _cap_matches(("if ", "elif ", "else:"))
#     if "loops" in slices:
#         out["loop_lines"] = _cap_matches(("for ", "while "))

#     return out


def get_graph_paths(active_project_root: str | Path) -> Tuple[Path, Path]:
    """The single source of truth for workspace graph cache and lock files.
    
    Always resolves the raw absolute directory path to prevent double-hashing.
    """
    # 1. Force resolve the raw directory path completely
    abs_path_str = str(Path(active_project_root).resolve())
    
    # 2. Compute a single MD5 hash of that absolute path string
    workspace_hash = hashlib.md5(abs_path_str.encode('utf-8')).hexdigest()
    
    # 3. Establish standard directories
    mcp_cache_dir = Path("~/.cursor_graph_rag/graphs").expanduser()
    mcp_cache_dir.mkdir(parents=True, exist_ok=True)
    
    return (
        mcp_cache_dir / f"graph_{workspace_hash}.json",
        mcp_cache_dir / f"graph_{workspace_hash}.lock"
    )


def _text_for_embedding(data: dict) -> str:
    """Prefer compact embedding_text; fall back to chunk_text for FILE/MODULE nodes."""
    return (data.get("embedding_text") or data.get("chunk_text") or "").strip()


def backfill_source_chunks(G: nx.DiGraph, repo_root: Path, rel_paths: set[str]) -> None:
    """Overlay real AST source segments and embedding_text onto graph nodes."""
    for rel_path in rel_paths:
        for node_id, data in extract_file_entities(rel_path, repo_root).items():
            if G.has_node(node_id):
                G.nodes[node_id]["chunk_text"] = data["chunk_text"]
                G.nodes[node_id]["embedding_text"] = data["embedding_text"]
            else:
                G.add_node(node_id, **data)


def _embed_file_nodes(G: nx.DiGraph, rel_path: str, embedder: LocalEmbeddingPipeline) -> int:
    """Batch-encode all nodes belonging to a single file."""
    nodes_to_encode = []
    texts_to_encode = []
    for node_id, data in G.nodes(data=True):
        if data.get("file_path") == rel_path:
            text = _text_for_embedding(data)
            if text:
                nodes_to_encode.append(node_id)
                texts_to_encode.append(text)
    if texts_to_encode:
        vectors = embedder.model.encode(texts_to_encode, convert_to_numpy=True).tolist()
        for node_id, vector in zip(nodes_to_encode, vectors):
            G.nodes[node_id]["embedding"] = vector
    return len(texts_to_encode)


def _parse_file_call_assets(repo_root: Path, rel_path: str, tracker: ImportTracker) -> dict:
    full_path = repo_root / rel_path
    ast_data = ASTParser(file_path=full_path).parse()
    import_data = tracker.get_dependencies(full_path)
    return {
        "classes": ast_data.get("classes", []),
        "functions": ast_data.get("functions", []),
        "top_level_calls": ast_data.get("top_level_calls", []),
        "internal_imports": import_data.get("internal_paths", []),
    }


def _rewire_all_calls(G: nx.DiGraph, repo_root: Path) -> None:
    strip_calls_edges(G)
    registry = build_func_registry_from_graph(G)
    tracked = {d.get("file_path") for _, d in G.nodes(data=True) if d.get("file_path")}
    python_files = WorkspaceScanner(repo_root).scan()
    tracker = ImportTracker(repo_root=repo_root, all_python_files=python_files)
    for rel_path in sorted(tracked):
        if (repo_root / rel_path).exists():
            wire_calls_for_file(G, rel_path, _parse_file_call_assets(repo_root, rel_path, tracker), registry)
    recompute_call_centrality(G)


def _rewire_file_calls(G: nx.DiGraph, repo_root: Path, rel_path: str, tracker: ImportTracker) -> None:
    strip_calls_edges_for_file(G, rel_path)
    registry = build_func_registry_from_graph(G)
    wire_calls_for_file(G, rel_path, _parse_file_call_assets(repo_root, rel_path, tracker), registry)
    recompute_call_centrality(G)


def execute_preflight_lazy_sync(repo_root: Path, G: nx.DiGraph, embedder: LocalEmbeddingPipeline) -> bool:
    """
    Surgically checks file timestamps and processes embedding vectors ONLY 
    for cold-started files or elements showing active post-save deltas.
    """
    if 'indexed_timestamps' not in G.graph:
        G.graph['indexed_timestamps'] = {}

    tracked_files = {data.get("file_path") for _, data in G.nodes(data=True) if data.get("file_path")}
    tracked_files.discard(None)

    schema_dirty = False

    if G.graph.get("chunk_schema") != 3:
        backfill_source_chunks(G, repo_root, tracked_files)
        for rel_path in tracked_files:
            if (repo_root / rel_path).exists():
                _embed_file_nodes(G, rel_path, embedder)
        G.graph["chunk_schema"] = 3
        schema_dirty = True

    if G.graph.get("calls_schema") != 1:
        _rewire_all_calls(G, repo_root)
        G.graph["calls_schema"] = 1
        schema_dirty = True
    elif G.graph.get("centrality_schema") != 1:
        recompute_call_centrality(G)
        schema_dirty = True

    if schema_dirty:
        return True

    dirty_files_detected = False
    first_run_initialization = False
    import_tracker: ImportTracker | None = None
    calls_graph_changed = False

    for rel_path in list(tracked_files):
        if not (repo_root / rel_path).exists():
            for nid, d in list(G.nodes(data=True)):
                if d.get("file_path") == rel_path:
                    G.remove_node(nid)
            G.graph["indexed_timestamps"].pop(rel_path, None)
            tracked_files.discard(rel_path)
            dirty_files_detected = True
            calls_graph_changed = True

    current_files = {str(p.relative_to(repo_root)) for p in WorkspaceScanner(repo_root).scan()}
    for rel_path in current_files - tracked_files:
        full_path = repo_root / rel_path
        file_id = rel_path
        if not G.has_node(file_id):
            with open(full_path, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            G.add_node(file_id, type="FILE", path=rel_path, line_count=line_count, docstring="")

        for node_id, data in extract_file_entities(rel_path, repo_root).items():
            G.add_node(node_id, **data)
            if not G.has_edge(file_id, node_id):
                G.add_edge(file_id, node_id, relationship="CONTAINS")

        _embed_file_nodes(G, rel_path, embedder)
        G.graph["indexed_timestamps"][rel_path] = os.path.getmtime(full_path)
        dirty_files_detected = True
        tracked_files.add(rel_path)

    print(f"[Sync Engine] Scanning metadata for {len(tracked_files)} files...", file=sys.stderr)

    for rel_path in tracked_files:
        if not rel_path:
            continue
            
        full_path = repo_root / rel_path
        
        # Verify file existence using absolute resolution path
        if not full_path.exists():
            continue
            
        current_mtime = os.path.getmtime(full_path)
        last_indexed_time = G.graph['indexed_timestamps'].get(rel_path)
        
        # =========================================================================
        # ❄️ CHANNELS FIX A: TARGETED COLD-START EMBEDDING BACKFILL
        # =========================================================================
        if last_indexed_time is None:
            count = _embed_file_nodes(G, rel_path, embedder)
            if count:
                print(f"[COLD BOOT] Batched {count} embeddings for '{rel_path}'.", file=sys.stderr)
            
            # Secure the timestamp boundary and move to next file smoothly
            G.graph['indexed_timestamps'][rel_path] = current_mtime
            first_run_initialization = True
            continue
            
        # =========================================================================
        # 🔥 CHANNELS FIX B: INCREMENTAL DELTA UPGRADES (MANUAL EDITS / AUTOSAVE)
        # =========================================================================
        if current_mtime > last_indexed_time:
            dirty_files_detected = True
            print(f"\n[⚠️ DIRTY FILE DETECTED] Change isolated in: '{rel_path}'")
            
            # Extract what the file looks like right now on disk
            fresh_entities = extract_file_entities(rel_path, repo_root)
            
            # Pull what the graph thinks the file looks like right now
            existing_file_nodes = {
                node_id: data for node_id, data in G.nodes(data=True) 
                if data.get("file_path") == rel_path
            }
            
            # --- SUB-STEP A: EVALUATE MODIFIED & ADDED ELEMENTS ---
            for node_id, fresh_data in fresh_entities.items():
                if node_id in existing_file_nodes:
                    # If chunk text differs, it's an inline structural edit
                    if fresh_data["chunk_text"] != existing_file_nodes[node_id].get("chunk_text"):
                        print(f"  -> [MODIFIED] Element updated: `{node_id}`. Regenerating vector...")
                        emb_text = _text_for_embedding(fresh_data)
                        fresh_data["embedding"] = embedder.model.encode(emb_text, convert_to_numpy=True).tolist()
                        G.add_node(node_id, **fresh_data)
                else:
                    # The node ID is brand new to the system
                    print(f"  -> [ADDED] New element detected: `{node_id}`. Generating vector...")
                    emb_text = _text_for_embedding(fresh_data)
                    fresh_data["embedding"] = embedder.model.encode(emb_text, convert_to_numpy=True).tolist()
                    G.add_node(node_id, **fresh_data)
                    
            # --- SUB-STEP B: EVALUATE DELETED ELEMENTS ---
            for old_node_id in existing_file_nodes.keys():
                if old_node_id not in fresh_entities:
                    print(f"  -> [DELETED] Element missing from file: `{old_node_id}`. Purging from graph...")
                    G.remove_node(old_node_id)
                    
            # Update our tracking timestamp baseline
            G.graph['indexed_timestamps'][rel_path] = current_mtime

            if import_tracker is None:
                python_files = WorkspaceScanner(repo_root).scan()
                import_tracker = ImportTracker(repo_root=repo_root, all_python_files=python_files)
            _rewire_file_calls(G, repo_root, rel_path, import_tracker)

    if calls_graph_changed:
        recompute_call_centrality(G)

    if first_run_initialization:
        return first_run_initialization

    return dirty_files_detected


# =========================================================================
# 3. NATIVE MCP CODE ENVIRONMENT TOOLS REGISTRATION
# =========================================================================
@mcp.tool()
def search_codebase_intent(
    search_queries: list[str],
    active_project_root: str,
    # targeted_symbols: list[str] = None,
    top_k: int = 8,
    format: str = "json",
    include_next_action: bool = True,
) -> str:
    """Find code by intent. Returns top 2 hits with capped source + neighbor IDs.

    Args:
        search_queries: Keyword-dense strings, e.g. ["redirect Location 301", "resolve_redirects"].
        active_project_root: Absolute path to the repo root.        
        top_k: Max candidates to return (top 2 include source_excerpt).
        format: "json" (default) or "markdown".
        include_next_action: If true, suggest fetch_node_source for the top hit.
    """
    workspace_root = Path(active_project_root).resolve()
    graph_path, lock_path = get_graph_paths(workspace_root)

    embedder = model_manager.acquire()
    reranker = model_manager.acquire_reranker()
    try:
        with FileLock(lock_path):
            G = GraphSerializer.load_from_json(workspace_root, graph_path)
            if execute_preflight_lazy_sync(workspace_root, G, embedder):
                GraphSerializer.save_to_json(G, workspace_root, graph_path)

        engine = AdvancedRetrievalEngine(
            graph_instance=G,
            embedder_instance=embedder,
            reranker=reranker,
        )
        return engine.compile_context_package(
            search_queries=search_queries,
            top_k=top_k,
            output_format=format,
            include_next_action=include_next_action,
        )
    finally:
        model_manager.release()


@mcp.tool()
def calculate_blast_radius(
    target_symbol: str,
    active_project_root: str,
    format: str = "markdown",
    max_items: int = 50,
) -> str:
    """List upstream dependents before editing code. Not for discovery-only questions.

    Args:
        target_symbol: Raw identifier only, e.g. "resolve_redirects" or "Session.request".
        active_project_root: Absolute path to the repo root.
    """
    workspace_root = Path(active_project_root).resolve()
    
    graph_path, lock_path = get_graph_paths(workspace_root)
    
    embedder = model_manager.acquire()
    try:
        with FileLock(lock_path):
            G = GraphSerializer.load_from_json(workspace_root,graph_path)
            if execute_preflight_lazy_sync(workspace_root, G, embedder):
                GraphSerializer.save_to_json(G, workspace_root, graph_path)
                
        engine = AdvancedRetrievalEngine(graph_instance=G, embedder_instance=embedder)
        
        matching_nodes = [nid for nid, d in G.nodes(data=True) if d.get("name") == target_symbol or nid == target_symbol]
        if not matching_nodes:
            return f"### [Graph-RAG System Message]\nTarget code symbol '{target_symbol}' could not be matched."
            
        impact_profiles = engine.compute_upstream_blast_radius(matching_nodes)
        
        impact_profiles = sorted(
            impact_profiles,
            key=lambda x: (x.get("file_path") or "", x.get("node_id") or ""),
        )

        if format == "json":
            items = impact_profiles[: max(0, max_items)]
            return _json_dumps(
                {
                    "target_symbol": target_symbol,
                    "matched_nodes": matching_nodes,
                    "total_affected": len(impact_profiles),
                    "items": items,
                    "more": max(0, len(impact_profiles) - len(items)),
                }
            )

        report = f"### UPSTREAM REGRESSION AUDIT FOR SYMBOL: `{target_symbol}`\n"
        report += f"Modifying this entity creates a direct domino risk across **{len(impact_profiles)}** elements.\n\n"

        shown = impact_profiles[: max(0, max_items)]
        for item in shown:
            report += f"- **[{item['type']}]** `{item['node_id']}` inside `{item['file_path']}`\n"
        more = len(impact_profiles) - len(shown)
        if more > 0:
            report += f"\n... and {more} more.\n"

        return report
    finally:
        model_manager.release()


def _build_node_payload(
    G: nx.DiGraph,
    node_id: str,
    *,
    neighbors_max: int = NEIGHBOR_ID_LIMIT,
) -> dict[str, Any]:
    """Single node: capped source_excerpt + 1-hop neighbor IDs."""
    if not G.has_node(node_id):
        return {"node_id": node_id, "error": f"Node ID '{node_id}' not found."}

    node_data = G.nodes[node_id]
    node_type = node_data.get("type", "UNKNOWN")
    resolved_id = node_id
    chunk_text = node_data.get("chunk_text", "")

    if node_type != "CLASS" and _is_stub_source(chunk_text):
        alt_id = _find_impl_alternate(G, node_id, node_data)
        if alt_id:
            resolved_id = alt_id
            node_data = G.nodes[resolved_id]
            chunk_text = node_data.get("chunk_text", chunk_text)

    callers, callees = get_call_neighbors(G, resolved_id, limit=neighbors_max)
    neighbors = {"callers": callers, "callees": callees}

    payload: dict[str, Any] = {
        "node_id": resolved_id,
        "type": node_type,
        "file_path": node_data.get("file_path"),
        "name": node_data.get("name"),
        "signature": node_data.get("signature", ""),
        "source_excerpt": extract_source_excerpt(chunk_text, node_type, SOURCE_EXCERPT_MAX_LINES),
        "neighbors": neighbors,
    }

    if node_type == "CLASS":
        payload["methods"] = [
            {
                "node_id": sub_id,
                "name": sub_data.get("name"),
                "signature": sub_data.get("signature", "()"),
            }
            for sub_id, sub_data in G.nodes(data=True)
            if sub_data.get("belongs_to_class") == node_data.get("name")
            and sub_data.get("file_path") == node_data.get("file_path")
        ][:50]

    if resolved_id != node_id:
        payload["resolved_stub_from"] = node_id

    return payload


def _format_node_source(
    G: nx.DiGraph,
    node_id: str,
    *,
    # mode: str = "excerpt",
    # slices: list[str] | None = None,
    # max_lines: int = SOURCE_EXCERPT_MAX_LINES,
    # include_neighbors: bool = True,
    neighbors_max: int = NEIGHBOR_ID_LIMIT,
    format: str = "json",
) -> str:
    """Backward-compatible wrapper; always returns capped excerpt + neighbor IDs."""
    payload = _build_node_payload(G, node_id, neighbors_max=neighbors_max)
    if format == "json":
        return _json_dumps(payload)

    if payload.get("error"):
        return f"### [Graph-RAG System Message]\n{payload['error']}"

    body = f"### Source for `{payload['node_id']}`\n\n```python\n{payload.get('source_excerpt', '')}\n```"
    neighbors = payload.get("neighbors") or {}
    rendered = _render_neighbors_markdown(neighbors)
    if rendered:
        body += f"\n\n### Graph neighbors\n{rendered}"
    return body


@mcp.tool()
def fetch_node_source(
    node_ids: list[str],
    active_project_root: str,
    # mode: str = "excerpt",
    # slices: list[str] | None = None,
    # max_lines: int = SOURCE_EXCERPT_MAX_LINES,
    # include_neighbors: bool = True,
    neighbors_max: int = NEIGHBOR_ID_LIMIT,
    format: str = "json",
) -> str:
    """Capped source excerpt + 1-hop neighbor IDs for one or more node_ids.

    Args:
        node_ids: Exact node_id(s) from search output neighbors or candidates.
        active_project_root: Absolute path to the repo root.
    """
    workspace_root = Path(active_project_root).resolve()
    graph_path, lock_path = get_graph_paths(workspace_root)

    embedder = model_manager.acquire()
    try:
        with FileLock(lock_path):
            G = GraphSerializer.load_from_json(workspace_root, graph_path)
            if execute_preflight_lazy_sync(workspace_root, G, embedder):
                GraphSerializer.save_to_json(G, workspace_root, graph_path)

        nodes = [
            _build_node_payload(G, node_id, neighbors_max=neighbors_max)
            for node_id in node_ids
        ]
        if format == "json":
            return _json_dumps({"nodes": nodes})

        parts = [
            _format_node_source(
                G,
                node_id,
                neighbors_max=neighbors_max,
                format="markdown",
            )
            for node_id in node_ids
        ]
        return "\n\n---\n\n".join(parts)
    finally:
        model_manager.release()


@mcp.tool()
def trace_callers(
    node_id: str,
    # active_project_root: str,
    format: str = "json",
    # max_items: int = 50,
) -> str:
    """Deprecated. Use fetch_node_source — it includes 1-hop callers and callees."""
    msg = {
        "deprecated": True,
        "message": "trace_callers is deprecated. Use fetch_node_source(node_ids=[...]) which includes 1-hop callers and callees.",
        "use_instead": {
            "tool": "fetch_node_source",
            "args": {"node_ids": [node_id]},
        },
    }
    if format == "json":
        return _json_dumps(msg)
    return msg["message"]


if __name__ == "__main__":
    mcp.run(transport="stdio")