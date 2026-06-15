import os
import ast
import sys
import gc
import logging
import hashlib
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
from advanced_engine import AdvancedRetrievalEngine, format_call_neighbors, get_call_neighbors
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


def _graph_neighbors_footer(G: nx.DiGraph, node_id: str) -> str:
    neighbors = format_call_neighbors(G, node_id)
    if not neighbors:
        return ""
    return f"\n\n### Graph neighbors\n{neighbors}"


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


def _rewire_file_calls(G: nx.DiGraph, repo_root: Path, rel_path: str, tracker: ImportTracker) -> None:
    strip_calls_edges_for_file(G, rel_path)
    registry = build_func_registry_from_graph(G)
    wire_calls_for_file(G, rel_path, _parse_file_call_assets(repo_root, rel_path, tracker), registry)


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

    if schema_dirty:
        return True

    dirty_files_detected = False
    first_run_initialization = False
    import_tracker: ImportTracker | None = None

    for rel_path in list(tracked_files):
        if not (repo_root / rel_path).exists():
            for nid, d in list(G.nodes(data=True)):
                if d.get("file_path") == rel_path:
                    G.remove_node(nid)
            G.graph["indexed_timestamps"].pop(rel_path, None)
            tracked_files.discard(rel_path)
            dirty_files_detected = True

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

    if first_run_initialization:
        return first_run_initialization

    return dirty_files_detected


# =========================================================================
# 3. NATIVE MCP CODE ENVIRONMENT TOOLS REGISTRATION
# =========================================================================
@mcp.tool()
def search_codebase_intent(search_queries: list[str], active_project_root: str, targeted_symbols: list[str] = None) -> str:
    """Find code by intent. Returns node_id, called_by, and calls — not source.
    Workflow: one search → fetch top hit → follow neighbors or trace_callers. No second search.

    Args:
        search_queries: One keyword-dense string per topic. Multiple only for separate subsystems.
        active_project_root: Absolute path to the repo root.
        targeted_symbols: Optional identifiers from the user's message, e.g. ["Session"].
    """
    # Workspace root is the project dir where cursor has been opened
    workspace_root = Path(active_project_root).resolve()
    
    # This function hashes the workspace dir value and then just gives us the graph and lock path of it
    graph_path, lock_path = get_graph_paths(workspace_root)
    
    embedder = model_manager.acquire()
    try:
        with FileLock(lock_path):
            G = GraphSerializer.load_from_json(workspace_root,graph_path)
            if execute_preflight_lazy_sync(workspace_root, G, embedder):
                GraphSerializer.save_to_json(G, workspace_root, graph_path)
                
        engine = AdvancedRetrievalEngine(graph_instance=G, embedder_instance=embedder)
        return engine.compile_context_package(search_queries=search_queries, targeted_symbols=targeted_symbols)
    finally:
        model_manager.release()


@mcp.tool()
def calculate_blast_radius(target_symbol: str, active_project_root: str) -> str:
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
        
        report = f"### UPSTREAM REGRESSION AUDIT FOR SYMBOL: `{target_symbol}`\n"
        report += f"Modifying this entity creates a direct domino risk across **{len(impact_profiles)}** elements.\n\n"
        
        for item in impact_profiles:
            report += f"- **[{item['type']}]** `{item['node_id']}` inside `{item['file_path']}`\n"
            
        return report
    finally:
        model_manager.release()


@mcp.tool()
def fetch_node_source(node_id: str, active_project_root: str) -> str:
    """Source for one node_id plus Graph neighbors (called_by, calls).
    Follow neighbors to walk the call graph. Do not Read/Grep after a successful fetch.

    Args:
        node_id: Exact node_id from search or trace_callers output.
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
                
        if G.has_node(node_id):
            node_data = G.nodes[node_id]
            node_type = node_data.get("type", "UNKNOWN")

            # ──> THE CLASS ROUTER PATCH
            if node_type == "CLASS":
                header = node_data.get("chunk_text", "# No header content available.")
                header_lines = header.splitlines()
                if len(header_lines) > 25:
                    header = "\n".join(header_lines[:25]) + f"\n# ... class header truncated ({len(header_lines)} lines total)"
                source_out = f"### RAW SOURCE CONTENT FOR CLASS NODE: `{node_id}`\n\n"
                source_out += f"```python\n{header}\n```\n\n"
                source_out += "### INTERFACE MAP: AVAILABLE MEMBER METHODS WITHIN THIS CLASS\n"
                source_out += "To read the operational logic of any method below, use 'fetch_node_source' with its absolute Node ID:\n"
                
                found_methods = False
                # Scan the graph for functions belonging to this class in the same file
                for sub_id, sub_data in G.nodes(data=True):
                    if sub_data.get("belongs_to_class") == node_data.get("name") and sub_data.get("file_path") == node_data.get("file_path"):
                        source_out += f"  - [METHOD] `def {sub_data.get('name')}{sub_data.get('signature', '()')}` | Node ID: `{sub_id}`\n"
                        found_methods = True
                        
                if not found_methods:
                    source_out += "  (No child methods indexed for this class block.)\n"
                return source_out + _graph_neighbors_footer(G, node_id)

            else:
                source = node_data.get("chunk_text", "# No content available.")
                alt_id = None
                if _is_stub_source(source):
                    alt_id = _find_impl_alternate(G, node_id, node_data)
                    if alt_id:
                        node_id = alt_id
                        node_data = G.nodes[node_id]
                        source = node_data.get("chunk_text", source)
                MAX_LINES = 150
                lines = source.splitlines()
                if len(lines) > MAX_LINES:
                    source = "\n".join(lines[:MAX_LINES])
                    source += f"\n\n# ... truncated ({len(lines)} lines total). Use trace_callers or fetch another node_id from search neighbors."
                note = f" (resolved from stub to `{node_id}`)" if alt_id else ""
                body = f"### RAW SOURCE CONTENT FOR NODE: `{node_id}`{note}\n\n```python\n{source}\n```"
                return body + _graph_neighbors_footer(G, node_id)

        return f"### [Graph-RAG System Message]\nNode ID '{node_id}' not found."
    finally:
        model_manager.release()


@mcp.tool()
def trace_callers(node_id: str, active_project_root: str) -> str:
    """List callers and callees for a node_id. Use when neighbors are missing or insufficient.
    Then fetch_node_source on the relevant caller/callee — do not search again.

    Args:
        node_id: Exact node_id from search or fetch.
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

        if not G.has_node(node_id):
            return f"### [Graph-RAG System Message]\nNode ID '{node_id}' not found."

        callers, callees = get_call_neighbors(G, node_id)
        out = f"### Call graph for `{node_id}`\n\n"
        if callers:
            out += "**Callers** (fetch these for orchestration logic):\n"
            for c in callers:
                out += f"- `{c}`\n"
        else:
            out += "*No CALLS edges found for callers.*\n"
        if callees:
            out += "\n**Callees**:\n"
            for c in callees:
                out += f"- `{c}`\n"
        return out
    finally:
        model_manager.release()


if __name__ == "__main__":
    mcp.run(transport="stdio")