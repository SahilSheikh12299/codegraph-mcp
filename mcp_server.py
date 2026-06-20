import os
import ast
import sys
import gc
import logging
import hashlib
import subprocess
import time
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
    get_call_neighbors,
    recompute_call_centrality,
    build_grep_text,
    grep_search_nodes,
    extract_source_excerpt,
    build_query_ranked_call_chain,
    NEIGHBOR_ID_LIMIT,
)
from markdown_format import (
    cursor_citation,
    format_error,
    format_search_markdown,
    format_grep_markdown,
    format_snippets_markdown,
    format_source_markdown,
    format_metadata_markdown,
    format_touch_set_markdown,
    format_repo_refs_markdown,
    format_blast_radius_markdown,
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

_NODE_CACHE_TTL_SEC = 1800
_NODE_CACHE_MAX = 500
_node_payload_cache: dict[tuple, tuple[float, dict[str, Any]]] = {}

_DEFAULT_REPO_REF_GLOBS = [
    "tests/**",
    "test/**",
    "docs/**",
    "doc/**",
    "**/*.md",
    "**/*.rst",
    "HISTORY.*",
    "CHANGELOG.*",
]


def _cache_get(key: tuple) -> dict[str, Any] | None:
    entry = _node_payload_cache.get(key)
    if not entry:
        return None
    ts, payload = entry
    if time.time() - ts > _NODE_CACHE_TTL_SEC:
        _node_payload_cache.pop(key, None)
        return None
    return payload


def _cache_set(key: tuple, payload: dict[str, Any]) -> None:
    if len(_node_payload_cache) >= _NODE_CACHE_MAX:
        oldest_key = min(_node_payload_cache, key=lambda k: _node_payload_cache[k][0])
        _node_payload_cache.pop(oldest_key, None)
    _node_payload_cache[key] = (time.time(), payload)


def _is_hollow_constant_source(source: str) -> bool:
    s = (source or "").strip()
    if not s:
        return True
    if s.startswith("Entity Type: CONSTANT"):
        return True
    return _is_stub_source(s)

def _is_stub_source(source: str) -> bool:
    s = source.strip()
    return s.endswith("...") or s.endswith(": ...")


def _parse_line_span(node_data: dict) -> tuple[int | None, int | None]:
    span = node_data.get("line_span")
    if not span or not isinstance(span, (list, tuple)) or len(span) < 2:
        return None, None
    return int(span[0]), int(span[1])


def _source_completeness_fields(source: str) -> dict[str, Any]:
    lines = (source or "").splitlines()
    return {
        "source_complete": True,
        "truncated": False,
        "line_count": len(lines),
        "source_kind": "ast_bound_full_body",
    }


def _to_snippet_payload(payload: dict[str, Any], max_lines: int) -> dict[str, Any]:
    """Trim a full node payload to a line-capped excerpt for skim / edit context."""
    if payload.get("error"):
        return payload
    full_source = (payload.get("source") or "").strip()
    node_type = payload.get("type", "FUNCTION")
    lines = full_source.splitlines()
    cap = max(1, int(max_lines))
    excerpt = extract_source_excerpt(full_source, node_type, max_lines=cap)
    truncated = len(lines) > cap or "# ... truncated" in excerpt
    out: dict[str, Any] = {
        "node_id": payload["node_id"],
        "type": node_type,
        "file_path": payload.get("file_path"),
        "name": payload.get("name"),
        "signature": payload.get("signature", ""),
        "start_line": payload.get("start_line"),
        "end_line": payload.get("end_line"),
        "snippet": excerpt,
        "line_count": len(lines),
        "truncated": truncated,
        "source_complete": not truncated,
        "source_kind": "ast_bound_excerpt",
        "max_lines": cap,
    }
    if payload.get("resolved_stub_from"):
        out["resolved_stub_from"] = payload["resolved_stub_from"]
    return out


def _constant_fallback(workspace_root: Path, node_id: str) -> dict[str, Any] | None:
    if "::" not in node_id:
        return None
    rel_path, name = node_id.rsplit("::", 1)
    if "." in name:
        return None
    full = workspace_root / rel_path
    if not full.is_file():
        return None
    for g in ASTParser(file_path=full).parse().get("globals", []):
        if g["name"] != name:
            continue
        file_lines = full.read_text(encoding="utf-8").splitlines()
        start, end = g["line_span"]
        source = "\n".join(file_lines[start - 1 : end])
        return {
            "node_id": node_id,
            "type": "CONSTANT",
            "file_path": rel_path,
            "name": name,
            "signature": "",
            "start_line": start,
            "end_line": end,
            "source": source,
            **_source_completeness_fields(source),
            "neighbors": {"callers": [], "callees": []},
            "resolved_via": "fallback_inline",
        }
    return None


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


def _make_snippet_loader(
    G: nx.DiGraph,
    workspace_root: Path,
    max_lines: int,
) -> Any:
    """Return a node_id -> cursor citation block loader."""
    cache: dict[str, str] = {}

    def load(node_id: str) -> str | None:
        if node_id in cache:
            return cache[node_id] or None
        payload = _build_node_payload(
            G, node_id, workspace_root=workspace_root, include_neighbors=False
        )
        if payload.get("error"):
            cache[node_id] = ""
            return None
        snippet = _to_snippet_payload(payload, max_lines)
        block = cursor_citation(
            snippet.get("file_path"),
            snippet.get("start_line"),
            snippet.get("end_line"),
            snippet.get("snippet") or "",
        )
        cache[node_id] = block
        return block or None

    return load


def _coerce_term_list(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        stripped = str(item).strip()
        if stripped:
            out.append(stripped)
    return out


def _resolve_grep_term_args(
    terms: list[str] | None,
    grep_terms: list[str] | None,
) -> list[str]:
    """Merge terms + grep_terms (search alias); dedupe preserving order."""
    seen: set[str] = set()
    merged: list[str] = []
    for term in _coerce_term_list(terms) + _coerce_term_list(grep_terms):
        if term not in seen:
            seen.add(term)
            merged.append(term)
    return merged


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


def _backfill_grep_fields(G: nx.DiGraph, rel_paths: set[str] | None = None) -> None:
    """Populate grep_text and name_embedding on CLASS/FUNCTION nodes."""
    for node_id, data in G.nodes(data=True):
        if data.get("type") not in ("CLASS", "FUNCTION", "CONSTANT"):
            continue
        if rel_paths is not None and data.get("file_path") not in rel_paths:
            continue
        data["grep_text"] = build_grep_text(data)


def _embed_name_vectors(G: nx.DiGraph, rel_path: str, embedder: LocalEmbeddingPipeline) -> int:
    """Batch-encode node names for fuzzy grep fallback."""
    nodes_to_encode: list[str] = []
    texts_to_encode: list[str] = []
    for node_id, data in G.nodes(data=True):
        if data.get("file_path") != rel_path:
            continue
        name = (data.get("name") or "").strip()
        if name:
            nodes_to_encode.append(node_id)
            texts_to_encode.append(name)
    if texts_to_encode:
        vectors = embedder.model.encode(texts_to_encode, convert_to_numpy=True).tolist()
        for node_id, vector in zip(nodes_to_encode, vectors):
            G.nodes[node_id]["name_embedding"] = vector
    return len(texts_to_encode)


def backfill_source_chunks(G: nx.DiGraph, repo_root: Path, rel_paths: set[str]) -> None:
    """Overlay real AST source segments and embedding_text onto graph nodes."""
    for rel_path in rel_paths:
        for node_id, data in extract_file_entities(rel_path, repo_root).items():
            if G.has_node(node_id):
                G.nodes[node_id]["chunk_text"] = data["chunk_text"]
                G.nodes[node_id]["embedding_text"] = data["embedding_text"]
                if data.get("line_span"):
                    G.nodes[node_id]["line_span"] = data["line_span"]
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
            data["grep_text"] = build_grep_text(data)
    if texts_to_encode:
        vectors = embedder.model.encode(texts_to_encode, convert_to_numpy=True).tolist()
        for node_id, vector in zip(nodes_to_encode, vectors):
            G.nodes[node_id]["embedding"] = vector
    _embed_name_vectors(G, rel_path, embedder)
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

    if G.graph.get("grep_schema") != 1:
        _backfill_grep_fields(G)
        for rel_path in tracked_files:
            if (repo_root / rel_path).exists():
                _embed_name_vectors(G, rel_path, embedder)
        G.graph["grep_schema"] = 1
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
                        fresh_data["grep_text"] = build_grep_text(fresh_data)
                        name = (fresh_data.get("name") or "").strip()
                        if name:
                            fresh_data["name_embedding"] = embedder.model.encode(name, convert_to_numpy=True).tolist()
                        G.add_node(node_id, **fresh_data)
                else:
                    # The node ID is brand new to the system
                    print(f"  -> [ADDED] New element detected: `{node_id}`. Generating vector...")
                    emb_text = _text_for_embedding(fresh_data)
                    fresh_data["embedding"] = embedder.model.encode(emb_text, convert_to_numpy=True).tolist()
                    fresh_data["grep_text"] = build_grep_text(fresh_data)
                    name = (fresh_data.get("name") or "").strip()
                    if name:
                        fresh_data["name_embedding"] = embedder.model.encode(name, convert_to_numpy=True).tolist()
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
    grep_terms: list[str] | None = None,
    include_snippets: bool = True,
    snippet_max_lines: int = 25,
) -> str:
    """Find code by intent. Returns markdown with hits, call chains, and citation blocks.

    Pass grep_terms whenever you have symbol names. Snippets are included by default.

    Args:
        search_queries: LLM-rewritten intent queries for semantic search.
        active_project_root: Absolute path to the repo root.
        grep_terms: Literal symbol names for graph grep buckets.
        include_snippets: Include line-capped citation blocks for hits and call chain.
        snippet_max_lines: Max lines per inline snippet (default 25).
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
        result = engine.compile_bucketed_context_package(
            search_queries=search_queries,
            grep_terms=grep_terms or [],
        )
        queries = result.get("rewritten_queries") or []
        call_chains: dict[str, list[dict[str, Any]]] = {}
        for node in (result.get("semantic_bucket") or {}).get("nodes", []):
            nid = node.get("node_id")
            if nid:
                call_chains[nid] = build_query_ranked_call_chain(
                    G,
                    nid,
                    queries,
                    embedder=embedder,
                    reranker=reranker,
                )

        snippet_loader = (
            _make_snippet_loader(G, workspace_root, snippet_max_lines)
            if include_snippets
            else None
        )
        return format_search_markdown(
            result,
            call_chains=call_chains,
            snippet_loader=snippet_loader,
        )
    finally:
        model_manager.release()


@mcp.tool()
def grep_graph_nodes(
    active_project_root: str,
    terms: list[str] | None = None,
    grep_terms: list[str] | None = None,
    case_sensitive: bool = False,
    max_hits_per_term: int = 3,
) -> str:
    """Graph symbol lookup: exact match metadata only (no neighbors, no source).

    Literal match returns top 1 per term; fuzzy fallback returns up to max_hits_per_term.
    Use for bare symbol names (e.g. \"rebuild_method\"). If you already have a node_id
    from search, use fetch_snippets instead — do not call this tool.

    Required args:
        active_project_root: Absolute path to the repo root.
        terms: Non-empty list of literal symbol strings, e.g. [\"Session.get\", \"rebuild_method\"].
        grep_terms: Alias for terms (same as search_codebase_intent); either terms or grep_terms required.
    """
    workspace_root = Path(active_project_root).resolve()
    resolved_terms = _resolve_grep_term_args(terms, grep_terms)
    if not resolved_terms:
        return format_error(
            "grep_graph_nodes requires terms: [\"SymbolName\", ...] "
            "(grep_terms is also accepted). "
            "If you have a node_id from search, use fetch_snippets with node_ids instead.",
            example=(
                f'active_project_root: "{workspace_root}"\n'
                'terms: ["rebuild_method"]'
            ),
        )

    graph_path, lock_path = get_graph_paths(workspace_root)

    embedder = model_manager.acquire()
    try:
        with FileLock(lock_path):
            G = GraphSerializer.load_from_json(workspace_root, graph_path)
            if execute_preflight_lazy_sync(workspace_root, G, embedder):
                GraphSerializer.save_to_json(G, workspace_root, graph_path)

        buckets: list[dict[str, Any]] = []
        for term in resolved_terms:
            term = (term or "").strip()
            if not term:
                continue
            nodes = grep_search_nodes(
                G,
                term,
                case_sensitive=case_sensitive,
                embedder=embedder,
                fuzzy_top_k=max(1, int(max_hits_per_term)),
            )
            buckets.append({
                "term": term,
                "match_mode": nodes[0]["match_mode"] if nodes else "none",
                "nodes": nodes,
            })

        return format_grep_markdown(buckets, resolved_terms)
    finally:
        model_manager.release()


@mcp.tool()
def calculate_blast_radius(
    target_symbol: str,
    active_project_root: str,
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
            return f"## Upstream dependents\n\nTarget symbol `{target_symbol}` could not be matched."
            
        impact_profiles = engine.compute_upstream_blast_radius(matching_nodes)
        
        impact_profiles = sorted(
            impact_profiles,
            key=lambda x: (x.get("file_path") or "", x.get("node_id") or ""),
        )

        return format_blast_radius_markdown(
            target_symbol,
            impact_profiles,
            len(impact_profiles),
            max_items,
        )
    finally:
        model_manager.release()


def _build_node_payload(
    G: nx.DiGraph,
    node_id: str,
    *,
    neighbors_max: int = NEIGHBOR_ID_LIMIT,
    workspace_root: Path | None = None,
    include_neighbors: bool = True,
) -> dict[str, Any]:
    """Single node: full source; optional 1-hop neighbor IDs."""
    if not G.has_node(node_id):
        if workspace_root is not None:
            fb = _constant_fallback(workspace_root, node_id)
            if fb:
                if not include_neighbors:
                    fb = {k: v for k, v in fb.items() if k != "neighbors"}
                return fb
        return {"node_id": node_id, "error": "not_indexed"}

    node_data = G.nodes[node_id]
    node_type = node_data.get("type", "UNKNOWN")
    resolved_id = node_id
    chunk_text = node_data.get("chunk_text", "")

    if node_type == "CONSTANT" and _is_hollow_constant_source(chunk_text) and workspace_root is not None:
        fb = _constant_fallback(workspace_root, node_id)
        if fb:
            if not include_neighbors:
                fb = {k: v for k, v in fb.items() if k != "neighbors"}
            return fb

    if node_type != "CLASS" and _is_stub_source(chunk_text):
        alt_id = _find_impl_alternate(G, node_id, node_data)
        if alt_id:
            resolved_id = alt_id
            node_data = G.nodes[resolved_id]
            chunk_text = node_data.get("chunk_text", chunk_text)

    source = (chunk_text or "").strip()
    start_line, end_line = _parse_line_span(node_data)
    callers, callees = (
        get_call_neighbors(G, resolved_id, limit=neighbors_max)
        if include_neighbors
        else ([], [])
    )
    payload: dict[str, Any] = {
        "node_id": resolved_id,
        "type": node_type,
        "file_path": node_data.get("file_path"),
        "name": node_data.get("name"),
        "signature": node_data.get("signature", ""),
        "start_line": start_line,
        "end_line": end_line,
        "source": source,
        **_source_completeness_fields(source),
    }
    if include_neighbors:
        payload["neighbors"] = {"callers": callers, "callees": callees}

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


def _build_node_metadata_payload(
    G: nx.DiGraph,
    node_id: str,
    *,
    neighbors_max: int = NEIGHBOR_ID_LIMIT,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Metadata-only: signature, line span, neighbors — no source body."""
    full = _build_node_payload(
        G, node_id, neighbors_max=neighbors_max, workspace_root=workspace_root
    )
    if full.get("error"):
        return full
    meta: dict[str, Any] = {
        "node_id": full["node_id"],
        "type": full.get("type"),
        "file_path": full.get("file_path"),
        "name": full.get("name"),
        "signature": full.get("signature", ""),
        "start_line": full.get("start_line"),
        "end_line": full.get("end_line"),
        "neighbors": full.get("neighbors", {"callers": [], "callees": []}),
    }
    if full.get("resolved_stub_from"):
        meta["resolved_stub_from"] = full["resolved_stub_from"]
    if full.get("type") == "CLASS" and full.get("methods"):
        meta["methods"] = full["methods"]
    if full.get("type") == "CONSTANT" and full.get("source"):
        snippet = (full["source"] or "").strip()
        if snippet and not _is_hollow_constant_source(snippet):
            meta["value_snippet"] = snippet[:200]
    return meta


def _fetch_node_payloads(
    G: nx.DiGraph,
    node_ids: list[str],
    *,
    mode: str,
    neighbors_max: int,
    workspace_root: Path,
    include_neighbors: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fetch node payloads with per-node server-side cache."""
    builder = _build_node_metadata_payload if mode == "metadata" else _build_node_payload
    nodes: list[dict[str, Any]] = []
    hits, misses = 0, 0
    for node_id in node_ids:
        cache_key = (mode, str(workspace_root), node_id, neighbors_max, include_neighbors)
        cached = _cache_get(cache_key)
        if cached is not None:
            nodes.append(cached)
            hits += 1
            continue
        if mode == "metadata":
            payload = builder(
                G, node_id, neighbors_max=neighbors_max, workspace_root=workspace_root
            )
        else:
            payload = builder(
                G,
                node_id,
                neighbors_max=neighbors_max,
                workspace_root=workspace_root,
                include_neighbors=include_neighbors,
            )
        if not payload.get("error"):
            _cache_set(cache_key, payload)
        nodes.append(payload)
        misses += 1
    return nodes, {"hits": hits, "misses": misses}


def _load_graph_for_workspace(workspace_root: Path) -> tuple[nx.DiGraph, Path, Path]:
    graph_path, lock_path = get_graph_paths(workspace_root)
    embedder = model_manager.acquire()
    try:
        with FileLock(lock_path):
            G = GraphSerializer.load_from_json(workspace_root, graph_path)
            if execute_preflight_lazy_sync(workspace_root, G, embedder):
                GraphSerializer.save_to_json(G, workspace_root, graph_path)
        return G, graph_path, lock_path
    finally:
        model_manager.release()


def _run_ripgrep_references(
    workspace_root: Path,
    terms: list[str],
    *,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_hits: int = 200,
    context_lines: int = 0,
) -> dict[str, Any]:
    """Repo-wide text search via ripgrep; returns path + line + snippet only."""
    cleaned = [t.strip() for t in terms if t and t.strip()]
    if not cleaned:
        return {"terms": terms, "hits": [], "coverage": {"searched_globs": []}}

    cmd = [
        "rg",
        "-n",
        "--no-heading",
        f"--max-count={max(1, max_hits)}",
        "--color=never",
    ]
    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])

    globs = include_globs if include_globs is not None else ["**/*"]
    for glob in globs:
        cmd.extend(["-g", glob])
    for glob in exclude_globs or []:
        cmd.extend(["-g", f"!{glob}"])
    for term in cleaned:
        cmd.extend(["-e", term])
    cmd.append(str(workspace_root))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return {
            "terms": cleaned,
            "hits": [],
            "error": "ripgrep (rg) not found on PATH",
            "coverage": {"searched_globs": globs},
        }
    except subprocess.TimeoutExpired:
        return {
            "terms": cleaned,
            "hits": [],
            "error": "ripgrep timed out",
            "coverage": {"searched_globs": globs},
        }

    hits: list[dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines():
        if len(hits) >= max_hits:
            break
        if not line.strip():
            continue
        if line.startswith("--") and context_lines > 0:
            continue
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        rel_path = Path(parts[0])
        try:
            rel_path = rel_path.relative_to(workspace_root)
        except ValueError:
            pass
        try:
            line_no = int(parts[1])
        except ValueError:
            continue
        hits.append({
            "file_path": str(rel_path).replace("\\", "/"),
            "line": line_no,
            "text": parts[2].strip()[:300],
        })

    return {
        "terms": cleaned,
        "hits": hits,
        "truncated": len(hits) >= max_hits,
        "coverage": {
            "searched_globs": globs,
            "rg_exit_code": proc.returncode,
        },
    }


def _format_node_source(
    G: nx.DiGraph,
    node_id: str,
    *,
    neighbors_max: int = NEIGHBOR_ID_LIMIT,
) -> str:
    """Returns full source as a cursor citation block."""
    payload = _build_node_payload(
        G, node_id, neighbors_max=neighbors_max, include_neighbors=False
    )
    if payload.get("error"):
        return f"### {node_id}\n{payload['error']}"

    block = cursor_citation(
        payload.get("file_path"),
        payload.get("start_line"),
        payload.get("end_line"),
        payload.get("source") or "",
    )
    return f"### {payload.get('name') or node_id}\nnode_id: {payload['node_id']}\n\n{block}"


@mcp.tool()
def fetch_node_metadata(
    node_ids: list[str],
    active_project_root: str,
    neighbors_max: int = NEIGHBOR_ID_LIMIT,
) -> str:
    """Lightweight fetch: signatures, line ranges, callers/callees — no source bodies.

    Use for edit planning and file-touch questions. Call fetch_node_source only when
    you need full implementation bodies.

    Args:
        node_ids: Exact node_id(s) from search or grep output.
        active_project_root: Absolute path to the repo root.
        neighbors_max: Max caller/callee node_ids per node (default 50).
    """
    workspace_root = Path(active_project_root).resolve()
    G = _load_graph_for_workspace(workspace_root)[0]

    nodes, _cache_stats = _fetch_node_payloads(
        G,
        node_ids,
        mode="metadata",
        neighbors_max=neighbors_max,
        workspace_root=workspace_root,
    )
    return format_metadata_markdown(nodes, G)


@mcp.tool()
def repo_references(
    terms: list[str],
    active_project_root: str,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_hits: int = 200,
    context_lines: int = 0,
) -> str:
    """Repo-wide symbol references via ripgrep (tests, docs, markdown, history).

    Returns file paths + line numbers + matched line snippets only — not full files.
    Use for planning questions that span non-Python paths not in the graph.

    Args:
        terms: Literal strings to search for (symbol names, config keys).
        active_project_root: Absolute path to the repo root.
        include_globs: Optional glob filters (default: all files).
        exclude_globs: Optional exclusion globs.
        max_hits: Cap total hits returned.
        context_lines: Extra context lines around each match (0 = match line only).
    """
    workspace_root = Path(active_project_root).resolve()
    result = _run_ripgrep_references(
        workspace_root,
        terms,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        max_hits=max_hits,
        context_lines=context_lines,
    )
    if result.get("error"):
        return format_error(str(result["error"]))
    return format_repo_refs_markdown(result.get("hits", []), result.get("terms"))


def _merge_touch_file(
    bucket: dict[str, dict[str, Any]],
    meta: dict[str, Any],
    *,
    reason: str,
) -> None:
    fp = meta.get("file_path")
    if not fp or meta.get("error"):
        return
    sl, el = meta.get("start_line"), meta.get("end_line")
    line_range = [sl, el] if sl and el else []
    if fp not in bucket:
        bucket[fp] = {
            "file_path": fp,
            "reason": reason,
            "node_ids": [meta["node_id"]],
            "line_ranges": [line_range] if line_range else [],
        }
        return
    entry = bucket[fp]
    if meta["node_id"] not in entry["node_ids"]:
        entry["node_ids"].append(meta["node_id"])
    if line_range and line_range not in entry["line_ranges"]:
        entry["line_ranges"].append(line_range)


@mcp.tool()
def plan_feature_touch_set(
    feature_description: str,
    grep_terms: list[str],
    active_project_root: str,
    max_files: int = 30,
) -> str:
    """Minimal PR surface: defining files + related callers + non-graph refs (tests/docs).

    Returns file paths and line ranges only — no full source bodies.

    Args:
        feature_description: Short description of the feature or change.
        grep_terms: Exact symbols to anchor the touch set.
        active_project_root: Absolute path to the repo root.
        max_files: Max files per required/related bucket.
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
        queries = [feature_description.strip()] if feature_description and feature_description.strip() else []
        bucketed = engine.compile_bucketed_context_package(
            search_queries=queries,
            grep_terms=grep_terms,
        )

        primary_ids: list[str] = []
        for node in bucketed.get("semantic_bucket", {}).get("nodes", []):
            primary_ids.append(node["node_id"])
        for bucket in bucketed.get("grep_buckets", []):
            for node in bucket.get("nodes", []):
                primary_ids.append(node["node_id"])

        seen: set[str] = set()
        ordered_primary: list[str] = []
        for nid in primary_ids:
            if nid not in seen:
                seen.add(nid)
                ordered_primary.append(nid)

        expanded: set[str] = set(ordered_primary)
        for nid in ordered_primary[:8]:
            meta = _build_node_metadata_payload(
                G, nid, workspace_root=workspace_root, neighbors_max=5
            )
            for caller in (meta.get("neighbors") or {}).get("callers", [])[:3]:
                expanded.add(caller)
            for callee in (meta.get("neighbors") or {}).get("callees", [])[:3]:
                expanded.add(callee)

        required_files: dict[str, dict[str, Any]] = {}
        related_files: dict[str, dict[str, Any]] = {}
        primary_set = set(ordered_primary)
        for nid in expanded:
            meta = _build_node_metadata_payload(
                G, nid, workspace_root=workspace_root, neighbors_max=5
            )
            if nid in primary_set:
                _merge_touch_file(required_files, meta, reason="semantic_or_grep_hit")
            else:
                _merge_touch_file(related_files, meta, reason="caller_or_callee")

        non_graph = _run_ripgrep_references(
            workspace_root,
            grep_terms,
            include_globs=_DEFAULT_REPO_REF_GLOBS,
            max_hits=100,
        )

        return format_touch_set_markdown(
            feature_description,
            list(required_files.values())[:max_files],
            list(related_files.values())[:max_files],
            non_graph.get("hits", []),
        )
    finally:
        model_manager.release()


@mcp.tool()
def fetch_node_source(
    node_ids: list[str],
    active_project_root: str,
    neighbors_max: int = NEIGHBOR_ID_LIMIT,
) -> str:
    """Full AST-bound source + citation lines for one or more node_ids (no neighbors).

    Module constants use node_id like path/file.py::CONSTANT_NAME.

    Args:
        node_ids: Exact node_id(s) from search or grep output.
        active_project_root: Absolute path to the repo root.
        neighbors_max: Deprecated; kept for API compatibility.
    """
    workspace_root = Path(active_project_root).resolve()
    G = _load_graph_for_workspace(workspace_root)[0]

    nodes, _cache_stats = _fetch_node_payloads(
        G,
        node_ids,
        mode="source",
        neighbors_max=neighbors_max,
        workspace_root=workspace_root,
        include_neighbors=False,
    )
    return format_source_markdown(nodes)


@mcp.tool()
def fetch_snippets(
    node_ids: list[str],
    active_project_root: str,
    max_lines: int = 30,
) -> str:
    """Line-capped AST-bound excerpts for many node_ids in one batched call (no neighbors).

    Use when search snippets are truncated or you need additional node_ids.

    Args:
        node_ids: Exact node_id(s) from search or grep — pass every skim target at once.
        active_project_root: Absolute path to the repo root.
        max_lines: Max lines per snippet (default 30); tail return lines included when truncated.
    """
    workspace_root = Path(active_project_root).resolve()
    G = _load_graph_for_workspace(workspace_root)[0]
    cap = max(1, int(max_lines))

    full_nodes, _cache_stats = _fetch_node_payloads(
        G,
        node_ids,
        mode="source",
        neighbors_max=NEIGHBOR_ID_LIMIT,
        workspace_root=workspace_root,
        include_neighbors=False,
    )
    nodes = [_to_snippet_payload(p, cap) for p in full_nodes]
    return format_snippets_markdown(nodes)


@mcp.tool()
def trace_callers(
    node_id: str,
) -> str:
    """Deprecated. Use search_codebase_intent for callers/callees, then fetch_snippets or fetch_node_source."""
    return (
        "## Deprecated\n\n"
        "trace_callers is deprecated. Use search_codebase_intent for graph neighbors "
        "(call chain is included in search results), then fetch_snippets or fetch_node_source for code."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")