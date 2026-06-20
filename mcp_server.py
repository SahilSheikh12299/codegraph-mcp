import os
import sys
import logging
import hashlib
from pathlib import Path
from typing import Any, Tuple

import networkx as nx
from filelock import FileLock
from mcp.server.fastmcp import FastMCP

from fileParsing import extract_file_entities, WorkspaceScanner, ASTParser, ImportTracker
from graph_io import GraphSerializer
from embeddingPipeline import EmbeddingModelLifecycleManager, LocalEmbeddingPipeline
from advanced_engine import (
    AdvancedRetrievalEngine,
    recompute_call_centrality,
    build_grep_text,
    build_query_ranked_call_chain,
)
from markdown_format import format_multi_term_paths_markdown
from buildGraph import (
    wire_calls_for_file,
    build_func_registry_from_graph,
    strip_calls_edges,
    strip_calls_edges_for_file,
)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

mcp = FastMCP("Cursor-Graph-RAG-Engine")
model_manager = EmbeddingModelLifecycleManager()

# =========================================================================
# Graph cache paths + incremental sync
# =========================================================================
def get_graph_paths(active_project_root: str | Path) -> Tuple[Path, Path]:
    abs_path_str = str(Path(active_project_root).resolve())
    workspace_hash = hashlib.md5(abs_path_str.encode("utf-8")).hexdigest()
    mcp_cache_dir = Path("~/.cursor_graph_rag/graphs").expanduser()
    mcp_cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        mcp_cache_dir / f"graph_{workspace_hash}.json",
        mcp_cache_dir / f"graph_{workspace_hash}.lock",
    )


def _text_for_embedding(data: dict) -> str:
    return (data.get("embedding_text") or data.get("chunk_text") or "").strip()


def _backfill_grep_fields(G: nx.DiGraph, rel_paths: set[str] | None = None) -> None:
    for _node_id, data in G.nodes(data=True):
        if data.get("type") not in ("CLASS", "FUNCTION", "CONSTANT"):
            continue
        if rel_paths is not None and data.get("file_path") not in rel_paths:
            continue
        data["grep_text"] = build_grep_text(data)


def _embed_name_vectors(G: nx.DiGraph, rel_path: str, embedder: LocalEmbeddingPipeline) -> int:
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
    nodes_to_encode: list[str] = []
    texts_to_encode: list[str] = []
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
            wire_calls_for_file(
                G, rel_path, _parse_file_call_assets(repo_root, rel_path, tracker), registry
            )
    recompute_call_centrality(G)


def _rewire_file_calls(G: nx.DiGraph, repo_root: Path, rel_path: str, tracker: ImportTracker) -> None:
    strip_calls_edges_for_file(G, rel_path)
    registry = build_func_registry_from_graph(G)
    wire_calls_for_file(G, rel_path, _parse_file_call_assets(repo_root, rel_path, tracker), registry)
    recompute_call_centrality(G)


def execute_preflight_lazy_sync(repo_root: Path, G: nx.DiGraph, embedder: LocalEmbeddingPipeline) -> bool:
    if "indexed_timestamps" not in G.graph:
        G.graph["indexed_timestamps"] = {}

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
        if not full_path.exists():
            continue

        current_mtime = os.path.getmtime(full_path)
        last_indexed_time = G.graph["indexed_timestamps"].get(rel_path)

        if last_indexed_time is None:
            count = _embed_file_nodes(G, rel_path, embedder)
            if count:
                print(f"[COLD BOOT] Batched {count} embeddings for '{rel_path}'.", file=sys.stderr)
            G.graph["indexed_timestamps"][rel_path] = current_mtime
            first_run_initialization = True
            continue

        if current_mtime > last_indexed_time:
            dirty_files_detected = True
            print(f"\n[DIRTY FILE] Change in: '{rel_path}'")
            fresh_entities = extract_file_entities(rel_path, repo_root)
            existing_file_nodes = {
                node_id: data
                for node_id, data in G.nodes(data=True)
                if data.get("file_path") == rel_path
            }

            for node_id, fresh_data in fresh_entities.items():
                if node_id in existing_file_nodes:
                    if fresh_data["chunk_text"] != existing_file_nodes[node_id].get("chunk_text"):
                        emb_text = _text_for_embedding(fresh_data)
                        fresh_data["embedding"] = embedder.model.encode(
                            emb_text, convert_to_numpy=True
                        ).tolist()
                        fresh_data["grep_text"] = build_grep_text(fresh_data)
                        name = (fresh_data.get("name") or "").strip()
                        if name:
                            fresh_data["name_embedding"] = embedder.model.encode(
                                name, convert_to_numpy=True
                            ).tolist()
                        G.add_node(node_id, **fresh_data)
                else:
                    emb_text = _text_for_embedding(fresh_data)
                    fresh_data["embedding"] = embedder.model.encode(
                        emb_text, convert_to_numpy=True
                    ).tolist()
                    fresh_data["grep_text"] = build_grep_text(fresh_data)
                    name = (fresh_data.get("name") or "").strip()
                    if name:
                        fresh_data["name_embedding"] = embedder.model.encode(
                            name, convert_to_numpy=True
                        ).tolist()
                    G.add_node(node_id, **fresh_data)

            for old_node_id in existing_file_nodes:
                if old_node_id not in fresh_entities:
                    G.remove_node(old_node_id)

            G.graph["indexed_timestamps"][rel_path] = current_mtime

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
# Term-scoped matches: top-2 per grep term + top-2 per search query
# =========================================================================
def _term_scoped_chain(
    G: nx.DiGraph,
    *,
    anchor_id: str,
    rank_queries: list[str],
    embedder: Any,
    reranker: Any,
) -> list[dict[str, Any]]:
    """Return a tiny, term-scoped caller → anchor → callee chain (1 hop each side)."""
    return build_query_ranked_call_chain(
        G,
        anchor_id,
        rank_queries,
        embedder=embedder,
        reranker=reranker,
        max_callers=1,
        max_callees=1,
        max_downstream_hops=0,
    )


def _compile_term_scoped_results(
    G: nx.DiGraph,
    bucketed: dict[str, Any],
    *,
    search_queries: list[str],
    grep_terms: list[str],
    embedder: Any,
    reranker: Any,
    per_item_top_k: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (grep_results, search_results) with top-K matches per item and tiny chains."""
    grep_results: list[dict[str, Any]] = []
    search_results: list[dict[str, Any]] = []

    # Grep term buckets (already per-term in `bucketed`).
    buckets = bucketed.get("grep_buckets") or []
    bucket_by_term = {b.get("term"): b for b in buckets if b.get("term")}
    for term in grep_terms:
        b = bucket_by_term.get(term) or {}
        nodes = (b.get("nodes") or [])[:per_item_top_k]
        matches: list[dict[str, Any]] = []
        # Rank caller/callee using the behavior queries when available; fallback to the grep term.
        rank_queries = search_queries if search_queries else [term]
        for node in nodes:
            anchor_id = node.get("node_id")
            if not anchor_id:
                continue
            chain = _term_scoped_chain(
                G, anchor_id=anchor_id, rank_queries=rank_queries, embedder=embedder, reranker=reranker
            )
            matches.append({"anchor": node, "chain": chain})
        grep_results.append({"term": term, "matches": matches})

    # Semantic per-query bucket (preferred).
    sem_by_query = bucketed.get("semantic_by_query") or []
    sem_map = {q.get("query"): (q.get("nodes") or []) for q in sem_by_query if q.get("query")}
    for q in search_queries:
        nodes = (sem_map.get(q) or [])[:per_item_top_k]
        matches = []
        rank_queries = [q]
        for node in nodes:
            anchor_id = node.get("node_id")
            if not anchor_id:
                continue
            chain = _term_scoped_chain(
                G, anchor_id=anchor_id, rank_queries=rank_queries, embedder=embedder, reranker=reranker
            )
            matches.append({"anchor": node, "chain": chain})
        search_results.append({"query": q, "matches": matches})

    return grep_results, search_results


@mcp.tool()
def search_codebase_intent(
    search_queries: list[str],
    active_project_root: str,
    grep_terms: list[str] | None = None,
) -> str:
    """Find code by intent. Returns top-2 matches per grep term + per search query.

    Output is term-scoped and minimal: anchor cite, tiny caller→anchor→callee flow,
    and caller/callee cites (when present). No read-next padding.

    Args:
        search_queries: Intent phrases for semantic search.
        active_project_root: Absolute path to the repo root.
        grep_terms: Known symbol names to anchor the search (optional).
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
        bucketed = engine.compile_bucketed_context_package(
            search_queries=search_queries,
            grep_terms=grep_terms or [],
        )
        rewritten_queries = bucketed.get("rewritten_queries") or []
        rewritten_grep_terms = bucketed.get("grep_terms") or []
        grep_results, search_results = _compile_term_scoped_results(
            G,
            bucketed,
            search_queries=rewritten_queries,
            grep_terms=rewritten_grep_terms,
            embedder=embedder,
            reranker=reranker,
        )
        return format_multi_term_paths_markdown(
            grep_results=grep_results,
            search_results=search_results,
        )
    finally:
        model_manager.release()


# =========================================================================
# Legacy MCP tools (removed — search + native Read only)
# =========================================================================
# grep_graph_nodes, repo_references, fetch_node_metadata, fetch_node_source,
# fetch_snippets, plan_feature_touch_set, calculate_blast_radius, trace_callers


if __name__ == "__main__":
    mcp.run(transport="stdio")
