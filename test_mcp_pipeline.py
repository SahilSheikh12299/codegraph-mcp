"""End-to-end check of Graph-RAG MCP tools on the requests test repo."""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))

try:
    from mcp_server import (
        calculate_blast_radius,
        search_codebase_intent,
        get_graph_paths,
        trace_callers,
        fetch_node_source,
        execute_preflight_lazy_sync,
    )
    from graph_io import GraphSerializer
    from advanced_engine import get_call_neighbors
    from embeddingPipeline import EmbeddingModelLifecycleManager
except ImportError as e:
    print(f"Import failed: {e}")
    print("Run this script from cursorGraphRag/ next to mcp_server.py.")
    sys.exit(1)

# --- test inputs ---
TEST_PROJECT_ROOT = "/Users/sahilsheikh/Documents/Djikstra-codebase/repo-understanding-engine/test_repos/requests"
SEARCH_QUERIES = ["How does Session retry on 429 status codes?"]
TARGETED_SYMBOLS = ["Session"]
BLAST_RADIUS_SYMBOL = "Session"
CALL_GRAPH_NODE = "src/requests/sessions.py::SessionRedirectMixin.resolve_redirects"
EXPECTED_CALLER_FRAGMENT = "Session.send"
EXPECTED_CALLEE_FRAGMENT = "get_redirect_target"


def _parse_search_hits(markdown: str) -> list[dict]:
    if "No relevant structural context" in markdown:
        return []
    hits = []
    blocks = re.split(r"### Hit \d+:", markdown)
    for block in blocks[1:]:
        node_match = re.match(r"\s*`([^`]+)`", block)
        score_match = re.search(
            r"hybrid_score:\s*([\d.]+)\s*\|\s*callers:\s*(\d+)\s*\|\s*match:\s*(\S+)",
            block,
        )
        if node_match and score_match:
            hits.append(
                {
                    "node_id": node_match.group(1),
                    "hybrid_score": float(score_match.group(1)),
                    "callers": int(score_match.group(2)),
                    "match": score_match.group(3),
                }
            )
    return hits


def _parse_blast_radius_count(markdown: str) -> int:
    return len(re.findall(r"^- \*\*\[", markdown, re.MULTILINE))


def _print_block(title: str, lines: list[str]) -> None:
    print(f"\n{title}")
    for line in lines:
        print(f"  {line}")


def _verdict(passed: bool, reason: str) -> bool:
    label = "PASS" if passed else "FAIL"
    print(f"  Verdict: {label} — {reason}")
    return passed


def run_system_audit() -> None:
    results: list[bool] = []

    if not Path(TEST_PROJECT_ROOT).exists():
        print(f"Repo path does not exist: {TEST_PROJECT_ROOT}")
        sys.exit(1)

    graph_path, _ = get_graph_paths(TEST_PROJECT_ROOT)

    _print_block(
        "Test setup",
        [
            f"repo: {TEST_PROJECT_ROOT}",
            f"search_queries: {SEARCH_QUERIES}",
            f"targeted_symbols: {TARGETED_SYMBOLS}",
            f"graph cache: {graph_path}",
        ],
    )

    # --- 1. search_codebase_intent ---
    _print_block(
        "Step 1 — search_codebase_intent",
        [
            "Purpose: find candidate node_ids (metadata only, no source code).",
            f"Input: search_queries={SEARCH_QUERIES}, targeted_symbols={TARGETED_SYMBOLS}",
        ],
    )

    search_output = search_codebase_intent(
        search_queries=SEARCH_QUERIES,
        active_project_root=TEST_PROJECT_ROOT,
        targeted_symbols=TARGETED_SYMBOLS,
    )

    hits = _parse_search_hits(search_output)
    print(f"  Returned: {len(hits)} hit(s)")
    for i, hit in enumerate(hits[:5], 1):
        print(
            f"    {i}. {hit['node_id']}"
            f" | hybrid_score={hit['hybrid_score']}"
            f" | callers={hit['callers']}"
            f" | match={hit['match']}"
        )
    if len(hits) > 5:
        print(f"    ... and {len(hits) - 5} more")

    top_node = hits[0]["node_id"] if hits else None
    results.append(
        _verdict(
            len(hits) > 0,
            "search returned at least one ranked node_id"
            if hits
            else "search returned no hits — check graph cache and similarity floor",
        )
    )

    # --- 2. calculate_blast_radius ---
    _print_block(
        "Step 2 — calculate_blast_radius",
        [
            "Purpose: list upstream dependents before editing code (not for discovery).",
            f"Input: target_symbol={BLAST_RADIUS_SYMBOL!r}",
        ],
    )

    blast_output = calculate_blast_radius(
        target_symbol=BLAST_RADIUS_SYMBOL,
        active_project_root=TEST_PROJECT_ROOT,
    )
    dependent_count = _parse_blast_radius_count(blast_output)
    symbol_found = "could not be matched" not in blast_output

    print(f"  Returned: {dependent_count} dependent node(s) for symbol {BLAST_RADIUS_SYMBOL!r}")
    if dependent_count:
        sample = [ln.strip() for ln in blast_output.splitlines() if ln.startswith("- **[")][:3]
        for line in sample:
            print(f"    {line}")
        if dependent_count > 3:
            print(f"    ... and {dependent_count - 3} more")

    results.append(
        _verdict(
            symbol_found and dependent_count > 0,
            f"symbol matched and {dependent_count} dependents listed"
            if symbol_found and dependent_count > 0
            else "symbol not found or no dependents listed",
        )
    )

    # --- 3. trace_callers ---
    _print_block(
        "Step 3 — trace_callers",
        [
            "Purpose: full caller/callee lists when search/fetch neighbors are not enough.",
            f"Input: node_id={CALL_GRAPH_NODE!r}",
        ],
    )

    trace_output = trace_callers(
        node_id=CALL_GRAPH_NODE,
        active_project_root=TEST_PROJECT_ROOT,
    )
    has_callers = "**Callers**" in trace_output and "No CALLS edges" not in trace_output
    has_callees = "**Callees**" in trace_output

    print("  Returned:")
    for line in trace_output.splitlines():
        if line.strip():
            print(f"    {line}")

    results.append(
        _verdict(
            has_callers and has_callees,
            "callers and callees sections present"
            if has_callers and has_callees
            else "missing callers or callees in trace output",
        )
    )

    # --- 4. fetch_node_source ---
    _print_block(
        "Step 4 — fetch_node_source",
        [
            "Purpose: return source code plus graph neighbors for one or more node_ids.",
            f"Input: node_ids=[{CALL_GRAPH_NODE!r}]",
        ],
    )

    fetch_output = fetch_node_source(
        node_ids=[CALL_GRAPH_NODE],
        active_project_root=TEST_PROJECT_ROOT,
    )
    has_source = "RAW SOURCE CONTENT" in fetch_output
    has_neighbors = "Graph neighbors" in fetch_output
    line_count = len(fetch_output.splitlines())

    print(f"  Returned: {line_count} lines of text")
    print(f"    contains source block: {has_source}")
    print(f"    contains graph neighbors footer: {has_neighbors}")

    results.append(
        _verdict(
            has_source and has_neighbors,
            "source code and graph neighbors both present"
            if has_source and has_neighbors
            else "missing source or neighbors in fetch output",
        )
    )

    # --- 5. graph CALLS edges (internal check, not an MCP tool) ---
    _print_block(
        "Step 5 — CALLS edge check (graph validation)",
        [
            "Purpose: confirm redirect flow is wired in the graph backing the MCP tools.",
            f"Node under test: {CALL_GRAPH_NODE}",
            f"Expect caller containing: {EXPECTED_CALLER_FRAGMENT!r}",
            f"Expect callee containing: {EXPECTED_CALLEE_FRAGMENT!r}",
        ],
    )

    mm = EmbeddingModelLifecycleManager()
    emb = mm.acquire()
    try:
        G = GraphSerializer.load_from_json(TEST_PROJECT_ROOT, graph_path)
        if execute_preflight_lazy_sync(Path(TEST_PROJECT_ROOT), G, emb):
            GraphSerializer.save_to_json(G, Path(TEST_PROJECT_ROOT), graph_path)

        callers, callees = get_call_neighbors(G, CALL_GRAPH_NODE)
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"  Callers ({len(callers)}): {callers}")
        print(f"  Callees ({len(callees)}): {callees[:5]}{'...' if len(callees) > 5 else ''}")

        caller_ok = any(EXPECTED_CALLER_FRAGMENT in c for c in callers)
        callee_ok = any(EXPECTED_CALLEE_FRAGMENT in c for c in callees)
        results.append(
            _verdict(
                caller_ok and callee_ok,
                "redirect caller and callee edges found"
                if caller_ok and callee_ok
                else f"caller_ok={caller_ok}, callee_ok={callee_ok}",
            )
        )
    finally:
        mm.release()

    # --- summary ---
    passed = sum(results)
    total = len(results)
    _print_block(
        "Summary",
        [
            f"Steps passed: {passed}/{total}",
            f"Top search hit: {top_node or 'none'}",
            "Overall: ALL PASS" if passed == total else "Overall: SOME FAILURES",
        ],
    )

    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    run_system_audit()
