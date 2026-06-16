"""End-to-end check of Graph-RAG MCP tools on the requests test repo."""

import json
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
SEARCH_QUERIES = ["Session retry 429 status code"]
BLAST_RADIUS_SYMBOL = "Session"
CALL_GRAPH_NODE = "src/requests/sessions.py::SessionRedirectMixin.resolve_redirects"
EXPECTED_CALLER_FRAGMENT = "Session.send"
EXPECTED_CALLEE_FRAGMENT = "get_redirect_target"

REDIRECT_SEARCH_QUERIES = [
    "HTTP redirect response follow Location header 301 302",
    "resolve_redirects allow_redirects",
    "is_redirect status code",
]
REDIRECT_TOP_NODE = "src/requests/sessions.py::SessionRedirectMixin.resolve_redirects"


def _parse_search_hits(markdown: str) -> list[dict]:
    if "No relevant structural context" in markdown:
        return []
    hits = []
    blocks = re.split(r"### Hit \d+:", markdown)
    for block in blocks[1:]:
        node_match = re.match(r"\s*`([^`]+)`", block)
        score_match = re.search(r"score:\s*([\d.]+)\s*\|\s*callers:\s*(\d+)", block)
        if node_match and score_match:
            hits.append(
                {
                    "node_id": node_match.group(1),
                    "hybrid_score": float(score_match.group(1)),
                    "callers": int(score_match.group(2)),
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


def _redirect_answerable_from_search(parsed: dict) -> bool:
    """Top hit should be redirect-related and include enough excerpt to answer."""
    candidates = parsed.get("candidates") or []
    if not candidates:
        return False
    top = candidates[0]
    node_id = top.get("node_id", "")
    excerpt = top.get("source_excerpt", "")
    redirectish = "redirect" in node_id.lower() or "redirect" in excerpt.lower()
    has_neighbors = bool(top.get("neighbors", {}).get("callers") or top.get("neighbors", {}).get("callees"))
    return redirectish and bool(excerpt) and has_neighbors


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
            f"graph cache: {graph_path}",
        ],
    )

    # --- 1. search_codebase_intent (json default) ---
    _print_block(
        "Step 1 — search_codebase_intent (json)",
        [
            "Purpose: return ranked candidates; top 2 include source_excerpt + neighbor IDs.",
            f"Input: search_queries={SEARCH_QUERIES}",
        ],
    )

    search_json = search_codebase_intent(
        search_queries=SEARCH_QUERIES,
        active_project_root=TEST_PROJECT_ROOT,
        top_k=8,
        include_next_action=True,
    )
    try:
        parsed = json.loads(search_json)
        candidates = parsed.get("candidates") or []
        ok = len(candidates) > 0
        top = candidates[0] if candidates else {}
        print(f"  Returned: {len(candidates)} candidate(s)")
        if top:
            print(f"    top: {top.get('node_id')} | score={top.get('score')}")
            print(f"    has source_excerpt: {bool(top.get('source_excerpt'))}")
            print(f"    neighbors: callers={len(top.get('neighbors', {}).get('callers', []))}, callees={len(top.get('neighbors', {}).get('callees', []))}")
        results.append(_verdict(ok, "json search returned candidates" if ok else "no candidates"))
        top_has_excerpt = bool(top.get("source_excerpt"))
        results.append(_verdict(top_has_excerpt, "top hit includes source_excerpt" if top_has_excerpt else "top hit missing source_excerpt"))
        top_node = top.get("node_id")
    except Exception as e:
        results.append(_verdict(False, f"json parse failed: {e}"))
        top_node = None

    # --- 1b. search_codebase_intent (markdown mode) ---
    _print_block(
        "Step 1b — search_codebase_intent (markdown mode)",
        ["Purpose: markdown fallback still returns ranked hits."],
    )
    search_md = search_codebase_intent(
        search_queries=SEARCH_QUERIES,
        active_project_root=TEST_PROJECT_ROOT,
        format="markdown",
    )
    hits = _parse_search_hits(search_md)
    results.append(_verdict(len(hits) > 0, f"markdown search returned {len(hits)} hit(s)"))

    # --- 1c. redirect-handling regression (1 search should be enough) ---
    _print_block(
        "Step 1c — redirect handling regression",
        [
            "Purpose: one search answers 'where does redirect handling happen after HTTP response?'",
            f"Input: search_queries={REDIRECT_SEARCH_QUERIES}",
        ],
    )
    redirect_json = search_codebase_intent(
        search_queries=REDIRECT_SEARCH_QUERIES,
        active_project_root=TEST_PROJECT_ROOT,
        top_k=8,
    )
    try:
        redirect_parsed = json.loads(redirect_json)
        candidates = redirect_parsed.get("candidates") or []
        print(f"  Returned: {len(candidates)} candidate(s)")
        if candidates:
            for i, c in enumerate(candidates[:3], 1):
                print(f"    {i}. {c.get('node_id')} | score={c.get('score')}")
        answerable = _redirect_answerable_from_search(redirect_parsed)
        top_id = candidates[0].get("node_id", "") if candidates else ""
        top_is_resolve = REDIRECT_TOP_NODE in top_id or "resolve_redirect" in top_id
        results.append(
            _verdict(
                answerable and top_is_resolve,
                "redirect question answerable from single search"
                if answerable and top_is_resolve
                else f"answerable={answerable}, top={top_id!r}",
            )
        )
    except Exception as e:
        results.append(_verdict(False, f"redirect json parse failed: {e}"))

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
    results.append(
        _verdict(
            symbol_found and dependent_count > 0,
            f"symbol matched and {dependent_count} dependents listed"
            if symbol_found and dependent_count > 0
            else "symbol not found or no dependents listed",
        )
    )

    # --- 3. trace_callers (deprecated) ---
    _print_block(
        "Step 3 — trace_callers (deprecated)",
        ["Purpose: should return deprecation message pointing to fetch_node_source."],
    )
    trace_output = trace_callers(
        node_id=CALL_GRAPH_NODE,
        active_project_root=TEST_PROJECT_ROOT,
        format="json",
    )
    try:
        trace_parsed = json.loads(trace_output)
        deprecated = trace_parsed.get("deprecated") is True
        has_redirect = trace_parsed.get("use_instead", {}).get("tool") == "fetch_node_source"
        results.append(_verdict(deprecated and has_redirect, "trace_callers returns deprecation redirect"))
    except Exception as e:
        results.append(_verdict(False, f"trace json parse failed: {e}"))

    # --- 4. fetch_node_source ---
    _print_block(
        "Step 4 — fetch_node_source",
        [
            "Purpose: capped source excerpt + 1-hop neighbor IDs.",
            f"Input: node_ids=[{CALL_GRAPH_NODE!r}]",
        ],
    )

    fetch_json = fetch_node_source(
        node_ids=[CALL_GRAPH_NODE],
        active_project_root=TEST_PROJECT_ROOT,
    )
    try:
        fetch_parsed = json.loads(fetch_json)
        nodes = fetch_parsed.get("nodes") or []
        node = nodes[0] if nodes else {}
        has_excerpt = bool(node.get("source_excerpt"))
        neighbors = node.get("neighbors") or {}
        has_neighbors = bool(neighbors.get("callers") or neighbors.get("callees"))
        print(f"  Returned: {len(nodes)} node(s)")
        print(f"    has source_excerpt: {has_excerpt}")
        print(f"    callers: {len(neighbors.get('callers', []))}, callees: {len(neighbors.get('callees', []))}")
        results.append(
            _verdict(
                has_excerpt and has_neighbors,
                "fetch returns source_excerpt and neighbor IDs"
                if has_excerpt and has_neighbors
                else "missing excerpt or neighbors",
            )
        )
    except Exception as e:
        results.append(_verdict(False, f"fetch json parse failed: {e}"))

    # --- 5. graph CALLS edges (internal check) ---
    _print_block(
        "Step 5 — CALLS edge check (graph validation)",
        [
            "Purpose: confirm redirect flow is wired in the graph backing the MCP tools.",
            f"Node under test: {CALL_GRAPH_NODE}",
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
        print(f"  Callers ({len(callers)}): {callers[:3]}{'...' if len(callers) > 3 else ''}")
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
