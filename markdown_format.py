"""Cursor-native markdown formatters for GraphRAG MCP tool output."""

from __future__ import annotations

from typing import Any, Callable


def short_name(node_id: str) -> str:
    if "::" not in node_id:
        return node_id
    _, sym = node_id.rsplit("::", 1)
    if "." in sym:
        return sym.rsplit(".", 1)[-1]
    return sym


def cite_ref(file_path: str | None, start: int | None, end: int | None) -> str:
    if not file_path or start is None or end is None:
        return ""
    return f"{start}:{end}:{file_path}"


def cursor_citation(
    file_path: str | None,
    start: int | None,
    end: int | None,
    code: str,
) -> str:
    ref = cite_ref(file_path, start, end)
    if not ref or not code:
        return ""
    return f"```{ref}\n{code.rstrip()}\n```"


def format_error(message: str, example: str = "") -> str:
    lines = [f"## Error\n\n{message}"]
    if example:
        lines.append(f"\nExample:\n{example}")
    return "\n".join(lines)


def _display_name(entry: dict[str, Any]) -> str:
    return entry.get("name") or short_name(entry.get("node_id", ""))


def format_call_chain(
    chain: list[dict[str, Any]],
    *,
    snippet_loader: Callable[[str], str | None] | None = None,
) -> str:
    if not chain:
        return ""
    lines = ["#### Call chain"]
    step = 0
    for entry in chain:
        step += 1
        name = _display_name(entry)
        ref = cite_ref(entry.get("file_path"), entry.get("start_line"), entry.get("end_line"))
        role = entry.get("role", "")
        if role == "anchor":
            label = f"**{name}** (anchor)"
        else:
            label = name
        lines.append(f"{step}. {label} → `{ref}`")

    if snippet_loader:
        seen: set[str] = set()
        for entry in chain:
            nid = entry.get("node_id")
            if not nid or nid in seen:
                continue
            seen.add(nid)
            snippet = snippet_loader(nid)
            if snippet:
                lines.append("")
                lines.append(snippet)

    return "\n".join(lines)


def format_search_markdown(
    result: dict[str, Any],
    *,
    call_chains: dict[str, list[dict[str, Any]]],
    snippet_loader: Callable[[str], str | None] | None = None,
) -> str:
    queries = result.get("rewritten_queries") or []
    grep_terms = result.get("grep_terms") or []
    semantic_nodes = (result.get("semantic_bucket") or {}).get("nodes") or []
    grep_buckets = result.get("grep_buckets") or []

    if not semantic_nodes and not any(b.get("nodes") for b in grep_buckets):
        q = ", ".join(queries) if queries else "(none)"
        return f"## Search\n\nNo graph matches for: {q}"

    lines = ["## Search", ""]
    if queries:
        lines.append(f"Queries: {', '.join(queries)}")
    if grep_terms:
        lines.append(f"Grep: {', '.join(grep_terms)}")
    lines.append("")

    for idx, node in enumerate(semantic_nodes, 1):
        name = _display_name(node)
        score = node.get("score")
        score_s = f" ({score})" if score is not None else ""
        ref = cite_ref(node.get("file_path"), node.get("start_line"), node.get("end_line"))
        lines.append(f"### Hit {idx} — {name}{score_s}")
        lines.append(f"node_id: {node.get('node_id')}")
        if ref:
            lines.append(f"cite: `{ref}`")
        sig = (node.get("signature") or "").strip()
        if sig:
            lines.append(f"signature: {sig}")

        if snippet_loader and node.get("node_id"):
            snippet = snippet_loader(node["node_id"])
            if snippet:
                lines.append("")
                lines.append(snippet)

        nid = node.get("node_id")
        if nid and nid in call_chains:
            chain_md = format_call_chain(
                call_chains[nid],
                snippet_loader=snippet_loader,
            )
            if chain_md:
                lines.append("")
                lines.append(chain_md)
        lines.append("")

    for bucket in grep_buckets:
        term = bucket.get("term", "")
        mode = bucket.get("match_mode", "none")
        nodes = bucket.get("nodes") or []
        lines.append(f"### Grep: {term} ({mode})")
        if not nodes:
            lines.append("- (no matches)")
        for node in nodes:
            name = _display_name(node)
            ref = cite_ref(node.get("file_path"), node.get("start_line"), node.get("end_line"))
            sig = (node.get("signature") or "").strip()
            sig_part = f" — {sig}" if sig else ""
            lines.append(f"- {name} — `{ref}`{sig_part}")
            lines.append(f"  node_id: {node.get('node_id')}")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_grep_markdown(buckets: list[dict[str, Any]], terms: list[str]) -> str:
    lines = ["## Grep Graph Results", ""]
    if terms:
        lines.append(f"Terms: {', '.join(terms)}")
        lines.append("")
    for bucket in buckets:
        term = bucket.get("term", "")
        mode = bucket.get("match_mode", "none")
        nodes = bucket.get("nodes") or []
        lines.append(f"### {term} ({mode})")
        if not nodes:
            lines.append("- (no matches)")
        for node in nodes:
            name = _display_name(node)
            ref = cite_ref(node.get("file_path"), node.get("start_line"), node.get("end_line"))
            sig = (node.get("signature") or "").strip()
            sig_part = f" — {sig}" if sig else ""
            lines.append(f"- {name} — `{ref}`{sig_part}")
            lines.append(f"  node_id: {node.get('node_id')}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_snippets_markdown(nodes: list[dict[str, Any]]) -> str:
    if not nodes:
        return "## Snippets\n\n(no nodes)"
    parts = ["## Snippets"]
    for node in nodes:
        if node.get("error"):
            parts.append(f"\n### {node.get('node_id', '?')}\n{node['error']}")
            continue
        name = _display_name(node)
        ref = cite_ref(node.get("file_path"), node.get("start_line"), node.get("end_line"))
        trunc = " (truncated)" if node.get("truncated") else ""
        parts.append(f"\n### {name}{trunc}")
        parts.append(f"node_id: {node.get('node_id')}")
        block = cursor_citation(
            node.get("file_path"),
            node.get("start_line"),
            node.get("end_line"),
            node.get("snippet") or "",
        )
        if block:
            parts.append(block)
    return "\n".join(parts)


def format_source_markdown(nodes: list[dict[str, Any]]) -> str:
    if not nodes:
        return "## Source\n\n(no nodes)"
    parts = ["## Source"]
    for node in nodes:
        if node.get("error"):
            parts.append(f"\n### {node.get('node_id', '?')}\n{node['error']}")
            continue
        name = _display_name(node)
        parts.append(f"\n### {name}")
        parts.append(f"node_id: {node.get('node_id')}")
        block = cursor_citation(
            node.get("file_path"),
            node.get("start_line"),
            node.get("end_line"),
            node.get("source") or "",
        )
        if block:
            parts.append(block)
    return "\n".join(parts)


def _neighbor_line(G: Any, nid: str) -> str:
    if G is not None and G.has_node(nid):
        data = G.nodes[nid]
        start, end = None, None
        span = data.get("line_span")
        if span and len(span) >= 2:
            start, end = int(span[0]), int(span[1])
        fp = data.get("file_path")
        name = data.get("name") or short_name(nid)
        ref = cite_ref(fp, start, end)
        return f"{name} — `{ref}`"
    return f"{short_name(nid)} — `{nid}`"


def format_metadata_markdown(nodes: list[dict[str, Any]], G: Any = None) -> str:
    if not nodes:
        return "## Node Metadata\n\n(no nodes)"
    lines = ["## Node Metadata", ""]
    for node in nodes:
        if node.get("error"):
            lines.append(f"- {node.get('node_id')}: {node['error']}")
            continue
        name = _display_name(node)
        ref = cite_ref(node.get("file_path"), node.get("start_line"), node.get("end_line"))
        lines.append(f"### {name}")
        lines.append(f"node_id: {node.get('node_id')}")
        lines.append(f"cite: `{ref}`")
        sig = (node.get("signature") or "").strip()
        if sig:
            lines.append(f"signature: {sig}")
        nb = node.get("neighbors") or {}
        callers = (nb.get("callers") or [])[:5]
        callees = (nb.get("callees") or [])[:5]
        if callers:
            lines.append("callers:")
            for i, cid in enumerate(callers, 1):
                lines.append(f"  {i}. {_neighbor_line(G, cid)}")
        if callees:
            lines.append("callees:")
            for i, cid in enumerate(callees, 1):
                lines.append(f"  {i}. {_neighbor_line(G, cid)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_touch_set_markdown(
    feature_description: str,
    required: list[dict[str, Any]],
    related: list[dict[str, Any]],
    non_graph_refs: list[dict[str, Any]],
) -> str:
    lines = [f"## Touch set: {feature_description}", "", "### Required"]
    if not required:
        lines.append("- (none)")
    for item in required:
        fp = item.get("file_path", "")
        ranges = item.get("line_ranges") or []
        if ranges:
            range_strs = [
                cite_ref(fp, r[0], r[1]) if len(r) >= 2 else fp for r in ranges
            ]
            lines.append(f"- `{fp}` — {', '.join(f'`{s}`' for s in range_strs if s)}")
        else:
            lines.append(f"- `{fp}`")
    lines.extend(["", "### Related"])
    if not related:
        lines.append("- (none)")
    for item in related:
        fp = item.get("file_path", "")
        lines.append(f"- `{fp}`")
    lines.extend(["", "### Non-graph refs"])
    if not non_graph_refs:
        lines.append("- (none)")
    for hit in non_graph_refs[:20]:
        lines.append(f"- `{hit.get('file_path')}`:{hit.get('line')} — {hit.get('text', '')}")
    return "\n".join(lines)


def format_repo_refs_markdown(hits: list[dict[str, Any]], terms: list[str] | None = None) -> str:
    lines = ["## Repo References", ""]
    if terms:
        lines.append(f"Terms: {', '.join(terms)}")
        lines.append("")
    if not hits:
        lines.append("(no matches)")
        return "\n".join(lines)
    for hit in hits:
        lines.append(f"- `{hit.get('file_path')}`:{hit.get('line')} — {hit.get('text', '')}")
    return "\n".join(lines)


def format_blast_radius_markdown(target_symbol: str, items: list[dict[str, Any]], total: int, max_items: int) -> str:
    lines = [
        f"## Upstream dependents: `{target_symbol}`",
        f"",
        f"Modifying this symbol may affect **{total}** graph element(s).",
        "",
    ]
    shown = items[:max_items]
    for item in shown:
        fp = item.get("file_path") or ""
        nid = item.get("node_id") or ""
        typ = item.get("type") or ""
        lines.append(f"- [{typ}] {short_name(nid)} — `{fp}`")
    more = total - len(shown)
    if more > 0:
        lines.append(f"\n... and {more} more.")
    return "\n".join(lines)
