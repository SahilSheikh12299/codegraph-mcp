"""Cursor-native markdown formatters for GraphRAG MCP search output."""

from __future__ import annotations

from typing import Any


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


def format_error(message: str, example: str = "") -> str:
    lines = [f"## Error\n\n{message}"]
    if example:
        lines.append(f"\nExample:\n{example}")
    return "\n".join(lines)


def _display_name(entry: dict[str, Any]) -> str:
    return entry.get("name") or short_name(entry.get("node_id", ""))


def _role_label(name: str, role: str) -> str:
    if role == "anchor":
        return f"**{name}** (anchor)"
    if role == "caller":
        return f"{name} (caller)"
    if role == "callee":
        return f"{name} (callee)"
    if role == "downstream":
        return f"{name} (downstream)"
    return name


def format_micro_search_markdown(
    *,
    queries: list[str],
    grep_terms: list[str],
    anchor: dict[str, Any] | None,
    path: list[dict[str, Any]],
    read_next: list[str],
) -> str:
    """Micro search: 1 anchor + ordered path + up to 3 read-next cite spans."""
    if not anchor:
        q = ", ".join(queries) if queries else "(none)"
        terms = ", ".join(grep_terms) if grep_terms else ""
        hint = f" and symbols: {terms}" if terms else ""
        return (
            f"## Search\n\n"
            f"No graph matches for: {q}{hint}\n\n"
            f"Refine `search_queries` or `grep_terms` and call search again."
        )

    lines = ["## Search", ""]
    if queries:
        lines.append(f"Queries: {', '.join(queries)}")
    if grep_terms:
        lines.append(f"Symbols: {', '.join(grep_terms)}")
    lines.append("")

    name = _display_name(anchor)
    ref = cite_ref(anchor.get("file_path"), anchor.get("start_line"), anchor.get("end_line"))
    lines.append("### Anchor")
    lines.append(f"- {name} — `{ref}`")
    sig = (anchor.get("signature") or "").strip()
    if sig:
        lines.append(f"- signature: {sig}")
    lines.append("")

    lines.append("### Path")
    if path:
        for idx, step in enumerate(path, 1):
            step_name = _display_name(step)
            step_ref = cite_ref(
                step.get("file_path"), step.get("start_line"), step.get("end_line")
            )
            label = _role_label(step_name, step.get("role", ""))
            lines.append(f"{idx}. {label} → `{step_ref}`")
    else:
        lines.append(f"1. {_role_label(name, 'anchor')} → `{ref}`")
    lines.append("")

    lines.append("### Read next")
    if read_next:
        for idx, span in enumerate(read_next, 1):
            lines.append(f"{idx}. `{span}`")
    else:
        lines.append(f"1. `{ref}`")

    return "\n".join(lines).rstrip()


def _chain_pick(chain: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    caller = next((s for s in chain if s.get("role") == "caller"), None)
    anchor = next((s for s in chain if s.get("role") == "anchor"), None)
    callee = next((s for s in chain if s.get("role") == "callee"), None)
    return caller, anchor, callee


def _ref_for(entry: dict[str, Any] | None) -> str:
    if not entry:
        return ""
    return cite_ref(entry.get("file_path"), entry.get("start_line"), entry.get("end_line"))


def _format_match_lines(
    match: dict[str, Any],
    *,
    seen_refs: set[str],
    seen_flows: set[str],
) -> list[str]:
    """Format one match, skipping cite refs and flows already emitted earlier."""
    anchor = match.get("anchor") or {}
    chain = match.get("chain") or []
    caller_step, anchor_step, callee_step = _chain_pick(chain)

    anchor_name = _display_name(anchor_step or anchor)
    anchor_ref = _ref_for(anchor_step) or cite_ref(
        anchor.get("file_path"), anchor.get("start_line"), anchor.get("end_line")
    )

    lines: list[str] = []
    emitted_anchor_cite = False
    if anchor_ref and anchor_ref not in seen_refs:
        lines.append(f"{anchor_name}: {anchor_ref}")
        seen_refs.add(anchor_ref)
        emitted_anchor_cite = True

    caller_name = _display_name(caller_step) if caller_step else ""
    callee_name = _display_name(callee_step) if callee_step else ""
    if caller_name and callee_name:
        flow = f"{caller_name} -> {anchor_name} (anchor) -> {callee_name}"
    elif caller_name:
        flow = f"{caller_name} -> {anchor_name} (anchor)"
    elif callee_name:
        flow = f"{anchor_name} (anchor) -> {callee_name}"
    else:
        flow = f"{anchor_name} (anchor)"

    has_chain_context = bool(caller_name or callee_name)
    if has_chain_context:
        if flow not in seen_flows:
            lines.append(flow)
            seen_flows.add(flow)
    elif emitted_anchor_cite and flow not in seen_flows:
        lines.append(flow)
        seen_flows.add(flow)

    if caller_step:
        caller_ref = _ref_for(caller_step)
        if caller_ref and caller_ref not in seen_refs:
            lines.append(f"{caller_name}: {caller_ref}")
            seen_refs.add(caller_ref)
    if callee_step:
        callee_ref = _ref_for(callee_step)
        if callee_ref and callee_ref not in seen_refs:
            lines.append(f"{callee_name}: {callee_ref}")
            seen_refs.add(callee_ref)

    return lines


def format_multi_term_paths_markdown(
    *,
    grep_results: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
) -> str:
    """Minimal, term-scoped output: top matches + tiny caller→anchor→callee flow.

    Expected output blocks:
    - grep#N: ... (top 2 matches)
    - searchQuery#N: ... (top 2 matches)
    """
    lines: list[str] = ["## Search", ""]
    seen_refs: set[str] = set()
    seen_flows: set[str] = set()

    def _emit_block(label: str, matches: list[dict[str, Any]]) -> None:
        block_lines: list[str] = []
        for match in matches:
            match_lines = _format_match_lines(
                match, seen_refs=seen_refs, seen_flows=seen_flows
            )
            if match_lines:
                block_lines.extend(match_lines)
                block_lines.append("")

        if not block_lines and not matches:
            lines.append(f"{label}:")
            lines.append("(no matches)")
            lines.append("")
            return
        if not block_lines:
            return

        lines.append(f"{label}:")
        lines.extend(block_lines)

    for idx, bucket in enumerate(grep_results or [], 1):
        _emit_block(f"grep#{idx}", (bucket.get("matches") or [])[:2])

    for idx, bucket in enumerate(search_results or [], 1):
        _emit_block(f"searchQuery#{idx}", (bucket.get("matches") or [])[:2])

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Legacy formatters (unused — kept for reference)
# ---------------------------------------------------------------------------
# def cursor_citation(...): ...
# def format_search_markdown(...): ...
# def format_grep_markdown(...): ...
# def format_snippets_markdown(...): ...
# def format_source_markdown(...): ...
# def format_metadata_markdown(...): ...
# def format_touch_set_markdown(...): ...
# def format_repo_refs_markdown(...): ...
# def format_blast_radius_markdown(...): ...
