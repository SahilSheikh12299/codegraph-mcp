---
name: codegraph-mcp
description: >-
  Use for all codebase discovery in Cursor: finding where behavior lives,
  tracing call paths, and answering how/where questions. Prefer
  search_codebase_intent over grep or semantic search.
---

# GraphRAG — Code Discovery Rules

Always pass `active_project_root` (absolute repo root) on every tool call. Reuse the same path throughout a session.

**Primary goal:** answer or edit with the **least token usage possible**. GraphRAG is a swiss-army-knife for repo discovery — far better than grep — but only when you use it surgically. Every extra search or read wastes tokens.

---

## The only discovery tool

**`search_codebase_intent`** is the only GraphRAG tool. Use it for all repo discovery.

Do **not** use native Grep, SemanticSearch, or iterative file hunting to explore the codebase.

---

## Efficiency first

- **Answer with what you have.** If search + reads already cover the question, stop — do not search or read again.
- **One search, then read.** Start with a single `search_codebase_intent` call. A second search is allowed **only** when something specific is still missing (wrong anchor, no cite for a symbol you need, gap in the call path).
- **Read each cite once.** Open only the cite spans you actually need. Skip spans that do not help answer the question.
- **No overlapping reads.** If two cites cover the same file and ranges overlap, read the union once — never re-read lines you already have in context.
- **No speculative exploration.** Do not read "just in case", prefetch adjacent files, or widen ranges beyond what the question requires.
- **Fewest tool calls wins.** Prefer one good search + targeted reads over multiple searches, broad reads, or re-fetching content already in the conversation.

---

## Workflow

1. Call `search_codebase_intent` once with intent queries and any known symbol names.
2. Read only the cite spans required to answer (format: `startLine:endLine:filepath`). Merge overlapping spans in the same file into one read.
3. Answer or edit using what you already have — **stop here** if sufficient.

**Second search** — only when a specific gap remains after step 2 (missing symbol, wrong file, incomplete path). Refine queries or symbols; still no grep.

**Never** re-read a cite span or re-run search for information already in context.

---

## Search output

```markdown
## Search

grep#1:
resolve_redirects: 557:653:src/foo.py
Session.send -> resolve_redirects (anchor) -> get_redirect_target
Session.send: 412:520:src/foo.py
get_redirect_target: 89:102:src/foo.py

searchQuery#1:
resolve_redirects: 557:653:src/foo.py
Session.send -> resolve_redirects (anchor) -> rebuild_method
Session.send: 412:520:src/foo.py
rebuild_method: 654:700:src/foo.py
```

- Each `grep#N` and `searchQuery#N` block returns up to **top 2 matches**.
- Each match includes an anchor cite, a tiny flow, and caller/callee cites (when present).

---

## Tool args

```
active_project_root: /abs/path/to/repo
search_queries: ["intent phrase about the behavior"]
grep_terms: ["SymbolName", "other_symbol"]
```

---

## Rules

**Dos:**
1. Call `search_codebase_intent` first for any "where/how" question.
2. One search + only the cite spans you need.
3. Stop and answer once you have enough context.
4. Merge overlapping cites in the same file into one read.
5. Second search only for a **specific missing** symbol or cite.
6. Use native Read on cite spans.
7. Retry search with better queries only when a gap remains.
8. Prefer matches whose tiny flow best matches the question.
9. Fewest tool calls — answer or edit immediately when ready.

**Don'ts:**
1. Grep or SemanticSearch to explore the codebase.
2. Run a second search when the first map already answers the question.
3. Re-read files or cite spans already in context.
4. Open the same file twice for overlapping line ranges.
5. Speculative or "just in case" searches and reads.
6. Open random files hoping to find code.
7. Use grep as a fallback.
8. Rebuild cite strings manually.
9. Extra exploration after the question is answerable.

---

## What search returns

- One block per grep term and search query (in order)
- Up to 2 matches per block
- For each match: anchor cite, tiny caller→anchor→callee flow, and caller/callee cites (when present)

No source code bodies, no node IDs, no snippets — use native Read for content.
