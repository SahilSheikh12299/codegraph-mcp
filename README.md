# codegraph-mcp

Intent search over your Python repo's call graph, inside Cursor. One MCP tool returns cite spans and caller→anchor→callee chains so the agent searches once, reads surgically, and burns fewer tokens than grep-then-read loops.

## Requirements

All of the following are **required**:

| Component | Purpose |
|-----------|---------|
| **Python 3.10+** | Runtime |
| **Ollama** | Generates intent docstrings during indexing |
| **`qwen2.5:1.5b`** | Ollama model for docstrings |
| **`BAAI/bge-large-en-v1.5`** | Embedding model (HuggingFace, downloaded on setup/first run) |
| **`mixedbread-ai/mxbai-rerank-base-v2`** | Cross-encoder reranker (HuggingFace) |

**Hardware:** ~8 GB RAM recommended; ~3–5 GB disk for models after first run.

**Scope:** Python repositories only (for now).

## Quick start

```bash
# 1. Ollama + required model
brew install ollama          # or https://ollama.com
ollama pull qwen2.5:1.5b

# 2. Install codegraph-mcp (once)
python -m venv .venv
source .venv/bin/activate
pip install "git+https://github.com/SahilSheikh12299/codegraph-mcp.git"

# 3. Global setup (once)
codegraph-mcp setup
```

Then **restart Cursor** (or reload MCP in Settings → MCP).

Open **any** Python repo and ask Cursor where behavior lives — e.g. *"Where is authentication handled?"* The first search indexes that repo; later searches use the cache at `~/.cursor_graph_rag/graphs/`.

No per-project configuration needed.

## Pinned install

```bash
pip install "git+https://github.com/SahilSheikh12299/codegraph-mcp.git@v0.1.0"
```

## Usage

The MCP server exposes one tool:

### `search_codebase_intent`

```python
search_codebase_intent(
    search_queries=["how redirects are resolved after HTTP response"],
    active_project_root="/absolute/path/to/repo",
    grep_terms=["resolve_redirects"],  # optional symbol anchors
)
```

Returns markdown with up to **2 matches per grep term and per search query**: anchor cite, a tiny call flow, and caller/callee cites. The agent reads those line ranges with native Read — no full-file dumps.

`active_project_root` is the absolute workspace root (Cursor provides this in context).

## What `setup` does

`codegraph-mcp setup` runs once globally:

1. Verifies Ollama is running and `qwen2.5:1.5b` is installed
2. Prefetches HuggingFace embedding + reranker models (warns if offline)
3. Merges `codegraph-mcp` into `~/.cursor/mcp.json`
4. Installs agent skill at `~/.cursor/skills/codegraph-mcp/SKILL.md`

## Documentation

- [Installation guide](docs/installation.md)
- [Configuration](docs/configuration.md)
- [Architecture](docs/architecture.md)

## Development

```bash
git clone https://github.com/SahilSheikh12299/codegraph-mcp.git
cd codegraph-mcp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
