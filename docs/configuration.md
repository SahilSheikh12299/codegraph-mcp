# Configuration

## Required models

These are not optional for normal operation:

| Model | Source | Used for |
|-------|--------|----------|
| `qwen2.5:1.5b` | Ollama | Intent docstrings on functions without native docstrings |
| `BAAI/bge-large-en-v1.5` | HuggingFace | Semantic embeddings |
| `mixedbread-ai/mxbai-rerank-base-v2` | HuggingFace | Cross-encoder reranking |

`codegraph-mcp setup` verifies Ollama and prefetches HuggingFace models when online.

## Environment variables

### `CURSOR_GRAPHRAG_AUTO_DOCSTRINGS`

Default: `1` (enabled).

When enabled, indexing calls Ollama to generate one-sentence intent docstrings for functions that lack native docstrings. Requires Ollama and `qwen2.5:1.5b`.

Disable only for debugging:

```bash
export CURSOR_GRAPHRAG_AUTO_DOCSTRINGS=0
```

### HuggingFace / PyTorch

Models download to the HuggingFace cache (`~/.cache/huggingface/` by default). Set standard HF env vars if you use a mirror or offline cache.

### Ollama URL

The server uses `http://localhost:11434` by default (see `codegraph_mcp.ollama_client`). Custom Ollama hosts are not yet exposed as CLI flags; set up a local proxy or patch if needed.

## Cache locations

| Path | Contents |
|------|----------|
| `~/.cursor_graph_rag/graphs/graph_<hash>.json` | Per-workspace code graph |
| `~/.cursor_graph_rag/graphs/graph_<hash>.doc_cache.json` | Ollama docstring cache |
| `~/.cursor_graph_rag/graphs/graph_<hash>.lock` | File lock for concurrent MCP calls |
| `~/.cache/huggingface/` | Embedding and reranker weights |

Graph hash is derived from the absolute `active_project_root` path.

## Cursor files written by `setup`

| Path | Purpose |
|------|---------|
| `~/.cursor/mcp.json` | MCP server registration (merged, not overwritten) |
| `~/.cursor/skills/codegraph-mcp/SKILL.md` | Agent workflow rules for efficient discovery |

## MCP server entry

Default after `setup`:

```json
{
  "mcpServers": {
    "codegraph-mcp": {
      "command": "codegraph-mcp",
      "args": []
    }
  }
}
```

## `active_project_root`

Every `search_codebase_intent` call must include the absolute path to the repository root being searched. The Cursor agent passes this from workspace context (`Workspace Path`). The same path must be reused within a session for consistent cache hits.
