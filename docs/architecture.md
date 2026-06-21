# Architecture

## Overview

codegraph-mcp is a **stdio MCP server** that indexes Python repositories into a **directed call graph** with embeddings, then answers intent queries with minimal cite-based navigation output.

```mermaid
flowchart LR
    subgraph index [Indexing per workspace]
        scan[WorkspaceScanner]
        ast[ASTParser]
        graph[NetworkX graph]
        ollama[Ollama qwen2.5:1.5b]
        embed[BGE embeddings]
        scan --> ast --> graph
        graph --> ollama
        graph --> embed
    end
    subgraph search [Search per query]
        intent[search_codebase_intent]
        retrieve[AdvancedRetrievalEngine]
        rerank[mxbai reranker]
        md[Markdown cites + call chains]
        intent --> retrieve --> rerank --> md
    end
    index --> search
```

## Pipeline phases

Each tool call runs two sync phases under a file lock:

### Phase 1: Docstrings (Ollama only)

`sync_docstrings` in `server.py`:

- Scans new or modified Python files
- Generates intent docstrings via Ollama for functions without native docs
- Persists doc cache to `graph_<hash>.doc_cache.json`
- Unloads Ollama from memory before Phase 2

### Phase 2: Embeddings and call graph

`sync_embeddings_and_graph`:

- Loads `BAAI/bge-large-en-v1.5` and `mixedbread-ai/mxbai-rerank-base-v2`
- Embeds chunk text and symbol names
- Wires `CALLS` edges between functions using import-aware resolution
- Recomputes call centrality for ranking
- Saves graph JSON on change

### Search

`AdvancedRetrievalEngine.compile_bucketed_context_package`:

- Semantic search over embeddings
- Keyword/grep scoring for symbol terms
- Cross-encoder reranking with AST semantic skeletons
- Returns top-2 matches per grep term and per intent query

Output is formatted as markdown cites (`startLine:endLine:filepath`) plus one-hop caller/callee chains — no source bodies in the tool response.

## Graph storage

- First access to a workspace builds the graph via `GraphSerializer.load_from_json` (cold build from AST + `RepositoryGraphCompiler`)
- Subsequent loads read `~/.cursor_graph_rag/graphs/graph_<md5(workspace)>.json`
- Incremental sync compares file mtimes in `G.graph["indexed_timestamps"]`

## Why `active_project_root` per call

The MCP server is a global process serving many workspaces. The project root selects which graph cache to load and which files to scan. Cursor knows the open folder; the agent forwards it as `active_project_root`.

## Module map

| Module | Role |
|--------|------|
| `server.py` | MCP tool, sync orchestration, graph cache paths |
| `graph_io.py` | Graph JSON load/save, cold-build bootstrap |
| `file_parsing/` | AST parse, workspace scan, import tracking |
| `build_graph/` | Graph compilation, call wiring, chunking |
| `embedding_pipeline/` | Sentence-transformer lifecycle |
| `advanced_engine.py` | Retrieval, reranking, call-chain ranking |
| `ollama_client.py` | Ollama generate + unload |
| `markdown_format.py` | Tool output formatting |

## Limitations

- Python source files only
- First index on a large repo can take minutes
- Global MCP uses one Python environment — install models once in that venv
- Cursor personal skills guide agent behavior but are not hard-enforced
