# Installation

## Prerequisites

1. **Python 3.10 or newer**
2. **Ollama** — [ollama.com](https://ollama.com)
3. **Disk and RAM** — models need ~3–5 GB disk and ~8 GB RAM for comfortable indexing

## Step 1: Ollama and qwen model

### macOS

```bash
brew install ollama
ollama serve   # if not already running as a service
ollama pull qwen2.5:1.5b
```

### Linux

Install Ollama from [ollama.com/download](https://ollama.com/download), then:

```bash
ollama pull qwen2.5:1.5b
```

Verify:

```bash
curl http://localhost:11434/api/tags
ollama list | grep qwen2.5
```

## Step 2: Install codegraph-mcp

Use a dedicated virtual environment (recommended):

```bash
python -m venv ~/.venvs/codegraph-mcp
source ~/.venvs/codegraph-mcp/bin/activate   # Windows: Scripts\activate
pip install --upgrade pip
pip install "git+https://github.com/SahilSheikh12299/codegraph-mcp.git"
```

Pinned to a release tag:

```bash
pip install "git+https://github.com/SahilSheikh12299/codegraph-mcp.git@v0.1.0"
```

Local development install:

```bash
git clone https://github.com/SahilSheikh12299/codegraph-mcp.git
cd codegraph-mcp
pip install -e ".[dev]"
```

## Step 3: Global setup (once)

```bash
codegraph-mcp setup
```

This checks Ollama, prefetches HuggingFace models, writes `~/.cursor/mcp.json`, and installs the Cursor agent skill.

Restart Cursor or reload MCP servers in **Settings → MCP**.

## Step 4: Use in any project

1. Open a Python repository in Cursor
2. Ask a discovery question (*"Where is X implemented?"*, *"How does Y work?"*)
3. The agent calls `search_codebase_intent` with your workspace path
4. First call indexes the repo (may take several minutes on large codebases)
5. Subsequent calls are incremental and faster

## Troubleshooting

### `codegraph-mcp is not on PATH`

Install into a venv whose `bin` directory is on your shell PATH, or point `~/.cursor/mcp.json` at the full path:

```json
{
  "mcpServers": {
    "codegraph-mcp": {
      "command": "/Users/you/.venvs/codegraph-mcp/bin/codegraph-mcp",
      "args": []
    }
  }
}
```

Cursor must inherit that PATH. Starting Cursor from a terminal where the venv is activated often works on macOS.

### Ollama not reachable

Ensure `ollama serve` is running and `curl http://localhost:11434/api/tags` succeeds.

### Model `qwen2.5:1.5b` missing

```bash
ollama pull qwen2.5:1.5b
codegraph-mcp setup
```

### MCP tool not visible in Cursor

- Restart Cursor completely
- Check **Settings → MCP** for `codegraph-mcp` with no errors
- Confirm `~/.cursor/mcp.json` contains the `codegraph-mcp` entry

### First search is very slow

Expected: initial index runs Ollama docstrings, builds the call graph, and downloads embedding/reranker models if `setup` could not prefetch them.

### Preflight only (no file writes)

```bash
codegraph-mcp setup-dry-run
```

Checks Ollama and model availability without modifying config files.
