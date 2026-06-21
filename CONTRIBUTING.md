# See docs/installation.md for full setup.

## Verify after clone

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
pip install "git+file://$(pwd)"
codegraph-mcp setup-dry-run   # requires Ollama + qwen2.5:1.5b
```

## Release

Tag `v0.1.0` on GitHub for pinned installs:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Users install with:

```bash
pip install "git+https://github.com/sahilsheikh/codegraph-mcp.git@v0.1.0"
```
