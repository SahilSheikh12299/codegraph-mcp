"""CLI entry point: run MCP server or one-time global setup."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
from importlib import resources
from pathlib import Path

from codegraph_mcp.ollama_client import _DEFAULT_BASE_URL, _DEFAULT_MODEL
from codegraph_mcp.server import run_server

OLLAMA_MODEL = _DEFAULT_MODEL
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
MCP_SERVER_KEY = "codegraph-mcp"
SKILL_DIR_NAME = "codegraph-mcp"


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _ollama_reachable(base_url: str = _DEFAULT_BASE_URL) -> bool:
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5.0):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _ollama_model_present(model: str, base_url: str = _DEFAULT_BASE_URL) -> bool:
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("models") or []
        names = {m.get("name", "").split(":")[0] for m in models if m.get("name")}
        target_base = model.split(":")[0]
        for m in models:
            name = m.get("name", "")
            if name == model or name.split(":")[0] == target_base:
                return True
        return target_base in names
    except Exception:
        return False


def _prefetch_hf_models() -> None:
    _eprint("[Setup] Prefetching HuggingFace models (embedding + reranker)...")
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        SentenceTransformer(EMBEDDING_MODEL)
        CrossEncoder(RERANKER_MODEL)
        _eprint("[Setup] HuggingFace models ready.")
    except Exception as exc:
        _eprint(f"[Setup] Warning: could not prefetch HF models: {exc}")
        _eprint("[Setup] Models will download on first search instead.")


def _resolve_command() -> str:
    cmd = shutil.which("codegraph-mcp")
    if cmd:
        return cmd
    raise SystemExit(
        "codegraph-mcp is not on PATH. Install with:\n"
        '  pip install "git+https://github.com/sahilsheikh/codegraph-mcp.git"\n'
        "Then activate the same venv Cursor uses, or use the full path in ~/.cursor/mcp.json."
    )


def _merge_mcp_config(command: str) -> Path:
    cursor_dir = Path.home() / ".cursor"
    mcp_path = cursor_dir / "mcp.json"
    cursor_dir.mkdir(parents=True, exist_ok=True)

    if mcp_path.exists():
        try:
            config = json.loads(mcp_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
    else:
        config = {}

    if "mcpServers" not in config or not isinstance(config["mcpServers"], dict):
        config["mcpServers"] = {}

    config["mcpServers"][MCP_SERVER_KEY] = {
        "command": command,
        "args": [],
    }

    mcp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return mcp_path


def _install_skill() -> Path:
    skill_root = Path.home() / ".cursor" / "skills" / SKILL_DIR_NAME
    skill_root.mkdir(parents=True, exist_ok=True)
    skill_path = skill_root / "SKILL.md"

    template = resources.files("codegraph_mcp.templates.skill").joinpath("SKILL.md")
    skill_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    return skill_path


def cmd_setup(_args: argparse.Namespace) -> None:
    _eprint("codegraph-mcp setup — one-time global install\n")

    if not _ollama_reachable():
        raise SystemExit(
            "Ollama is not reachable at http://localhost:11434\n"
            "Install Ollama (https://ollama.com) and start it, then re-run:\n"
            "  codegraph-mcp setup"
        )
    _eprint("[Setup] Ollama is running.")

    if not _ollama_model_present(OLLAMA_MODEL):
        raise SystemExit(
            f"Required Ollama model '{OLLAMA_MODEL}' is not installed.\n"
            f"Run:  ollama pull {OLLAMA_MODEL}\n"
            "Then re-run:  codegraph-mcp setup"
        )
    _eprint(f"[Setup] Ollama model '{OLLAMA_MODEL}' is available.")

    _prefetch_hf_models()

    command = _resolve_command()
    mcp_path = _merge_mcp_config(command)
    skill_path = _install_skill()

    print("\ncodegraph-mcp setup complete.\n")
    print(f"  MCP config:  {mcp_path}")
    print(f"  Agent skill: {skill_path}")
    print("\nNext steps:")
    print("  1. Restart Cursor (or reload MCP in Settings → MCP).")
    print("  2. Open any Python repo and ask Cursor where behavior lives.")
    print("  3. First search indexes the repo; later searches use the cache.")
    print(f"\nGraph cache: ~/.cursor_graph_rag/graphs/")


def cmd_setup_dry_run(_args: argparse.Namespace) -> None:
    """Validate environment without writing config files."""
    ok = _ollama_reachable() and _ollama_model_present(OLLAMA_MODEL)
    if not ok:
        raise SystemExit(1)
    print("Preflight OK (dry run).")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="codegraph-mcp",
        description="MCP server for intent-based Python code discovery in Cursor.",
    )
    sub = parser.add_subparsers(dest="command")

    setup_parser = sub.add_parser("setup", help="One-time global install (MCP config + agent skill)")
    setup_parser.set_defaults(func=cmd_setup)

    dry_parser = sub.add_parser(
        "setup-dry-run",
        help="Check Ollama/model prerequisites without writing files",
    )
    dry_parser.set_defaults(func=cmd_setup_dry_run)

    args = parser.parse_args()
    if args.command is None:
        run_server()
        return
    args.func(args)


if __name__ == "__main__":
    main()
