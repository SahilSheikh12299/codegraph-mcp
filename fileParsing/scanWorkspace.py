import os
from pathlib import Path
from typing import List, Set


class WorkspaceScanner:
    """Recursively crawls a workspace directory to find Python files,

    safely handling standard ignore patterns and local .gitignore files.
    """

    def __init__(self, root_path: str | Path):
        self.root_path = Path(root_path).resolve()

        # Industry standard directories to skip to prevent performance lag
        self.default_ignores: Set[str] = {
            ".git",
            ".venv",
            "venv",
            "env",
            "__pycache__",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            "build",
            "dist",
            ".*"
        }
        self.ignore_patterns = self._load_gitignore_patterns()

    def _load_gitignore_patterns(self) -> Set[str]:
        """Reads local .gitignore if it exists and merges it with default ignores."""
        patterns = set(self.default_ignores)
        gitignore_path = self.root_path / ".gitignore"

        if gitignore_path.exists():
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith("#"):
                            # Normalize path separators and strip trailing slashes
                            clean_pattern = line.rstrip("/")
                            patterns.add(clean_pattern)
            except Exception as e:
                print(f"Warning: Could not read .gitignore: {e}")

        return patterns

    def _is_ignored(self, path: Path) -> bool:
        """Determines if a given path matches any entry in the ignore patterns."""
        try:
            # Get path segments relative to the root workspace directory
            relative_path = path.relative_to(self.root_path)
            parts = relative_path.parts

            # Check if any individual folder in the path tree is in our ignore set
            for part in parts:
                if part in self.ignore_patterns:
                    return True
        except ValueError:
            # Path is outside the root directory, ignore it for safety
            return True
        return False

    def scan(self) -> List[Path]:
        """Executes the recursive crawl, pruning ignored branches instantly."""
        python_files: List[Path] = []

        def _recursive_walk(current_dir: Path):
            try:
                for item in current_dir.iterdir():
                    if self._is_ignored(item):
                        continue

                    if item.is_dir():
                        # Deep-dive into the unignored subdirectory
                        _recursive_walk(item)
                    elif item.is_file() and item.suffix == ".py":
                        python_files.append(item)

            except PermissionError:
                # Safely bypass OS-protected directories
                pass

        _recursive_walk(self.root_path)
        return python_files