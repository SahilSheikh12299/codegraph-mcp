import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

class ImportTracker:
    """A plug-and-play import resolver that maps absolute and relative imports

    by matching import tokens against the repository's known file layout.
    """

    def __init__(self, repo_root: str | Path, all_python_files: List[Path]):
        self.repo_root = Path(repo_root).resolve()
        # Map absolute file paths to their full dot-notation strings relative to the repo root
        self.file_dot_paths: Dict[Path, str] = {}
        
        for file_path in all_python_files:
            abs_path = file_path.resolve()
            try:
                rel_path = abs_path.relative_to(self.repo_root)
                # Convert path parts to dot notation, stripping the file extension
                if rel_path.name == "__init__.py":
                    # Package boundaries drop the __init__ token (e.g., src/requests/__init__.py -> src.requests)
                    dot_path = ".".join(rel_path.parent.parts)
                else:
                    dot_path = ".".join(rel_path.with_suffix("").parts)
                
                self.file_dot_paths[abs_path] = dot_path
            except ValueError:
                continue

    def get_dependencies(self, file_path: str | Path) -> Dict[str, Any]:
        """Parses a file's AST and resolves imports using the pre-computed file layout map."""
        file_path = Path(file_path).resolve()
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except (UnicodeDecodeError, SyntaxError) as e:
            return {"internal_paths": [], "external_modules": [], "error": str(e)}

        internal_dependencies: Set[Path] = set()
        external_dependencies: Set[str] = set()

        for node in ast.walk(tree):
            # Scenario A: Standard imports (e.g., 'import database.connection')
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = self._resolve_absolute(alias.name)
                    if resolved:
                        internal_dependencies.add(resolved)
                    else:
                        external_dependencies.add(alias.name.split(".")[0])

            # Scenario B: From-Imports (e.g., 'from ..models import user', 'from security import auth')
            elif isinstance(node, ast.ImportFrom):
                # 1. Explicit Relative Imports (level > 0)
                if node.level > 0:
                    resolved_rel = self._resolve_relative(file_path, node.module, node.level)
                    if resolved_rel:
                        internal_dependencies.add(resolved_rel)
                    
                    # Also check if the specific symbols being imported are actually separate sub-module files
                    for alias in node.names:
                        combined_rel = f"{node.module}.{alias.name}" if node.module else alias.name
                        resolved_sub = self._resolve_relative(file_path, combined_rel, node.level)
                        if resolved_sub:
                            internal_dependencies.add(resolved_sub)
                    continue

                # 2. Absolute From-Imports (level == 0)
                if node.module:
                    has_resolved = False
                    # Check base module (e.g., 'from billing import processors')
                    resolved_base = self._resolve_absolute(node.module)
                    if resolved_base:
                        internal_dependencies.add(resolved_base)
                        has_resolved = True
                    
                    # Check explicit sub-module symbols (e.g., checking if 'processors' inside 'from billing import processors' is a file)
                    for alias in node.names:
                        combined_abs = f"{node.module}.{alias.name}"
                        resolved_alias = self._resolve_absolute(combined_abs)
                        if resolved_alias:
                            internal_dependencies.add(resolved_alias)
                            has_resolved = True
                    
                    if not has_resolved:
                        external_dependencies.add(node.module.split(".")[0])

        return {
            "file_path": str(file_path),
            "internal_paths": [str(p.relative_to(self.repo_root)) for p in internal_dependencies],
            "external_modules": sorted(list(external_dependencies))
        }

    def _resolve_absolute(self, import_str: str) -> Optional[Path]:
        """Matches an absolute import string against the end of any known repository file's dot-path."""
        for abs_file_path, full_dot_path in self.file_dot_paths.items():
            # Match if the import string maps perfectly, or forms a valid trailing subpath sequence
            if full_dot_path == import_str or full_dot_path.endswith("." + import_str):
                return abs_file_path
        return None

    def _resolve_relative(self, source_file: Path, module_str: Optional[str], level: int) -> Optional[Path]:
        """Handles explicit dot-notation directory jumps deterministically."""
        base_dir = source_file.parent
        for _ in range(level - 1):
            if base_dir.parent != base_dir:
                base_dir = base_dir.parent
        
        module_parts = module_str.split(".") if module_str else []
        target_path = base_dir.joinpath(*module_parts)

        file_candidate = target_path.with_suffix(".py")
        if file_candidate.exists() and file_candidate.is_file():
            return file_candidate.resolve()

        dir_candidate = target_path / "__init__.py"
        if dir_candidate.exists() and dir_candidate.is_file():
            return dir_candidate.resolve()
            
        return None
        