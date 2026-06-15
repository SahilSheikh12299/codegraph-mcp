import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

class ASTParser:
    """Parses a single Python file using the native AST module to extract

    high-level code architecture entities.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path).resolve()

    def parse(self) -> Dict[str, Any]:
        """Reads the source file, compiles the AST, and extracts structural metadata."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            
            # Compile the raw text into a syntax tree
            tree = ast.parse(source_code, filename=str(self.file_path))
        except (UnicodeDecodeError, SyntaxError) as e:
            return {
                "file_path": str(self.file_path),
                "error": f"Failed to parse: {str(e)}",
                "docstring": "",
                "classes": [],
                "functions": [],
                "globals": [],
                "top_level_calls": []
            }

        # EXTRACT FILE/MODULE LEVEL DOCSTRING
        file_docstring = ast.get_docstring(tree) or ""

        classes: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []
        globals_list: List[Dict[str, Any]] = []
        top_level_calls: Set[str] = set()

        # Iterate through the top-level nodes of the file
        for node in tree.body:
            
            # Extract root-level script execution calls
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.Call):
                        if isinstance(sub_node.func, ast.Name):
                            top_level_calls.add(sub_node.func.id)
                        elif isinstance(sub_node.func, ast.Attribute):
                            top_level_calls.add(sub_node.func.attr)

            # 1. Extract Top-Level Classes
            if isinstance(node, ast.ClassDef):
                class_meta = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",  # <--- FIXED
                    "line_span": (node.lineno, node.end_lineno),
                    "bases": [self._get_source_segment(base) for base in node.bases],
                    "methods": self._extract_methods(node)  # Make sure your internal _extract_methods also pulls sub_node docstrings!
                }
                classes.append(class_meta)

            # 2. Extract Top-Level Functions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_meta = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",  # <--- FIXED
                    "line_span": (node.lineno, node.end_lineno),
                    "arguments": [arg.arg for arg in node.args.args],
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "decorators": [self._get_source_segment(dec) for dec in node.decorator_list],
                    "calls": self._extract_calls_from_body(node.body)
                }
                functions.append(func_meta)

            # 3. Extract Top-Level Global Variables
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        globals_list.append({"name": target.id, "line_span": (node.lineno, node.end_lineno)})
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    globals_list.append({"name": node.target.id, "line_span": (node.lineno, node.end_lineno)})

        return {
            "file_path": str(self.file_path),
            "docstring": file_docstring,  # <--- FIXED
            "classes": classes,
            "functions": functions,
            "globals": globals_list,
            "top_level_calls": sorted(list(top_level_calls))
        }
    def _extract_methods(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Helper to extract methods defined inside a specific class block."""
        methods: List[Dict[str, Any]] = []
        for sub_node in class_node.body:
            if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append({
                    "name": sub_node.name,
                    "line_span": (sub_node.lineno, sub_node.end_lineno),  # <--- UPGRADED
                    "arguments": [arg.arg for arg in sub_node.args.args],
                    "is_async": isinstance(sub_node, ast.AsyncFunctionDef),
                    "calls": self._extract_calls_from_body(sub_node.body),
                })
        return methods
    # Add this helper method inside your ASTParser class
    def _extract_calls_from_body(self, body_nodes: list[ast.stmt]) -> list[str]:
        """Scans the inner statements of a function body to find called function/method names."""
        called_names = set()
        for sub_node in ast.walk(ast.Module(body=body_nodes, type_ignores=[])):
            if isinstance(sub_node, ast.Call):
                # Case A: Direct call (e.g., 'extract_cookies_to_jar(x)')
                if isinstance(sub_node.func, ast.Name):
                    called_names.add(sub_node.func.id)
                # Case B: Attribute call (e.g., 'self.prepare_cookies()' or 'models.Response()')
                elif isinstance(sub_node.func, ast.Attribute):
                    called_names.add(sub_node.func.attr)
        return sorted(list(called_names))
        
    def _get_source_segment(self, node: ast.AST) -> str:
        """Helper to convert complex AST nodes (like nested decorators or base classes)

        back into their original string representation.
        """
        try:
            return ast.unparse(node)
        except Exception:
            return "Unknown"


def extract_file_entities(rel_path: str, repo_root: Path) -> dict:
    """Parses a python file using its absolute path, but labels the nodes

    using the relative path format to match the existing graph ledger.
    """
    full_path = repo_root / rel_path
    if not full_path.exists():
        return {}
        
    with open(full_path, "r", encoding="utf-8") as f:
        source = f.read()
        
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  [Parser Error] Syntax error while reading {rel_path}: {e}")
        return {}
        
    entities = {}
    
    class RepoASTVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_class = None
            
        def visit_ClassDef(self, node):
            node_id = f"{rel_path}::{node.name}"
            entities[node_id] = {
                "type": "CLASS",
                "name": node.name,
                "file_path": rel_path,
                "chunk_text": ast.get_source_segment(source, node),
                "bases": [ast.unparse(b) for b in node.bases]
            }
            old_class = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = old_class
            
        def visit_FunctionDef(self, node):
            if self.current_class:
                node_id = f"{rel_path}::{self.current_class}.{node.name}"
            else:
                node_id = f"{rel_path}::{node.name}"
                
            args = [arg.arg for arg in node.args.args]
            entities[node_id] = {
                "type": "FUNCTION",
                "name": node.name,
                "file_path": rel_path,
                "chunk_text": ast.get_source_segment(source, node),
                "signature": f"({', '.join(args)})"
            }
            self.generic_visit(node)

    visitor = RepoASTVisitor()
    visitor.visit(tree)
    return entities