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


def build_embedding_text(
    node_type: str,
    name: str,
    *,
    signature: str = "()",
    docstring: str = "",
    bases: list | None = None,
    belongs_to_class: str | None = None,
) -> str:
    """Compact text for vector search — not returned on fetch."""
    lines = [f"Type: {node_type}", f"Name: {name}"]
    if belongs_to_class:
        lines.append(f"Class: {belongs_to_class}")
    if node_type == "CLASS" and bases:
        lines.append(f"Inherits: {', '.join(bases)}")
    if node_type == "FUNCTION":
        lines.append(f"Signature: def {name}{signature}")
    if docstring:
        first_line = docstring.strip().split("\n")[0]
        lines.append(f"Summary: {first_line[:300]}")
    return "\n".join(lines)


_MAX_UNPARSE_LEN = 80
_MAX_LIST_ITEMS = 30


def _unparse_expr(node: ast.AST | None, max_len: int = _MAX_UNPARSE_LEN) -> str:
    if node is None:
        return ""
    try:
        text = ast.unparse(node)
    except Exception:
        return ""
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _assign_target_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        names.append(elt.id)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names.append(node.target.id)
    elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
        names.append(node.target.id)
    return names


def _format_call_list(calls: set[str]) -> str:
    return ", ".join(sorted(calls)[:_MAX_LIST_ITEMS])


def _function_skeleton_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    lines: list[str] = []
    try:
        header = ast.unparse(node).split("\n", 1)[0].rstrip(":")
    except Exception:
        args = [arg.arg for arg in node.args.args]
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        header = f"{prefix} {node.name}({', '.join(args)})"

    lines.append(f"Signature: {header}")

    if node.decorator_list:
        decorators = ", ".join(_unparse_expr(d) for d in node.decorator_list)
        lines.append(f"Decorators: {decorators}")

    docstring = ast.get_docstring(node)
    if docstring:
        lines.append(f"Docstring: {docstring.strip()[:500]}")

    if node.returns:
        lines.append(f"Return type: {_unparse_expr(node.returns)}")

    param_names = {arg.arg for arg in node.args.args}
    calls: set[str] = set()
    variables: set[str] = set()
    yields: list[str] = []
    returns: list[str] = []

    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            name = _call_name(sub)
            if name:
                calls.add(name)
        elif isinstance(sub, ast.Yield):
            yields.append(_unparse_expr(sub.value) or "value")
        elif isinstance(sub, ast.YieldFrom):
            yields.append(f"from {_unparse_expr(sub.value)}")
        elif isinstance(sub, ast.Return) and sub.value is not None:
            returns.append(_unparse_expr(sub.value))
        elif isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for var_name in _assign_target_names(sub):
                if var_name not in param_names and var_name != "self":
                    variables.add(var_name)

    if calls:
        lines.append(f"Calls: {_format_call_list(calls)}")
    if variables:
        lines.append(f"Variables: {', '.join(sorted(variables)[:_MAX_LIST_ITEMS])}")
    if yields:
        lines.append(f"Yields: {', '.join(yields[:5])}")
    if returns:
        lines.append(f"Return expressions: {', '.join(returns[:5])}")

    return lines


def _class_skeleton_lines(node: ast.ClassDef) -> list[str]:
    lines = [f"Type: CLASS", f"Name: {node.name}"]

    if node.bases:
        lines.append(f"Inherits: {', '.join(_unparse_expr(b) for b in node.bases)}")

    docstring = ast.get_docstring(node)
    if docstring:
        lines.append(f"Docstring: {docstring.strip()[:500]}")

    attributes: list[str] = []
    methods: list[str] = []
    all_calls: set[str] = set()

    for sub in node.body:
        if isinstance(sub, (ast.Assign, ast.AnnAssign)):
            for name in _assign_target_names(sub):
                attributes.append(name)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                methods.append(ast.unparse(sub).split("\n", 1)[0].rstrip(":"))
            except Exception:
                args = [arg.arg for arg in sub.args.args]
                prefix = "async def" if isinstance(sub, ast.AsyncFunctionDef) else "def"
                methods.append(f"{prefix} {sub.name}({', '.join(args)})")
            method_doc = ast.get_docstring(sub)
            if method_doc:
                first = method_doc.strip().split("\n")[0][:120]
                methods[-1] = f"{methods[-1]}  # {first}"
            for call_node in ast.walk(sub):
                if isinstance(call_node, ast.Call):
                    name = _call_name(call_node)
                    if name:
                        all_calls.add(name)

    if attributes:
        lines.append(f"Attributes: {', '.join(attributes[:_MAX_LIST_ITEMS])}")
    if methods:
        lines.append("Methods:")
        lines.extend(f"  - {m}" for m in methods[:_MAX_LIST_ITEMS])
    if all_calls:
        lines.append(f"Calls: {_format_call_list(all_calls)}")

    return lines


def build_semantic_skeleton(
    chunk_text: str,
    node_type: str = "FUNCTION",
    *,
    data: Dict[str, Any] | None = None,
) -> str:
    """AST-extracted symbols for cross-encoder reranking (no loop/if bodies)."""
    data = data or {}
    if not chunk_text or not chunk_text.strip():
        return (data.get("embedding_text") or "").strip()

    try:
        tree = ast.parse(chunk_text.strip())
    except SyntaxError:
        return (data.get("embedding_text") or chunk_text[:500]).strip()

    if not tree.body:
        return (data.get("embedding_text") or "").strip()

    top = tree.body[0]
    if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
        lines = _function_skeleton_lines(top)
    elif isinstance(top, ast.ClassDef):
        lines = _class_skeleton_lines(top)
    else:
        return (data.get("embedding_text") or "").strip()

    if data.get("belongs_to_class"):
        lines.insert(1, f"Class: {data['belongs_to_class']}")
    if data.get("file_path"):
        lines.append(f"File: {data['file_path']}")

    return "\n".join(lines)


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
            docstring = ast.get_docstring(node) or ""
            bases = [ast.unparse(b) for b in node.bases]
            entities[node_id] = {
                "type": "CLASS",
                "name": node.name,
                "file_path": rel_path,
                "chunk_text": ast.get_source_segment(source, node),
                "docstring": docstring,
                "bases": bases,
                "embedding_text": build_embedding_text(
                    "CLASS", node.name, docstring=docstring, bases=bases
                ),
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
            signature = f"({', '.join(args)})"
            docstring = ast.get_docstring(node) or ""
            entities[node_id] = {
                "type": "FUNCTION",
                "name": node.name,
                "file_path": rel_path,
                "chunk_text": ast.get_source_segment(source, node),
                "signature": signature,
                "docstring": docstring,
                "belongs_to_class": self.current_class,
                "embedding_text": build_embedding_text(
                    "FUNCTION",
                    node.name,
                    signature=signature,
                    docstring=docstring,
                    belongs_to_class=self.current_class,
                ),
            }
            self.generic_visit(node)

    visitor = RepoASTVisitor()
    visitor.visit(tree)
    return entities