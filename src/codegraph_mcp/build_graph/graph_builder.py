from pathlib import Path
from typing import Any, Dict, List, Tuple
import networkx as nx

class CodeGraph:
    """An advanced directed property graph schema that manages files, classes,

    functions, and automatically maps the underlying folder structure (MODULES).
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    # ==========================================
    # AUTOMATIC TOPOLOGY RESOLUTION
    # ==========================================

    def _ensure_folder_tree(self, file_relative_path: str) -> str:
        """Surgically analyzes a file path, builds missing parent MODULE nodes,

        and connects the directory hierarchy. Returns the immediate parent module ID.
        """
        path_obj = Path(file_relative_path)
        parts = path_obj.parent.parts

        # If the file sits directly in the root folder, no module nodes are required
        if not parts or parts == (".",) or parts == ("",):
            return ""

        current_tree_path = Path("")
        previous_module_id = None

        # Traverse down the folder segments (e.g., 'src' -> 'src/requests')
        for part in parts:
            current_tree_path = current_tree_path / part
            module_id = str(current_tree_path)

            # 1. Dynamically inject the MODULE node if it doesn't exist yet
            if not self.graph.has_node(module_id):
                self.graph.add_node(
                    module_id,
                    type="MODULE",
                    path=module_id,
                    name=part
                )

            # 2. Wire the directory to its parent directory if it's a subfolder
            if previous_module_id:
                self.graph.add_edge(previous_module_id, module_id, relationship="SUBMODULE_OF")

            previous_module_id = module_id

        return previous_module_id  # Returns the innermost folder containing the file

    # ==========================================
    # NOUNS: Node Creation Methods
    # ==========================================

    def add_file_node(self, relative_path: str, docstring: str | None, line_count: int) -> str:
        """Adds a physical file node to the graph schema and links it to its

        automatically generated parent module folder tree.
        """
        file_node_id = relative_path
        
        self.graph.add_node(
            file_node_id,
            type="FILE",
            path=relative_path,
            docstring=docstring or "",
            line_count=line_count,
        )

        # Automatically resolve and wire the folder topology
        parent_module_id = self._ensure_folder_tree(relative_path)
        if parent_module_id:
            # Draw the connection stating this file belongs inside that specific folder module
            self.graph.add_edge(parent_module_id, file_node_id, relationship="MEMBER_OF")

        return file_node_id

    def add_class_node(
        self,
        class_name: str,
        file_path: str,
        docstring: str | None,
        bases: List[str],
        line_span: Tuple[int, int],
    ) -> str:
        """Adds a class definition node to the graph schema using a scoped ID."""
        node_id = f"{file_path}::{class_name}"
        self.graph.add_node(
            node_id,
            type="CLASS",
            name=class_name,
            file_path=file_path,
            docstring=docstring or "",
            bases=bases,
            line_span=line_span,
        )
        return node_id

    def add_function_node(
        self,
        func_name: str,
        file_path: str,
        docstring: str | None,
        signature: str,
        line_span: Tuple[int, int],
        is_method: bool = False,
        class_name: str | None = None,
    ) -> str:
        """Adds a function or method node to the graph schema using a scoped ID."""
        if is_method and class_name:
            node_id = f"{file_path}::{class_name}.{func_name}"
        else:
            node_id = f"{file_path}::{func_name}"

        self.graph.add_node(
            node_id,
            type="FUNCTION",
            name=func_name,
            file_path=file_path,
            docstring=docstring or "",
            signature=signature,
            line_span=line_span,
            is_method=is_method,
            belongs_to_class=class_name,
        )
        return node_id

    def add_constant_node(
        self,
        const_name: str,
        file_path: str,
        line_span: Tuple[int, int],
    ) -> str:
        node_id = f"{file_path}::{const_name}"
        self.graph.add_node(
            node_id,
            type="CONSTANT",
            name=const_name,
            file_path=file_path,
            line_span=line_span,
            signature="",
        )
        return node_id

    # ==========================================
    # VERBS: Edge Creation Methods
    # ==========================================

    def add_contains_edge(self, parent_file_id: str, child_node_id: str) -> None:
        """Draws a structural relationship stating a file contains a class/function."""
        self.graph.add_edge(parent_file_id, child_node_id, relationship="CONTAINS")

    def add_imports_edge(self, source_file_id: str, target_file_id: str) -> None:
        """Draws a dependency relationship stating file A imports file B."""
        self.graph.add_edge(source_file_id, target_file_id, relationship="IMPORTS")

    def add_calls_edge(self, caller_id: str, callee_id: str) -> None:
        """Draws an execution relationship stating function A calls function B."""
        self.graph.add_edge(caller_id, callee_id, relationship="CALLS")

    def add_inherits_edge(self, child_class_id: str, parent_class_id: str) -> None:
        """Draws an object-oriented relationship stating class A inherits class B."""
        self.graph.add_edge(child_class_id, parent_class_id, relationship="INHERITS_FROM")

    # ==========================================
    # Diagnostics & Verification Utilities
    # ==========================================

    def get_summary(self) -> Dict[str, Any]:
        """Returns structural count metrics to verify the complete schema footprint."""
        node_types: Dict[str, int] = {"MODULE": 0, "FILE": 0, "CLASS": 0, "FUNCTION": 0}
        edge_relationships: Dict[str, int] = {
            "SUBMODULE_OF": 0,
            "MEMBER_OF": 0,
            "CONTAINS": 0,
            "IMPORTS": 0,
            "CALLS": 0,
            "INHERITS_FROM": 0,
        }

        for _, data in self.graph.nodes(data=True):
            n_type = data.get("type", "UNKNOWN")
            if n_type in node_types:
                node_types[n_type] += 1

        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relationship", "UNKNOWN")
            if rel in edge_relationships:
                edge_relationships[rel] += 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "breakdown_nodes": node_types,
            "breakdown_edges": edge_relationships,
        }


def resolve_callee_id(
    G: nx.DiGraph,
    file_path: str,
    called_symbol: str,
    class_name: str | None,
    global_func_registry: Dict[str, str],
    internal_imports: List[str],
) -> str | None:
    """Resolve a called symbol to a graph node_id."""
    candidates: List[str] = []
    if class_name:
        candidates.append(f"{file_path}::{class_name}.{called_symbol}")
    candidates.append(f"{file_path}::{called_symbol}")

    for cid in candidates:
        if G.has_node(cid):
            return cid

    for nid, d in G.nodes(data=True):
        if d.get("file_path") == file_path and d.get("name") == called_symbol and d.get("type") == "FUNCTION":
            return nid

    if called_symbol in global_func_registry:
        defining_file = global_func_registry[called_symbol]
        if defining_file in internal_imports:
            callee_id = f"{defining_file}::{called_symbol}"
            if G.has_node(callee_id):
                return callee_id
    return None


def wire_calls_for_file(
    G: nx.DiGraph,
    file_path: str,
    assets: dict,
    global_func_registry: Dict[str, str],
) -> None:
    """Wire CALLS edges for one file's functions and class methods."""
    internal_imports: List[str] = assets.get("internal_imports", [])
    file_id = file_path

    for func in assets.get("functions", []):
        caller_id = f"{file_path}::{func['name']}"
        if not G.has_node(caller_id):
            continue
        for called_symbol in func.get("calls", []):
            callee_id = resolve_callee_id(
                G, file_path, called_symbol, None, global_func_registry, internal_imports
            )
            if callee_id:
                G.add_edge(caller_id, callee_id, relationship="CALLS")

    for cls in assets.get("classes", []):
        for method in cls.get("methods", []):
            caller_id = f"{file_path}::{cls['name']}.{method['name']}"
            if not G.has_node(caller_id):
                continue
            for called_symbol in method.get("calls", []):
                callee_id = resolve_callee_id(
                    G, file_path, called_symbol, cls["name"], global_func_registry, internal_imports
                )
                if callee_id:
                    G.add_edge(caller_id, callee_id, relationship="CALLS")

    for called_symbol in assets.get("top_level_calls", []):
        callee_id = resolve_callee_id(
            G, file_path, called_symbol, None, global_func_registry, internal_imports
        )
        if callee_id:
            G.add_edge(file_id, callee_id, relationship="CALLS")


def build_func_registry_from_graph(G: nx.DiGraph) -> Dict[str, str]:
    """Map bare function/class names to defining file paths."""
    registry: Dict[str, str] = {}
    for _, d in G.nodes(data=True):
        ntype = d.get("type")
        name, fpath = d.get("name"), d.get("file_path")
        if not name or not fpath:
            continue
        if ntype == "CLASS" or (ntype == "FUNCTION" and not d.get("is_method")):
            registry[name] = fpath
    return registry


def strip_calls_edges(G: nx.DiGraph) -> None:
    to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get("relationship") == "CALLS"]
    G.remove_edges_from(to_remove)


def strip_calls_edges_for_file(G: nx.DiGraph, rel_path: str) -> None:
    file_nodes = {
        nid for nid, d in G.nodes(data=True)
        if d.get("file_path") == rel_path or nid == rel_path
    }
    to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("relationship") == "CALLS" and (u in file_nodes or v in file_nodes)
    ]
    G.remove_edges_from(to_remove)


class RepositoryGraphCompiler:
    """Consumes parsed structural metadata JSON and compiles a unified,

    cross-referenced NetworkX directed property graph.
    """

    def __init__(self, raw_data: Dict):
        self.raw_data = raw_data
        self.cg = CodeGraph()

    def compile(self) -> CodeGraph:
        """Executes the two-phase graph compilation pipeline."""

        # --- PHASE 1: NODE & CONTAINMENT HYDRATION ---
        print("Phase 1: Hydrating nodes and containment structural layers...")
        for file_path, assets in self.raw_data.items():
            # 1. Inject the File Node (which automatically triggers parent MODULE node trees)
            file_id = self.cg.add_file_node(
                relative_path=file_path,
                docstring=assets.get("docstring", ""),
                line_count=assets.get("line_count", 0)
            )

            # 2. Inject Top-Level Functions
            for func in assets.get("functions", []):
                func_id = self.cg.add_function_node(
                    func_name=func["name"],
                    file_path=file_path,
                    docstring=func.get("docstring", ""),
                    signature=str(func.get("arguments", [])),
                    line_span=tuple(func.get("line_span", (0, 0))),
                    is_method=False
                )
                self.cg.add_contains_edge(file_id, func_id)

            # 3. Inject Classes and their internal Methods
            for cls in assets.get("classes", []):
                class_id = self.cg.add_class_node(
                    class_name=cls["name"],
                    file_path=file_path,
                    docstring=cls.get("docstring", ""),
                    bases=cls.get("bases", []),
                    line_span=tuple(cls.get("line_span", (0, 0)))
                )
                self.cg.add_contains_edge(file_id, class_id)

                # Inject individual methods belonging to this class
                for method in cls.get("methods", []):
                    method_id = self.cg.add_function_node(
                        func_name=method["name"],
                        file_path=file_path,
                        docstring=method.get("docstring", ""),
                        signature=str(method.get("arguments", [])),
                        line_span=tuple(method.get("line_span", (0, 0))),
                        is_method=True,
                        class_name=cls["name"]
                    )
                    # File contains the method text footprint structurally
                    self.cg.add_contains_edge(file_id, method_id)

            for g in assets.get("globals", []):
                const_id = self.cg.add_constant_node(
                    g["name"],
                    file_path,
                    tuple(g.get("line_span", (0, 0))),
                )
                self.cg.add_contains_edge(file_id, const_id)

        # --- PHASE 2: CROSS-FILE DEPENDENCY RESOLUTION ---
        print("Phase 2: Resolving cross-file imports and object inheritance maps...")
        # 1. Build rapid global lookup maps for both Classes and Functions before resolving
        global_class_registry: Dict[str, str] = {}
        global_func_registry: Dict[str, str] = {}
        
        for file_path, assets in self.raw_data.items():
            for cls in assets.get("classes", []):
                global_class_registry[cls["name"]] = file_path
                # Class names can be called to instantiate them, so add them to the call registry too
                global_func_registry[cls["name"]] = file_path
            for func in assets.get("functions", []):
                global_func_registry[func["name"]] = file_path

        # 2. Loop through every file to wire up the relationships
        for file_path, assets in self.raw_data.items():
            file_id = file_path
            internal_imports: List[str] = assets.get("internal_imports", [])

            # Step A: Wire File-to-File IMPORTS relationships
            for imported_file_path in internal_imports:
                if self.cg.graph.has_node(imported_file_path):
                    self.cg.add_imports_edge(file_id, imported_file_path)

            # Step B: Wire Cross-File Class Inheritance (INHERITS_FROM)
            for cls in assets.get("classes", []):
                class_id = f"{file_path}::{cls['name']}"
                
                base_names = cls.get("base_names") or []
                for base_class_name in base_names:
                    # Scenario A: Parent class is defined inside the exact same file
                    local_match = f"{file_path}::{base_class_name}"
                    if self.cg.graph.has_node(local_match):
                        self.cg.add_inherits_edge(class_id, local_match)
                        continue

                    # Scenario B: Parent class is imported from another internal repository file
                    if base_class_name in global_class_registry:
                        defining_file = global_class_registry[base_class_name]
                        if defining_file in internal_imports:
                            parent_class_id = f"{defining_file}::{base_class_name}"
                            self.cg.add_inherits_edge(class_id, parent_class_id)

            # Step C + D: Wire CALLS for functions, methods, and top-level calls
            wire_calls_for_file(
                self.cg.graph, file_path, assets, global_func_registry
            )
        print("[Compilation Complete] Repository network successfully mapped.")
        return self.cg