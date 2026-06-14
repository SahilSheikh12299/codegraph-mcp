from pathlib import Path
from typing import Any, Dict, List, Tuple
import networkx as nx
import json

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



class RepositoryGraphCompiler:
    """Consumes parsed structural metadata JSON and compiles a unified,

    cross-referenced NetworkX directed property graph.
    """

    def __init__(self, raw_data: Dict):
        self.raw_data = raw_data
        self.cg = CodeGraph()

    def compile(self) -> CodeGraph:
        """Executes the two-phase graph compilation pipeline."""
        # with open(self.json_path, "r", encoding="utf-8") as f:
        #     self.raw_data = json.load(f)

        # Build a rapid global lookup table to map where classes are defined across the project
        # Key: "ClassName" -> Value: "src/requests/auth.py"
        global_class_registry: Dict[str, str] = {}
        for file_path, assets in self.raw_data.items():
            for cls in assets.get("classes", []):
                global_class_registry[cls["name"]] = file_path

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
                
                for base_class_name in cls.get("bases", []):
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

            # Step C: Wire Function-to-Function / Function-to-Class CALLS relationships
            for func in assets.get("functions", []):
                caller_id = f"{file_path}::{func['name']}"
                
                for called_symbol in func.get("calls", []):
                    # Scenario A: The called target lives in the exact same file
                    local_id = f"{file_path}::{called_symbol}"
                    if self.cg.graph.has_node(local_id):
                        self.cg.add_calls_edge(caller_id, local_id)
                        continue
                        
                    # Scenario B: The called target was imported from another internal repository file
                    if called_symbol in global_func_registry:
                        defining_file = global_func_registry[called_symbol]
                        # Verify the caller file actually imports the file where the target is defined
                        if defining_file in internal_imports:
                            callee_id = f"{defining_file}::{called_symbol}"
                            if self.cg.graph.has_node(callee_id):
                                self.cg.add_calls_edge(caller_id, callee_id)

            # Step D: Wire FILE-to-FUNCTION / FILE-to-CLASS root-level execution calls
            for called_symbol in assets.get("top_level_calls", []):
                # Scenario A: The called script target lives in the exact same file
                local_id = f"{file_path}::{called_symbol}"
                if self.cg.graph.has_node(local_id):
                    self.cg.add_calls_edge(file_id, local_id) # Edge originates from file_id!
                    continue
                    
                # Scenario B: The called script target was imported from an internal module
                if called_symbol in global_func_registry:
                    defining_file = global_func_registry[called_symbol]
                    if defining_file in internal_imports:
                        callee_id = f"{defining_file}::{called_symbol}"
                        if self.cg.graph.has_node(callee_id):
                            self.cg.add_calls_edge(file_id, callee_id) # Edge originates from file_id!
        print("[Compilation Complete] Repository network successfully mapped.")
        return self.cg




class CodeChunker:
    """Traverses a compiled CodeGraph layout and generates dense, hybrid

    semantic text strings for every node, combining attributes and graph context.
    """

    def __init__(self, code_graph: CodeGraph):
        self.cg = code_graph
        self.graph = code_graph.graph

    def extract_and_bind_chunks(self) -> int:
        """Generates semantic context blocks for all valid code nodes

        and binds them directly to the graph structure. Returns the count of processed nodes.
        """
        processed_count = 0

        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get("type", "UNKNOWN")
            
            # Skip pure directory module layers from heavy embedding blocks if desired,
            # or give them a clean structural layout
            if node_type == "MODULE":
                chunk = self._build_module_chunk(node_id, data)
            elif node_type == "FILE":
                chunk = self._build_file_chunk(node_id, data)
            elif node_type == "CLASS":
                chunk = self._build_class_chunk(node_id, data)
            elif node_type == "FUNCTION":
                chunk = self._build_function_chunk(node_id, data)
            else:
                continue

            # Bind the text representation cleanly back into the graph node properties
            self.graph.nodes[node_id]["chunk_text"] = chunk
            processed_count += 1

        return processed_count

    def _build_module_chunk(self, node_id: str, data: dict) -> str:
        """Compiles a structural description for directory packages."""
        return (
            f"Entity Type: MODULE\n"
            f"Package Path: {data.get('path', node_id)}\n"
            f"Module Name: {data.get('name', '')}\n"
            f"Description: Directory container managing local package source implementations."
        )

    def _build_file_chunk(self, node_id: str, data: dict) -> str:
        """Compiles a complete description for a physical Python file module."""
        docstring = data.get("docstring", "").strip() or "No module documentation provided."
        return (
            f"Entity Type: FILE\n"
            f"File Workspace Path: {data.get('path', node_id)}\n"
            f"Total Lines of Code: {data.get('line_count', 0)}\n"
            f"Module Level Documentation:\n{docstring}"
        )

    def _build_class_chunk(self, node_id: str, data: dict) -> str:
        """Compiles a description for a class structure using inherited ancestors."""
        docstring = data.get("docstring", "").strip() or "No class documentation provided."
        bases = data.get("bases", [])
        bases_str = ", ".join(bases) if bases else "None (Base Object)"
        
        # Query Graph Layer: Find what file contains this class
        containing_files = [source for source, target, edge_data in self.graph.in_edges(node_id, data=True) 
                            if edge_data.get("relationship") == "CONTAINS"]
        file_context = containing_files[0] if containing_files else data.get("file_path", "Unknown")

        return (
            f"Entity Type: CLASS\n"
            f"Class Identifier Name: {data.get('name', '')}\n"
            f"Defined in Workspace File: {file_context}\n"
            f"Inherits From Ancestors: [{bases_str}]\n"
            f"Class Core Documentation:\n{docstring}"
        )

    def _build_function_chunk(self, node_id: str, data: dict) -> str:
        """Compiles a function signature blueprint enriched with structural execution context."""
        docstring = data.get("docstring", "").strip() or "No functional documentation provided."
        func_name = data.get("name", "")
        signature = data.get("signature", "()")
        
        # 1. Base identity layout
        chunk_lines = [
            f"Entity Type: FUNCTION / METHOD",
            f"Function Name: {func_name}",
            f"Execution Signature Profile: def {func_name}{signature}",
            f"Is Object Method: {data.get('is_method', False)}"
        ]
        if data.get("belongs_to_class"):
            chunk_lines.append(f"Belongs to Class Structure: {data.get('belongs_to_class')}")

        # 2. GRAPH POWER: Trace operational execution context neighbors
        called_targets = []
        for _, target, edge_data in self.graph.out_edges(node_id, data=True):
            if edge_data.get("relationship") == "CALLS":
                # Extract clean token name from target ID (path::name -> name)
                called_targets.append(target.split("::")[-1])

        calling_sources = []
        for source, _, edge_data in self.graph.in_edges(node_id, data=True):
            if edge_data.get("relationship") == "CALLS":
                calling_sources.append(source.split("::")[-1])

        if called_targets:
            chunk_lines.append(f"Under the hood, this function explicitly invokes: {called_targets}")
        if calling_sources:
            chunk_lines.append(f"This function is triggered and relied upon by: {calling_sources}")

        # 3. Append core documentation strings
        chunk_lines.append(f"Functional Operational Summary:\n{docstring}")
        
        return "\n".join(chunk_lines)