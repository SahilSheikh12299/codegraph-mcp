import json
from pathlib import Path
from typing import Any
import networkx as nx
from networkx.readwrite import json_graph
from codegraph_mcp.file_parsing import WorkspaceScanner, ImportTracker, parse_python_file, read_python_ast
from codegraph_mcp.build_graph import RepositoryGraphCompiler


class GraphSerializer:
    """Manages workspace path isolation using deterministic SHA-256 hashing

    along with the serialization and de-serialization of codebase graphs.
    """
    @staticmethod
    def save_to_json(G: nx.DiGraph, workspace_path: str | Path, graph_path: str | Path) -> Path:
        """Converts a working NetworkX graph into standard node-link data schemas

        and commits the resulting JSON payload cleanly to the global storage cache.
        """
        abs_path_str = str(Path(workspace_path).resolve())
        
        # Attach the raw workspace path directly into the graph attributes for traceability
        G.graph["workspace_path"] = abs_path_str
        
        # Resolve the destination file hash signature
        serialized_data = json_graph.node_link_data(G)
        
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(serialized_data, f, indent=2)
            
        return graph_path

    @staticmethod
    def load_from_json(workspace_path: str | Path, graph_path: str | Path) -> nx.DiGraph:
        """Resolves the hash signature for the target workspace, reads its JSON cache,

        and reconstructs a live, operational directed graph tree.
        
        Returns an initialized blank graph if no previous index ledger exists.
        """
        abs_path_str = str(Path(workspace_path).resolve())
        #target_file_path = GraphSerializer.get_graph_path_for_workspace(abs_path_str)
        
        if not graph_path.exists():
            # Graph does not exist, we need to build it for the first time and return
            repo_root = Path(abs_path_str).resolve()
            scanner = WorkspaceScanner(root_path=repo_root)
            python_files = scanner.scan()
            tracker = ImportTracker(repo_root=repo_root, all_python_files=python_files)

            evaluation_report = {}
            entities_by_file: dict[str, dict[str, Any]] = {}

            for file_path in python_files:
                repo_relative_path = str(file_path.relative_to(repo_root))
                print(f"Analyzing: {repo_relative_path}...")

                parsed = read_python_ast(file_path)
                if parsed is None:
                    continue
                source, tree = parsed

                parsed_file = parse_python_file(
                    file_path,
                    rel_path=repo_relative_path,
                    source=source,
                    tree=tree,
                    build_entities=True,
                    enable_auto_docstrings=False,
                )
                if parsed_file.get("error"):
                    continue

                import_data = tracker.get_dependencies(file_path, tree=tree)

                evaluation_report[repo_relative_path] = {
                    "docstring": parsed_file.get("docstring", ""),
                    "line_count": source.count("\n") + 1 if source else 0,
                    "classes": parsed_file.get("classes", []),
                    "functions": parsed_file.get("functions", []),
                    "globals": parsed_file.get("globals", []),
                    "top_level_calls": parsed_file.get("top_level_calls", []),
                    "internal_imports": import_data.get("internal_paths", []),
                    "external_imports": import_data.get("external_modules", []),
                }
                entities_by_file[repo_relative_path] = parsed_file.get("entities", {})

            compiler = RepositoryGraphCompiler(evaluation_report)
            code_graph_instance = compiler.compile()
            G = code_graph_instance.graph

            for _repo_relative_path, entities in entities_by_file.items():
                for node_id, data in entities.items():
                    if G.has_node(node_id):
                        G.nodes[node_id]["chunk_text"] = data["chunk_text"]
                        G.nodes[node_id]["embedding_text"] = data["embedding_text"]
                        if data.get("line_span"):
                            G.nodes[node_id]["line_span"] = data["line_span"]
                        if data.get("body_hash"):
                            G.nodes[node_id]["body_hash"] = data["body_hash"]

            # Output detailed diagnostics to verify your metrics
            summary = code_graph_instance.get_summary()
            print("\n==========================================")
            print("      COMPILED GRAPH DIAGNOSTICS REPORT   ")
            print("==========================================")
            print(f"Total Network Entities (Nodes): {summary['total_nodes']}")
            print(f"Total Structural Links (Edges): {summary['total_edges']}\n")
            
            print("Entity Breakdown:")
            for node_type, count in summary["breakdown_nodes"].items():
                print(f"  -> {node_type}: {count}")
                
            print("\nRelationship Connection Breakdown:")
            for rel_type, count in summary["breakdown_edges"].items():
                print(f"  -> {rel_type}: {count}")
            print("==========================================")
            return G
            
        with open(graph_path, "r", encoding="utf-8") as f:
            raw_json_data = json.load(f)
            
        # Re-materialize the live NetworkX structure from the json schema
        G = json_graph.node_link_graph(raw_json_data)
        
        # Ensure graph typing properties remain strictly directed
        if not G.is_directed():
            G = G.to_directed()
            
        return G
