import os
import json
import hashlib
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph
from typing import Any, Dict, List, Tuple
from fileParsing import WorkspaceScanner, ImportTracker, ASTParser, extract_file_entities
from buildGraph import RepositoryGraphCompiler, CodeChunker


class GraphSerializer:
    """Manages workspace path isolation using deterministic SHA-256 hashing

    along with the serialization and de-serialization of codebase graphs.
    """

    @staticmethod
    def get_global_storage_dir() -> Path:
        """Returns and creates the centralized, hidden global tool storage cache folder."""
        storage_dir = Path("~/.cursor_graph_rag/storage").expanduser()
        storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir

    # @staticmethod
    # def get_graph_path_for_workspace(workspace_path: str | Path) -> Path:
    #     """Generates a deterministic SHA-256 hash from an absolute workspace path

    #     to guarantee unique file identification without naming collisions.
    #     """
    #     abs_path_str = str(Path(workspace_path).resolve())
    #     # Generate a distinct cryptographic hash of the file system path string
    #     path_hash = hashlib.sha256(abs_path_str.encode("utf-8")).hexdigest()
        
    #     storage_dir = GraphSerializer.get_global_storage_dir()
    #     return storage_dir / f"{path_hash}.json"

    @staticmethod
    def save_to_json(G: nx.DiGraph, workspace_path: str | Path, graph_path: str | Path) -> Path:
        """Converts a working NetworkX graph into standard node-link data schemas

        and commits the resulting JSON payload cleanly to the global storage cache.
        """
        abs_path_str = str(Path(workspace_path).resolve())
        
        # Attach the raw workspace path directly into the graph attributes for traceability
        G.graph["workspace_path"] = abs_path_str
        
        # Resolve the destination file hash signature
        #target_file_path = GraphSerializer.get_graph_path_for_workspace(abs_path_str)
        # Transform graph entities and structural edges to primitive python dictionaries
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
    
            for file_path in python_files:
                repo_relative_path = str(file_path.relative_to(repo_root))
                print(f"Analyzing: {repo_relative_path}...")
                
                # Parse structural tokens
                parser = ASTParser(file_path=file_path)
                ast_data = parser.parse()
                
                # Track import linkages seamlessly
                import_data = tracker.get_dependencies(file_path)
                
                evaluation_report[repo_relative_path] = {
                    "classes": ast_data.get("classes", []),
                    "functions": ast_data.get("functions", []),
                    "globals": ast_data.get("globals", []),
                    "top_level_calls": ast_data.get("top_level_calls", []),
                    "internal_imports": import_data.get("internal_paths", []),
                    "external_imports": import_data.get("external_modules", [])
                }

            compiler = RepositoryGraphCompiler(evaluation_report)
            code_graph_instance = compiler.compile()
            #G = code_graph_instance.graph
            

            chunker = CodeChunker(code_graph=code_graph_instance)
            # This will update the graph with code chunks
            _ = chunker.extract_and_bind_chunks() 
            G = chunker.graph

            for file_path in python_files:
                repo_relative_path = str(file_path.relative_to(repo_root))
                for node_id, data in extract_file_entities(repo_relative_path, repo_root).items():
                    if G.has_node(node_id):
                        G.nodes[node_id]["chunk_text"] = data["chunk_text"]
            G.graph["chunk_schema"] = 2
            G.graph["calls_schema"] = 1

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
