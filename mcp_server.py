import os
import ast
import sys
import gc
import logging
import hashlib
from pathlib import Path
import networkx as nx
from mcp.server.fastmcp import FastMCP
from filelock import FileLock
import re
import os
from typing import Dict, Any, Tuple, List
from fileParsing import extract_file_entities

# Completely muzzle third-party library progress bars before they can print
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import your validated local analytical engine modules
from graph_io import GraphSerializer
from embeddingPipeline import EmbeddingModelLifecycleManager, LocalEmbeddingPipeline
from advanced_engine import AdvancedRetrievalEngine

# Configure silent logging to avoid corrupting standard output
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

# =========================================================================
# 1. MASTER MODEL MCP INITIALIZATION & LIFECYCLE
# =========================================================================
mcp = FastMCP("Cursor-Graph-RAG-Engine")



model_manager = EmbeddingModelLifecycleManager()



def get_graph_paths(active_project_root: str | Path) -> Tuple[Path, Path]:
    """The single source of truth for workspace graph cache and lock files.
    
    Always resolves the raw absolute directory path to prevent double-hashing.
    """
    # 1. Force resolve the raw directory path completely
    abs_path_str = str(Path(active_project_root).resolve())
    
    # 2. Compute a single MD5 hash of that absolute path string
    workspace_hash = hashlib.md5(abs_path_str.encode('utf-8')).hexdigest()
    
    # 3. Establish standard directories
    mcp_cache_dir = Path("~/.cursor_graph_rag/graphs").expanduser()
    mcp_cache_dir.mkdir(parents=True, exist_ok=True)
    
    return (
        mcp_cache_dir / f"graph_{workspace_hash}.json",
        mcp_cache_dir / f"graph_{workspace_hash}.lock"
    )


def execute_preflight_lazy_sync(repo_root: Path, G: nx.DiGraph, embedder: LocalEmbeddingPipeline) -> bool:
    """
    Surgically checks file timestamps and processes embedding vectors ONLY 
    for cold-started files or elements showing active post-save deltas.
    """
    # Initialize our internal timestamp tracking map inside the graph's metadata if missing
    if 'indexed_timestamps' not in G.graph:
        G.graph['indexed_timestamps'] = {}
    
    # ──> FIXED: Wiped out the unconditional global embedder call from this line
    
    # Gather all unique file paths currently tracked across your graph nodes
    tracked_files = {data.get("file_path") for _, data in G.nodes(data=True) if data.get("file_path")}

    dirty_files_detected = False
    first_run_initialization = False

    print(f"[Sync Engine] Scanning metadata for {len(tracked_files)} files...")

    for rel_path in tracked_files:
        if not rel_path:
            continue
            
        full_path = repo_root / rel_path
        
        # Verify file existence using absolute resolution path
        if not full_path.exists():
            continue
            
        current_mtime = os.path.getmtime(full_path)
        last_indexed_time = G.graph['indexed_timestamps'].get(rel_path)
        
        # =========================================================================
        # ❄️ CHANNELS FIX A: TARGETED COLD-START EMBEDDING BACKFILL
        # =========================================================================
        if last_indexed_time is None:
            print(f"🧬 [COLD BOOT] Initializing tracking and batch vectors for: '{rel_path}'")
            
            nodes_to_encode = []
            texts_to_encode = []
            
            # 1. Collect all cold chunks needing an embedding pass
            for node_id, data in G.nodes(data=True):
                if data.get("file_path") == rel_path:
                    if "embedding" not in data or not data["embedding"]:
                        chunk_text = data.get("chunk_text", "").strip()
                        if chunk_text:
                            nodes_to_encode.append(node_id)
                            texts_to_encode.append(chunk_text)
            
            # 2. Compute embeddings in ONE single model forward pass
            if texts_to_encode:
                # The model internally parallelizes this entire array across a matrix batch
                vectors = embedder.model.encode(texts_to_encode, convert_to_numpy=True).tolist()
                
                # 3. Zip and map the vectors back to the memory graph using index alignment
                for node_id, vector in zip(nodes_to_encode, vectors):
                    G.nodes[node_id]["embedding"] = vector
                    
                print(f"   ✅ Batched {len(texts_to_encode)} embeddings for '{rel_path}' successfully.")
            
            # Secure the timestamp boundary and move to next file smoothly
            G.graph['indexed_timestamps'][rel_path] = current_mtime
            first_run_initialization = True
            continue
            
        # =========================================================================
        # 🔥 CHANNELS FIX B: INCREMENTAL DELTA UPGRADES (MANUAL EDITS / AUTOSAVE)
        # =========================================================================
        if current_mtime > last_indexed_time:
            dirty_files_detected = True
            print(f"\n[⚠️ DIRTY FILE DETECTED] Change isolated in: '{rel_path}'")
            
            # Extract what the file looks like right now on disk
            fresh_entities = extract_file_entities(rel_path, repo_root)
            
            # Pull what the graph thinks the file looks like right now
            existing_file_nodes = {
                node_id: data for node_id, data in G.nodes(data=True) 
                if data.get("file_path") == rel_path
            }
            
            # --- SUB-STEP A: EVALUATE MODIFIED & ADDED ELEMENTS ---
            for node_id, fresh_data in fresh_entities.items():
                if node_id in existing_file_nodes:
                    # If chunk text differs, it's an inline structural edit
                    if fresh_data["chunk_text"] != existing_file_nodes[node_id].get("chunk_text"):
                        print(f"  -> [MODIFIED] Element updated: `{node_id}`. Regenerating vector...")
                        fresh_data["embedding"] = embedder.model.encode(fresh_data["chunk_text"], convert_to_numpy=True).tolist()
                        G.add_node(node_id, **fresh_data)
                else:
                    # The node ID is brand new to the system
                    print(f"  -> [ADDED] New element detected: `{node_id}`. Generating vector...")
                    fresh_data["embedding"] = embedder.model.encode(fresh_data["chunk_text"], convert_to_numpy=True).tolist()
                    G.add_node(node_id, **fresh_data)
                    
            # --- SUB-STEP B: EVALUATE DELETED ELEMENTS ---
            for old_node_id in existing_file_nodes.keys():
                if old_node_id not in fresh_entities:
                    print(f"  -> [DELETED] Element missing from file: `{old_node_id}`. Purging from graph...")
                    G.remove_node(old_node_id)
                    
            # Update our tracking timestamp baseline
            G.graph['indexed_timestamps'][rel_path] = current_mtime

    if first_run_initialization:
        return first_run_initialization

    return dirty_files_detected


# =========================================================================
# 3. NATIVE MCP CODE ENVIRONMENT TOOLS REGISTRATION
# =========================================================================
@mcp.tool()
def search_codebase_intent(search_queries: list[str], active_project_root: str, targeted_symbols: list[str] = None) -> str:
    """CRITICAL CODEBASE ARCHITECTURE DISCOVERY ENGINE.
    
    Deconstructs complex multi-layered human prompts into highly targeted, 
    orthogonal, dense vector search operations to map intent to source code layers.

    WHEN TO USE THIS TOOL:
    - Invoke this at the start of EVERY repository analysis, feature implementation, or bug audit.
    - Use this to map general human feature descriptions straight to concrete code layout positions.

    QUERY GENERATION ALGORITHM FOR THE LLM:
    1. **Deconstruct Compound Intents**: Analyze the user's prompt for multi-part objectives or distinct 
       subsystems (e.g., "Where is data cached AND how are validation errors raised?").
    2. **Isolate Architectural Layers**: Break compound questions into an array of completely separate, 
       non-overlapping search queries. Generate exactly ONE dedicated query string per technical concept.
    3. **Populate with Dense Semantic Tokens**: For each query string in the list, construct a precise, 
       keyword-dense vector target using the mandatory structural formula below.

    THE SEARCH QUERY STRUCTURAL FORMULA:
    Every string inside the `search_queries` list must adhere to this strict layout:
    `[Target Component/Class/Module Noun] [Functional Action/Process Verb] [State/Exception/Error Condition]`

    - EXECUTABLE COMPLIANCE EXAMPLES:
      * Human: "Find where we check user tokens and return expired session exceptions."
        -> search_queries: ["Auth Token Validator verify authenticate", "Session Expiration timeout exception error handler"]
      * Human: "Where are database writes handled and how do we retry failed connections?"
        -> search_queries: ["Database Storage transaction write commit pool", "Connection Retry backoff circuit breaker failure"]

    - PROHIBITED ANTI-PATTERNS (DO NOT DO):
      - Never include conversational filler, greeting words, polite punctuation, or question pronouns 
        (e.g., strip 'please find', 'how do we', 'where is', 'show me').
      - Never smash completely independent architectural layers into a single query string.

    Args:
        search_queries (list[str]): A list of highly focused, atomized technical keyword vectors.
                                    Generate a unique string item for each structural subsystem or 
                                    isolated engineering concept present in the user's prompt.
        active_project_root (str): The absolute file path to the user's current active project repository.
        targeted_symbols (list[str], optional): An explicit array of case-sensitive code identifiers 
                                                (Class names, method names, standalone files) isolated 
                                                directly from the chat history. Leave as None if no symbols 
                                                are explicitly named.
    """
    # Workspace root is the project dir where cursor has been opened
    workspace_root = Path(active_project_root).resolve()
    
    # This function hashes the workspace dir value and then just gives us the graph and lock path of it
    graph_path, lock_path = get_graph_paths(workspace_root)
    
    embedder = model_manager.acquire()
    try:
        with FileLock(lock_path):
            G = GraphSerializer.load_from_json(workspace_root,graph_path)
            if execute_preflight_lazy_sync(workspace_root, G, embedder):
                GraphSerializer.save_to_json(G, workspace_root, graph_path)
                
        engine = AdvancedRetrievalEngine(graph_instance=G, embedder_instance=embedder)
        return engine.compile_context_package(search_queries=search_queries, targeted_symbols=targeted_symbols)
    finally:
        model_manager.release()


@mcp.tool()
def calculate_blast_radius(target_symbol: str, active_project_root: str) -> str:
    """CRITICAL UPSTREAM STATIC-ANALYSIS TRAVERSAL ENGINE.
    
    DO NOT ATTEMPT TO MODIFY, DELETE, OR ALTER THE SIGNATURE OF ANY CODE ELEMENT BLINDLY.
    You must invoke this tool before proposing a code modification or refactor to audit the 
    cascading regression risks across the wider codebase.

    WHAT THIS TOOL PROVIDES TO YOU:
    1. A complete recursive 'Domino Effect' impact map tracing all upstream dependent entities.
    2. The exact relative file paths where the affected elements reside.
    3. The concrete structural typings (Class vs Function) and execution footprints 
       of every single entity that will break if your target component is changed.

    WHEN TO USE THIS TOOL:
    - You MUST call this the moment you identify a target component to change, right BEFORE writing code.
    - Use this when an edit might alter an API contract, method argument list, or return type behavior.
    - Use this to guarantee zero-regression stability for multi-file code updates.

    Args:
        target_symbol (str): The single, exact, case-sensitive technical name of the code component 
                             being audited for downstream failures. 
                             
                             CRITICAL INSTRUCTIONS:
                             1. This must be a raw code identifier ONLY. 
                             2. Do NOT pass descriptive human sentences, file paths, or multi-word search phrases.
                             3. If the target is a class method, provide either the bare method name or the 
                                dot-notation format if known from history.
                             
                             - EXAMPLES:
                               * CORRECT: "resolve_redirects"
                               * CORRECT: "HTTPAdapter"
                               * CORRECT: "Session.request"
                               * INCORRECT: "the redirect function in sessions"
                               * INCORRECT: "modify the timeout configuration parameter"
                               * INCORRECT: "src/requests/sessions.py"
        
        active_project_root (str): The absolute path to the user's current active project repository.

    """
    workspace_root = Path(active_project_root).resolve()
    
    graph_path, lock_path = get_graph_paths(workspace_root)
    
    embedder = model_manager.acquire()
    try:
        with FileLock(lock_path):
            G = GraphSerializer.load_from_json(workspace_root,graph_path)
            if execute_preflight_lazy_sync(workspace_root, G, embedder):
                GraphSerializer.save_to_json(G, workspace_root, graph_path)
                
        engine = AdvancedRetrievalEngine(graph_instance=G, embedder_instance=embedder)
        
        matching_nodes = [nid for nid, d in G.nodes(data=True) if d.get("name") == target_symbol or nid == target_symbol]
        if not matching_nodes:
            return f"### [Graph-RAG System Message]\nTarget code symbol '{target_symbol}' could not be matched."
            
        impact_profiles = engine.compute_upstream_blast_radius(matching_nodes)
        
        report = f"### UPSTREAM REGRESSION AUDIT FOR SYMBOL: `{target_symbol}`\n"
        report += f"Modifying this entity creates a direct domino risk across **{len(impact_profiles)}** elements.\n\n"
        
        for item in impact_profiles:
            report += f"- **[{item['type']}]** `{item['node_id']}` inside `{item['file_path']}`\n"
            
        return report
    finally:
        model_manager.release()


@mcp.tool()
def fetch_node_source(node_id: str, active_project_root: str) -> str:
    """CRITICAL GRANULAR SOURCE EXTRACTION ENGINE.
    
    DO NOT GUESS NODE IDENTIFIERS, ASSUME LINE CONTENT BOUNDARIES, OR PROPOSE REFLEXIVE 
    CODE MODIFICATIONS BLINDLY. You must call this tool to extract the exact, raw, un-truncated 
    internal Python implementation of a code element before reviewing it or modifying it.

    WHAT THIS TOOL PROVIDES TO YOU:
    1. The absolute, un-truncated, pristine functional source code block for the target element.
    2. Deep implementation visibility (internal variable definitions, inner logic expressions, 
       and sub-method execution patterns) that were truncated or skipped in general search previews.

    WHEN TO USE THIS TOOL:
    - You MUST use this tool after identifying critical component boundaries via `search_codebase_intent`.
    - Use this right before writing a code change or comprehensive refactoring block to ensure 
      you see 100% of the internal logic details.
    - This is your final target lookup tool before delivering code modifications back to the user chat.

    Args:
        node_id (str): The exact, unique, case-sensitive internal graph identifier string.
                       
                       CRITICAL INSTRUCTIONS:
                       1. This is a strict systemic key. Do NOT pass bare human keywords, general expressions, 
                          or simple file names.
                       2. You must copy-paste this string EXACTLY as it appears inside the node_id fields 
                          returned from previous tool runs (`search_codebase_intent` or `calculate_blast_radius`).
                       3. The format must always follow our absolute ledger schema: 
                          'relative_file_path::ClassName.method_name' or 'relative_file_path::function_name'.
                       
                       - EXAMPLES:
                         * CORRECT: "src/requests/sessions.py::SessionRedirectMixin.resolve_redirects"
                         * CORRECT: "src/requests/utils.py::to_key_val_list"
                         * INCORRECT: "resolve_redirects"
                         * INCORRECT: "def resolve_redirects(self, resp)"
                         * INCORRECT: "src/requests/sessions.py"

        active_project_root (str): The absolute path to the user's current active project repository.
        
    """
    workspace_root = Path(active_project_root).resolve()
    
    graph_path, lock_path = get_graph_paths(workspace_root)

    embedder = model_manager.acquire()
    try:
        with FileLock(lock_path):
            G = GraphSerializer.load_from_json(workspace_root,graph_path)
            if execute_preflight_lazy_sync(workspace_root, G, embedder):
                GraphSerializer.save_to_json(G, workspace_root, graph_path)
                
        if G.has_node(node_id):
            node_data = G.nodes[node_id]
            node_type = node_data.get("type", "UNKNOWN")

            # ──> THE CLASS ROUTER PATCH
            if node_type == "CLASS":
                # Start with the core class signature and docstring chunk text
                source_out = f"### RAW SOURCE CONTENT FOR CLASS NODE: `{node_id}`\n\n"
                source_out += f"```python\n{node_data.get('chunk_text', '# No header content available.')}\n```\n\n"
                source_out += "### INTERFACE MAP: AVAILABLE MEMBER METHODS WITHIN THIS CLASS\n"
                source_out += "To read the operational logic of any method below, use 'fetch_node_source' with its absolute Node ID:\n"
                
                found_methods = False
                # Scan the graph for functions belonging to this class in the same file
                for sub_id, sub_data in G.nodes(data=True):
                    if sub_data.get("belongs_to_class") == node_data.get("name") and sub_data.get("file_path") == node_data.get("file_path"):
                        source_out += f"  - [METHOD] `def {sub_data.get('name')}{sub_data.get('signature', '()')}` | Node ID: `{sub_id}`\n"
                        found_methods = True
                        
                if not found_methods:
                    source_out += "  (No child methods indexed for this class block.)\n"
                return source_out

            # ──> STANDARD FUNCTION/FILE FALLBACK (Keeps your original behavior intact)
            else:
                return f"### RAW SOURCE CONTENT FOR NODE: `{node_id}`\n\n```python\n{node_data.get('chunk_text', '# No content available.')}\n```"

        return f"### [Graph-RAG System Message]\nNode ID '{node_id}' not found."
    finally:
        model_manager.release()


if __name__ == "__main__":
    mcp.run(transport="stdio")