import sys
from pathlib import Path

# Ensure the current directory is in the Python path so we can import our modules
sys.path.append(str(Path(__file__).parent.resolve()))

try:
    # Import the fixed functions directly from your server file
    from mcp_server import calculate_blast_radius, search_codebase_intent, get_graph_paths
    from graph_io import GraphSerializer
except ImportError as e:
    print(f"❌ Import Error: Make sure this script is placed next to your mcp_server.py and graph_io.py files.")
    print(f"Details: {e}")
    sys.exit(1)

# =========================================================================
# CONFIGURATION PANEL (Adjust these paths to match your machine setup)
# =========================================================================
# The absolute path to the local repository you want to test/index
TEST_PROJECT_ROOT = "/Users/sahilsheikh/Documents/Djikstra-codebase/repo-understanding-engine/test_repos/requests"
TARGET_SYMBOL_TO_AUDIT = "Session"
SEARCH_QUERY_TEST = ["How does Session retry on 429 status codes?"]

def run_system_audit():
    print("=" * 80)
    print("🚀 STARTING GRAPH-RAG ENGINE LOCAL INTEGRATION AUDIT")
    print("=" * 80)
    
    project_path = Path(TEST_PROJECT_ROOT)
    if not project_path.exists():
        print(f"❌ ERROR: Configured project path does not exist:\n   {TEST_PROJECT_ROOT}")
        print("Please update TEST_PROJECT_ROOT inside this script to a valid local repo path.")
        return

    # -------------------------------------------------------------------------
    # 🔍 STEP 1: Verify Cache File Paths & Hashing Alignment
    # -------------------------------------------------------------------------
    print("\n▶️ STEP 1: Verifying Cache File Locations...")
    try:
        json_path, lock_path = get_graph_paths(TEST_PROJECT_ROOT)
        print(f"  📂 Target Graph Cache File: {json_path}")
        print(f"  🔒 Target Graph Lock File:  {lock_path}")
        
        # Check if a cache file already exists from previous buggy runs
        if json_path.exists():
            print(f"  ⚠️  Found an existing graph cache file ({json_path.stat().st_size / 1024:.2f} KB).")
            print("  💡 Recommendation: If testing a cold boot, delete this JSON file manually first!")
    except Exception as e:
        print(f"  ❌ Error calculating cache paths: {e}")
        return

    # -------------------------------------------------------------------------
    # 🏗️ STEP 2: Test Cold Ingestion via Search Engine
    # -------------------------------------------------------------------------
    print("\n▶️ STEP 2: Testing Codebase Search & Automatic Graph Hydration...")
    print("  (This will trigger the AST Manifest Parser and 2-Phase Build if needed...)")
    
    try:
        search_output = search_codebase_intent(
            search_queries=SEARCH_QUERY_TEST,
            active_project_root=TEST_PROJECT_ROOT,
            targeted_symbols=[TARGET_SYMBOL_TO_AUDIT]
        )
        print("  🟢 Search Intent Executed Successfully!")
        print("-" * 50)
        print("📄 BRIEF SAMPLE OF RETRIEVAL OUTPUT LAYER:")
        print("\n".join(search_output.split("\n")[:12])) # Print just the top context packet headers
        print("...")
        print("-" * 50)
    except Exception as e:
        print(f"  ❌ Crash during Search / Ingestion Pipeline!")
        import traceback
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # 🕸️ STEP 3: Validate the Edges via Blast Radius Engine
    # -------------------------------------------------------------------------
    print("\n▶️ STEP 3: Executing Upstream Blast Radius Dependency Analysis...")
    print(f"  Auditing target token: `{TARGET_SYMBOL_TO_AUDIT}`")
    
    try:
        # Load the newly synchronized graph into memory manually to inspect its dimensions
        G = GraphSerializer.load_from_json(project_path, json_path)
        print(f"  📊 Graph Metrics Post-Sync:")
        print(f"     • Total Nodes Present: {G.number_of_nodes()}")
        print(f"     • Total Edges Linked: {G.number_of_edges()}")
        
        if G.number_of_edges() == 0:
            print("     ❌ FATAL: Graph has 0 edges. The Phase 2 Edge-Weaving engine is still failing to connect the nodes.")
        else:
            print("     ✅ SUCCESS: Edge-weaving connected highways across your code components.")

        print("-" * 50)
        # Execute the primary tool call with your new standardized argument ordering
        report = calculate_blast_radius(
            active_project_root=TEST_PROJECT_ROOT,
            target_symbol=TARGET_SYMBOL_TO_AUDIT
        )
        print("📊 LIVE BLAST RADIUS OUTPUT REPORT:")
        print(report)
        print("-" * 50)
        
    except Exception as e:
        print(f"  ❌ Crash during Blast Radius Execution Phase!")
        import traceback
        traceback.print_exc()
        return

    print("\n=" * 80)
    print("🏁 SYSTEM INTEGRATION AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_system_audit()