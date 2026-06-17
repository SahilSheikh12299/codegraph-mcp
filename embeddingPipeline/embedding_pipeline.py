import json
from pathlib import Path
import networkx as nx
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
import sys
import torch
import threading
import os
import contextlib
import gc


RERANKER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-base-v2"


class LocalEmbeddingPipeline:
    """Encodes CodeGraph text chunks into high-dimensional vector embeddings

    completely locally using vanilla, top-tier open-source models.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        print(f"[Embedding Engine] Initializing local model: {model_name}...")
        # No trust_remote_code or model_kwargs needed! It's a standard vanilla architecture.
        self.model = SentenceTransformer(model_name)
        print("[Embedding Engine] Model loaded successfully and ready for encoding.")

    def embed_graph_nodes(self, graph: nx.DiGraph) -> int:
        """Extracts text chunks from all nodes, encodes them in optimized batches,

        and binds the resulting vector arrays back into the graph properties.
        """
        nodes_to_encode = []
        texts_to_encode = []

        for node_id, data in graph.nodes(data=True):
            text = (data.get("embedding_text") or data.get("chunk_text") or "").strip()
            if text:
                nodes_to_encode.append(node_id)
                texts_to_encode.append(text)

        if not texts_to_encode:
            print("[Warning] No text chunks discovered in the graph.")
            return 0

        print(f"[Embedding Engine] Batch encoding {len(texts_to_encode)} chunks...")
        
        embeddings = self.model.encode(
            texts_to_encode, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print("[Embedding Engine] Mapping vectorized footprints back into NetworkX nodes...")
        for node_id, embedding_array in zip(nodes_to_encode, embeddings):
            graph.nodes[node_id]["embedding"] = embedding_array.tolist()

        return len(nodes_to_encode)


class EmbeddingModelLifecycleManager:
    """Thread-safe on-demand lifecycle manager for heavy deep learning models."""
    def __init__(self):
        self.model = None
        self.reranker = None
        self.active_tasks = 0
        self.lock = threading.Lock()

    def acquire(self) -> LocalEmbeddingPipeline:
        """Safely bumps reference counter and lazy-loads the model if missing."""
        with self.lock:
            if self.model is None:
                # Safeguard: Redirect stdout to stderr temporarily to protect the JSON channel
                old_stdout = sys.stdout
                sys.stdout = sys.stderr
                try:
                    # Muzzle ALL standard prints and progress bars from showing up as red errors
                    with open(os.devnull, "w") as fnull:
                        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                            self.model = LocalEmbeddingPipeline(model_name="BAAI/bge-large-en-v1.5")
                except Exception as e:
                    print(f"[CRITICAL FAIL] Model load crashed: {e}", file=sys.stderr)
                    raise e
                finally:
                    sys.stdout = old_stdout
            self.active_tasks += 1
            return self.model

    def acquire_reranker(self) -> CrossEncoder:
        """Lazy-load cross-encoder for search reranking."""
        with self.lock:
            if self.reranker is None:
                old_stdout = sys.stdout
                sys.stdout = sys.stderr
                try:
                    with open(os.devnull, "w") as fnull:
                        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                            self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
                except Exception as e:
                    print(f"[CRITICAL FAIL] Reranker load crashed: {e}", file=sys.stderr)
                    raise e
                finally:
                    sys.stdout = old_stdout
            return self.reranker

    def release(self):
        """Reduces usage reference. Completely offloads model if no jobs remain."""
        with self.lock:
            self.active_tasks -= 1
            if self.active_tasks <= 0:
                self.model = None
                self.reranker = None
                self.active_tasks = 0
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except ImportError:
                    pass