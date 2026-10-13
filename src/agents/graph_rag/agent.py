"""
src/agents/graph_rag/agent.py

Graph RAG Agent (Neo4j-backed)
------------------------------
Same three operations and same public interface as the original pickle agent:

  1. deduplicate() — remove near-duplicate images (edge weight >= threshold)
  2. expand()      — pull in graph neighbors of top seeds as extra candidates
  3. rerank()      — boost images well-connected to the candidate set

What changed:
  * The graph is no longer unpickled into memory on startup. Each operation asks
    the graph store only for the edges it needs (GRAPH_BACKEND=neo4j sends one
    Cypher query per operation instead of loading all 25K nodes).
  * deduplicate() no longer rescans the full candidate list for every neighbor
    (was O(n^2)); a photo_id -> candidate lookup makes it O(n + edges).
  * expand() builds the caption lookup once instead of scanning the 25K-row
    dataset for every new neighbor.

Results are identical to the original agent; tests/test_parity_*.py check that.

Recommended pipeline order (unchanged):
    candidates = text_agent.merge_with_siglip(grounding, siglip_agent)
    candidates = graph_agent.deduplicate(candidates)
    candidates = graph_agent.expand(candidates, text_agent, grounding)
    candidates = graph_agent.rerank(candidates, grounding)
    verified   = verification_agent.run(candidates, grounding)
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from agents.graph_rag.graph_store import make_store

load_dotenv()

BASE_DIR        = Path(__file__).resolve().parents[2]
GRAPH_PATH      = BASE_DIR / "data/graph/image_graph.pkl"

GRAPH_WEIGHT    = float(os.getenv("GRAPH_WEIGHT",    "0.25"))
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.60"))
EXPAND_SEEDS    = int(os.getenv("EXPAND_SEEDS",      "5"))
EXPAND_MAX_ADD  = int(os.getenv("EXPAND_MAX_ADD",    "10"))
MAX_PER_SEED    = 2   # max new candidates per seed, keeps expansion diverse


class GraphRAGAgent:
    """
    Graph-based retrieval enhancement for mood board candidate sets.

    Usage
    -----
        agent = GraphRAGAgent()                   # backend from GRAPH_BACKEND
        agent = GraphRAGAgent(backend="neo4j")    # or pick explicitly
        agent = GraphRAGAgent(store=my_store)     # or inject one (tests)
    """

    def __init__(self, graph_path: Path | None = None, backend: str | None = None,
                 store=None, **store_kwargs):
        self.store = store or make_store(
            backend, graph_path=Path(graph_path) if graph_path else GRAPH_PATH,
            **store_kwargs,
        )
        self._caption_cache_for = None   # id(dataset) the cache was built from
        self._caption_cache: dict[str, str] = {}
        print(f"[GraphRAG] Graph store ready ({type(self.store).__name__}) — "
              f"{self.store.node_count()} nodes.")

    def close(self):
        self.store.close()

    # ── Public interface ──────────────────────────────────────────────────────

    def deduplicate(self, candidates: list[dict], threshold: float | None = None) -> list[dict]:
        """
        Remove near-duplicate images from the candidate set.

        Two images are near-duplicates if their graph edge weight >= threshold
        (default DEDUP_THRESHOLD, 0.60). Walking candidates in order, each kept
        image removes its near-duplicate neighbors that score no higher than it.
        """
        if not candidates:
            return candidates

        thresh = threshold if threshold is not None else DEDUP_THRESHOLD
        ids    = [str(c["photo_id"]) for c in candidates]
        close_pairs = self.store.edges_within(ids, thresh)   # one query for the whole set

        first_by_id = {}
        for c in candidates:
            first_by_id.setdefault(str(c["photo_id"]), c)

        kept, removed_ids = [], set()
        for c in candidates:
            pid = str(c["photo_id"])
            if pid in removed_ids:
                continue
            kept.append(c)
            for neighbor_id, _w in close_pairs.get(pid, []):
                if neighbor_id in removed_ids:
                    continue
                other = first_by_id[neighbor_id]
                if c.get("score", 0.0) >= other.get("score", 0.0):
                    removed_ids.add(neighbor_id)

        n_removed = len(candidates) - len(kept)
        if n_removed > 0:
            print(f"[GraphRAG] Deduplicated: removed {n_removed} near-duplicate(s) "
                  f"(threshold={thresh}). {len(kept)} remain.")
        else:
            print(f"[GraphRAG] Deduplicate: no near-duplicates found (threshold={thresh}).")
        return kept

    def expand(self, candidates: list[dict], text_agent, grounding_output: dict | None = None,
               n_seeds: int | None = None, max_add: int | None = None) -> list[dict]:
        """
        Expand candidate pool by pulling in graph neighbors of the top seeds.

        With grounding_output, new candidates get real text-similarity scores so
        they compete on the same scale as hybrid retrieval results; otherwise
        they keep proxy score = seed_score * edge_weight.
        """
        if not candidates:
            return candidates

        seeds   = n_seeds if n_seeds is not None else EXPAND_SEEDS
        max_new = max_add if max_add is not None else EXPAND_MAX_ADD

        existing_ids      = {str(c["photo_id"]) for c in candidates}
        seed_nodes        = candidates[:seeds]
        seed_ids          = [str(s["photo_id"]) for s in seed_nodes]
        seed_neighbors    = self.store.neighbors(seed_ids)       # one query for all seeds
        captions          = self._captions(text_agent)
        new_candidates    = {}
        seed_contribution = defaultdict(int)

        for seed in seed_nodes:
            seed_id    = str(seed["photo_id"])
            seed_score = float(seed.get("score", 0.0))

            for neighbor_id, edge_weight in seed_neighbors.get(seed_id, []):
                if neighbor_id in existing_ids:
                    continue

                proxy_score = seed_score * float(edge_weight)

                if neighbor_id in new_candidates:
                    # already added by another seed: keep the higher proxy score
                    if proxy_score > new_candidates[neighbor_id]["_proxy"]:
                        new_candidates[neighbor_id]["_proxy"] = proxy_score
                    continue

                if seed_contribution[seed_id] >= MAX_PER_SEED:
                    continue

                new_candidates[neighbor_id] = {
                    "photo_id":     neighbor_id,
                    "image_url":    text_agent.url_lookup.get(neighbor_id, ""),
                    "caption":      captions.get(neighbor_id, ""),
                    "score":        round(proxy_score, 4),
                    "siglip_score": 0.0,
                    "text_score":   0.0,
                    "graph_score":  round(float(edge_weight), 4),
                    "source":       "graph_expand",
                    "field_scores": {},
                    "_proxy":       proxy_score,
                }
                seed_contribution[seed_id] += 1

        new_sorted = sorted(new_candidates.values(), key=lambda x: x["_proxy"],
                            reverse=True)[:max_new]

        if new_sorted and grounding_output is not None:
            try:
                text_scores, _ = text_agent._score_all(grounding_output)
                id_to_idx = {
                    str(row.get("photo_id", f"idx_{i}")): i
                    for i, row in enumerate(text_agent.dataset)
                }
                text_weight = float(os.getenv("TEXT_WEIGHT", "0.3"))
                for c in new_sorted:
                    idx = id_to_idx.get(str(c["photo_id"]))
                    if idx is not None:
                        real_text_score = float(text_scores[idx])
                        c["text_score"] = round(real_text_score, 4)
                        c["score"]      = round(real_text_score * text_weight, 4)
                print(f"[GraphRAG] Expanded: +{len(new_sorted)} candidates from graph "
                      f"neighbors of top-{seeds} seeds (real text scores assigned).")
            except Exception as e:
                print(f"[GraphRAG] Expand: real text scoring failed ({e}) "
                      f"— falling back to proxy scores.")
        elif new_sorted:
            print(f"[GraphRAG] Expanded: +{len(new_sorted)} candidates from graph "
                  f"neighbors of top-{seeds} seeds (proxy scores).")
        else:
            print("[GraphRAG] Expand: no new neighbors found.")

        for c in new_sorted:
            c.pop("_proxy", None)

        expanded = candidates + new_sorted
        expanded.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return expanded

    def rerank(self, candidates: list[dict], grounding_output: dict) -> list[dict]:
        """
        Rerank by blending hybrid score with graph connectivity
        (average edge weight to the OTHER candidates in this set).
        """
        if not candidates:
            return candidates

        conn = self.store.connectivity([str(c["photo_id"]) for c in candidates])

        scored = []
        for c in candidates:
            graph_score = conn.get(str(c["photo_id"]), 0.0)
            final_score = (1 - GRAPH_WEIGHT) * float(c.get("score", 0.0)) + GRAPH_WEIGHT * graph_score
            scored.append({**c, "graph_score": round(graph_score, 4),
                           "score": round(final_score, 4)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        print(f"[GraphRAG] Reranked {len(scored)} candidates. "
              f"Top: {scored[0]['score']:.4f} (graph={scored[0]['graph_score']:.4f})")
        return scored

    def get_cluster(self, photo_id: str, max_size: int = 10) -> list[str]:
        """Return top neighbors of an image by edge weight."""
        pid = str(photo_id)
        return [nid for nid, _ in self.store.neighbors([pid], limit=max_size).get(pid, [])]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _captions(self, text_agent) -> dict[str, str]:
        """photo_id -> visual_description, first matching row wins (as before)."""
        if self._caption_cache_for != id(text_agent.dataset):
            cache = {}
            for row in text_agent.dataset:
                cache.setdefault(
                    str(row.get("photo_id", "")),
                    row.get("grounding_output", {}).get("visual_description", ""),
                )
            self._caption_cache, self._caption_cache_for = cache, id(text_agent.dataset)
        return self._caption_cache
