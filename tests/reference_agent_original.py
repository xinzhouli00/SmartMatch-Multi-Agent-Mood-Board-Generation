"""
src/agents/graph_rag/agent.py

Graph RAG Agent
----------------
Three operations on hybrid retrieval candidates:

  1. deduplicate() — remove near-duplicate images (edge weight > threshold)
                     Solves: concrete queries returning near-identical images

  2. expand()      — pull in graph neighbors of top seeds as extra candidates
                     Solves: abstract queries with few relevant results

  3. rerank()      — boost images well-connected to the candidate set
                     General coherence improvement

Recommended pipeline order:
    candidates = text_agent.merge_with_siglip(grounding, siglip_agent)
    candidates = graph_agent.deduplicate(candidates)
    candidates = graph_agent.expand(candidates, text_agent)
    candidates = graph_agent.rerank(candidates, grounding)
    verified   = verification_agent.run(candidates, grounding)
"""

from collections import defaultdict
import os
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from agents.field_text_retrieval.agent import TEXT_WEIGHT

load_dotenv()

BASE_DIR        = Path(__file__).resolve().parents[2]
GRAPH_PATH      = BASE_DIR / "data/graph/image_graph.pkl"

GRAPH_WEIGHT    = float(os.getenv("GRAPH_WEIGHT",    "0.25"))
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.60"))
EXPAND_SEEDS    = int(os.getenv("EXPAND_SEEDS",      "5"))
EXPAND_MAX_ADD  = int(os.getenv("EXPAND_MAX_ADD",    "10"))


class GraphRAGAgent:
    """
    Graph-based retrieval enhancement for mood board candidate sets.

    Usage
    -----
        agent = GraphRAGAgent()

        candidates = text_agent.merge_with_siglip(grounding, siglip_agent)
        candidates = agent.deduplicate(candidates)
        candidates = agent.expand(candidates, text_agent)
        candidates = agent.rerank(candidates, grounding)
    """

    def __init__(self, graph_path: Path | None = None):
        path = Path(graph_path) if graph_path else GRAPH_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Graph not found: {path}\n"
                f"Run: python src/data/data_prep/build_graph.py --threshold 0.0 --top_k 30"
            )
        print(f"[GraphRAG] Loading graph from {path.name}...")
        with open(path, "rb") as f:
            self.graph = pickle.load(f)
        print(f"[GraphRAG] Graph loaded — {len(self.graph)} nodes.")

    # ── Public interface ──────────────────────────────────────────────────────

    def deduplicate(
        self,
        candidates: list[dict],
        threshold: float | None = None,
    ) -> list[dict]:
        """
        Remove near-duplicate images from the candidate set.

        Two images are near-duplicates if their graph edge weight >= threshold.
        When duplicates found, keep the one with the higher score.

        Solves: concrete queries (spring blossoms, beach sunset) where top
        candidates are essentially the same shot from different angles.

        Args:
            candidates : list of dicts from hybrid retrieval
            threshold  : edge weight cutoff for duplicate detection (default 0.85)

        Returns:
            Deduplicated list, same dict format, sorted by score descending.
        """
        if not candidates:
            return candidates

        thresh      = threshold if threshold is not None else DEDUP_THRESHOLD
        kept        = []
        removed_ids = set()

        for c in candidates:
            pid = str(c["photo_id"])
            if pid in removed_ids:
                continue

            kept.append(c)

            # Mark all near-duplicates of this image for removal
            for neighbor_id, edge_weight in self.graph.get(pid, []):
                if edge_weight < thresh or neighbor_id in removed_ids:
                    continue
                # Check if neighbor is in candidate set with lower score
                for other in candidates:
                    if str(other["photo_id"]) == neighbor_id:
                        if c.get("score", 0.0) >= other.get("score", 0.0):
                            removed_ids.add(neighbor_id)
                        break

        n_removed = len(candidates) - len(kept)
        if n_removed > 0:
            print(f"[GraphRAG] Deduplicated: removed {n_removed} near-duplicate(s) "
                  f"(threshold={thresh}). {len(kept)} remain.")
        else:
            print(f"[GraphRAG] Deduplicate: no near-duplicates found "
                  f"(threshold={thresh}).")

        return kept

    def expand(
            self,
            candidates: list[dict],
            text_agent,
            grounding_output: dict | None = None,
            n_seeds: int | None = None,
            max_add: int | None = None,
        ) -> list[dict]:
            """
            Expand candidate pool by pulling in graph neighbors of top seeds.

            Takes the top n_seeds candidates as seed nodes, traverses their graph
            neighbors, and adds previously unseen neighbors as extra candidates.

            Key improvement over proxy scores:
            If grounding_output is provided, graph-expanded candidates are scored
            using the real text similarity (via text_agent._score_all) so they
            compete on the same scale as hybrid retrieval candidates.
            Without grounding_output, falls back to proxy score = seed_score * edge_weight.

            Args:
                candidates       : current hybrid retrieval candidate list
                text_agent       : FieldTextRetrievalAgent instance
                grounding_output : grounding dict for real text scoring (optional but recommended)
                n_seeds          : number of top candidates to use as seeds (default: EXPAND_SEEDS)
                max_add          : max new candidates to add (default: EXPAND_MAX_ADD)

            Returns:
                Expanded list (original + new), sorted by score descending.
                New candidates have source="graph_expand".
            """
            if not candidates:
                return candidates

            seeds   = n_seeds if n_seeds is not None else EXPAND_SEEDS
            max_new = max_add if max_add  is not None else EXPAND_MAX_ADD

            existing_ids      = {str(c["photo_id"]) for c in candidates}
            seed_nodes        = candidates[:seeds]
            new_candidates    = {}
            seed_contribution = defaultdict(int)  # tracks how many new images each seed has contributed
            MAX_PER_SEED      = 2                 # max new candidates per seed to ensure diversity

            for seed in seed_nodes:
                seed_id    = str(seed["photo_id"])
                seed_score = float(seed.get("score", 0.0))

                for neighbor_id, edge_weight in self.graph.get(seed_id, []):
                    if neighbor_id in existing_ids:
                        continue

                    proxy_score = seed_score * float(edge_weight)

                    if neighbor_id in new_candidates:
                        # Neighbor already added by another seed — update proxy score if higher
                        if proxy_score > new_candidates[neighbor_id]["_proxy"]:
                            new_candidates[neighbor_id]["_proxy"] = proxy_score
                        continue

                    # Skip if this seed has already contributed MAX_PER_SEED new candidates
                    if seed_contribution[seed_id] >= MAX_PER_SEED:
                        continue

                    # Look up metadata from text_agent
                    url     = text_agent.url_lookup.get(neighbor_id, "")
                    caption = ""
                    for row in text_agent.dataset:
                        if str(row.get("photo_id", "")) == neighbor_id:
                            caption = row.get("grounding_output", {}).get(
                                "visual_description", ""
                            )
                            break

                    new_candidates[neighbor_id] = {
                        "photo_id":     neighbor_id,
                        "image_url":    url,
                        "caption":      caption,
                        "score":        round(proxy_score, 4),
                        "siglip_score": 0.0,
                        "text_score":   0.0,
                        "graph_score":  round(float(edge_weight), 4),
                        "source":       "graph_expand",
                        "field_scores": {},
                        "_proxy":       proxy_score,   # internal, removed before returning
                    }

                    # Increment this seed contribution count
                    seed_contribution[seed_id] += 1

            # Sort by proxy score and take top max_new
            new_sorted = sorted(
                new_candidates.values(),
                key=lambda x: x["_proxy"],
                reverse=True,
            )[:max_new]

            # Replace proxy scores with real text similarity scores so expanded
            # candidates compete fairly with hybrid retrieval results.
            # _score_all scores all 25k images in one pass — fast enough to run here.
            if new_sorted and grounding_output is not None:
                try:
                    text_scores, _ = text_agent._score_all(grounding_output)

                    # Build photo_id → dataset index lookup for O(1) access
                    id_to_idx = {
                        str(row.get("photo_id", f"idx_{i}")): i
                        for i, row in enumerate(text_agent.dataset)
                    }

                    for c in new_sorted:
                        idx = id_to_idx.get(str(c["photo_id"]))
                        if idx is not None:
                            real_text_score = float(text_scores[idx])
                            c["text_score"] = round(real_text_score, 4)
                            # Use real text score as final score
                            # (no siglip since we don't have image embeddings at query time)
                            TEXT_WEIGHT = float(os.getenv("TEXT_WEIGHT", "0.3"))
                            c["score"] = round(real_text_score * TEXT_WEIGHT, 4)

                    print(f"[GraphRAG] Expanded: +{len(new_sorted)} candidates "
                        f"from graph neighbors of top-{seeds} seeds "
                        f"(real text scores assigned).")

                except Exception as e:
                    print(f"[GraphRAG] Expand: real text scoring failed ({e}) "
                        f"— falling back to proxy scores.")

            elif new_sorted:
                print(f"[GraphRAG] Expanded: +{len(new_sorted)} candidates "
                    f"from graph neighbors of top-{seeds} seeds (proxy scores).")
            else:
                print(f"[GraphRAG] Expand: no new neighbors found.")

            # Remove internal proxy field before returning
            for c in new_sorted:
                c.pop("_proxy", None)

            expanded = candidates + new_sorted
            expanded.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return expanded

    def rerank(
        self,
        candidates: list[dict],
        grounding_output: dict,
    ) -> list[dict]:
        """
        Rerank candidates by combining hybrid score with graph connectivity.

        Graph connectivity score = average edge weight to OTHER candidates
        in this set. Images well-connected to peers score higher.

        Args:
            candidates       : list of candidate dicts
            grounding_output : unused, reserved for future use

        Returns:
            Same list with graph_score updated and score recomputed, sorted descending.
        """
        if not candidates:
            return candidates

        candidate_ids = {str(c["photo_id"]) for c in candidates}

        scored = []
        for c in candidates:
            pid          = str(c["photo_id"])
            hybrid_score = float(c.get("score", 0.0))
            graph_score  = self._connectivity_score(pid, candidate_ids)
            final_score  = (1 - GRAPH_WEIGHT) * hybrid_score + GRAPH_WEIGHT * graph_score

            scored.append({
                **c,
                "graph_score": round(graph_score, 4),
                "score":       round(final_score, 4),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        print(f"[GraphRAG] Reranked {len(scored)} candidates. "
              f"Top: {scored[0]['score']:.4f} (graph={scored[0]['graph_score']:.4f})")

        return scored

    def get_cluster(self, photo_id: str, max_size: int = 10) -> list[str]:
        """Return top neighbors of an image sorted by edge weight."""
        return [pid for pid, _ in self.graph.get(photo_id, [])[:max_size]]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connectivity_score(self, photo_id: str, candidate_ids: set) -> float:
        """Average edge weight to other candidates in the set."""
        neighbors = self.graph.get(photo_id, [])
        connected = [
            w for nid, w in neighbors
            if nid in candidate_ids and nid != photo_id
        ]
        return float(np.mean(connected)) if connected else 0.0