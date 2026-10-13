"""
src/agents/graph_rag/graph_store.py

Storage layer for the image-similarity graph used by GraphRAGAgent.

The agent only needs four questions answered about the graph, so both backends
implement the same four methods:

    neighbors(ids)            -> {pid: [(nid, weight), ...]}  in rank order
    edges_within(ids, min_w)  -> {pid: [(nid, weight), ...]}  only nids in ids, weight >= min_w
    connectivity(ids)         -> {pid: mean weight to OTHER ids}  (missing pid = 0.0)
    node_count()              -> int

"Rank order" = the order of the original adjacency list written by
build_graph.py (strongest neighbor first). Neo4j stores it as r.rank so
results come back in exactly the same order as the pickle, including ties.

Backends
--------
PickleGraphStore : original behaviour, whole graph loaded into a Python dict
Neo4jGraphStore  : graph lives in Neo4j, every lookup is a Cypher query
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np


class PickleGraphStore:
    """Original backend: {photo_id: [(neighbor_id, weight), ...]} held in memory."""

    def __init__(self, graph_path: Path):
        path = Path(graph_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Graph not found: {path}\n"
                f"Run: python src/data/data_prep/build_graph.py --threshold 0.0 --top_k 30"
            )
        with open(path, "rb") as f:
            self.graph: dict[str, list[tuple[str, float]]] = pickle.load(f)

    def node_count(self) -> int:
        return len(self.graph)

    def neighbors(self, ids, limit: int | None = None):
        out = {}
        for pid in ids:
            nbrs = self.graph.get(pid, [])
            out[pid] = list(nbrs[:limit] if limit is not None else nbrs)
        return out

    def edges_within(self, ids, min_weight: float):
        id_set = set(ids)
        return {
            pid: [(nid, w) for nid, w in self.graph.get(pid, [])
                  if nid in id_set and w >= min_weight]
            for pid in id_set
        }

    def connectivity(self, ids):
        id_set = set(ids)
        out = {}
        for pid in id_set:
            ws = [w for nid, w in self.graph.get(pid, [])
                  if nid in id_set and nid != pid]
            out[pid] = float(np.mean(ws)) if ws else 0.0
        return out

    def close(self):
        pass


class Neo4jGraphStore:
    """
    Neo4j backend. Graph model (written by load_graph_to_neo4j.py):

        (:Image {photo_id})-[:SIMILAR {weight, rank}]->(:Image {photo_id})

    Edges are directed, one per entry in the original adjacency list, so a
    top-k graph that isn't symmetric stays exactly as it was built.
    """

    def __init__(self, uri=None, user=None, password=None,
                 database=None, label: str = "Image"):
        from neo4j import GraphDatabase  # imported lazily so pickle mode needs no driver

        if not label.isidentifier():
            raise ValueError(f"Unsafe label: {label!r}")  # label is formatted into Cypher
        self.label = label
        self.database = database or os.getenv("NEO4J_DATABASE") or None
        self.driver = GraphDatabase.driver(
            uri or os.environ["NEO4J_URI"],
            auth=(user or os.environ["NEO4J_USER"],
                  password or os.environ["NEO4J_PASSWORD"]),
        )
        self.driver.verify_connectivity()

    # -- helpers ---------------------------------------------------------------

    def _run(self, query: str, **params):
        records, _, _ = self.driver.execute_query(
            query, params, database_=self.database, routing_="r"
        )
        return records

    @staticmethod
    def _group(records, ids):
        out = {pid: [] for pid in ids}
        for rec in records:  # records arrive ordered by pid, rank
            out[rec["pid"]].append((rec["nid"], float(rec["w"])))
        return out

    # -- interface -------------------------------------------------------------

    def node_count(self) -> int:
        rec = self._run(f"MATCH (i:{self.label}) RETURN count(i) AS n")
        return int(rec[0]["n"])

    def neighbors(self, ids, limit: int | None = None):
        ids = list(dict.fromkeys(ids))
        if limit is None:
            q = f"""
            UNWIND $ids AS pid
            MATCH (:{self.label} {{photo_id: pid}})-[r:SIMILAR]->(b:{self.label})
            RETURN pid, b.photo_id AS nid, r.weight AS w
            ORDER BY pid, r.rank
            """
            return self._group(self._run(q, ids=ids), ids)
        q = f"""
        UNWIND $ids AS pid
        CALL (pid) {{
            MATCH (:{self.label} {{photo_id: pid}})-[r:SIMILAR]->(b:{self.label})
            RETURN b.photo_id AS nid, r.weight AS w, r.rank AS rank
            ORDER BY rank LIMIT $limit
        }}
        RETURN pid, nid, w
        ORDER BY pid, rank
        """
        return self._group(self._run(q, ids=ids, limit=int(limit)), ids)

    def edges_within(self, ids, min_weight: float):
        ids = list(dict.fromkeys(ids))
        q = f"""
        MATCH (a:{self.label})-[r:SIMILAR]->(b:{self.label})
        WHERE a.photo_id IN $ids AND b.photo_id IN $ids AND r.weight >= $min_w
        RETURN a.photo_id AS pid, b.photo_id AS nid, r.weight AS w
        ORDER BY pid, r.rank
        """
        return self._group(self._run(q, ids=ids, min_w=float(min_weight)), ids)

    def connectivity(self, ids):
        # Neo4j does the filtering; the mean is taken in numpy over weights in rank
        # order so the float result is bit-identical to the pickle path (Cypher avg()
        # sums in a different order and can differ in the last digit).
        ids = list(dict.fromkeys(ids))
        q = f"""
        MATCH (a:{self.label})-[r:SIMILAR]->(b:{self.label})
        WHERE a.photo_id IN $ids AND b.photo_id IN $ids AND a <> b
        WITH a, r ORDER BY r.rank
        RETURN a.photo_id AS pid, collect(r.weight) AS ws
        """
        out = {pid: 0.0 for pid in ids}
        for rec in self._run(q, ids=ids):
            out[rec["pid"]] = float(np.mean(rec["ws"]))
        return out

    def close(self):
        self.driver.close()


def make_store(backend: str | None = None, graph_path: Path | None = None, **kwargs):
    """GRAPH_BACKEND=neo4j | pickle (default pickle, so nothing breaks without a DB)."""
    backend = (backend or os.getenv("GRAPH_BACKEND", "pickle")).lower()
    if backend == "neo4j":
        return Neo4jGraphStore(**kwargs)
    if backend == "pickle":
        return PickleGraphStore(graph_path)
    raise ValueError(f"Unknown GRAPH_BACKEND: {backend!r}")
