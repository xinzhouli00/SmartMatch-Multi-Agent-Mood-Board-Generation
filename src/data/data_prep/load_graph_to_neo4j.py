"""
src/data/data_prep/load_graph_to_neo4j.py

Load the image-similarity graph built by build_graph.py (image_graph.pkl) into Neo4j.

Graph model
    (:Image {photo_id})-[:SIMILAR {weight, rank}]->(:Image {photo_id})

    rank = position in the original adjacency list (0 = strongest neighbor), so
    Neo4j returns neighbors in exactly the order the pickle had them.

Usage
    export NEO4J_URI=neo4j+s://<id>.databases.neo4j.io
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=...
    python src/data/data_prep/load_graph_to_neo4j.py --reset
    python src/data/data_prep/load_graph_to_neo4j.py --graph data/graph/image_graph.pkl --batch 10000

Writes in batches with UNWIND so ~25K nodes / ~750K edges go in a few dozen
round trips instead of one query per edge.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# same base dir as agents/graph_rag/agent.py (parents[2] of this file = src/)
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = BASE_DIR / "data/graph/image_graph.pkl"


def chunks(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def load_graph(driver, graph: dict, label: str = "Image", batch: int = 10000,
               reset: bool = False, database: str | None = None) -> dict:
    if not label.isidentifier():
        raise ValueError(f"Unsafe label: {label!r}")

    def run(query, **params):
        return driver.execute_query(query, params, database_=database)

    if reset:
        # delete in slices so a big graph doesn't blow the transaction memory limit.
        # CALL {...} IN TRANSACTIONS only works in an auto-commit transaction, so this
        # one goes through session.run(), not execute_query() (which wraps a managed tx).
        with driver.session(database=database) as session:
            session.run(f"MATCH (n:{label}) CALL (n) {{ DETACH DELETE n }} "
                        f"IN TRANSACTIONS OF 5000 ROWS").consume()

    run(f"CREATE CONSTRAINT {label.lower()}_photo_id IF NOT EXISTS "
        f"FOR (i:{label}) REQUIRE i.photo_id IS UNIQUE")

    # every id that appears anywhere, including neighbor-only ids
    all_ids = set(map(str, graph))
    for nbrs in graph.values():
        all_ids.update(str(nid) for nid, _ in nbrs)
    node_rows = [{"id": pid} for pid in sorted(all_ids)]

    edge_rows = []
    for pid, nbrs in graph.items():
        for rank, (nid, w) in enumerate(nbrs):
            edge_rows.append({"src": str(pid), "dst": str(nid),
                              "w": float(w), "rank": rank})

    t0 = time.time()
    for part in chunks(node_rows, batch):
        run(f"UNWIND $rows AS row MERGE (:{label} {{photo_id: row.id}})", rows=part)
    t1 = time.time()
    for part in chunks(edge_rows, batch):
        run(f"""
            UNWIND $rows AS row
            MATCH (a:{label} {{photo_id: row.src}})
            MATCH (b:{label} {{photo_id: row.dst}})
            MERGE (a)-[r:SIMILAR {{rank: row.rank}}]->(b)
            SET r.weight = row.w
            """, rows=part)
    t2 = time.time()

    return {"nodes": len(node_rows), "edges": len(edge_rows),
            "node_secs": round(t1 - t0, 1), "edge_secs": round(t2 - t1, 1)}


def verify(driver, graph: dict, label: str = "Image", database: str | None = None) -> None:
    """Row-count check: what went in must match what the pickle holds."""
    expected_edges = sum(len(v) for v in graph.values())
    records, _, _ = driver.execute_query(
        f"MATCH (:{label})-[r:SIMILAR]->(:{label}) RETURN count(r) AS n", database_=database)
    got = records[0]["n"]
    status = "OK" if got == expected_edges else "MISMATCH"
    print(f"[verify] edges in Neo4j = {got}, in pickle = {expected_edges} -> {status}")
    if got != expected_edges:
        raise SystemExit(1)


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default=str(DEFAULT_GRAPH))
    p.add_argument("--label", default="Image")
    p.add_argument("--batch", type=int, default=10000)
    p.add_argument("--reset", action="store_true", help="delete existing :<label> nodes first")
    args = p.parse_args()

    with open(args.graph, "rb") as f:
        graph = pickle.load(f)

    database = os.getenv("NEO4J_DATABASE") or None
    with GraphDatabase.driver(os.environ["NEO4J_URI"],
                              auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])) as driver:
        driver.verify_connectivity()
        stats = load_graph(driver, graph, label=args.label, batch=args.batch,
                           reset=args.reset, database=database)
        print(f"[load] {stats['nodes']} nodes in {stats['node_secs']}s, "
              f"{stats['edges']} edges in {stats['edge_secs']}s")
        verify(driver, graph, label=args.label, database=database)


if __name__ == "__main__":
    main()
