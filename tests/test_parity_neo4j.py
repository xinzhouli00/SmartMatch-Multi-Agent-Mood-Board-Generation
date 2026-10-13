"""
Parity: Neo4j store must answer every graph question exactly like the pickle store.
Needs a live Neo4j (AuraDB Free works). Skipped unless NEO4J_URI is set.

    export NEO4J_URI=neo4j+s://<id>.databases.neo4j.io NEO4J_USER=neo4j NEO4J_PASSWORD=...
    pytest tests/test_parity_neo4j.py -v

Uses its own label (:ParityTestImage) and deletes it afterwards, so it never
touches the real :Image graph.
"""

import copy
import os
import random

import pytest

from agents.graph_rag.agent import GraphRAGAgent
from agents.graph_rag.graph_store import Neo4jGraphStore, PickleGraphStore
from conftest import FakeTextAgent, make_candidates

pytestmark = pytest.mark.skipif(not os.getenv("NEO4J_URI"), reason="NEO4J_URI not set")

LABEL = "ParityTestImage"
N_TRIALS = 15


@pytest.fixture
def stores(graph_case):
    from data.data_prep.load_graph_to_neo4j import load_graph

    neo = Neo4jGraphStore(label=LABEL)
    load_graph(neo.driver, graph_case["graph"], label=LABEL, batch=2000,
               reset=True, database=neo.database)
    yield PickleGraphStore(graph_case["path"]), neo
    with neo.driver.session(database=neo.database) as s:
        s.run(f"MATCH (n:{LABEL}) DETACH DELETE n").consume()
        s.run(f"DROP CONSTRAINT {LABEL.lower()}_photo_id IF EXISTS").consume()
    neo.close()


def test_store_answers_match(graph_case, stores):
    pkl, neo = stores
    assert neo.node_count() == len(graph_case["all_ids"])   # includes neighbor-only ids
    rng = random.Random(graph_case["seed"])
    for _ in range(N_TRIALS):
        ids = rng.sample(graph_case["all_ids"], rng.randint(1, 40)) + ["missing-id"]
        assert neo.neighbors(ids) == pkl.neighbors(ids)
        assert neo.neighbors(ids, limit=5) == pkl.neighbors(ids, limit=5)
        t = rng.choice([0.3, 0.6, 0.9])
        got, want = neo.edges_within(ids, t), pkl.edges_within(ids, t)
        assert {k: v for k, v in got.items() if v} == {k: v for k, v in want.items() if v}
        got_c, want_c = neo.connectivity(ids), pkl.connectivity(ids)
        for pid in set(ids):
            assert got_c.get(pid, 0.0) == pytest.approx(want_c.get(pid, 0.0), abs=1e-9)


def test_agent_pipeline_matches(graph_case, stores):
    pkl, neo = stores
    a_pkl, a_neo = GraphRAGAgent(store=pkl), GraphRAGAgent(store=neo)
    text_agent = FakeTextAgent(graph_case["all_ids"], seed=graph_case["seed"])
    rng = random.Random(50 + graph_case["seed"])
    for _ in range(N_TRIALS):
        cands = make_candidates(rng, graph_case["all_ids"], rng.randint(5, 50))
        outs = []
        for agent in (a_pkl, a_neo):
            c = agent.deduplicate(copy.deepcopy(cands))
            c = agent.expand(c, text_agent, {"q": "x"})
            c = agent.rerank(c, {"q": "x"})
            outs.append(c)
        assert outs[0] == outs[1]
