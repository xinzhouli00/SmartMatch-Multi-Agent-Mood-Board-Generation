"""
Parity: refactored agent (pickle store) must return exactly what the original agent returns.
Runs anywhere, no database needed.
"""

import copy
import random

from agents.graph_rag.agent import GraphRAGAgent
from agents.graph_rag.graph_store import PickleGraphStore
from conftest import FakeTextAgent, make_candidates, reference

N_TRIALS = 40


def _agents(case):
    old = reference.GraphRAGAgent(graph_path=case["path"])
    new = GraphRAGAgent(store=PickleGraphStore(case["path"]))
    return old, new


def test_deduplicate_matches_original(graph_case):
    old, new = _agents(graph_case)
    rng = random.Random(graph_case["seed"])
    for _ in range(N_TRIALS):
        cands = make_candidates(rng, graph_case["all_ids"], rng.randint(1, 60))
        thresh = rng.choice([None, 0.3, 0.6, 0.9])
        assert new.deduplicate(copy.deepcopy(cands), thresh) == \
               old.deduplicate(copy.deepcopy(cands), thresh)


def test_expand_matches_original(graph_case):
    old, new = _agents(graph_case)
    rng = random.Random(100 + graph_case["seed"])
    text_agent = FakeTextAgent(graph_case["all_ids"], seed=graph_case["seed"])
    for _ in range(N_TRIALS):
        cands = make_candidates(rng, graph_case["all_ids"], rng.randint(1, 40))
        grounding = rng.choice([None, {"q": "spring blossoms"}])
        kw = dict(n_seeds=rng.choice([None, 1, 3, 8]), max_add=rng.choice([None, 2, 25]))
        assert new.expand(copy.deepcopy(cands), text_agent, grounding, **kw) == \
               old.expand(copy.deepcopy(cands), text_agent, grounding, **kw)


def test_rerank_matches_original(graph_case):
    old, new = _agents(graph_case)
    rng = random.Random(200 + graph_case["seed"])
    for _ in range(N_TRIALS):
        cands = make_candidates(rng, graph_case["all_ids"], rng.randint(1, 60))
        assert new.rerank(copy.deepcopy(cands), {}) == old.rerank(copy.deepcopy(cands), {})


def test_get_cluster_matches_original(graph_case):
    old, new = _agents(graph_case)
    for pid in graph_case["all_ids"][:50] + ["missing-id"]:
        for size in (1, 5, 10, 100):
            assert new.get_cluster(pid, size) == old.get_cluster(pid, size)


def test_full_pipeline_matches_original(graph_case):
    old, new = _agents(graph_case)
    rng = random.Random(300 + graph_case["seed"])
    text_agent = FakeTextAgent(graph_case["all_ids"], seed=graph_case["seed"])
    for _ in range(N_TRIALS):
        cands = make_candidates(rng, graph_case["all_ids"], rng.randint(5, 50))
        grounding = {"q": "beach sunset"}
        outs = []
        for agent in (old, new):
            c = agent.deduplicate(copy.deepcopy(cands))
            c = agent.expand(c, text_agent, grounding)
            c = agent.rerank(c, grounding)
            outs.append(c)
        assert outs[0] == outs[1]


def test_empty_inputs():
    # empty candidate lists short-circuit before touching the store
    new = GraphRAGAgent.__new__(GraphRAGAgent)
    assert new.deduplicate([]) == []
    assert new.expand([], text_agent=None) == []
    assert new.rerank([], {}) == []
