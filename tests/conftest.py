"""
Shared test fixtures: a synthetic similarity graph shaped like build_graph.py's
output, a fake text agent, and the ORIGINAL pickle agent as the reference.
"""

from __future__ import annotations

import importlib.util
import pickle
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The original agent imports TEXT_WEIGHT from the text-retrieval agent; stub it so
# the reference file can be imported without the rest of the project.
_stub = types.ModuleType("agents.field_text_retrieval.agent")
_stub.TEXT_WEIGHT = 0.3
sys.modules.setdefault("agents.field_text_retrieval", types.ModuleType("agents.field_text_retrieval"))
sys.modules["agents.field_text_retrieval.agent"] = _stub

_spec = importlib.util.spec_from_file_location(
    "reference_agent_original", Path(__file__).with_name("reference_agent_original.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def make_graph(seed: int, n_nodes: int = 300, top_k: int = 15) -> dict:
    """
    Adjacency list {pid: [(nid, weight), ...]} sorted strongest-first, like build_graph.py.
    Deliberately includes the awkward cases: tied weights, asymmetric edges,
    neighbor-only ids that have no adjacency list of their own, and self-loops.
    """
    rng = random.Random(seed)
    ids = [f"p{i}" for i in range(n_nodes)]
    extra = [f"x{i}" for i in range(20)]            # appear only as neighbors
    graph = {}
    for pid in ids:
        k = rng.randint(0, top_k)
        nbrs = rng.sample(ids + extra, k)
        if rng.random() < 0.02:
            nbrs.append(pid)                        # self-loop
        weights = [round(rng.uniform(0.1, 0.99), 2) for _ in nbrs]  # 2dp -> ties
        pairs = sorted(zip(nbrs, weights), key=lambda t: t[1], reverse=True)
        graph[pid] = pairs
    return graph


def make_candidates(rng: random.Random, ids: list[str], n: int) -> list[dict]:
    picks = rng.sample(ids, n)
    if rng.random() < 0.3:
        picks.append(picks[0])                      # duplicate photo_id
    cands = [{"photo_id": pid, "score": round(rng.uniform(0, 1), 3), "source": "hybrid"}
             for pid in picks]
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


class FakeTextAgent:
    def __init__(self, all_ids, seed=0):
        rng = random.Random(seed)
        self.dataset = [{"photo_id": pid,
                         "grounding_output": {"visual_description": f"caption of {pid}"}}
                        for pid in all_ids]
        self.dataset.append({"photo_id": all_ids[3],   # duplicate row: first one must win
                             "grounding_output": {"visual_description": "SHADOWED"}})
        self.url_lookup = {pid: f"https://img/{pid}.jpg" for pid in all_ids}
        self._scores = np.array([rng.uniform(0, 1) for _ in self.dataset])

    def _score_all(self, grounding_output):
        return self._scores, None


@pytest.fixture(params=[0, 1, 2, 3, 4])
def graph_case(request, tmp_path):
    graph = make_graph(request.param)
    path = tmp_path / "image_graph.pkl"
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    all_ids = sorted(set(graph) | {n for v in graph.values() for n, _ in v})
    return {"seed": request.param, "graph": graph, "path": path, "all_ids": all_ids}
