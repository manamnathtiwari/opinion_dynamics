"""
graph_network.py — Interaction Mode 4
---------------------------------------
Agents interact only with their neighbours in a fixed
Barabási-Albert (scale-free) network topology.

Scale-free networks are chosen because:
  - They model real social networks (Twitter/X, news comment sections)
  - A few hubs (high-degree nodes) dominate information flow
  - Minority opinions can persist in peripheral clusters

Expected research outcome:
  - Intermediate polarization (between random & social-feed)
  - Hub agents become opinion leaders regardless of personality
  - Communities may diverge if hubs hold opposing views
"""

import random
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

from sim.llm_caller import call_llm, build_interaction_prompt

_EMBEDDER: SentenceTransformer | None = None
_GRAPH: nx.Graph | None = None   # graph is built once and reused across turns


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _build_graph(n_agents: int, m: int = 2) -> nx.Graph:
    """
    Build a Barabási-Albert scale-free graph.

    Parameters
    ----------
    n_agents : int
        Number of nodes.
    m : int
        Edges added per new node (controls average degree).
    """
    global _GRAPH
    if _GRAPH is None or _GRAPH.number_of_nodes() != n_agents:
        _GRAPH = nx.barabasi_albert_graph(n_agents, m=m, seed=42)
    return _GRAPH


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def run(agents: list, turn: int, mock: bool = False) -> None:
    """
    Execute one turn of graph-network interactions.

    Each agent interacts with ONE randomly selected neighbour
    (sampled each turn so the conversation is not static).
    """
    embedder = _get_embedder()
    agent_map = {a.id: a for a in agents}
    G = _build_graph(len(agents))

    for node in G.nodes():
        agent = agent_map.get(node)
        if agent is None or not agent.opinion_text:
            continue

        neighbours = list(G.neighbors(node))
        random.shuffle(neighbours)

        for nb_id in neighbours:
            peer = agent_map.get(nb_id)
            if peer is None or not peer.opinion_text:
                continue

            prev_embedding = agent.opinion_embedding.copy() if agent.opinion_embedding is not None else None
            user_prompt = build_interaction_prompt(peer.opinion_text, peer.mbti)
            new_text = call_llm(agent.system_prompt, user_prompt, mock=mock)
            new_embedding = embedder.encode(new_text)

            shifted = False
            if prev_embedding is not None:
                sim = _cosine_sim(prev_embedding, new_embedding)
                shifted = sim < agent.shift_threshold   # personality-driven threshold

            agent.update_opinion(new_text, new_embedding, confidence_delta=0.02 if shifted else 0.0)
            agent.record_interaction(turn, peer.id, shifted)
            peer.record_interaction(turn, agent.id, shifted)
            break   # one interaction per agent per turn
