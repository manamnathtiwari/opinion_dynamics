"""
metrics.py
----------
Three core analytics metrics, computed after every simulation turn:

┌─────────────────────────────────────────────────────────────────┐
│  P(t) — Polarization index                                      │
│    Mean variance across embedding dimensions.                   │
│    High → opinions are spread far apart in semantic space.      │
│                                                                 │
│  I(agent) — Influence score                                     │
│    Fraction of interactions where the agent caused a shift.     │
│    High → "opinion leader".                                     │
│                                                                 │
│  E(t) — Echo chamber index                                      │
│    # connected components in an epsilon-neighborhood graph.     │
│    (Nodes connected if opinion similarity > 0.90)               │
│    High → agents are isolated in completely distinct silos.     │
└─────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


def compute_metrics(agents: list, turn: int) -> dict:
    """
    Compute P(t), I(agent), E(t) for the current turn.

    Parameters
    ----------
    agents : list[Agent]
        All agents in the simulation.
    turn : int
        Current turn index.

    Returns
    -------
    dict with keys:
        turn, polarization, echo_chamber, influence_scores, embeddings
    """
    valid_agents = [a for a in agents if a.opinion_embedding is not None]

    if not valid_agents:
        return {
            "turn": turn,
            "polarization": 0.0,
            "echo_chamber": 0.0,
            "influence_scores": {},
            "embeddings": [],
        }

    embeddings = np.array([a.opinion_embedding for a in valid_agents])

    # ------------------------------------------------------------------
    # P(t) — mean variance across embedding dimensions
    # ------------------------------------------------------------------
    polarization = float(np.var(embeddings, axis=0).mean())

    # ------------------------------------------------------------------
    # E(t) — Echo chamber index (Opinion Fragmentation)
    # ------------------------------------------------------------------
    # We define an echo chamber as a disconnected component in a graph
    # where edges indicate a cosine similarity > 0.90 (high agreement).
    G = nx.Graph()
    n = len(valid_agents)
    G.add_nodes_from(range(n))
    
    if n > 1:
        sim_matrix = cosine_similarity(embeddings)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > 0.90:
                    G.add_edge(i, j)
                    
    echo_chamber = nx.number_connected_components(G) / n if n > 0 else 0.0

    # ------------------------------------------------------------------
    # I(agent) — per-agent influence score
    # ------------------------------------------------------------------
    influence_scores = {a.id: a.calculate_influence(agents) for a in agents}

    return {
        "turn": turn,
        "polarization": round(polarization, 6),
        "echo_chamber": round(echo_chamber, 4),
        "influence_scores": influence_scores,
        "embeddings": embeddings.tolist(),
        "agent_ids": [a.id for a in valid_agents],
        "agent_mbtis": [a.mbti for a in valid_agents],
        "agent_confidences": [a.confidence for a in valid_agents],
    }


def polarization_delta(logs: list[dict]) -> float:
    """
    Return P(final) − P(initial): the net change in polarization.
    Positive → framing increased polarization.
    """
    if len(logs) < 2:
        return 0.0
    return logs[-1]["polarization"] - logs[0]["polarization"]


def influence_leaderboard(agents: list, top_n: int = 5) -> list[dict]:
    """
    Return the top-n agents by influence score.

    Returns
    -------
    list of dicts with keys: rank, id, mbti, influence_score.
    """
    ranked = sorted(agents, key=lambda a: a.calculate_influence(agents), reverse=True)
    return [
        {"rank": i + 1, "id": a.id, "mbti": a.mbti, "influence_score": a.calculate_influence(agents)}
        for i, a in enumerate(ranked[:top_n])
    ]
