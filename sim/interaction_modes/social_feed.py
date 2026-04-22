"""
social_feed.py — Interaction Mode 2
-------------------------------------
Each turn: each agent sees the top-k opinion most SIMILAR to its own
(an echo-chamber-inducing feed). Agent updates opinion based on that feed.

This mode is expected to:
  - Reinforce existing views
  - Increase E(t) by fragmenting agents into opinion clusters
  - Show lower ΔP(t) oscillation but higher final P(t)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from sim.llm_caller import call_llm, build_interaction_prompt

_EMBEDDER: SentenceTransformer | None = None
TOP_K = 2   # each agent interacts with top-2 most similar peers


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def _find_similar_agents(agent, agents: list, k: int = TOP_K) -> list:
    """Return the k agents most similar to *agent* (excluding itself)."""
    if agent.opinion_embedding is None:
        return []
    scored = []
    for other in agents:
        if other.id == agent.id or other.opinion_embedding is None:
            continue
        sim = _cosine_sim(agent.opinion_embedding, other.opinion_embedding)
        scored.append((sim, other))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [o for _, o in scored[:k]]


def run(agents: list, turn: int, mock: bool = False) -> None:
    """
    Execute one turn of social-feed interactions.

    Each agent reads from the top-k most similar peers and responds once,
    combining their opinions into a synthesised user prompt.
    """
    embedder = _get_embedder()

    for agent in agents:
        if not agent.opinion_text:
            continue

        similar_peers = _find_similar_agents(agent, agents)
        if not similar_peers:
            continue

        # Synthesise prompt from most similar peer (or first of top-k)
        peer = similar_peers[0]
        prev_embedding = agent.opinion_embedding.copy() if agent.opinion_embedding is not None else None

        user_prompt = build_interaction_prompt(peer.opinion_text, peer.mbti)
        new_text = call_llm(agent.system_prompt, user_prompt, mock=mock)
        new_embedding = embedder.encode(new_text)

        shifted = False
        if prev_embedding is not None:
            sim = _cosine_sim(prev_embedding, new_embedding)
            shifted = sim < agent.shift_threshold   # personality-driven threshold

        agent.update_opinion(new_text, new_embedding, confidence_delta=0.03 if shifted else 0.0)
        agent.record_interaction(turn, peer.id, shifted)
        peer.record_interaction(turn, agent.id, shifted)
