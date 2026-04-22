"""
influencer_hub.py — Interaction Mode 3
----------------------------------------
High-Extraversion agents (E > 0.65) act as broadcasters:
they share their opinion with ALL other agents in the same turn.
Low-E agents receive but do not broadcast.

Expected research outcome:
  - EXTJ / ESTP / ENFJ rise to top of influence leaderboard
  - P(t) may converge faster as high-E agents pull others toward their view
  - A single dominant opinion may emerge (lower E(t))
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from sim.llm_caller import call_llm, build_interaction_prompt

_EMBEDDER: SentenceTransformer | None = None
EXTRAVERSION_THRESHOLD = 0.65


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def run(agents: list, turn: int, mock: bool = False) -> None:
    """
    Execute one turn of influencer-hub interactions.

    Each high-E 'influencer' broadcasts to every other agent.
    Each receiver generates a response (LLM call) and may shift.
    """
    embedder = _get_embedder()

    influencers = [a for a in agents if a.ocean.get("E", 0) > EXTRAVERSION_THRESHOLD and a.opinion_text]
    listeners = [a for a in agents if a.ocean.get("E", 0) <= EXTRAVERSION_THRESHOLD and a.opinion_text]

    if not influencers:
        # Fallback: pick agent with highest E as sole influencer
        valid = [a for a in agents if a.opinion_text]
        if not valid:
            return
        influencers = [max(valid, key=lambda a: a.ocean.get("E", 0))]
        listeners = [a for a in valid if a.id != influencers[0].id]

    for influencer in influencers:
        for listener in listeners:
            if listener.id == influencer.id:
                continue

            prev_embedding = listener.opinion_embedding.copy() if listener.opinion_embedding is not None else None
            user_prompt = build_interaction_prompt(influencer.opinion_text, influencer.mbti)
            new_text = call_llm(listener.system_prompt, user_prompt, mock=mock)
            new_embedding = embedder.encode(new_text)

            shifted = False
            if prev_embedding is not None:
                sim = _cosine_sim(prev_embedding, new_embedding)
                shifted = sim < listener.shift_threshold   # personality-driven threshold

            listener.update_opinion(new_text, new_embedding, confidence_delta=0.02 if shifted else 0.0)
            listener.record_interaction(turn, influencer.id, shifted)
            influencer.record_interaction(turn, listener.id, shifted)
