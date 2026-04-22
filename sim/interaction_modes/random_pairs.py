"""
random_pairs.py — Interaction Mode 1 (Baseline)
------------------------------------------------
Each turn: randomly pair agents and run one interaction per pair.
Agent B reads Agent A's opinion and decides (via LLM) whether to update.
This is the fully randomised baseline — no topology, no echo-chamber bias.
"""

import random
import numpy as np
from sentence_transformers import SentenceTransformer

from sim.llm_caller import call_llm, build_interaction_prompt
from sim.convergence_check import cosine_check

_EMBEDDER: SentenceTransformer | None = None


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
    Execute one turn of random-pair interactions.

    For each pair (A, B):
      1. B reads A's opinion_text.
      2. B's LLM generates a response opinion.
      3. Cosine distance between old and new embedding decides if B "shifted".
      4. Influence log updated on both sides.

    Parameters
    ----------
    agents : list[Agent]
        Mutable list — opinion_text, opinion_embedding, confidence are updated.
    turn : int
        Current turn index (used for influence_log timestamps).
    mock : bool
        If True, skip Ollama and use canned opinions.
    """
    embedder = _get_embedder()
    shuffled = agents.copy()
    random.shuffle(shuffled)

    # Form pairs from shuffled list (last agent skipped if odd count)
    pairs = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled) - 1, 2)]

    for agent_a, agent_b in pairs:
        if not agent_a.opinion_text or not agent_b.opinion_text:
            continue  # skip agents who haven't formed opinions yet

        prev_embedding = agent_b.opinion_embedding.copy() if agent_b.opinion_embedding is not None else None

        # B responds to A
        user_prompt = build_interaction_prompt(agent_a.opinion_text, agent_a.mbti)
        new_text = call_llm(agent_b.system_prompt, user_prompt, mock=mock)
        new_embedding = embedder.encode(new_text)

        # Determine if B shifted (cosine distance threshold)
        shifted = False
        if prev_embedding is not None:
            sim = _cosine_sim(prev_embedding, new_embedding)
            shifted = sim < agent_b.shift_threshold   # personality-driven threshold

        agent_b.update_opinion(
            new_text, new_embedding,
            confidence_delta=0.05 if shifted else 0.0
        )
        agent_b.record_interaction(turn, agent_a.id, shifted)
        agent_a.record_interaction(turn, agent_b.id, shifted)   # A gets credit if B shifted
