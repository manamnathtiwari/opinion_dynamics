"""
convergence_check.py
--------------------
Detects when opinion dynamics have stabilised by monitoring the
polarization index P(t) across recent turns.

Two methods are provided:
  - delta_check : fast — if max(P[-3:]) - min(P[-3:]) < threshold → converged
  - cosine_check: semantic — mean cosine similarity between consecutive
                  opinion embeddings is above a high threshold
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def delta_check(polarization_history: list[float], threshold: float = 0.005) -> bool:
    """
    Return True if P(t) has not varied by more than *threshold* in the last 3 turns.

    Parameters
    ----------
    polarization_history : list[float]
        Ordered list of P(t) values from turn 0 onward.
    threshold : float
        Maximum allowed swing before declaring convergence.
    """
    if len(polarization_history) < 3:
        return False
    recent = polarization_history[-3:]
    return (max(recent) - min(recent)) < threshold


def cosine_check(agents, prev_embeddings: np.ndarray, threshold: float = 0.995) -> bool:
    """
    Return True if mean cosine similarity between current and previous
    opinion embeddings exceeds *threshold*.

    Parameters
    ----------
    agents : list[Agent]
        Current agent list with updated opinion_embedding.
    prev_embeddings : np.ndarray
        Embeddings from the previous turn (shape: n_agents × dim).
    threshold : float
        Mean similarity above which opinions are considered stable.
    """
    curr = np.array([
        a.opinion_embedding for a in agents if a.opinion_embedding is not None
    ])
    if curr.shape != prev_embeddings.shape:
        return False

    sims = []
    for c, p in zip(curr, prev_embeddings):
        sim = cosine_similarity(c.reshape(1, -1), p.reshape(1, -1))[0, 0]
        sims.append(sim)

    return float(np.mean(sims)) > threshold
