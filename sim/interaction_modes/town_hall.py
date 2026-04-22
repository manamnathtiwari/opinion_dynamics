"""
town_hall.py
------------
Interaction mode where EVERY agent hears the summarized opinions of ALL
other agents simultaneously, representing an open debate forum.
They then state their support or opposition to the collective views.
"""

from typing import Any
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

# Mute sentence_transformers internal warnings
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def _get_embedder() -> Any:
    from sim.environment import _get_embedder as get_global_embedder
    return get_global_embedder()

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics.pairwise import cosine_similarity
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])

def run(agents: list, turn: int, mock: bool = False) -> None:
    """
    Every agent reads a summary of all other agents' past opinions,
    then generates a new synthesized opinion in parallel.
    """
    from sim.llm_caller import call_llm
    embedder = _get_embedder()

    # Create the collective context of what everyone currently believes
    peer_statements = []
    for a in agents:
        if a.opinion_text:
            text = f"- Agent {a.id} ({a.mbti}): {a.opinion_text}"
            peer_statements.append(text)
    collective_context = "\n".join(peer_statements)
    
    # Define generation task for parallel execution
    def _agent_task(agent):
        prompt = (
            f"You are in an open town hall debate. Here are the current views of everyone else:\n"
            f"{collective_context}\n\n"
            "--- TASK ---\n"
            "Consider ALL the views presented. Who do you agree with? Who do you oppose? "
            "Synthesize your updated argument based on your personality type. "
            "Respond in exactly 2-4 sentences. Do not ask questions."
        )
        
        # Keep old embedding to detect shifts
        prev_embedding = agent.opinion_embedding

        new_text = call_llm(
            agent.system_prompt,
            prompt,
            mock=mock,
            seed=agent.id + turn * 100,
        )
        
        if "[OLLAMA FAILED]" in new_text:
            logger.warning("Agent %d skipped update due to LLM failure.", agent.id)
            return
            
        new_embedding = embedder.encode(new_text)

        shifted = False
        if prev_embedding is not None:
            sim = _cosine_sim(prev_embedding, new_embedding)
            shifted = sim < agent.shift_threshold

        agent.update_opinion(new_text, new_embedding, confidence_delta=0.04 if shifted else 0.0)
        
        # In a town hall, you are theoretically influenced by everyone, but for logging we just mark self-shift
        agent.record_interaction(turn, -1, shifted) 
        
    # Run all agent updates in parallel to massive speedup LLM generation!
    with ThreadPoolExecutor(max_workers=min(len(agents), 8)) as pool:
        pool.map(_agent_task, agents)
        
    logger.debug("Turn %d | Town Hall complete.", turn)
