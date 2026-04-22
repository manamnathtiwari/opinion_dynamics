from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Agent:
    """
    Represents a single participant in the opinion dynamics simulation.

    Attributes
    ----------
    id : int
        Unique agent identifier.
    mbti : str
        16-type MBTI label (e.g. "INTJ").
    ocean : dict
        OCEAN trait scores {O, C, E, A, N} all floats in [0, 1].
    system_prompt : str
        LLM system prompt generated from ocean by prompt_generator.
    opinion_text : str
        Current natural-language opinion held by the agent.
    opinion_embedding : np.ndarray | None
        384-dim sentence-transformer embedding of opinion_text.
    confidence : float
        How strongly the agent holds its current opinion (0–1).
    influence_log : list
        Records of the form (turn, other_agent_id, shifted: bool).
        'shifted' = True when THIS agent changed its opinion after
        hearing other_agent_id.
    chroma_collection_name : str
        Key into the ChromaDB collection storing this agent's memory.
    mock : bool
        If True the agent never calls Ollama (useful for fast testing).
    """

    id: int
    mbti: str
    ocean: dict                    # {O, C, E, A, N}
    system_prompt: str
    opinion_text: str = ""
    opinion_embedding: Optional[np.ndarray] = None
    confidence: float = 0.5
    influence_log: list = field(default_factory=list)   # [(turn, other_id, shifted)]
    chroma_collection_name: str = ""
    mock: bool = False

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def calculate_influence(self, all_agents: list["Agent"]) -> float:
        """
        Fraction of interactions where THIS agent successfully caused 
        another agent to shift their opinion.
        """
        caused_shifts = 0
        total_times_heard = 0
        
        for p in all_agents:
            if p.id == self.id: continue
            for _, speaker_id, shifted in p.influence_log:
                if speaker_id == self.id:
                    total_times_heard += 1
                    if shifted:
                        caused_shifts += 1
                        
        return round(caused_shifts / total_times_heard, 4) if total_times_heard > 0 else 0.0

    @property
    def stubbornness(self) -> float:
        """
        How resistant this agent is to changing their opinion.
        Derived from OCEAN: High C + Low O + Low A = very stubborn.
        Range [0, 1]. Higher = harder to shift.

        Shift threshold formula used in interaction modes:
            threshold = 0.98 - stubbornness * 0.13
        ENFP (~0.22 stubborn) → threshold 0.951 (shifts fairly easily)
        ISTJ (~0.77 stubborn) → threshold 0.880 (shifts rarely)
        """
        O = self.ocean.get("O", 0.5)
        C = self.ocean.get("C", 0.5)
        A = self.ocean.get("A", 0.5)
        return round((1 - O) * 0.4 + C * 0.3 + (1 - A) * 0.3, 4)

    @property
    def shift_threshold(self) -> float:
        """
        Cosine-similarity threshold below which an opinion change counts as a 'shift'.
        Lower threshold = harder for this agent to shift (more stubborn).
        """
        return round(0.98 - self.stubbornness * 0.13, 4)

    @property
    def search_tendency(self) -> float:
        """
        How likely this agent is to search the web for evidence.
        High-C (methodical) and high-O (curious) agents search.
        Range [0, 1]. Agents above 0.55 will search.
        """
        C = self.ocean.get("C", 0.5)
        O = self.ocean.get("O", 0.5)
        return round(C * 0.55 + O * 0.45, 4)

    @property
    def stability_score(self) -> float:
        """Fraction of turns where THIS agent did NOT shift opinion."""
        if not self.influence_log:
            return 1.0
        non_shifts = sum(1 for _, _, shifted in self.influence_log if not shifted)
        return round(non_shifts / len(self.influence_log), 4)

    def record_interaction(self, turn: int, other_id: int, shifted: bool) -> None:
        """Append an interaction record to influence_log."""
        self.influence_log.append((turn, other_id, shifted))

    def update_opinion(self, new_text: str, new_embedding: np.ndarray,
                       confidence_delta: float = 0.0) -> None:
        """Replace opinion text, embedding and adjust confidence."""
        self.opinion_text = new_text
        self.opinion_embedding = new_embedding
        self.confidence = float(np.clip(self.confidence + confidence_delta, 0.0, 1.0))

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.id}, mbti={self.mbti}, "
            f"conf={self.confidence:.2f})"
        )
