"""
environment.py
--------------
Main simulation controller. Orchestrates:
  1. Agent spawning (MBTI/OCEAN profiles)
  2. Initial opinion formation (each agent reads the article once)
  3. Turn-based interaction loop (delegated to selected mode)
  4. Per-turn metrics + logging
  5. Convergence detection

Uses LangGraph as the state machine backbone.
"""

import yaml
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

from sim.agent import Agent
from sim.prompt_generator import generate_system_prompt
from sim.llm_caller import call_llm, build_opinion_prompt
from sim.run_logger import RunLogger
from sim.convergence_check import delta_check
from sim.interaction_modes import get_mode
from sim.web_searcher import maybe_search
from analytics.metrics import compute_metrics

logger = logging.getLogger(__name__)

# Mute sentence_transformers internal load reports
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

_EMBEDDER: SentenceTransformer | None = None

POPULATION_PRESETS = {
    "uniform": None,   # cycle through all types evenly
    "majority_neurotic":  ["INFP", "INFJ", "ENFP", "ENTP"],
    "majority_agreeable": ["ENFJ", "ESFJ", "ISFJ", "ISFP"],
}


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


class SimulationEnvironment:
    """
    Manages one complete simulation run.

    Parameters
    ----------
    framing_config_path : str
        Path to config/framing_config.yaml.
    personality_config_path : str
        Path to config/personality_profiles.yaml.
    framing_id : str
        Which article variant to use: 'fear' | 'neutral' | 'solution'.
    mode : str
        Interaction mode: 'random_pairs' | 'social_feed' |
        'influencer_hub' | 'graph_network'.
    n_agents : int
        How many agents to spawn.
    mock : bool
        If True, all LLM calls are replaced with canned opinions.
    population_preset : str
        'uniform' | 'majority_neurotic' | 'majority_agreeable'
    """

    def __init__(
        self,
        framing_config_path: str = "config/framing_config.yaml",
        personality_config_path: str = "config/personality_profiles.yaml",
        framing_id: str = "fear",
        mode: str = "random_pairs",
        n_agents: int = 15,
        mock: bool = False,
        population_preset: str = "uniform",
        custom_article: dict | None = None,
    ):
        """
        Parameters
        ----------
        custom_article : dict | None
            If provided, bypasses the YAML entirely.
            Must have keys: 'id', 'label', 'headline', 'body'.
            Example:
                {"id": "custom", "label": "custom",
                 "headline": "My headline",
                 "body": "Full article text..."}
        """
        self.framing_cfg = yaml.safe_load(open(framing_config_path))
        self.personality_cfg = yaml.safe_load(open(personality_config_path))
        self.article = custom_article if custom_article else self._load_article(framing_id)
        self.mode = mode
        self.mock = mock
        self.n_agents = n_agents
        self.agents = self._spawn_agents(n_agents, population_preset)
        self.turn = 0
        self.logs: list[dict] = []
        self._polarization_history: list[float] = []

        label = custom_article.get("label", "custom") if custom_article else framing_id
        logger.info(
            "SimulationEnvironment ready | framing=%s mode=%s agents=%d mock=%s",
            label, mode, n_agents, mock
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_article(self, framing_id: str) -> dict:
        for v in self.framing_cfg["article_variants"]:
            if v["id"] == framing_id:
                return v
        raise ValueError(f"Framing ID '{framing_id}' not found in config.")

    def _spawn_agents(self, n: int, preset: str) -> list[Agent]:
        profiles = self.personality_cfg["profiles"]
        mbti_pool = POPULATION_PRESETS.get(preset)
        if mbti_pool is None:
            mbti_pool = list(profiles.keys())   # uniform → all 16

        agents = []
        for i in range(n):
            mbti = mbti_pool[i % len(mbti_pool)]
            ocean = profiles[mbti]["ocean"]
            prompt = generate_system_prompt(mbti, ocean)
            agents.append(Agent(
                id=i, mbti=mbti, ocean=ocean, system_prompt=prompt, mock=self.mock
            ))
        logger.info("Spawned %d agents (preset=%s)", n, preset)
        return agents

    # ------------------------------------------------------------------
    # Phase 1 — Initial opinion formation
    # ------------------------------------------------------------------

    def _form_initial_opinions(self) -> None:
        """Each agent independently reads the article and forms an opinion.
        High-C/O agents search the web for evidence first.
        """
        embedder = _get_embedder()
        # Extract a search topic from the article headline
        topic = self.article.get("headline", "the topic")

        from concurrent.futures import ThreadPoolExecutor

        logger.info("Forming initial opinions (mock=%s, web_search=enabled, parallel=True) ...", self.mock)
        
        def _task(agent):
            # Web search for qualifying agents
            evidence = maybe_search(
                agent_ocean=agent.ocean,
                topic=topic,
                mock=self.mock,
                seed=agent.id,
            )
            if evidence:
                logger.debug(
                    "Agent %d (%s) searched the web (search_tendency=%.2f)",
                    agent.id, agent.mbti, agent.search_tendency,
                )

            article_prompt = build_opinion_prompt(self.article, evidence=evidence)
            text = call_llm(agent.system_prompt, article_prompt, mock=self.mock, seed=agent.id)
            if "[OLLAMA FAILED]" in text:
                logger.error("Agent %d failed initial opinion formation.", agent.id)
                text = "I am unsure due to an internal error."
            
            embedding = embedder.encode(text)
            agent.update_opinion(text, embedding)
            logger.debug("Agent %d (%s): %s", agent.id, agent.mbti, text[:80])

        with ThreadPoolExecutor(max_workers=min(len(self.agents), 8)) as pool:
            pool.map(_task, self.agents)

    # ------------------------------------------------------------------
    # Phase 2 — Turn loop
    # ------------------------------------------------------------------

    def _run_interactions(self) -> None:
        mode_fn = get_mode(self.mode)
        mode_fn(self.agents, self.turn, mock=self.mock)

    def run(
        self,
        max_turns: int = 30,
        delta: float = 0.005,
        output_path: str | None = None,
    ) -> list[dict]:
        """
        Run the simulation for up to *max_turns* turns.

        Parameters
        ----------
        max_turns : int
            Hard limit on turns.
        delta : float
            Convergence threshold for polarization delta check.
        output_path : str | None
            If given, logs are written to this NDJSON file.

        Returns
        -------
        list[dict]
            Per-turn metrics list.
        """
        self._form_initial_opinions()

        logger_ctx = RunLogger(output_path) if output_path else None

        for t in range(max_turns):
            self.turn = t
            self._run_interactions()

            metrics = compute_metrics(self.agents, t)
            self.logs.append(metrics)
            self._polarization_history.append(metrics["polarization"])

            logger.info(
                "Turn %d | P(t)=%.4f | E(t)=%.3f",
                t, metrics["polarization"], metrics["echo_chamber"]
            )

            if logger_ctx:
                logger_ctx.log_turn(t, self.agents, metrics)

            # Convergence check (only after 3 turns)
            if t > 3 and delta_check(self._polarization_history, threshold=delta):
                logger.info("Converged at turn %d", t)
                break

        if logger_ctx:
            logger_ctx.close()

        return self.logs
