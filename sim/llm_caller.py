"""
llm_caller.py
-------------
Single entry point for all LLM calls in the simulation.

CPU strategy
------------
- Default model: mistral:7b-instruct-q4_K_M  (4-bit quantised, ~4 GB)
- Mock mode:  returns a deterministic canned opinion — no Ollama required.
  Activate via:
      export MOCK_LLM=1          (Linux/Mac)
      set MOCK_LLM=1             (Windows)
  or pass mock=True to call_llm().
- Ollama errors auto-fall back to mock so the simulation never crashes.

Speed tips (Windows, CPU only)
-------------------------------
  set OLLAMA_NUM_THREAD=<your_cpu_cores>   before starting ollama serve
"""

import os
import random
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canned opinions used in mock mode
# ---------------------------------------------------------------------------
MOCK_OPINIONS = [
    "I believe AI regulation is urgently necessary to protect democratic institutions.",
    "The risks of AI are overstated; excessive regulation will stifle innovation and harm the economy.",
    "A balanced, evidence-based framework is needed — neither blanket restriction nor total laissez-faire.",
    "Governments lack the technical expertise to regulate AI effectively; self-regulation has proven adequate.",
    "Corporate self-regulation has failed in every major industry; AI requires binding international oversight.",
    "The focus should be on specific harmful applications rather than broad restrictions on AI research.",
    "Without global coordination, national AI regulation will only disadvantage compliant countries.",
    "Transparency requirements and algorithmic auditing are far more practical than outright bans.",
]

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    mock: bool = False,
    seed: int | None = None,
) -> str:
    """
    Call the local Ollama LLM or return a mock opinion.

    Parameters
    ----------
    system_prompt : str
        Agent personality + behaviour instructions.
    user_prompt : str
        The article text (initial opinion) or peer opinion (interaction turn).
    mock : bool
        If True, skip Ollama entirely and return a canned opinion.
    seed : int | None
        Optional random seed for reproducible mock outputs.

    Returns
    -------
    str
        The generated (or mocked) opinion text.
    """
    # Respect environment override
    if mock or os.getenv("MOCK_LLM", "0") != "0":
        rng = random.Random(seed)
        return rng.choice(MOCK_OPINIONS)

    try:
        import ollama  # imported lazily so mock mode works without ollama installed

        response = ollama.chat(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={
                "temperature": 0.7,
                "num_predict": 120,   # ~3 sentences max — keeps CPU time low
            },
        )
        return response["message"]["content"].strip()

    except ImportError:
        logger.warning("ollama package not installed — falling back to mock mode.")
        return random.choice(MOCK_OPINIONS)
        
    except Exception as exc:
        err_msg = f"[OLLAMA FAILED]: {exc}"
        logger.error(err_msg)
        return err_msg


def build_opinion_prompt(article: dict, evidence: str = "") -> str:
    """
    Construct the user prompt for initial opinion formation.

    Parameters
    ----------
    article : dict
        Must contain keys: 'headline', 'body', 'label'.
    evidence : str
        Optional web-search evidence block to inject (from web_searcher.py).
        Passed only to agents whose search_tendency qualifies them.

    Returns
    -------
    str
        Formatted user prompt.
    """
    evidence_block = ""
    if evidence:
        evidence_block = (
            f"\n\n--- RELEVANT EVIDENCE YOU FOUND ---\n"
            f"{evidence}\n"
            f"--- END EVIDENCE ---\n"
            f"(Use this evidence to sharpen your argument if relevant, "
            f"but speak from your own perspective.)"
        )

    return (
        f"--- NEWS ARTICLE ({article['label'].upper()}) ---\n"
        f"Headline: {article['headline']}\n\n"
        f"{article['body'].strip()}"
        f"{evidence_block}\n\n"
        "--- TASK ---\n"
        "Read the article above and state your opinion on the issue it raises. "
        "Be specific. Anchor your view in your personality and prior beliefs. "
        "State your opinion in exactly 2-4 sentences. Do not ask questions."
    )


def build_interaction_prompt(peer_opinion: str, peer_mbti: str) -> str:
    """
    Construct the user prompt for a peer-interaction turn.

    Parameters
    ----------
    peer_opinion : str
        The text opinion of the peer agent.
    peer_mbti : str
        MBTI label of the peer (adds social context).

    Returns
    -------
    str
        Formatted user prompt.
    """
    return (
        f"A peer ({peer_mbti} personality type) has shared the following opinion:\n\n"
        f"\"{peer_opinion}\"\n\n"
        "Respond to this view. You may agree, partially agree, or disagree. "
        "State YOUR updated opinion in exactly 2-4 sentences. Do not ask questions."
    )
