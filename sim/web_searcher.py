"""
web_searcher.py
---------------
Gives agents the ability to search the internet for evidence before
forming or defending their opinions.

Design:
  - High-Conscientiousness agents (C > 0.6) fact-check before opining.
  - High-Openness agents (O > 0.8) search to discover new angles.
  - Uses DuckDuckGo (free, no API key needed).
  - In mock mode, returns realistic-looking canned evidence snippets.
  - All errors are silently caught — search failure never crashes the sim.

Typical usage:
    evidence = search_for_evidence("AI regulation", mock=False)
    # Returns a formatted string of 2-3 web snippets ready to inject into prompt
"""

import logging
import random

logger = logging.getLogger(__name__)

# Silence excessively noisy HTTP and search libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ddgs").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

# Global cache for web searches so agents reading the same article don't
# spam the DDG API and trigger rate limits.
_SEARCH_CACHE: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Mock evidence snippets (used when mock=True or DDG fails)
# ---------------------------------------------------------------------------
MOCK_EVIDENCE = [
    (
        "• Nature (2024): A study of 47 countries found that stricter AI oversight "
        "frameworks correlated with a 23% reduction in AI-related fraud incidents, "
        "though innovation metrics showed a 9% dip in the same cohort.\n"
        "• MIT Tech Review (2023): Researchers argue that voluntary corporate AI ethics "
        "boards have failed in 78% of audited cases to flag material harms before deployment.\n"
        "• OECD (2024): Nations that adopted early AI governance saw GDP-adjusted tech "
        "sector growth outpace unregulated peers by 1.4% over 5 years."
    ),
    (
        "• Stanford HAI Index (2024): The number of countries introducing AI legislation "
        "jumped from 12 to 69 in a single year, the fastest regulatory expansion in tech history.\n"
        "• Brookings Institution: Analysis suggests that heavy AI regulation risks relocating "
        "AI development to jurisdictions with weaker rules, a 'regulation arbitrage' effect.\n"
        "• Pew Research (2024): 62% of adults in G20 countries support binding AI safety laws, "
        "up from 41% two years prior, driven largely by concerns about deepfakes and job automation."
    ),
    (
        "• European Parliament (2024): The EU AI Act mandates risk-tiered compliance, "
        "requiring high-risk AI systems to undergo third-party auditing — biopharma and autonomous "
        "vehicles are the first sectors targeted.\n"
        "• World Economic Forum: Industry self-regulation has produced measurable safety improvements "
        "in some sectors (fintech), but critics note these gains lag behind what statutory law achieved "
        "in aviation and pharmaceuticals.\n"
        "• AI Safety Institute (UK, 2024): Red-teaming exercises revealed critical vulnerabilities "
        "in frontier models, strengthening the case for mandatory pre-deployment audits."
    ),
    (
        "• McKinsey Global Institute: AI could automate 30% of current work tasks by 2030, "
        "with low-skill workers disproportionately affected — heightening calls for regulation "
        "paired with retraining mandates.\n"
        "• Cato Institute: Economic analysis shows regulatory compliance costs for SMEs in the "
        "EU AI Act framework average €340,000 — potentially cementing Big Tech dominance.\n"
        "• UN Secretary-General (2024): Called for an international AI governance body similar "
        "to the IAEA, arguing no single nation can manage cross-border AI risks alone."
    ),
]


def _should_search(ocean: dict) -> bool:
    """
    Return True if this agent's personality drives them to seek evidence.
    High-C (methodical) and high-O (curious) agents search; others may not.
    """
    C = ocean.get("C", 0.5)
    O = ocean.get("O", 0.5)
    # Score: 0→1. Agents above 0.55 search.
    search_drive = C * 0.55 + O * 0.45
    return search_drive > 0.55


def search_for_evidence(
    topic: str,
    mock: bool = False,
    max_results: int = 3,
    seed: int | None = None,
) -> str:
    """
    Search DuckDuckGo for real evidence about *topic*.

    Parameters
    ----------
    topic : str
        The article headline or subject matter.
    mock : bool
        If True, return canned evidence (no network call).
    max_results : int
        Number of DDG snippets to fetch.
    seed : int | None
        For deterministic mock selection.

    Returns
    -------
    str
        Formatted evidence block ready to inject into an LLM prompt.
        Empty string if search failed or agent type doesn't search.
    """
    if mock:
        rng = random.Random(seed)
        return rng.choice(MOCK_EVIDENCE)

    if topic in _SEARCH_CACHE:
        logger.debug("Using cached web search for '%s'", topic)
        return _SEARCH_CACHE[topic]

    try:
        from ddgs import DDGS

        # Build a focused query
        query = f"{topic} research data evidence policy"
        results = list(DDGS().text(query, max_results=max_results))

        if not results:
            return ""

        lines = []
        for r in results:
            title = r.get("title", "Source")
            body  = r.get("body", "")[:220].strip()
            if body:
                lines.append(f"• {title}: {body}")

        evidence = "\n".join(lines)
        logger.debug("Web search for '%s' returned %d results.", topic, len(results))
        _SEARCH_CACHE[topic] = evidence
        return evidence

    except Exception as exc:
        logger.debug("Web search failed (%s) — proceeding without evidence.", exc)
        _SEARCH_CACHE[topic] = ""  # Cache the failure too, to prevent infinite re-tries
        return ""


def maybe_search(
    agent_ocean: dict,
    topic: str,
    mock: bool = False,
    seed: int | None = None,
) -> str:
    """
    Call search_for_evidence only if this agent's personality calls for it.
    Wrapper used in environment.py to avoid searching for every agent.

    Returns
    -------
    str — evidence block, or "" if this agent type wouldn't search.
    """
    if _should_search(agent_ocean):
        return search_for_evidence(topic, mock=mock, seed=seed)
    return ""
