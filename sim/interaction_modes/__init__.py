# sim/interaction_modes/__init__.py
"""
Dispatcher for simulation interaction modes.
Each mode module must expose a function:
    run(agents: list[Agent], turn: int, mock: bool = False) -> None
"""

from sim.interaction_modes.random_pairs import run as _random_pairs
from sim.interaction_modes.social_feed import run as _social_feed
from sim.interaction_modes.influencer_hub import run as _influencer_hub
from sim.interaction_modes.graph_network import run as _graph_network
from sim.interaction_modes.town_hall import run as _town_hall

_MODES = {
    "random_pairs":   _random_pairs,
    "social_feed":    _social_feed,
    "influencer_hub": _influencer_hub,
    "graph_network":  _graph_network,
    "town_hall":      _town_hall,
}


def get_mode(mode_name: str):
    """Return the run() function for *mode_name*."""
    if mode_name not in _MODES:
        raise ValueError(
            f"Unknown mode '{mode_name}'. Valid options: {list(_MODES.keys())}"
        )
    return _MODES[mode_name]
