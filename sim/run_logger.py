"""
run_logger.py
-------------
Appends a JSON snapshot of the simulation state at each turn to a log file.
The log is newline-delimited JSON (NDJSON) — one record per line.
"""

import json
import os
from pathlib import Path


class RunLogger:
    """
    Writes per-turn state snapshots to a NDJSON log file.

    Parameters
    ----------
    output_path : str | Path
        Path to the output .json file. Created (with parent dirs) if absent.
    """

    def __init__(self, output_path: str | Path):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")

    def log_turn(self, turn: int, agents: list, metrics: dict) -> None:
        """
        Serialise agent states + metrics for *turn* and append to the log.

        Parameters
        ----------
        turn : int
            Current simulation turn index.
        agents : list[Agent]
            All agents in the simulation.
        metrics : dict
            Output of analytics.metrics.compute_metrics().
        """
        record = {
            "turn": turn,
            "polarization": metrics.get("polarization"),
            "echo_chamber": metrics.get("echo_chamber"),
            "influence_scores": metrics.get("influence_scores"),
            "agents_state": [
                {
                    "id": a.id,
                    "mbti": a.mbti,
                    "confidence": a.confidence,
                    "influence_score": a.calculate_influence(agents),
                    "opinion_text": a.opinion_text,
                    "opinion_embedding": (
                        a.opinion_embedding.tolist()
                        if a.opinion_embedding is not None else None
                    ),
                }
                for a in agents
            ],
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush and close the log file."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
