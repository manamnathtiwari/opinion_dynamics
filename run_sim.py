"""
run_sim.py
----------
CLI entry point for the Opinion Dynamics Simulator.

Usage examples:
    # Quick mock test (no Ollama needed)
    python run_sim.py --framing fear --mode random_pairs --agents 8 --turns 5 --mock

    # Real Ollama run
    python run_sim.py --framing fear --mode random_pairs --agents 15 --turns 30

    # A/B comparison run
    python run_sim.py --framing fear   --mode random_pairs --agents 15 --turns 30 --output results/fear_random.json
    python run_sim.py --framing neutral --mode random_pairs --agents 15 --turns 30 --output results/neutral_random.json

    # Full experiment batch (all 12 combos)
    python run_sim.py --batch

CPU tip: set OLLAMA_NUM_THREAD=<cores> before running for best speed.
         Or use --mock for instant pipeline testing.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Opinion Dynamics Simulator CLI")
    p.add_argument("--framing",  default="fear",
                   choices=["fear", "neutral", "solution"],
                   help="Article framing variant")
    p.add_argument("--mode",     default="random_pairs",
                   choices=["random_pairs", "social_feed", "influencer_hub", "graph_network"],
                   help="Agent interaction mode")
    p.add_argument("--agents",   type=int,  default=15, help="Number of agents")
    p.add_argument("--turns",    type=int,  default=30, help="Max simulation turns")
    p.add_argument("--delta",    type=float, default=0.005, help="Convergence threshold")
    p.add_argument("--mock",     action="store_true",
                   help="Use canned mock opinions (no Ollama needed)")
    p.add_argument("--output",   default=None,
                   help="Path for NDJSON log output (auto-generated if omitted)")
    p.add_argument("--preset",   default="uniform",
                   choices=["uniform", "majority_neurotic", "majority_agreeable"],
                   help="Population composition preset")
    p.add_argument("--batch",    action="store_true",
                   help="Run all 12 framing×mode combos sequentially")
    p.add_argument("--verbose",  action="store_true", help="Enable DEBUG logging")
    return p.parse_args()


def single_run(
    framing: str,
    mode: str,
    agents: int,
    turns: int,
    delta: float,
    mock: bool,
    output: str | None,
    preset: str,
) -> list[dict]:
    from sim.environment import SimulationEnvironment
    from analytics.metrics import polarization_delta, influence_leaderboard
    from analytics.graph_viz import build_influence_graph, export_pyvis_html
    from analytics.umap_viz import save_three_panel_png

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output is None:
        output = f"results/{framing}_{mode}_{timestamp}.json"

    logger.info("=" * 60)
    logger.info("Starting run: framing=%s mode=%s agents=%d turns=%d mock=%s",
                framing, mode, agents, turns, mock)
    logger.info("Output: %s", output)
    logger.info("=" * 60)

    env = SimulationEnvironment(
        framing_id=framing,
        mode=mode,
        n_agents=agents,
        mock=mock,
        population_preset=preset,
    )

    logs = env.run(max_turns=turns, delta=delta, output_path=output)

    # ── Post-run analytics ─────────────────────────────────────────────────
    delta_p = polarization_delta(logs)
    leaderboard = influence_leaderboard(env.agents, top_n=5)

    logger.info("─" * 60)
    logger.info("Run complete | turns=%d | ΔP(t)=%.4f", len(logs), delta_p)
    logger.info("Influence leaderboard:")
    for row in leaderboard:
        logger.info("  #%d Agent %d (%s) — score %.4f",
                    row["rank"], row["id"], row["mbti"], row["influence_score"])

    # ── Export social graph ────────────────────────────────────────────────
    G = build_influence_graph(env.agents)
    graph_html = f"results/{framing}_{mode}_{timestamp}_graph.html"
    export_pyvis_html(G, graph_html, title=f"{framing} / {mode} Influence Graph")

    # ── Export UMAP 3-panel PNG ────────────────────────────────────────────
    umap_png = f"results/{framing}_{mode}_{timestamp}_umap.png"
    try:
        save_three_panel_png(logs, umap_png, framing=framing, mode=mode)
    except Exception as exc:
        logger.warning("UMAP PNG export failed: %s", exc)

    # ── Save summary JSON ──────────────────────────────────────────────────
    summary = {
        "framing": framing, "mode": mode, "agents": agents,
        "turns_run": len(logs), "delta_p": delta_p,
        "leaderboard": leaderboard,
        "final_polarization": logs[-1]["polarization"] if logs else None,
        "final_echo_chamber": logs[-1]["echo_chamber"] if logs else None,
    }
    summary_path = output.replace(".json", "_summary.json")
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    logger.info("Summary saved → %s", summary_path)

    return logs


def batch_run(agents: int, turns: int, delta: float, mock: bool, preset: str) -> None:
    framings = ["fear", "neutral", "solution"]
    modes    = ["random_pairs", "social_feed", "influencer_hub", "graph_network"]
    total    = len(framings) * len(modes)

    logger.info("BATCH MODE: %d runs total", total)

    for i, framing in enumerate(framings):
        for j, mode in enumerate(modes):
            run_n = i * len(modes) + j + 1
            logger.info("─── Batch run %d/%d ───", run_n, total)
            single_run(
                framing=framing, mode=mode, agents=agents, turns=turns,
                delta=delta, mock=mock, output=None, preset=preset,
            )

    logger.info("Batch complete.")


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.mock:
        os.environ["MOCK_LLM"] = "1"
        logger.info("MOCK MODE enabled — no Ollama calls will be made.")

    Path("results").mkdir(exist_ok=True)

    if args.batch:
        batch_run(
            agents=args.agents, turns=args.turns, delta=args.delta,
            mock=args.mock, preset=args.preset,
        )
    else:
        single_run(
            framing=args.framing, mode=args.mode, agents=args.agents,
            turns=args.turns, delta=args.delta, mock=args.mock,
            output=args.output, preset=args.preset,
        )


if __name__ == "__main__":
    main()
