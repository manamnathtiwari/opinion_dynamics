"""
umap_viz.py
-----------
UMAP-based 2D projection of agent opinion embeddings.

Produces:
  - Interactive Plotly scatter (used in Streamlit dashboard)
  - 3-panel static PNG: t=0 | t=mid | t=final  (poster figure)
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path


def project_umap(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Reduce *embeddings* (n × d) to 2D using UMAP.

    Returns np.ndarray of shape (n, 2). Returns None if n < 4.
    """
    if embeddings.shape[0] < 4:
        return None
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=random_state, n_neighbors=min(5, embeddings.shape[0]-1))
    return reducer.fit_transform(embeddings)


def make_scatter_fig(
    log_entry: dict,
    turn: int,
    framing: str = "",
    mode: str = "",
) -> go.Figure:
    """
    Build a Plotly scatter figure from a single turn's log entry.

    Parameters
    ----------
    log_entry : dict
        Output of analytics.metrics.compute_metrics().
    turn : int
        Turn number (for title).
    framing : str
        Framing variant label (for subtitle).
    mode : str
        Interaction mode label (for subtitle).

    Returns
    -------
    plotly Figure or None if too few agents.
    """
    embs = np.array(log_entry["embeddings"])
    if embs.shape[0] < 4:
        return None

    proj = project_umap(embs)
    if proj is None:
        return None

    df = pd.DataFrame(proj, columns=["x", "y"])
    df["mbti"]        = log_entry.get("agent_mbtis", ["?"] * len(df))
    df["confidence"]  = log_entry.get("agent_confidences", [0.5] * len(df))
    df["agent_id"]    = log_entry.get("agent_ids", list(range(len(df))))

    title = f"Opinion clusters — Turn {turn}"
    if framing:
        title += f" | {framing} framing"
    if mode:
        title += f" | {mode} mode"

    fig = px.scatter(
        df, x="x", y="y",
        color="mbti",
        size="confidence",
        hover_data=["agent_id", "mbti", "confidence"],
        title=title,
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def save_three_panel_png(
    logs: list[dict],
    output_path: str,
    framing: str = "",
    mode: str = "",
) -> None:
    """
    Save a 3-panel PNG: opinion clusters at t=0, t=mid, t=final.
    Used for poster Figure 1.

    Parameters
    ----------
    logs : list[dict]
        Full list of per-turn metric dicts.
    output_path : str
        Where to write the PNG.
    """
    if len(logs) < 3:
        return

    indices = [0, len(logs) // 2, len(logs) - 1]
    labels  = ["Initial (t=0)", f"Mid (t={indices[1]})", f"Final (t={indices[2]})"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=labels,
    )

    for col_idx, (log_idx, label) in enumerate(zip(indices, labels), start=1):
        entry = logs[log_idx]
        embs = np.array(entry["embeddings"])
        proj = project_umap(embs)
        if proj is None:
            continue

        mbtis = entry.get("agent_mbtis", ["?"] * len(proj))
        trace = go.Scatter(
            x=proj[:, 0], y=proj[:, 1],
            mode="markers+text",
            text=mbtis,
            textposition="top center",
            marker=dict(size=10),
            showlegend=(col_idx == 1),
        )
        fig.add_trace(trace, row=1, col=col_idx)

    main_title = "Opinion Cluster Evolution"
    if framing:
        main_title += f" | {framing} framing"
    if mode:
        main_title += f" | {mode} mode"

    fig.update_layout(title_text=main_title, height=400, width=1200)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_path)
    print(f"[umap_viz] Saved 3-panel PNG -> {output_path}")
