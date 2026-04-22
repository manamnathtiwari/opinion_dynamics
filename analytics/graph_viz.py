"""
graph_viz.py
------------
Social influence graph visualisation.

Builds a NetworkX directed graph from agent influence logs and exports:
  - PyVis interactive HTML (animated node colours by opinion cluster)
  - Plotly static network chart (for Streamlit embed)
"""

import json
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path


def build_influence_graph(agents: list, up_to_turn: int | None = None) -> nx.DiGraph:
    """
    Build a directed influence graph from agent influence logs.

    Edge A → B means "A influenced B to shift opinion".
    Edge weight = number of successful shifts.

    Parameters
    ----------
    agents : list[Agent]
    up_to_turn : int | None
        If given, only include interactions from turns ≤ up_to_turn.
    """
    G = nx.DiGraph()
     # 1. Add nodes
    for a in agents:
        G.add_node(a.id, mbti=a.mbti, influence_score=a.calculate_influence(agents))

    for agent in agents:
        listener_id = agent.id
        for (turn_t, speaker_id, shifted) in agent.influence_log:
            if up_to_turn is not None and turn_t > up_to_turn:
                continue
            if shifted:
                # speaker influenced listener → directed edge speaker_id → listener_id
                if G.has_edge(speaker_id, listener_id):
                    G[speaker_id][listener_id]["weight"] += 1
                else:
                    G.add_edge(speaker_id, listener_id, weight=1)

    return G


def export_pyvis_html(graph: nx.DiGraph, output_path: str, title: str = "Influence Graph") -> None:
    """
    Export an interactive PyVis HTML force-directed graph.

    Parameters
    ----------
    graph : nx.DiGraph
    output_path : str
        Path to write the .html file.
    title : str
        Page title shown in the HTML.
    """
    from pyvis.network import Network

    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.heading = title

    # Add nodes — size proportional to influence score
    for node_id, data in graph.nodes(data=True):
        size = 10 + data.get("influence_score", 0) * 40
        label = f"{data.get('mbti', '?')}\n({node_id})"
        net.add_node(node_id, label=label, size=size, title=label)

    # Add edges — width proportional to weight
    for src, dst, data in graph.edges(data=True):
        w = data.get("weight", 1)
        net.add_edge(src, dst, width=min(w * 2, 10), title=f"shifts: {w}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    net.write_html(output_path)
    print(f"[graph_viz] Saved PyVis HTML -> {output_path}")


def make_plotly_graph(graph: nx.DiGraph) -> go.Figure:
    """
    Build a Plotly network figure (spring layout) for Streamlit embedding.

    Returns a go.Figure.
    """
    pos = nx.spring_layout(graph, seed=42)

    edge_x, edge_y = [], []
    for src, dst in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#aaa"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = [pos[n][0] for n in graph.nodes()]
    node_y = [pos[n][1] for n in graph.nodes()]
    node_text = [
        f"Agent {n}<br>MBTI: {graph.nodes[n].get('mbti','?')}<br>"
        f"Influence: {graph.nodes[n].get('influence_score', 0):.3f}"
        for n in graph.nodes()
    ]
    node_sizes = [
        10 + graph.nodes[n].get("influence_score", 0) * 30
        for n in graph.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[graph.nodes[n].get("mbti", str(n)) for n in graph.nodes()],
        textposition="top center",
        hovertext=node_text,
        marker=dict(size=node_sizes, color="#6366f1", line=dict(width=1, color="white")),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Social Influence Graph",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        ),
    )
    return fig
