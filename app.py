"""
app.py — Streamlit Dashboard
-----------------------------
Live dashboard with support for:
  - Built-in article framings (fear / neutral / solution)
  - Paste-your-own custom news article
  - Live UMAP opinion scatter
  - P(t) & E(t) time series
  - Agent chat log
  - Influence leaderboard + social graph
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

from sim.environment import SimulationEnvironment
from analytics.metrics import compute_metrics, polarization_delta, influence_leaderboard
from analytics.umap_viz import project_umap
from analytics.graph_viz import build_influence_graph, make_plotly_graph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Opinion Dynamics Simulator",
    page_icon="🧠",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .agent-card {
        background: #1e2130;
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        color: #e2e8f0;
    }
    .metric-pill {
        display: inline-block;
        background: #6366f1;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Opinion Dynamics")
    st.caption("Multi-agent news framing simulator")
    st.divider()

    # --- Article source ---
    st.subheader("📰 Article input")
    article_source = st.radio(
        "Source",
        ["Built-in framing", "Paste my own article"],
        horizontal=True,
    )

    custom_article = None

    if article_source == "Built-in framing":
        framing = st.selectbox(
            "Framing variant",
            ["fear", "neutral", "solution"],
            help="fear = alarming tone | neutral = balanced | solution = optimistic"
        )
    else:
        st.markdown("**Paste your article below**")
        custom_headline = st.text_input(
            "Headline",
            placeholder="e.g. 'Scientists warn of catastrophic consequences...'",
        )
        custom_body = st.text_area(
            "Article body",
            height=200,
            placeholder="Paste the full article text here. The agents will read this and form opinions about it.",
        )
        custom_label = st.text_input(
            "Label (short tag for graphs)",
            value="custom",
            max_chars=20,
        )
        framing = "custom"
        if custom_headline and custom_body:
            custom_article = {
                "id": "custom",
                "label": custom_label or "custom",
                "headline": custom_headline,
                "body": custom_body,
            }

    st.divider()

    # --- Sim config ---
    st.subheader("⚙️ Simulation config")
    mode = st.selectbox(
        "Interaction mode",
        ["random_pairs", "social_feed", "influencer_hub", "graph_network", "town_hall"],
        help=(
            "random_pairs = baseline random | "
            "social_feed = echo chamber | "
            "influencer_hub = high-E agents broadcast | "
            "graph_network = scale-free topology | "
            "town_hall = ALL agents hear EVERYONE's views concurrently in parallel"
        ),
    )
    n_agents = st.slider("Number of agents", 4, 20, 10)
    max_turns = st.slider("Max turns", 5, 50, 15)
    preset = st.selectbox(
        "Population preset",
        ["uniform", "majority_neurotic", "majority_agreeable"],
        help="Controls which MBTI types are over-represented"
    )
    mock = st.checkbox(
        "🚀 Mock mode (instant, no Ollama)",
        value=True,
        help="Use canned opinions — ideal for CPU or when Ollama is not running"
    )

    st.divider()

    # Validation check
    ready = True
    if article_source == "Paste my own article":
        if not custom_headline:
            st.warning("Add a headline to run.")
            ready = False
        elif not custom_body:
            st.warning("Paste article text to run.")
            ready = False

    run_btn = st.button(
        "▶ Run Simulation",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )

    st.caption(
        "💡 Mock mode ON = instant pipeline test.\n"
        "For real LLM opinions: pull Ollama model and uncheck mock.\n"
        "`set OLLAMA_NUM_THREAD=8` for CPU speed."
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 News Framing · Opinion Dynamics Simulator")
st.caption(
    "**Research question:** Which news framing causes the most polarization? "
    "Which personality types become opinion leaders?"
)

# Show article preview if custom
if article_source == "Paste my own article" and custom_article:
    with st.expander("📄 Article preview", expanded=False):
        st.markdown(f"**{custom_article['headline']}**")
        st.markdown(custom_article["body"])

# ── Main layout ───────────────────────────────────────────────────────────────
col_chat, col_scatter, col_metrics = st.columns([1.2, 1.4, 1.4])

col_chat.markdown("### 💬 Agent Opinions")
col_scatter.markdown("### 🔵 Opinion Clusters (UMAP)")
col_metrics.markdown("### 📈 Metrics")

chat_ph    = col_chat.empty()
scatter_ph = col_scatter.empty()
metrics_ph = col_metrics.empty()

# ── Run simulation ─────────────────────────────────────────────────────────────
if run_btn:
    if mock:
        os.environ["MOCK_LLM"] = "1"
    else:
        os.environ.pop("MOCK_LLM", None)

    # Show article being used
    if custom_article:
        st.info(f"📰 Running with your article: **{custom_article['headline'][:80]}**")
    else:
        st.info(f"📰 Running with built-in **{framing}** framing article.")

    with st.spinner("Spawning agents and forming initial opinions…"):
        env = SimulationEnvironment(
            framing_id=framing if not custom_article else "fear",  # fallback ID
            mode=mode,
            n_agents=n_agents,
            mock=mock,
            population_preset=preset,
            custom_article=custom_article,
        )
        env._form_initial_opinions()

    # Show initial opinions
    with col_chat:
        initial_lines = []
        for agent in env.agents:
            if agent.opinion_text:
                snippet = agent.opinion_text[:150] + ("…" if len(agent.opinion_text) > 150 else "")
                initial_lines.append(
                    f'<div class="agent-card">'
                    f'<span class="metric-pill">T0 · {agent.mbti}</span> '
                    f'{snippet}</div>'
                )
        chat_ph.markdown("\n".join(initial_lines), unsafe_allow_html=True)

    polarization_over_time = []
    echo_over_time         = []
    chat_lines_html        = list(initial_lines)

    progress = st.progress(0, text="Running turns…")

    for t in range(max_turns):
        env.turn = t
        env._run_interactions()

        m = compute_metrics(env.agents, t)
        polarization_over_time.append({"turn": t, "P(t)": round(m["polarization"], 6)})
        echo_over_time.append({"turn": t, "E(t)": round(m["echo_chamber"], 4)})

        # ── Chat log ──────────────────────────────────────────────────────
        for agent in env.agents:
            if agent.opinion_text:
                snippet = agent.opinion_text[:140] + ("…" if len(agent.opinion_text) > 140 else "")
                chat_lines_html.append(
                    f'<div class="agent-card">'
                    f'<span class="metric-pill">T{t+1} · {agent.mbti}</span> '
                    f'{snippet}</div>'
                )
        # Show last 15 lines
        chat_ph.markdown("\n".join(chat_lines_html[-15:]), unsafe_allow_html=True)

        # ── UMAP scatter ──────────────────────────────────────────────────
        embs = np.array(m["embeddings"])
        if embs.shape[0] >= 4:
            proj = project_umap(embs)
            if proj is not None:
                df_scatter = pd.DataFrame(proj, columns=["x", "y"])
                df_scatter["mbti"] = m.get("agent_mbtis", ["?"] * len(df_scatter))
                df_scatter["conf"] = [round(c, 2) for c in m.get("agent_confidences", [0.5]*len(df_scatter))]
                df_scatter["id"]   = m.get("agent_ids", list(range(len(df_scatter))))
                scatter_fig = px.scatter(
                    df_scatter, x="x", y="y",
                    color="mbti", size="conf",
                    hover_data={"id": True, "mbti": True, "conf": True, "x": False, "y": False},
                    title=f"Turn {t+1} | {framing} framing · {mode}",
                    height=340,
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                )
                scatter_fig.update_layout(
                    paper_bgcolor="#1e2130",
                    plot_bgcolor="#1e2130",
                    font_color="#e2e8f0",
                    legend=dict(bgcolor="#1e2130"),
                )
                scatter_ph.plotly_chart(scatter_fig, use_container_width=True)

        progress.progress((t + 1) / max_turns, text=f"Simulating turn {t+1} of {max_turns}...")

        # ── Dual-axis metrics line chart ──────────────────────────────────
        df_p = pd.DataFrame(polarization_over_time)
        df_e = pd.DataFrame(echo_over_time)

        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=df_p["turn"], y=df_p["P(t)"],
            name="Polarization P(t)",
            line=dict(color="#f43f5e", width=2.5),
            fill="tozeroy", fillcolor="rgba(244,63,94,0.08)",
        ))
        fig_m.add_trace(go.Scatter(
            x=df_e["turn"], y=df_e["E(t)"],
            name="Echo chamber E(t)",
            line=dict(color="#818cf8", width=2.5, dash="dot"),
            yaxis="y2",
        ))
        fig_m.update_layout(
            height=340,
            paper_bgcolor="#1e2130",
            plot_bgcolor="#1e2130",
            font_color="#e2e8f0",
            yaxis=dict(
                title=dict(text="P(t)", font=dict(color="#f43f5e")),
                gridcolor="#2a2f45",
            ),
            yaxis2=dict(
                title=dict(text="E(t)", font=dict(color="#818cf8")),
                overlaying="y",
                side="right",
            ),
            legend=dict(x=0, y=1.08, orientation="h", bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        metrics_ph.plotly_chart(fig_m, use_container_width=True)

        progress.progress((t + 1) / max_turns, text=f"Turn {t+1}/{max_turns}")

    progress.empty()

    # ── Summary banner ────────────────────────────────────────────────────
    final_p   = polarization_over_time[-1]["P(t)"]
    initial_p = polarization_over_time[0]["P(t)"]
    delta_p   = round(final_p - initial_p, 5)
    direction = "↑ increased" if delta_p > 0 else "↓ decreased"

    st.success(
        f"✅ **Simulation complete** — {len(polarization_over_time)} turns | "
        f"Polarization {direction} by **{abs(delta_p):.5f}** | "
        f"Final P(t) = {final_p:.5f}"
    )

    # ── Influence Leaderboard ─────────────────────────────────────────────
    st.markdown("---")
    lb_col, graph_col = st.columns([1, 2])

    with lb_col:
        st.markdown("### 🏆 Influence Leaderboard")
        st.caption("Which MBTI types shaped others' views the most?")
        lb = influence_leaderboard(env.agents, top_n=n_agents)
        lb_df = pd.DataFrame(lb)
        lb_fig = px.bar(
            lb_df, x="influence_score", y="mbti",
            orientation="h",
            color="influence_score",
            color_continuous_scale="Viridis",
            labels={"influence_score": "Score", "mbti": "MBTI"},
            height=350,
        )
        lb_fig.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            font_color="#e2e8f0", showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=20),
        )
        st.plotly_chart(lb_fig, use_container_width=True)

    with graph_col:
        st.markdown("### 🕸️ Social Influence Graph")
        st.caption("Directed edges = who shifted whom")
        G = build_influence_graph(env.agents)
        graph_fig = make_plotly_graph(G)
        graph_fig.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            font_color="#e2e8f0",
        )
        st.plotly_chart(graph_fig, use_container_width=True)

    # ── Full agent opinion table ──────────────────────────────────────────
    st.markdown("### 📋 Final Agent Opinions")
    rows = []
    for a in env.agents:
        searched = a.search_tendency > 0.55
        rows.append({
            "ID": a.id,
            "MBTI": a.mbti,
            "Stubbornness": round(a.stubbornness, 2),
            "🔍 Searched Web": "Yes" if searched else "No",
            "Confidence": round(a.confidence, 3),
            "Influence Score": round(a.calculate_influence(env.agents), 3),
            "Final Opinion": a.opinion_text[:200] + ("..." if len(a.opinion_text) > 200 else ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Save logs ─────────────────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = custom_label if (article_source == "Paste my own article" and custom_article) else framing
    log_path = f"results/{tag}_{mode}_{ts}_dashboard.json"
    with open(log_path, "w") as f:
        json.dump({
            "framing": tag,
            "mode": mode,
            "turns": len(polarization_over_time),
            "polarization": polarization_over_time,
            "echo_chamber": echo_over_time,
        }, f, indent=2)
    st.caption(f"Run log saved -> `{log_path}`")

else:
    # ── Placeholder state ────────────────────────────────────────────────
    with col_chat:
        st.markdown("""
        <div style='text-align:center;padding:40px;color:#4b5563;'>
            <div style='font-size:2.5rem'>📰</div>
            <div>Agent opinions will stream here</div>
        </div>
        """, unsafe_allow_html=True)
    with col_scatter:
        st.markdown("""
        <div style='text-align:center;padding:40px;color:#4b5563;'>
            <div style='font-size:2.5rem'>🔵</div>
            <div>UMAP opinion clusters will appear here</div>
        </div>
        """, unsafe_allow_html=True)
    with col_metrics:
        st.markdown("""
        <div style='text-align:center;padding:40px;color:#4b5563;'>
            <div style='font-size:2.5rem'>📈</div>
            <div>P(t) and E(t) charts will appear here</div>
        </div>
        """, unsafe_allow_html=True)

    if article_source == "Paste my own article":
        st.info("👈 Paste your article headline and body in the sidebar, then press **▶ Run Simulation**.")
    else:
        st.info("👈 Configure settings in the sidebar and press **▶ Run Simulation**.")
