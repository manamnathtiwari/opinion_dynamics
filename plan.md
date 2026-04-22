# News Framing & Opinion Dynamics Simulator
### Build Plan — Antigravity AIML Project

---

## Project summary

A configurable multi-agent simulation where MBTI/OCEAN-profiled LLM agents read a news article, form independent opinions, and interact across multiple turns. The primary research question is: **which news framing causes the most polarization, and which personality types become opinion leaders?**

The end deliverable is a live Streamlit dashboard showing opinion clusters shifting in real time alongside polarization metrics and a social influence graph.

---

## Repository structure

```
opinion-dynamics-sim/
├── plan.md                        ← this file
├── requirements.txt
├── config/
│   ├── framing_config.yaml        ← article variants + framing labels
│   └── personality_profiles.yaml  ← MBTI → OCEAN mappings
├── sim/
│   ├── agent.py                   ← Agent class
│   ├── prompt_generator.py        ← OCEAN → system prompt
│   ├── environment.py             ← simulation controller (LangGraph)
│   ├── interaction_modes/
│   │   ├── __init__.py
│   │   ├── random_pairs.py
│   │   ├── social_feed.py
│   │   ├── influencer_hub.py
│   │   └── graph_network.py
│   ├── run_logger.py              ← JSON state logging per turn
│   └── convergence_check.py
├── analytics/
│   ├── metrics.py                 ← P(t), I(agent), E(t)
│   ├── umap_viz.py                ← UMAP projection + scatter
│   └── graph_viz.py              ← NetworkX social graph
├── app.py                         ← Streamlit dashboard
├── run_sim.py                     ← CLI entry point
└── results/                       ← auto-created, gitignored
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running locally
- ~8 GB RAM for Mistral 7B

### 2. Install Ollama model

```bash
ollama pull mistral
# or for better reasoning:
ollama pull llama3
```

### 3. Clone and install Python dependencies

```bash
git clone <your-repo>
cd opinion-dynamics-sim
pip install -r requirements.txt
```

### 4. `requirements.txt`

```
langgraph>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.0
sentence-transformers>=2.7.0
umap-learn>=0.5.6
networkx>=3.2
pyvis>=0.3.2
streamlit>=1.35.0
plotly>=5.20.0
pyyaml>=6.0
numpy>=1.26.0
scikit-learn>=1.4.0
```

---

## Configuration files

### `config/framing_config.yaml`

Define multiple framing variants of the same story here. The simulator runs one variant at a time; run multiple times for A/B comparison.

```yaml
article_variants:
  - id: fear
    label: "fear-based"
    headline: "Scientists warn of catastrophic consequences if AI regulation fails"
    body: |
      Researchers at a leading institute have issued stark warnings that
      unregulated artificial intelligence poses an existential threat to
      democratic institutions and individual privacy...

  - id: neutral
    label: "neutral"
    headline: "Policymakers debate scope of new AI oversight framework"
    body: |
      A cross-party committee met this week to discuss a proposed framework
      for AI oversight. Experts presented a range of positions on the
      appropriate level of regulatory intervention...

  - id: solution
    label: "solution-focused"
    headline: "New international AI safety coalition launches with industry backing"
    body: |
      A coalition of governments and technology companies announced a
      collaborative framework designed to promote responsible AI development
      while preserving innovation...

active_variant: fear   # change this to switch framing
```

### `config/personality_profiles.yaml`

```yaml
profiles:
  INTJ:
    ocean: {O: 0.75, C: 0.90, E: 0.20, A: 0.25, N: 0.35}
    description: "Strategic, independent thinker. Argues from logic, resists emotional appeals."

  ENFP:
    ocean: {O: 0.92, C: 0.40, E: 0.85, A: 0.75, N: 0.55}
    description: "Enthusiastic, empathetic. Open to new views, communicates warmly."

  ISTJ:
    ocean: {O: 0.30, C: 0.92, E: 0.25, A: 0.55, N: 0.28}
    description: "Detail-oriented, traditional. Relies on precedent, slow to change opinion."

  ESTP:
    ocean: {O: 0.60, C: 0.38, E: 0.90, A: 0.40, N: 0.42}
    description: "Bold, pragmatic. Challenges others directly, high influence potential."

  INFP:
    ocean: {O: 0.88, C: 0.42, E: 0.22, A: 0.82, N: 0.65}
    description: "Idealistic, values-driven. Avoids conflict, very open but deeply principled."

  ESTJ:
    ocean: {O: 0.28, C: 0.90, E: 0.78, A: 0.42, N: 0.30}
    description: "Decisive, traditional. High broadcast rate, low update probability."

# Add remaining MBTI types following the same pattern
# Population preset used by run_sim.py:
population_preset: uniform   # options: uniform | majority_neurotic | majority_agreeable
```

---

## Core modules

### `sim/agent.py`

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Agent:
    id: int
    mbti: str
    ocean: dict            # {O, C, E, A, N} all floats in [0,1]
    system_prompt: str     # generated from ocean by prompt_generator
    opinion_text: str = ""
    opinion_embedding: Optional[np.ndarray] = None
    confidence: float = 0.5
    influence_log: list = field(default_factory=list)   # [(turn, agent_id, shifted: bool)]
    chroma_collection_name: str = ""

    @property
    def influence_score(self) -> float:
        shifts = sum(1 for _, _, shifted in self.influence_log if shifted)
        total = len(self.influence_log)
        return shifts / total if total > 0 else 0.0
```

### `sim/prompt_generator.py`

```python
def generate_system_prompt(mbti: str, ocean: dict) -> str:
    O, C, E, A, N = ocean["O"], ocean["C"], ocean["E"], ocean["A"], ocean["N"]

    openness_str = (
        "You engage curiously with unfamiliar viewpoints and are willing to update your beliefs."
        if O > 0.65 else
        "You prefer familiar frameworks and are cautious about radical new ideas."
    )
    agree_str = (
        "You listen carefully to others and prefer to find common ground."
        if A > 0.65 else
        "You hold your ground and don't soften your view just to avoid conflict."
    )
    neuro_str = (
        "You feel threatened by direct challenges to your beliefs and may react defensively."
        if N > 0.60 else
        "You engage with criticism calmly and without taking it personally."
    )
    extrav_str = (
        "You readily share your opinions and actively try to persuade others."
        if E > 0.65 else
        "You prefer to listen before speaking and share opinions selectively."
    )
    consc_str = (
        "You reason carefully from evidence and facts, not emotions."
        if C > 0.65 else
        "You trust your gut and respond to the emotional tone of arguments."
    )

    return (
        f"You are an agent with an {mbti} personality type. "
        f"{openness_str} {agree_str} {neuro_str} {extrav_str} {consc_str} "
        f"Respond in 2-4 sentences. Be direct. Do not use bullet points."
    )
```

### `sim/environment.py` (skeleton)

```python
from langgraph.graph import StateGraph
from sim.agent import Agent
from sim.prompt_generator import generate_system_prompt
from analytics.metrics import compute_metrics
import yaml, random

class SimulationEnvironment:
    def __init__(self, config_path: str, framing_id: str, mode: str, n_agents: int = 15):
        self.config = yaml.safe_load(open(config_path))
        self.article = self._load_article(framing_id)
        self.mode = mode
        self.agents = self._spawn_agents(n_agents)
        self.turn = 0
        self.logs = []

    def _load_article(self, framing_id):
        for v in self.config["article_variants"]:
            if v["id"] == framing_id:
                return v
        raise ValueError(f"Framing ID '{framing_id}' not found in config.")

    def _spawn_agents(self, n):
        profiles = yaml.safe_load(open("config/personality_profiles.yaml"))["profiles"]
        agents = []
        mbti_types = list(profiles.keys())
        for i in range(n):
            mbti = mbti_types[i % len(mbti_types)]
            ocean = profiles[mbti]["ocean"]
            prompt = generate_system_prompt(mbti, ocean)
            agents.append(Agent(id=i, mbti=mbti, ocean=ocean, system_prompt=prompt))
        return agents

    def run(self, max_turns: int = 30, delta: float = 0.005):
        self._form_initial_opinions()
        for t in range(max_turns):
            self.turn = t
            self._run_interactions()
            metrics = compute_metrics(self.agents, t)
            self.logs.append(metrics)
            if t > 3 and self._converged(delta):
                print(f"Converged at turn {t}")
                break
        return self.logs

    def _form_initial_opinions(self):
        # Each agent reads article independently, forms opinion via LLM
        pass   # implementation: call ollama with system_prompt + article text

    def _run_interactions(self):
        from sim.interaction_modes import get_mode
        mode_fn = get_mode(self.mode)
        mode_fn(self.agents, self.turn)

    def _converged(self, delta):
        recent = [log["polarization"] for log in self.logs[-3:]]
        return max(recent) - min(recent) < delta
```

### `analytics/metrics.py`

```python
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def compute_metrics(agents, turn: int) -> dict:
    embeddings = np.array([a.opinion_embedding for a in agents if a.opinion_embedding is not None])

    # Polarization index: mean variance across embedding dimensions
    polarization = float(np.var(embeddings, axis=0).mean())

    # Build listening graph for this turn
    G = nx.Graph()
    G.add_nodes_from([a.id for a in agents])
    for a in agents:
        for turn_t, other_id, shifted in a.influence_log:
            if turn_t == turn:
                G.add_edge(a.id, other_id)

    n = len(agents)
    echo_chamber = nx.number_connected_components(G) / n if n > 0 else 0

    influence_scores = {a.id: a.influence_score for a in agents}

    return {
        "turn": turn,
        "polarization": polarization,
        "echo_chamber": round(echo_chamber, 3),
        "influence_scores": influence_scores,
        "embeddings": embeddings.tolist(),
    }
```

### `app.py` — Streamlit dashboard (skeleton)

```python
import streamlit as st
import plotly.express as px
import pandas as pd
from umap import UMAP
from sim.environment import SimulationEnvironment
import yaml

st.set_page_config(layout="wide", page_title="Opinion Dynamics Simulator")
st.title("News Framing · Opinion Dynamics Simulator")

# --- Sidebar config ---
with st.sidebar:
    framing = st.selectbox("Framing variant", ["fear", "neutral", "solution"])
    mode = st.selectbox("Interaction mode", ["random_pairs", "social_feed", "influencer_hub", "graph_network"])
    n_agents = st.slider("Number of agents", 8, 20, 15)
    max_turns = st.slider("Max turns", 10, 50, 30)
    run = st.button("Run simulation")

col_chat, col_charts = st.columns([1, 1])

# --- Chat log placeholder ---
chat_placeholder = col_chat.empty()

# --- Chart placeholders ---
scatter_placeholder = col_charts.empty()
metrics_placeholder = col_charts.empty()

if run:
    env = SimulationEnvironment("config/framing_config.yaml", framing, mode, n_agents)

    chat_lines = []
    polarization_over_time = []
    echo_over_time = []

    for t in range(max_turns):
        # run one turn
        env.turn = t
        env._run_interactions()
        from analytics.metrics import compute_metrics
        m = compute_metrics(env.agents, t)

        polarization_over_time.append({"turn": t, "P(t)": round(m["polarization"], 4)})
        echo_over_time.append({"turn": t, "E(t)": round(m["echo_chamber"], 3)})

        # Update UMAP scatter
        import numpy as np
        embs = np.array(m["embeddings"])
        if embs.shape[0] >= 4:
            reducer = UMAP(n_components=2, random_state=42)
            proj = reducer.fit_transform(embs)
            df = pd.DataFrame(proj, columns=["x", "y"])
            df["mbti"] = [a.mbti for a in env.agents]
            fig = px.scatter(df, x="x", y="y", color="mbti",
                             title=f"Opinion clusters — turn {t}",
                             height=300)
            scatter_placeholder.plotly_chart(fig, use_container_width=True)

        # Update metrics chart
        df_p = pd.DataFrame(polarization_over_time)
        fig2 = px.line(df_p, x="turn", y="P(t)", title="Polarization index over time", height=200)
        metrics_placeholder.plotly_chart(fig2, use_container_width=True)

    st.success(f"Simulation complete — {t+1} turns")
```

---

## CLI usage

```bash
# Run a single simulation from command line
python run_sim.py --framing fear --mode random_pairs --agents 15 --turns 30

# Run an A/B framing comparison (fear vs neutral)
python run_sim.py --framing fear --mode random_pairs --agents 15 --turns 30 --output results/fear_run1.json
python run_sim.py --framing neutral --mode random_pairs --agents 15 --turns 30 --output results/neutral_run1.json

# Launch the live Streamlit dashboard
streamlit run app.py
```

---

## Build roadmap

### Week 1–2 — Agent engine
- [ ] `agent.py` dataclass with all fields
- [ ] `personality_profiles.yaml` with all 16 MBTI types
- [ ] `prompt_generator.py` — OCEAN → system prompt
- [ ] Ollama connection test (`ollama.chat()` with Mistral)
- [ ] Smoke test: 5 agents read same article, verify ENFP ≠ ISTJ outputs
- [ ] `config/framing_config.yaml` with 3 article variants

### Week 3–4 — Simulation loop
- [ ] `environment.py` — LangGraph state graph
- [ ] `sim/interaction_modes/random_pairs.py` (baseline)
- [ ] `sim/interaction_modes/social_feed.py`
- [ ] `sim/interaction_modes/influencer_hub.py`
- [ ] `sim/interaction_modes/graph_network.py`
- [ ] ChromaDB per-agent memory integration
- [ ] `run_logger.py` — JSON state snapshot per turn
- [ ] `convergence_check.py`
- [ ] First full A/B run: fear vs neutral, 15 agents, 20 turns
- [ ] Verify P(t) curves differ between framings

### Week 5–6 — Analytics + Streamlit
- [ ] `analytics/metrics.py` — P(t), I(agent), E(t)
- [ ] `analytics/umap_viz.py` — UMAP 2D projection pipeline
- [ ] `analytics/graph_viz.py` — NetworkX + PyVis export
- [ ] `app.py` — full Streamlit dashboard with live updates
- [ ] Run full experiment: 3 framings × 4 modes = 12 simulation runs
- [ ] Save all results to `results/`
- [ ] Generate poster figure 1: UMAP cluster evolution (t=0, t=mid, t=final)
- [ ] Generate poster figure 2: social graph fracture sequence

### Week 7–8 — Analysis + polish
- [ ] Comparative analysis writeup: which framing → highest ΔP?
- [ ] Influence leaderboard: which MBTI type → highest I(agent)?
- [ ] Does Mode 2 (social feed) polarize more than Mode 1 (random pairs)?
- [ ] Dashboard UX polish for demo day
- [ ] 4-page project report
- [ ] 3-minute live demo script
- [ ] Antigravity submission package

---

## Key design decisions

**Why Ollama over OpenAI API**
With 15 agents × 30 turns × 2 LLM calls per interaction = ~900 calls per run, and 12 total runs for the full experiment, that is ~10,800 LLM calls. At GPT-4 pricing this is significant cost. With Ollama running Mistral 7B locally it is zero cost and fully reproducible.

**Why sentence-transformers for embeddings**
`all-MiniLM-L6-v2` produces 384-dimensional embeddings fast enough to compute after every turn without slowing the sim loop. Cosine similarity between these vectors is the backbone of all three metrics.

**Why LangGraph over vanilla Python**
LangGraph handles the turn-taking state machine natively — each agent's state, the inter-agent message passing, and the loop control are all first-class primitives. Rolling this by hand in plain Python produces fragile spaghetti by week 3.

**The epsilon decision as an LLM call**
Instead of computing |x_i - x_j| < ε numerically, Agent B's LLM decides "will I engage with this view?" given its personality. This is the core semantic contribution — the same opinion can be ignored by an ISTJ and embraced by an ENFP, which is not possible in any scalar model.

---

## Expected outputs

| Output | Format | When |
|---|---|---|
| Raw run logs | `results/run_<id>.json` | After each sim run |
| Polarization comparison | Plotly line chart | Live in dashboard |
| UMAP cluster evolution | 3-panel PNG | End of run |
| Social graph fracture | PyVis HTML animation | End of run |
| Influence leaderboard | CSV + bar chart | End of run |
| Final analysis | PDF report | Week 8 |

---

## Troubleshooting

**Ollama not responding**
```bash
ollama serve          # start the server
ollama list           # verify mistral is pulled
```

**ChromaDB collection conflicts across runs**
Each run creates per-agent collections named `agent_{id}_run_{timestamp}`. Delete stale collections with:
```python
import chromadb
client = chromadb.Client()
for col in client.list_collections():
    client.delete_collection(col.name)
```

**UMAP fails with < 4 samples**
UMAP needs at least 4 points. The dashboard skips the scatter plot for turns where fewer than 4 agents have formed opinions yet.

**LLM producing non-opinion outputs**
Tighten the user prompt: end it with `"State your opinion in exactly 2-4 sentences. Do not ask questions."` This reduces the rate of non-opinion outputs to near zero with Mistral.
