# 🧠 Opinion Dynamics Simulator

> **Simulating How Personalities Shape Belief Evolution in Multi-Agent Societies**

A cutting-edge multi-agent opinion dynamics simulator powered by advanced LLMs and personality theory, running on GPU with real-time analytics and visualization.

---

## 🎯 What This Does

This simulator creates a **digital society** of 160 AI agents with distinct personality types (MBTI + OCEAN traits) who engage in opinion discussions about any news article. Watch as:

- 📰 **Agents read and digest** news articles through their personality lens
- 💬 **Agents interact dynamically**, supporting or opposing each other's views
- 🧭 **Opinions evolve** based on personality traits, bounded confidence, and social influence
- 📊 **Polarization & echo chambers emerge** organically from local interactions
- 🎨 **Beautiful visualizations** show opinion clusters, influence networks, and convergence patterns

### Key Innovation: The Hegselmann-Krause Model

Traditional opinion models assume everyone converges to consensus. This simulator implements the **Hegselmann-Krause bounded confidence model**, which is far more realistic:

```
Agent A only considers Agent B's opinion if they're "similar enough"
(confidence threshold = 0.65 cosine similarity)
```

This simple rule creates **natural clustering** and **persistent disagreement clusters**—just like real societies! 🌍

---

## 🧬 The Science Behind It

### 1. **Personality-Driven Behavior** (MBTI + OCEAN)
Each agent has a unique personality profile:

| Trait | What It Controls |
|-------|-----------------|
| **Openness** | Willingness to consider new ideas |
| **Conscientiousness** | How structured and rule-following |
| **Extraversion** | Participation in discussions |
| **Agreeableness** | Tendency to support vs. oppose others |
| **Neuroticism** | Emotional reactivity |

**Example personality snapshots:**
- 🧠 **INTJ (The Architect)**: High conscientiousness, very skeptical, changes mind only with iron-clad logic
- 💫 **ENFP (The Campaigner)**: High openness, flexible, loves exploring alternatives
- 🛡️ **ISFJ (The Defender)**: High agreeableness, protective, values harmony

### 2. **Hegselmann-Krause Dynamics**

At each turn, agents:

1. **Look inward**: Review all other agents' opinions
2. **Filter by confidence**: Only consider agents with opinions within ε = 0.65 cosine similarity
3. **Update via averaging**: Pool opinions of "similar-thinking" peers
4. **Evolve beliefs**: Shift toward the average of accepted opinions
5. **Personality modulation**: OCEAN traits bias the direction and magnitude of shift

This creates **emergent polarization** without explicit conflict or hate speech—just incompatible worldviews! 🪞

### 3. **LLM-Powered Opinions**

Each agent uses an instruction-tuned LLM (TinyLlama-1.1B-Chat or Zephyr-7B) to:
- ✍️ **Write coherent, personality-consistent opinions** in natural language
- 🎭 **Respond to others' views** while maintaining character
- 🧠 **Reason through arguments** (not just keyword matching)
- 💾 **Remember previous interactions** via per-agent RAG memory (ChromaDB)

---

## 📊 What You Get

### Live Metrics
- ✅ **Polarization Index P(t)**: Measures opinion diversity over time
- ✅ **Echo Chamber Index E(t)**: Quantifies clustering and insularity
- ✅ **Agent Influence Scores**: Who shapes opinions most?
- ✅ **Per-MBTI Analytics**: Which personality types are most influential?
- ✅ **Convergence Detection**: Early stopping when beliefs stabilize

### Visualizations
```
📈 MBTI Influence Comparison
   └─ Average influence by personality type
   └─ Activity (# interactions) by type
   └─ Opinion change frequency

📊 Decision Patterns
   └─ Support/Oppose/Neutral rates by MBTI
   └─ Stacked bar showing preference distributions

🔄 Opinion Evolution Over Time
   └─ Interactions per turn (trend)
   └─ Influence accumulated per turn

🎨 UMAP Opinion Clusters
   └─ 2D projection of opinion embeddings
   └─ Color-coded by MBTI type
   └─ See clusters & polarization visually
```

### Raw Data Export
- Complete JSON export: opinions, interaction logs, statistics
- Per-agent opinion evolution history
- Interaction network (who influenced whom)
- Final opinion embeddings for further analysis

---

## 🚀 Quick Start

### 1. **Local Installation**
```bash
# Clone or navigate to project folder
cd "Opinion Dynamic"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the simulator
python kaggle.py
```

### 2. **On Kaggle**
1. Create a new notebook
2. Copy [kaggle.py](kaggle.py) into first cell
3. Add your Hugging Face token to environment (optional, for faster downloads)
4. Run! ✨

### 3. **Customize the Article**
When prompted, enter any news article text. Example:
```
Scientists discover potential cure for disease X...
[multiline input, press Enter twice to finish]
```

Or press Enter immediately to use the default political controversy article.

---

## ⚙️ Configuration

Edit `CONFIG` dict in `kaggle.py`:

```python
CONFIG = {
    "n_agents_per_type": 10,      # 10 × 16 types = 160 agents (↑ for massive sims)
    "n_turns": 20,                 # Interaction rounds (↑ for slow convergence)
    "llm_model": "TinyLlama-...",  # LLM backbone (swap for Zephyr-7B for quality)
    "embedding_model": "all-MiniLM-L6-v2",  # Semantic similarity
    "epsilon": 0.65,               # Confidence threshold (0.5-0.8 realistic)
    "max_opinion_length": 100,     # Tokens per opinion
    "batch_size": 4,               # GPU batch size
    "enable_rag_memory": True,     # Per-agent memory
    "convergence_threshold": 0.95, # Early stopping threshold
}
```

### Scaling Guide
| Config | Agents | Turns | Time (T4) | Realism |
|--------|--------|-------|-----------|---------|
| **Micro** | 32 | 10 | 3 min | Demo |
| **Standard** | 160 | 20 | 15 min | Good |
| **Large** | 480 | 30 | 60 min | Excellent |
| **Mega** | 1280 | 50 | 4+ hrs | Research-grade |

---

## 📚 The 16 MBTI Archetypes

### Thinking-Feelers (T/F axis controls logic vs. empathy)

| Type | Archetype | Stance | Key Strength |
|------|-----------|--------|--------------|
| **INTJ** | The Architect | Skeptical, evidence-driven | Logical consistency |
| **INTP** | The Logician | Ambivalent, open to trade-offs | Systemic modeling |
| **ENTJ** | The Commander | Pro-structure, efficiency-focused | Strategic clarity |
| **ENTP** | The Debater | Provocateur, anti-consensus | Finding weak arguments |
| **INFJ** | The Advocate | Values-driven regulation | Human dignity focus |
| **INFP** | The Mediator | Idealistic, anti-capture | Moral intuition |
| **ENFJ** | The Protagonist | Community-focused | Building consensus |
| **ENFP** | The Campaigner | Optimistic, creative | Exploring alternatives |
| **ISTJ** | The Logistician | Rule-based, practical | Institutional stability |
| **ISFJ** | The Defender | Protective, traditional | Social harmony |
| **ESTJ** | The Executive | Authority-respecting | Clear hierarchies |
| **ESFJ** | The Consul | Community bonds | Group welfare |
| **ISTP** | The Virtuoso | Pragmatic, minimalist | Real-world solutions |
| **ISFP** | The Adventurer | Values-driven, non-confrontational | Personal authenticity |
| **ESTP** | The Entrepreneur | Adaptable, flexible | Rapid response |
| **ESFP** | The Entertainer | People-focused, warm | Individual compassion |

Each type has distinct argument styles, stubbornness profiles, and confidence dynamics! 🎭

---

## 🔬 Example Runs

### Example 1: Political Controversy Article
**Article**: "Opposition leader calls PM a 'terrorist' (then clarifies)"

**Results**:
- INTJ/ENTJ cluster together (logical, skeptical)
- ENFJ/ISFJ cluster separately (empathy-driven)
- Persistent disagreement emerges even after 20 turns
- Polarization Index remains ~0.6 (moderate-to-high)

### Example 2: Scientific Breakthrough Article
**Article**: "AI researchers achieve breakthrough in medical diagnosis"

**Results**:
- Much faster consensus (turns 1-5)
- Weaker personality clustering (easier to agree on facts)
- Echo chamber index remains low
- All agents converge to similar positions

### Example 3: Tech Policy Article
**Article**: "Should AI systems be regulated before deployment?"

**Results**:
- Persistent two-cluster equilibrium
- INTJ/INTP favor light-touch; INFJ/ENFJ favor heavy regulation
- Boundary agents (ENTJ, ISFP) waver
- Opinion embeddings show clear "regulation-yes" vs "regulation-no" poles

---

## 🎓 Research Applications

This simulator is useful for:

✅ **Social Science Research**
- Understanding personality's role in polarization
- Testing interventions (modify epsilon or OCEAN traits)
- Studying opinion cascades and tipping points

✅ **LLM Behavior Analysis**
- Do models exhibit personality-like traits?
- How do instruction-tuned vs. base models differ?
- Bias detection in multi-turn interactions

✅ **AI Safety & Alignment**
- Simulating multi-agent agreement processes
- Testing consensus-building algorithms
- Understanding failure modes in belief formation

✅ **Education & Visualization**
- Teach opinion dynamics, complex systems, personality theory
- Beautiful interactive diagrams for presentations
- Demo "how societies form conflicting beliefs"

---

## 📦 Dependencies

```
torch>=2.0.0              # GPU acceleration
transformers>=4.30       # LLM loading
sentence-transformers    # Opinion embeddings
plotly                   # Visualization
numpy, pandas            # Data handling
tqdm                     # Progress bars
networkx                 # Network analysis
chromadb                 # Per-agent RAG memory (optional)
umap-learn               # UMAP clustering (optional)
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

### "CUDA kernel not available"
→ Switch Kaggle GPU to **T4 × 2** (not P100 or V100)

### "bitsandbytes ImportError"
→ Automatic fallback to regular model loading (no quantization needed)

### "Out of memory"
→ Lower `n_agents_per_type` or `batch_size` in CONFIG

### "Very slow generation"
→ Model is running on CPU; check GPU detection message at startup

### "Opinion embeddings all zeros"
→ Sentence-transformer failing; check internet connection and HF token

---

## 🎨 Output Files

After running, you'll find:

```
simulation_results.json
├─ config: Parameters used
├─ mbti_stats: Per-type analytics
├─ turn_logs: Interaction records per turn
├─ final_opinions: Each agent's final belief + metadata
└─ convergence_info: Whether & when beliefs stabilized

visualizations/
├─ influence_chart.html
├─ decision_patterns.html
├─ evolution_over_time.html
└─ opinion_clusters_umap.html
```

---

## 🔮 Future Enhancements

- 🌐 **Multi-article cascades**: How beliefs persist across topics
- 🎯 **Influence interventions**: Add "influencers" or "media outlets"
- 🔗 **Social network topology**: Small-world vs. scale-free structures
- 🎓 **Belief strength**: Not just direction, but confidence
- 🗣️ **Real web search**: Connect to live news APIs
- 🤖 **Heterogeneous LLMs**: Mix different model families
- 📊 **Causal analysis**: "What changed this agent's mind?"

---

## 📜 License & Citation

This project implements the **Hegselmann-Krause model** (Hegselmann & Krause, 2002) and MBTI/OCEAN personality theory.

**Citation**:
```bibtex
@article{hegselmann2002opinion,
  title={Opinion dynamics and bounded confidence},
  author={Hegselmann, R. and Krause, U.},
  journal={Journal of Artificial Societies and Social Simulation},
  year={2002}
}
```

**MBTI Reference**: Myers-Briggs Type Indicator® (MBTI®)
**OCEAN Reference**: Costa & McCrae's Five-Factor Personality Model

---

## 💡 Key Insights

> **"Polarization doesn't require malice, just diversity in what people find convincing."**

When agents only update beliefs based on similar thinkers (bounded confidence), the system naturally fragments. This is not a bug—it's a feature! It shows how **healthy democracies require bridging institutions** that connect opinion clusters.

---

## 🤝 Contributing

Found a bug? Have an idea? Open an issue or PR!

### Development Roadmap
- [ ] Multi-GPU distributed simulation
- [ ] Real-time streaming dashboard
- [ ] API for external opinion providers
- [ ] Interactive web interface

---

## 📧 Questions?

Check the code comments—they're extensive! Or review the docstrings in key classes:
- `Agent`: Individual agent behavior
- `LLMInterface`: Opinion generation logic
- `OpinionDynamicsSimulator`: Orchestration & HK update
- `analyze_results()`: Metrics computation

---

## 🎉 Enjoy Simulating!

Run the simulator, explore the opinions, and watch as **emergent societal dynamics** unfold from simple local rules.

**Remember**: Polarization is not inevitable—it's a consequence of structure. Change the structure, change the outcome. 🧩

---

<div align="center">

**Made with 🧠 + ⚡ + 📊**

*Where personality meets opinion dynamics*

</div>
#   o p i n i o n _ d y n a m i c s  
 