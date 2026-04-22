# 🧠 Opinion Dynamics Simulator

> **A multi-agent AI system where personalities shape beliefs, interactions create influence, and polarization emerges naturally.**

---

## 🌍 Overview

The **Opinion Dynamics Simulator** models how beliefs evolve inside a society of AI agents.

Each agent:

* Has a **personality (MBTI + OCEAN)**
* Reads and interprets a **shared news article**
* Interacts with other agents using **LLMs**
* Updates beliefs using a **bounded confidence model**

The result?
👉 A fully simulated digital society where **consensus, disagreement, and echo chambers emerge organically**.

---

## ⚡ Key Highlights

* 🧠 **LLM-powered reasoning agents**
* 🧬 **Personality-driven behavior (MBTI + OCEAN)**
* 🔗 **Hegselmann–Krause opinion dynamics**
* 📊 **Real-time analytics & metrics**
* 🎨 **Visualization of belief clusters**
* 🧪 **Research-ready simulation framework**

---

## 🧩 System Architecture

```mermaid
flowchart TD
    A[News Article Input] --> B[Agent Initialization]

    B --> C[Assign Personality]
    C --> D[MBTI + OCEAN Traits]

    D --> E[LLM Opinion Generation]
    E --> F[Initial Opinions]

    F --> G[Interaction Loop]

    G --> H[Agent Communication]
    H --> I[Opinion Similarity Check]

    I -->|Within ε| J[Accept Influence]
    I -->|Outside ε| K[Ignore Opinion]

    J --> L[Opinion Update]
    K --> G

    L --> M[Updated Beliefs]
    M --> G

    G --> N[Convergence Check]

    N -->|Stable| O[Final State]
    N -->|Continue| G
```

---

## 🧠 Opinion Update Logic

```mermaid
flowchart LR
    A[Agent A Opinion] --> B[Compare with Others]
    B --> C{Similarity ≥ ε ?}

    C -->|Yes| D[Include in Update Pool]
    C -->|No| E[Discard]

    D --> F[Average Accepted Opinions]
    F --> G[Apply Personality Bias]

    G --> H[New Opinion]
```

---

## 🧬 Personality Model

Each agent combines:

* **MBTI Archetype** → behavioral style
* **OCEAN Traits** → numerical influence factors

| Trait             | Role                  |
| ----------------- | --------------------- |
| Openness          | Willingness to change |
| Conscientiousness | Stability & structure |
| Extraversion      | Interaction frequency |
| Agreeableness     | Support vs opposition |
| Neuroticism       | Emotional sensitivity |

---

## 🔄 Simulation Flow

```mermaid
sequenceDiagram
    participant A as Agent A
    participant B as Agent B
    participant C as Agent C
    participant LLM as LLM Engine

    A->>LLM: Generate opinion
    B->>LLM: Generate opinion
    C->>LLM: Generate opinion

    A->>B: Share opinion
    B->>C: Share opinion
    C->>A: Share opinion

    A->>A: Evaluate similarity
    B->>B: Evaluate similarity
    C->>C: Evaluate similarity

    A->>A: Update belief
    B->>B: Update belief
    C->>C: Update belief
```

---

## 📊 Metrics & Insights

* 📈 **Polarization Index (P)**
* 🧱 **Echo Chamber Index (E)**
* 🎯 **Agent Influence Scores**
* 🧬 **MBTI-based analytics**
* ⏹️ **Convergence detection**

---

## 🎨 Visual Outputs

* Opinion evolution graphs
* Influence distribution charts
* Decision pattern breakdowns
* UMAP-based opinion clustering

---

## ⚙️ Configuration

```python
CONFIG = {
    "n_agents_per_type": 10,
    "n_turns": 20,
    "llm_model": "TinyLlama-...",
    "embedding_model": "all-MiniLM-L6-v2",
    "epsilon": 0.65,
    "batch_size": 4,
    "enable_rag_memory": True,
    "convergence_threshold": 0.95
}
```

---

## 🚀 Getting Started

### 🖥️ Local Setup

```bash
git clone <repo-url>
cd opinion-dynamics

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
python kaggle.py
```

---

### ☁️ Kaggle Setup

1. Create a new notebook
2. Paste `kaggle.py`
3. (Optional) Add Hugging Face token
4. Run 🚀

---

## 🧪 Example Behaviors

### 🏛️ Political Topics

* Strong clustering
* Persistent disagreement
* High polarization

### 🔬 Scientific Topics

* Faster convergence
* Lower conflict
* Weak clustering

### 🤖 Tech Policy

* Two stable opinion groups
* Boundary agents fluctuate

---

## 🧰 Tech Stack

* **PyTorch** → GPU acceleration
* **Transformers** → LLMs
* **Sentence Transformers** → embeddings
* **Plotly** → visualization
* **NetworkX** → interaction graphs
* **ChromaDB** → agent memory
* **UMAP** → clustering

---

## 📦 Output Structure

```bash
simulation_results.json
visualizations/
```

---

## 🔮 Future Work

* 🌐 Multi-topic simulations
* 🎯 Influencer injection
* 🔗 Real network topologies
* 📊 Causal reasoning
* 🤖 Multi-model ecosystems
* 🌍 Live news integration

---

## 💡 Key Insight

> **Polarization doesn’t require conflict—only selective trust.**

---

## 🤝 Contributing

PRs welcome!

---

## 📜 License

Hegselmann–Krause model (2002)

---

## 👨‍💻 About Me

**Manamnath Tiwari**  
AI/ML Developer  

📧 manamnathtiwari@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/manamnath-tiwari  
💻 GitHub: https://github.com/manamnathtiwari 

---

**Made with 🧠 + ⚡ + 📊**
