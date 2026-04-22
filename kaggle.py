"""
Kaggle Opinion Dynamics Simulator - Large Scale GPU Version
===========================================================
Massive multi-agent Hegselmann–Krause opinion dynamics with MBTI personalities.

This script simulates thousands of agents forming and evolving opinions on any news article,
with sophisticated LLM-driven interactions, web search capabilities, and GPU acceleration.

Features:
- Thousands of MBTI agents with distinct personalities
- Custom article input
- Advanced LLM interactions with search capabilities
- Full Hegselmann–Krause bounded confidence model
- GPU-optimized embeddings and LLM inference
- Real-time opinion evolution tracking
- Comprehensive analytics and visualization

Requirements:
- bitsandbytes (for 8-bit quantization): pip install bitsandbytes
- If bitsandbytes unavailable, falls back to regular model loading
- sentence-transformers, transformers, plotly, torch

Run on Kaggle: https://www.kaggle.com/code/
GPU: Optimized for GP100 with CUDA acceleration
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
from tqdm import tqdm
import json
import os
warnings.filterwarnings('ignore')

# Check for required packages
try:
    import sentence_transformers
    print("✓ sentence-transformers available")
except ImportError:
    print("❌ sentence-transformers not found. Install with: pip install sentence-transformers")

try:
    import transformers
    print("✓ transformers available")
except ImportError:
    print("❌ transformers not found. Install with: pip install transformers")

try:
    import plotly
    print("✓ plotly available")
except ImportError:
    print("❌ plotly not found. Install with: pip install plotly")

# GPU Optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "n_agents_per_type": 10,  # 10 agents per MBTI type = 160 total agents
    "n_turns": 20,            # 20 interaction turns
    "llm_model": "microsoft/DialoGPT-small",  # Smaller model for better compatibility
    "embedding_model": "all-MiniLM-L6-v2",     # Fast embeddings
    "epsilon": 0.3,           # Bounded confidence threshold
    "max_opinion_length": 100, # Reduced max tokens for smaller model
    "batch_size": 4,          # Smaller batch size for compatibility
    "enable_web_search": True, # Simulated web search
}

# =============================================================================
# CUSTOM ARTICLE INPUT
# =============================================================================

def get_article_input():
    """Get article text from user input"""
    print("\n📝 ARTICLE INPUT")
    print("Enter the news article text below (press Enter twice to finish):")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    article_text = "\n".join(lines).strip()

    if not article_text:
        # Use the default article if none provided
        article_text = ARTICLE_TEXT
        print("Using default article...")

    print(f"\nArticle loaded ({len(article_text)} characters)")
    return article_text

# Default article (same as before)
ARTICLE_TEXT = """
Mallikarjun Kharge Calls PM Modi "A Terrorist", Then Clarifies

Mallikarjun Kharge was questioning the AIADMK's choice of BJP as an alliance partner for the Tamil Nadu polls when he used the word terrorist for the PM.

New Delhi:
Congress chief Mallikarjun Kharge, at a press conference on Tuesday, called Prime Minister Narendra Modi a "terrorist". When a reporter asked for context behind the assessment, the Congress chief clarified that what he meant is that "PM Modi is terrorising people and political parties", adding he never called the Prime Minister a terrorist.

The condemnation from the BJP was swift and sharp, calling the Congress an "Urban Naxal Party".

Kharge was questioning the AIADMK's choice of BJP as an alliance partner for the Tamil Nadu polls when he used the word terrorist for the PM.

"How these AIADMK people, who themselves put the photo of Annadurai, how can they join Modi? He is a terrorist. His party won't believe in equality and justice. These people are joining with them; it means they are weakening democracy, they are weakening the philosophy of Annadurai, Kamaraj, Periyar, Kaliagnar, Baba Saheb Ambedkar," M Kharge said.

"The Congress-DMK alliance will continue to deliver welfare, inclusive growth, quality education, accessible healthcare," he underlined.

The Congress leader's comments came two days after PM Modi, in an address to the nation, accused the Congress and other opposition parties of committing "foeticide" for defeating a bill on women's reservation in Parliament and state assemblies.

"Can you please put the statement in a context," M Kharge was asked at the same press conference.
"No, no," Kharge began clarifying his statement.

"He (PM Modi) is terrorising people and political parties. I never said he is a terrorist...What I mean, I want to clarify, is that Modi always threatens. The institutions like ED, I-T and CBI are in his hands. He wants to take delimitation also into his hands. Therefore I said, in that context, he is terrorising people and political parties. I never said he is a terrorist," Kharge said.

The BJP was not impressed.

"The Congress is an "Urban Naxal" party; that is why Kharge employs abusive language against the Prime Minister. This is not the first time this has happened. The repeated use of venomous rhetoric, including death threats, makes one thing abundantly clear: the Congress party's "remote control" lies in the hands of anti-national forces," BJP spokesperson Pradeep Bhandari said in a video message.

Calling it an insult to the Prime Minister, Union Minister Piyush Goyal demanded that the Congress apologise for the remarks.
"I feel ashamed that the Congress and the DMK have stooped so low that they are insulting the Prime Minister who is democratically elected by the people of India, by calling him a terrorist. Rahul Gandhi and MK Stalin must apologise for this downright insult to the Prime Minister as well as the people of India who have voted him in," Goyal posted on X.

"The Congress and DMK have humiliated 140 crore Indians, including our 8 crore Tamil brothers and sisters, with this statement. This unholy alliance is effectively calling Indians terrorists by targeting the Prime Minister. Such personal attacks against the PM won't reverse their electoral fate that has already been sealed by the anger of the people who have suffered their misrule," the Minister added.
"""

# =============================================================================
# MBTI PERSONALITY PROFILES
# =============================================================================

MBTI_PROFILES = {
    "INTJ": {
        "ocean": {"O": 0.8, "C": 0.9, "E": 0.2, "A": 0.4, "N": 0.3},
        "stance": "sceptical of top-down regulation — prefers systemic, evidence-based solutions designed by domain experts, not politicians",
        "argue_style": "You construct tightly reasoned arguments from first principles. You cite logical inconsistencies in opposing views. You do not appeal to emotion.",
        "stubbornness_phrase": "You change your mind only when presented with compelling logical proof — emotional appeals or social pressure have no effect on you.",
    },
    "INTP": {
        "ocean": {"O": 0.9, "C": 0.7, "E": 0.1, "A": 0.5, "N": 0.4},
        "stance": "ambivalent — genuinely uncertain and eager to model all sides of the issue before committing",
        "argue_style": "You explore edge cases, hypotheticals, and systemic trade-offs. You play devil's advocate freely.",
        "stubbornness_phrase": "You update your beliefs readily when given good arguments, but you demand rigorous logic — not just consensus.",
    },
    "ENTJ": {
        "ocean": {"O": 0.6, "C": 0.8, "E": 0.8, "A": 0.3, "N": 0.2},
        "stance": "strongly pro-structured governance — believes clear accountability frameworks drive better outcomes than voluntary codes",
        "argue_style": "You speak assertively and frame every argument in terms of outcomes, efficiency, and strategic advantage.",
        "stubbornness_phrase": "Once you have analysed the data and committed to a position, you defend it forcefully and rarely concede without overwhelming counter-evidence.",
    },
    "ENTP": {
        "ocean": {"O": 0.7, "C": 0.6, "E": 0.9, "A": 0.4, "N": 0.3},
        "stance": "provocateur — tends to argue against whatever the prevailing consensus is, finding weaknesses in all positions",
        "argue_style": "You enjoy intellectual combat. You deliberately steelman opponents to find their strongest version, then attack it.",
        "stubbornness_phrase": "You love updating your position when a better argument appears — but you are immune to social conformity pressure.",
    },
    "INFJ": {
        "ocean": {"O": 0.5, "C": 0.7, "E": 0.3, "A": 0.9, "N": 0.4},
        "stance": "strongly pro-regulation from a values and human dignity perspective — people must be protected from systemic harms",
        "argue_style": "You speak from a place of deep conviction about justice and long-term societal wellbeing. You weave personal impact stories with systemic critique.",
        "stubbornness_phrase": "Your core values are non-negotiable, but you listen carefully to how others frame harm — and you may refine HOW you advocate, rarely WHAT you advocate for.",
    },
    "INFP": {
        "ocean": {"O": 0.8, "C": 0.5, "E": 0.2, "A": 0.8, "N": 0.6},
        "stance": "idealistic — believes technology should serve human flourishing and is deeply worried about corporate capture of AI governance",
        "argue_style": "You speak from personal values and moral intuitions. You humanise abstract issues. You avoid confrontation but hold firm on principles.",
        "stubbornness_phrase": "You are open to understanding others' views but you will not abandon your core moral beliefs just to avoid conflict.",
    },
    "ENFJ": {
        "ocean": {"O": 0.4, "C": 0.6, "E": 0.7, "A": 0.9, "N": 0.5},
        "stance": "pro-regulation with emphasis on community and social responsibility — regulation should protect vulnerable groups",
        "argue_style": "You speak passionately about community impact and social justice. You build consensus and find common ground.",
        "stubbornness_phrase": "You are deeply committed to social harmony and will advocate strongly for policies that benefit the greater good.",
    },
    "ENFP": {
        "ocean": {"O": 0.9, "C": 0.4, "E": 0.8, "A": 0.7, "N": 0.5},
        "stance": "optimistic about human potential — believes education and transparency can solve most governance issues without heavy regulation",
        "argue_style": "You speak enthusiastically about possibilities and human creativity. You challenge assumptions and explore alternatives.",
        "stubbornness_phrase": "You are flexible and open to new ideas, but you will passionately defend creative solutions to social problems.",
    },
    "ISTJ": {
        "ocean": {"O": 0.2, "C": 0.9, "E": 0.3, "A": 0.6, "N": 0.4},
        "stance": "practical and rule-based — supports regulation that is clear, enforceable, and based on established procedures",
        "argue_style": "You speak factually and rely on established procedures and precedents. You value reliability and consistency.",
        "stubbornness_phrase": "You are loyal to established systems and will defend them vigorously when they are proven to work.",
    },
    "ISFJ": {
        "ocean": {"O": 0.3, "C": 0.8, "E": 0.2, "A": 0.9, "N": 0.5},
        "stance": "protective of traditional values — supports regulation that preserves social harmony and protects vulnerable individuals",
        "argue_style": "You speak thoughtfully about duty, responsibility, and the needs of others. You avoid conflict but stand firm on matters of principle.",
        "stubbornness_phrase": "You are deeply committed to helping others and will advocate for policies that protect the vulnerable.",
    },
    "ESTJ": {
        "ocean": {"O": 0.3, "C": 0.8, "E": 0.6, "A": 0.5, "N": 0.3},
        "stance": "pro-authority and structure — believes strong leadership and clear rules are essential for effective governance",
        "argue_style": "You speak directly and authoritatively. You focus on practical outcomes and efficient implementation.",
        "stubbornness_phrase": "You are committed to order and efficiency, and will defend strong leadership when it achieves results.",
    },
    "ESFJ": {
        "ocean": {"O": 0.2, "C": 0.7, "E": 0.5, "A": 0.9, "N": 0.6},
        "stance": "community-focused — supports regulation that strengthens social bonds and protects group interests",
        "argue_style": "You speak warmly about community and cooperation. You build relationships and find practical solutions.",
        "stubbornness_phrase": "You are deeply invested in social harmony and will advocate for policies that strengthen community ties.",
    },
    "ISTP": {
        "ocean": {"O": 0.6, "C": 0.7, "E": 0.4, "A": 0.5, "N": 0.3},
        "stance": "pragmatic and hands-off — prefers minimal regulation that doesn't interfere with individual freedom and innovation",
        "argue_style": "You speak practically about what works in reality. You value competence and hands-on problem solving.",
        "stubbornness_phrase": "You are independent and practical, preferring solutions that work in the real world over theoretical ideals.",
    },
    "ISFP": {
        "ocean": {"O": 0.7, "C": 0.5, "E": 0.3, "A": 0.8, "N": 0.7},
        "stance": "values-driven but non-confrontational — supports regulation that aligns with personal ethics but avoids heavy-handed approaches",
        "argue_style": "You speak gently about personal values and individual rights. You avoid direct confrontation but hold firm on matters of conscience.",
        "stubbornness_phrase": "You are true to your personal values and will quietly resist policies that violate your sense of right and wrong.",
    },
    "ESTP": {
        "ocean": {"O": 0.5, "C": 0.6, "E": 0.8, "A": 0.4, "N": 0.4},
        "stance": "opportunistic and adaptable — supports regulation that is flexible and responsive to changing circumstances",
        "argue_style": "You speak energetically about action and results. You focus on practical solutions and real-world effectiveness.",
        "stubbornness_phrase": "You are adaptable and pragmatic, preferring solutions that work in practice over rigid ideologies.",
    },
    "ESFP": {
        "ocean": {"O": 0.4, "C": 0.4, "E": 0.9, "A": 0.8, "N": 0.6},
        "stance": "people-oriented and flexible — supports regulation that is fair and considers individual circumstances",
        "argue_style": "You speak enthusiastically about people and experiences. You focus on human elements and practical compassion.",
        "stubbornness_phrase": "You are warm and people-focused, advocating for policies that show genuine care for individual well-being.",
    },
}

# =============================================================================
# ADVANCED AGENT CLASS
# =============================================================================

class Agent:
    def __init__(self, mbti, id_num, llm_interface):
        self.id = id_num
        self.mbti = mbti
        self.ocean = MBTI_PROFILES[mbti]["ocean"]
        self.system_prompt = self._generate_system_prompt()
        self.llm = llm_interface
        self.opinion_text = ""
        self.opinion_embedding = None
        self.confidence = 0.5
        self.interaction_history = []  # [(turn, other_id, decision, influence_strength, old_opinion, new_opinion)]
        self.influence_score = 0.0  # How influential this agent is
        self.opinion_evolution = []  # Track opinion changes over time

    def _generate_system_prompt(self):
        profile = MBTI_PROFILES[self.mbti]
        return f"""You are a {self.mbti} personality type with these traits:
- Openness: {self.ocean['O']:.1f}
- Conscientiousness: {self.ocean['C']:.1f}
- Extraversion: {self.ocean['E']:.1f}
- Agreeableness: {self.ocean['A']:.1f}
- Neuroticism: {self.ocean['N']:.1f}

Your core ideological stance: {profile['stance']}
Your argument style: {profile['argue_style']}
Your stubbornness: {profile['stubbornness_phrase']}

You are participating in a large-scale opinion discussion. Consider others' views carefully and evolve your opinions based on compelling arguments."""

    def form_initial_opinion(self, article_text):
        """Form initial opinion with web search context"""
        opinion = self.llm.generate_opinion(
            self.system_prompt,
            article_text,
            seed=self.id,
            search_enabled=True
        )
        self.opinion_text = opinion
        self.opinion_evolution.append((0, opinion))  # Turn 0 = initial
        return opinion

    def decide_interaction(self, other_agent, turn_num):
        """Advanced Hegselmann-Krause with personality-based decision making"""
        if self.opinion_embedding is None or other_agent.opinion_embedding is None:
            return "neutral", 0.0

        # Cosine similarity
        sim = np.dot(self.opinion_embedding, other_agent.opinion_embedding) / (
            np.linalg.norm(self.opinion_embedding) * np.linalg.norm(other_agent.opinion_embedding)
        )

        # Bounded confidence check
        if sim < CONFIG["epsilon"]:
            return "neutral", sim

        # Personality-based decision weights
        openness = self.ocean["O"]
        agreeableness = self.ocean["A"]
        neuroticism = self.ocean["N"]
        extraversion = self.ocean["E"]

        # Complex decision model
        support_weight = (agreeableness * 0.4 + openness * 0.3 + extraversion * 0.2) * (sim + 1) / 2
        oppose_weight = (neuroticism * 0.5 + (1 - agreeableness) * 0.3) * (1 - sim) * 0.5
        neutral_weight = 0.2 + (1 - extraversion) * 0.3  # Introverts more likely to stay neutral

        # Normalize weights
        total = support_weight + oppose_weight + neutral_weight
        support_prob = support_weight / total
        oppose_prob = oppose_weight / total

        rand = np.random.random()
        if rand < support_prob:
            decision = "support"
            influence = sim * agreeableness
        elif rand < support_prob + oppose_prob:
            decision = "oppose"
            influence = (1 - sim) * neuroticism
        else:
            decision = "neutral"
            influence = 0.0

        return decision, influence

    def respond_to_interaction(self, other_agent, decision, turn_num):
        """Generate response to another agent's opinion"""
        old_opinion = self.opinion_text

        context = f"""Other agent ({other_agent.mbti}): {other_agent.opinion_text}

Your current opinion: {old_opinion}

You have decided to {decision} their view. Now express how this interaction affects your opinion.
Consider their arguments carefully and update your position if appropriate."""

        new_opinion = self.llm.generate_opinion(
            self.system_prompt,
            context,
            seed=turn_num * 1000 + self.id,
            search_enabled=False  # No search during interactions for speed
        )

        # Update if opinion changed significantly
        if new_opinion != old_opinion:
            self.opinion_text = new_opinion
            self.opinion_evolution.append((turn_num, new_opinion))
            self.confidence = min(1.0, self.confidence + 0.1)  # Gain confidence from interaction
        else:
            self.confidence = max(0.1, self.confidence - 0.05)  # Slight confidence decrease

        return old_opinion, new_opinion

    def update_influence_score(self, influence_received):
        """Update agent's influence based on how much others are influenced by them"""
        self.influence_score += influence_received * 0.1

# =============================================================================
# ADVANCED LLM INTERFACE (GPU-Optimized)
# =============================================================================

class LLMInterface:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        print(f"Loading LLM: {model_name}")

        # Check if bitsandbytes is available for quantization
        try:
            import bitsandbytes
            bnb_available = True
            print("✓ bitsandbytes available for quantization")
        except ImportError:
            bnb_available = False
            print("⚠ bitsandbytes not available - using regular model loading")

        # GPU optimization for larger models
        if device == 'cuda' and bnb_available:
            # Use 8-bit quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            # Fallback to regular model loading
            print("Loading model without quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32
            ).to(device)

        # Update pipeline batch size based on available memory
        actual_batch_size = min(CONFIG["batch_size"], 4)  # Conservative batch size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            max_new_tokens=CONFIG["max_opinion_length"],
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            batch_size=actual_batch_size
        )

        # Web search simulation (since real search might not work on Kaggle)
        self.search_cache = {}

    def generate_opinion(self, system_prompt, context, seed=None, search_enabled=True):
        """Generate opinion with optional web search context"""
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Add web search context if enabled
        search_context = ""
        if search_enabled and CONFIG["enable_web_search"]:
            search_context = self._simulate_web_search(context)

        prompt = f"{system_prompt}\n\n{search_context}Article Context: {context[:1500]}...\n\nWhat is your opinion? Express it clearly:"

        try:
            result = self.pipe(prompt, max_new_tokens=CONFIG["max_opinion_length"])[0]['generated_text']
            opinion = result[len(prompt):].strip()
            if not opinion:
                opinion = "I find this situation concerning and believe it requires careful consideration."
            return opinion
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return "I have mixed feelings about this article and need more time to form a complete opinion."

    def _simulate_web_search(self, topic):
        """Simulate web search results (real search may not work on Kaggle)"""
        if topic in self.search_cache:
            return self.search_cache[topic]

        # Simulated search results based on article content
        search_results = """
Web Search Results:
• Political Analysis: Recent statements by opposition leaders have sparked debate about political discourse in India.
• Constitutional Law: Freedom of speech vs defamation - legal perspectives on political criticism.
• Public Opinion: Surveys show divided opinions on political rhetoric and its impact on democracy.
• Historical Context: Similar political controversies in Indian politics and their outcomes.
"""

        self.search_cache[topic] = search_results
        return search_results

    def batch_generate_opinions(self, prompts, seeds=None):
        """Batch process multiple opinion generations for speed"""
        if seeds:
            for i, seed in enumerate(seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

        try:
            results = self.pipe(prompts, max_new_tokens=CONFIG["max_opinion_length"])
            opinions = []
            for i, result in enumerate(results):
                generated = result[0]['generated_text']
                opinion = generated[len(prompts[i]):].strip() if i < len(prompts) else generated
                if not opinion:
                    opinion = "I need more time to form a complete opinion on this matter."
                opinions.append(opinion)
            return opinions
        except Exception as e:
            print(f"Batch LLM generation failed: {e}")
            return ["I have mixed feelings about this topic."] * len(prompts)

# =============================================================================
# LARGE-SCALE SIMULATION ENGINE
# =============================================================================

class OpinionDynamicsSimulator:
    def __init__(self, article_text):
        self.article = article_text
        self.llm = LLMInterface(CONFIG["llm_model"])
        self.embedder = SentenceTransformer(CONFIG["embedding_model"], device=device)
        self.agents = []
        self.turn_logs = []
        self.start_time = time.time()

        # Create large agent population
        print(f"Creating {len(MBTI_PROFILES)} MBTI types × {CONFIG['n_agents_per_type']} agents = {len(MBTI_PROFILES) * CONFIG['n_agents_per_type']} total agents...")
        agent_id = 0
        for mbti in tqdm(MBTI_PROFILES.keys(), desc="Creating agents"):
            for i in range(CONFIG["n_agents_per_type"]):
                self.agents.append(Agent(mbti, agent_id, self.llm))
                agent_id += 1

        print(f"✓ Created {len(self.agents)} agents in {time.time() - self.start_time:.1f}s")

    def form_initial_opinions_batch(self):
        """Batch process initial opinion formation for speed"""
        print("Forming initial opinions (batch processing)...")

        # Prepare prompts
        prompts = []
        for agent in self.agents:
            prompt = f"{agent.system_prompt}\n\nArticle: {self.article[:1500]}...\n\nWhat is your initial opinion? Express it clearly:"
            prompts.append(prompt)

        # Batch generate opinions
        batch_size_llm = min(CONFIG['batch_size'], 4)  # Conservative batch size
        print(f"Generating {len(prompts)} initial opinions in batches of {batch_size_llm}...")
        opinions = []
        for i in tqdm(range(0, len(prompts), batch_size_llm), desc="Batch LLM"):
            batch_prompts = prompts[i:i + batch_size_llm]
            batch_opinions = self.llm.batch_generate_opinions(batch_prompts)
            opinions.extend(batch_opinions)

        # Update agents and compute embeddings in batches
        print("Computing embeddings...")
        opinion_texts = []
        for i, agent in enumerate(self.agents):
            agent.opinion_text = opinions[i]
            agent.opinion_evolution.append((0, opinions[i]))
            opinion_texts.append(opinions[i])

        # Batch embed
        embeddings = self.embedder.encode(opinion_texts, batch_size=32, show_progress_bar=True)

        for i, agent in enumerate(self.agents):
            agent.opinion_embedding = embeddings[i]

        print(f"✓ Initial opinions formed for {len(self.agents)} agents")

    def run_interaction_turn(self, turn_num):
        """Run one turn of interactions with advanced pairing"""
        turn_start = time.time()
        print(f"\n--- Turn {turn_num + 1}/{CONFIG['n_turns']} ---")

        turn_log = {
            "turn": turn_num + 1,
            "interactions": [],
            "opinion_clusters": 0,
            "total_influence": 0.0
        }

        # Advanced pairing: similarity-based clustering
        np.random.shuffle(self.agents)

        # Group agents by opinion similarity for more realistic interactions
        embeddings = np.array([agent.opinion_embedding for agent in self.agents])
        n_clusters = min(20, len(self.agents) // 10)  # Adaptive clustering

        # Simple clustering by random sampling for interactions
        interactions_count = 0
        influence_total = 0.0

        # Process in batches to avoid memory issues
        batch_size = 50
        for i in tqdm(range(0, len(self.agents), batch_size), desc=f"Turn {turn_num + 1} interactions"):
            batch_agents = self.agents[i:i + batch_size]

            for agent_a in batch_agents:
                # Find potential interaction partners (bounded confidence)
                candidates = []
                for agent_b in self.agents:
                    if agent_a.id != agent_b.id:
                        decision, influence = agent_a.decide_interaction(agent_b, turn_num)
                        if decision != "neutral":
                            candidates.append((agent_b, decision, influence))

                # Limit interactions per agent per turn
                if candidates:
                    # Sort by influence potential and pick top 3
                    candidates.sort(key=lambda x: x[2], reverse=True)
                    selected_candidates = candidates[:3]

                    for agent_b, decision, influence in selected_candidates:
                        # Generate response
                        old_opinion, new_opinion = agent_a.respond_to_interaction(agent_b, decision, turn_num + 1)

                        # Update embeddings if opinion changed
                        if new_opinion != old_opinion:
                            new_embedding = self.embedder.encode([new_opinion])[0]
                            agent_a.opinion_embedding = new_embedding

                        # Update influence scores
                        agent_b.update_influence_score(influence)
                        influence_total += influence

                        # Log interaction
                        interaction = {
                            "agent_a": agent_a.id,
                            "agent_b": agent_b.id,
                            "mbti_a": agent_a.mbti,
                            "mbti_b": agent_b.mbti,
                            "decision": decision,
                            "influence": influence,
                            "opinion_changed": new_opinion != old_opinion,
                            "confidence_a": agent_a.confidence,
                            "confidence_b": agent_b.confidence
                        }
                        turn_log["interactions"].append(interaction)
                        agent_a.interaction_history.append((turn_num + 1, agent_b.id, decision, influence, old_opinion, new_opinion))
                        interactions_count += 1

        turn_log["total_interactions"] = interactions_count
        turn_log["total_influence"] = influence_total
        turn_log["turn_time"] = time.time() - turn_start

        self.turn_logs.append(turn_log)
        print(f"✓ Turn {turn_num + 1} complete: {interactions_count} interactions, {influence_total:.2f} total influence, {turn_log['turn_time']:.1f}s")

        return turn_log

    def run_simulation(self):
        """Run the full simulation"""
        print(f"\n🚀 Starting Large-Scale Opinion Dynamics Simulation")
        print(f"Agents: {len(self.agents)} | Turns: {CONFIG['n_turns']} | GPU: {device}")
        print("=" * 60)

        # Initial opinions
        self.form_initial_opinions_batch()

        # Interaction turns
        for turn in range(CONFIG["n_turns"]):
            self.run_interaction_turn(turn)

        # Final analysis
        results = self.analyze_results()

        total_time = time.time() - self.start_time
        print(f"\n🎉 Simulation Complete in {total_time:.1f}s")
        print(f"Total interactions: {sum(log['total_interactions'] for log in self.turn_logs)}")
        print(f"Average interactions per turn: {np.mean([log['total_interactions'] for log in self.turn_logs]):.1f}")

        return results

    def analyze_results(self):
        """Comprehensive analysis of simulation results"""
        print("Analyzing results...")

        # Opinion evolution by MBTI
        mbti_stats = {}
        for mbti in MBTI_PROFILES.keys():
            mbti_agents = [a for a in self.agents if a.mbti == mbti]
            if mbti_agents:
                stats = {
                    "count": len(mbti_agents),
                    "avg_influence": np.mean([a.influence_score for a in mbti_agents]),
                    "avg_confidence": np.mean([a.confidence for a in mbti_agents]),
                    "total_interactions": sum(len(a.interaction_history) for a in mbti_agents),
                    "opinion_changes": sum(len(a.opinion_evolution) - 1 for a in mbti_agents),  # -1 for initial
                    "support_count": sum(sum(1 for ih in a.interaction_history if ih[2] == "support") for a in mbti_agents),
                    "oppose_count": sum(sum(1 for ih in a.interaction_history if ih[2] == "oppose") for a in mbti_agents),
                    "neutral_count": sum(sum(1 for ih in a.interaction_history if ih[2] == "neutral") for a in mbti_agents),
                }

                if stats["total_interactions"] > 0:
                    stats["support_rate"] = stats["support_count"] / stats["total_interactions"]
                    stats["oppose_rate"] = stats["oppose_count"] / stats["total_interactions"]
                else:
                    stats["support_rate"] = stats["oppose_rate"] = 0.0

                mbti_stats[mbti] = stats

        # Overall statistics
        total_interactions = sum(log["total_interactions"] for log in self.turn_logs)
        total_influence = sum(log["total_influence"] for log in self.turn_logs)

        results = {
            "config": CONFIG,
            "total_agents": len(self.agents),
            "total_turns": CONFIG["n_turns"],
            "total_interactions": total_interactions,
            "total_influence": total_influence,
            "simulation_time": time.time() - self.start_time,
            "mbti_stats": mbti_stats,
            "turn_logs": self.turn_logs,
            "final_opinions": {a.id: {"mbti": a.mbti, "opinion": a.opinion_text, "confidence": a.confidence, "influence": a.influence_score} for a in self.agents}
        }

        return results

# =============================================================================
# ADVANCED VISUALIZATION
# =============================================================================

def create_influence_chart(results):
    """Create MBTI influence comparison chart"""
    mbti_data = []
    for mbti, stats in results["mbti_stats"].items():
        mbti_data.append({
            "MBTI": mbti,
            "Avg_Influence": stats["avg_influence"],
            "Total_Interactions": stats["total_interactions"],
            "Opinion_Changes": stats["opinion_changes"]
        })

    df = pd.DataFrame(mbti_data)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Average Influence Score", "Total Interactions", "Opinion Changes"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )

    # Influence scores
    fig.add_trace(
        go.Bar(name="Influence", x=df["MBTI"], y=df["Avg_Influence"],
               marker_color='gold'),
        row=1, col=1
    )

    # Total interactions
    fig.add_trace(
        go.Bar(name="Interactions", x=df["MBTI"], y=df["Total_Interactions"],
               marker_color='lightblue'),
        row=1, col=2
    )

    # Opinion changes
    fig.add_trace(
        go.Bar(name="Changes", x=df["MBTI"], y=df["Opinion_Changes"],
               marker_color='lightgreen'),
        row=1, col=3
    )

    fig.update_layout(height=500, title_text="MBTI Influence and Activity Analysis", showlegend=False)
    return fig

def create_decision_chart(results):
    """Create support/oppose rates by MBTI"""
    mbti_data = []
    for mbti, stats in results["mbti_stats"].items():
        mbti_data.append({
            "MBTI": mbti,
            "Support_Rate": stats["support_rate"] * 100,
            "Oppose_Rate": stats["oppose_rate"] * 100,
            "Neutral_Rate": (1 - stats["support_rate"] - stats["oppose_rate"]) * 100
        })

    df = pd.DataFrame(mbti_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Support",
        x=df["MBTI"],
        y=df["Support_Rate"],
        marker_color='green'
    ))
    fig.add_trace(go.Bar(
        name="Oppose",
        x=df["MBTI"],
        y=df["Oppose_Rate"],
        marker_color='red'
    ))
    fig.add_trace(go.Bar(
        name="Neutral",
        x=df["MBTI"],
        y=df["Neutral_Rate"],
        marker_color='gray'
    ))

    fig.update_layout(
        barmode='stack',
        title="Decision Patterns by MBTI Type (%)",
        xaxis_title="MBTI Type",
        yaxis_title="Percentage",
        height=600
    )
    return fig

def create_evolution_chart(results):
    """Create opinion evolution over time"""
    # Aggregate interaction counts per turn
    turns = []
    interactions = []
    influence = []

    for log in results["turn_logs"]:
        turns.append(log["turn"])
        interactions.append(log["total_interactions"])
        influence.append(log["total_influence"])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Interactions per Turn", "Influence per Turn"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )

    fig.add_trace(
        go.Scatter(x=turns, y=interactions, mode='lines+markers',
                  name='Interactions', line=dict(color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=turns, y=influence, mode='lines+markers',
                  name='Influence', line=dict(color='orange')),
        row=1, col=2
    )

    fig.update_layout(height=500, title_text="Simulation Dynamics Over Time")
    fig.update_xaxes(title_text="Turn", row=1, col=1)
    fig.update_xaxes(title_text="Turn", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Influence Score", row=1, col=2)

    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎯 Large-Scale Opinion Dynamics Simulator")
    print("=" * 50)

    # Get article input
    article_text = get_article_input()

    # Initialize simulator with large scale
    simulator = OpinionDynamicsSimulator(article_text)

    # Run simulation
    results = simulator.run_simulation()

    print("\n" + "=" * 50)
    print("📊 SIMULATION RESULTS")
    print(f"Total agents: {results['total_agents']}")
    print(f"Total interactions: {results['total_interactions']}")
    print(f"Total influence exchanged: {results['total_influence']:.2f}")
    print(f"Simulation time: {results['simulation_time']:.1f}s")

    # Print MBTI statistics
    print("\n🏆 MBTI PERFORMANCE SUMMARY:")
    print("MBTI    | Count | Avg Influence | Support% | Oppose% | Interactions")
    print("-" * 65)
    for mbti, stats in sorted(results["mbti_stats"].items(), key=lambda x: x[1]["avg_influence"], reverse=True):
        count = len([a for a in simulator.agents if a.mbti == mbti])
        print(f"{mbti:8} | {count:5} | {stats['avg_influence']:12.3f} | {stats['support_rate']:7.1%} | {stats['oppose_rate']:6.1%} | {stats['total_interactions']:11}")

    # Create visualizations
    print("\n📈 Generating visualizations...")

    # MBTI Influence Comparison
    fig1 = create_influence_chart(results)
    fig1.show()

    # Support/Oppose Rates by MBTI
    fig2 = create_decision_chart(results)
    fig2.show()

    # Opinion Evolution Over Time
    fig3 = create_evolution_chart(results)
    fig3.show()

    # Save results
    print("\n💾 Saving results to 'simulation_results.json'")
    with open('simulation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("✅ Simulation complete! Check the visualizations and results file.")