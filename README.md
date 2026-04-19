# Reinforcement Learning Tutor

An interactive Streamlit application that teaches Reinforcement Learning concepts from the ground up — through engaging content, embedded quizzes, live animations, and an AI tutor powered by your choice of LLM.

---

## What's Inside

The tutorial is organized into **8 interactive modules**:

| Module | Content |
|--------|---------|
| 🏠 Home | Course overview, learning objectives, progress dashboard |
| 🧠 What is RL? | Dog-trick analogy, paradigm comparison (Supervised/Unsupervised/RL), real-world success stories |
| 🗺️ RL Framework | Agent, Environment, State, Action, Reward, Episode, Discount Factor (γ) |
| 📐 Key Concepts | Policies, Value Functions V(s) & Q(s,a), Bellman Equation |
| 🔑 Q-Learning | Q-Table, update rule, TD Error, ε-greedy exploration strategy |
| 🎮 Grid World Demo | Animated agent navigation on a 5×5 grid, Q-table heatmap, policy arrows |
| 🚀 Train Your Agent | Interactive hyperparameter tuning, live training, results comparison |
| 🎓 Summary & Challenges | Concept recap, RL roadmap, final 5-question assessment |

Every module includes:
- **Clear explanations** with analogies and LaTeX formulas
- **Interactive visualizations** using Plotly
- **Embedded quizzes** (17 questions total) with instant feedback
- **Ask AI Tutor** — ask any question, answered in context

---

## AI Tutor — LLM Support

The app supports four LLM providers. Choose one in the sidebar:

| Provider | Models Available |
|----------|----------------|
| **Claude (Anthropic)** | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001 |
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| **Groq** | llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it |
| **Gemini (Google)** | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash |

Enter your API key in the sidebar — the app works fully without an LLM (quizzes and animations still function).

---

## Grid World Example

The main demo environment is a **5×5 Grid World**:

```
┌─────┬─────┬─────┬─────┬─────┐
│  S  │     │     │  💀 │     │
├─────┼─────┼─────┼─────┼─────┤
│     │  💀 │     │     │     │
├─────┼─────┼─────┼─────┼─────┤
│     │     │  💀 │     │     │
├─────┼─────┼─────┼─────┼─────┤
│     │     │     │  💀 │     │
├─────┼─────┼─────┼─────┼─────┤
│     │     │     │     │  🏆 │
└─────┴─────┴─────┴─────┴─────┘
S = Start (0,0)   🏆 = Goal (4,4)   💀 = Traps
```

**Rewards:** Goal = +10 | Trap = -5 | Step = -0.1 | Boundary = -0.5

The Q-Learning agent trains to ~99% success rate in 2,000 episodes. Watch it learn in the **Grid World Demo** tab.

---

## Getting Started

### Prerequisites

- Python 3.9+
- `pip`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/snerur/Reinforcement-Learning-Tutor.git
cd Reinforcement-Learning-Tutor

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Adding Your API Key

1. Open the sidebar (click `>` if collapsed)
2. Select your **LLM Provider**
3. Select a **Model**
4. Paste your **API Key**

The key is used only for in-session API calls and is never stored.

---

## How to Use

1. **Follow the tabs in order** — each builds on the previous
2. **Read the content**, then expand "Test Your Knowledge" to take the quiz
3. **Use "Ask AI Tutor"** at the bottom of each section if anything is unclear
4. **Visit Grid World Demo** to see the animated agent
5. **Experiment in Train Your Agent** — change hyperparameters and see the effect
6. **Complete the Final Challenge** in the Summary tab to earn your completion badge

---

## Project Structure

```
Reinforcement-Learning-Tutor/
├── app.py                          # Main Streamlit application (1645 lines)
├── rl_engine.py                    # GridWorld environment + Q-Learning agent
├── llm_utils.py                    # LLM provider integration (4 providers)
├── requirements.txt                # Python dependencies
├── reinforcement_learning_intro.ipynb  # Source Jupyter notebook
└── README.md                       # This file
```

---

## Key RL Concepts Covered

- **Markov Decision Process (MDP)** — states, actions, rewards, transitions
- **Policy π** — deterministic and stochastic strategies
- **Value functions** — V(s) and Q(s,a), and why they matter
- **Bellman Equation** — the recursive heart of RL
- **Q-Learning** — tabular, model-free, off-policy algorithm (Watkins, 1989)
- **TD Error** — the learning signal
- **ε-greedy** — balancing exploration and exploitation
- **Discount factor γ** — valuing future rewards

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `numpy` | Numerical computing (Q-table, training) |
| `plotly` | Interactive charts and grid animations |
| `anthropic` | Claude API client |
| `openai` | OpenAI API client |
| `groq` | Groq API client |
| `google-generativeai` | Gemini API client |

---

## Next Steps After This Tutorial

```
[1] Tabular Q-Learning   ← You are here
[2] Deep Q-Networks (DQN)
[3] Policy Gradient Methods (REINFORCE, PPO)
[4] Actor-Critic (A2C, SAC, TD3)
[5] RLHF, Multi-Agent RL, Model-Based RL
```

**Recommended Resources:**
- Sutton & Barto — *Reinforcement Learning: An Introduction* (free at incompleteideas.net)
- OpenAI Gymnasium — standard RL environments
- Stable-Baselines3 — production RL in PyTorch
- DeepMind × UCL Lectures — free on YouTube

---

## License

MIT License — see [LICENSE](LICENSE) for details.
