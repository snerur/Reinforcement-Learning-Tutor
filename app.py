import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from rl_engine import GridWorld, QLearningAgent, train_agent, run_greedy_episode
from llm_utils import get_llm_response

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RL Interactive Tutor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
body, .stApp {
    background-color: #1a1a2e;
    color: #e0e0e0;
}
/* ── card ── */
.rl-card {
    background: #16213e;
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    color: #e0e0e0 !important;
}
.rl-card *, .rl-card p, .rl-card small, .rl-card li, .rl-card b, .rl-card span {
    color: #e0e0e0 !important;
}
/* ── section header ── */
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #4fc3f7;
    margin-bottom: 8px;
}
/* ── feature card ── */
.feature-card {
    background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
    border: 1px solid #4fc3f7;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    height: 100%;
    color: #e0e0e0 !important;
}
.feature-card h3 { color: #4fc3f7 !important; margin-bottom: 6px; }
.feature-card p, .feature-card small { color: #cccccc !important; }
/* ── info boxes ── */
.info-box {
    background: #0d47a1;
    border-left: 4px solid #4fc3f7;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
    color: #e0e0e0 !important;
}
.info-box *, .info-box b, .info-box p { color: #e0e0e0 !important; }
.success-box {
    background: #1b5e20;
    border-left: 4px solid #4caf50;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
    color: #e0e0e0 !important;
}
.success-box *, .success-box b, .success-box h3 { color: #e0e0e0 !important; }
.warning-box {
    background: #4a2c00;
    border-left: 4px solid #ff9800;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
    color: #e0e0e0 !important;
}
.warning-box *, .warning-box b { color: #e0e0e0 !important; }
/* ── quiz ── */
.quiz-card {
    background: #1e2a45;
    border: 1px solid #0f3460;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #e0e0e0 !important;
}
.quiz-correct { color: #4caf50 !important; font-weight: bold; }
.quiz-wrong   { color: #f44336 !important; font-weight: bold; }
/* ── sidebar ── */
.sidebar-score {
    font-size: 0.9rem;
    padding: 4px 0;
    color: #e0e0e0 !important;
}
/* ── formula ── */
.formula-box {
    background: #0a0a1a;
    border: 1px solid #4fc3f7;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    margin: 10px 0;
    color: #e0e0e0 !important;
}
/* ── component badge ── */
.comp-badge {
    display: inline-block;
    background: #0f3460;
    border: 1px solid #4fc3f7;
    border-radius: 20px;
    padding: 6px 14px;
    margin: 4px;
    font-size: 0.9rem;
    color: #4fc3f7;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
SECTIONS = ["what_is_rl", "rl_framework", "key_concepts", "q_learning", "final"]
SECTION_LABELS = {
    "what_is_rl":    "What is RL?",
    "rl_framework":  "RL Framework",
    "key_concepts":  "Key Concepts",
    "q_learning":    "Q-Learning",
    "final":         "Final Challenge",
}
SECTION_MAX = {"what_is_rl": 3, "rl_framework": 3, "key_concepts": 3, "q_learning": 3, "final": 5}

def _init_state():
    for sec in SECTIONS:
        for i in range(SECTION_MAX[sec]):
            k = f"quiz_answers_{sec}_{i}"
            if k not in st.session_state:
                st.session_state[k] = None
        if f"quiz_submitted_{sec}" not in st.session_state:
            st.session_state[f"quiz_submitted_{sec}"] = False
        if f"quiz_score_{sec}" not in st.session_state:
            st.session_state[f"quiz_score_{sec}"] = 0

    for key in ["default_agent", "default_env", "default_history",
                "custom_agent", "custom_history", "training_runs",
                "llm_provider", "llm_api_key", "llm_model"]:
        if key not in st.session_state:
            st.session_state[key] = None

    if st.session_state.training_runs is None:
        st.session_state.training_runs = []

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Quiz data
# ─────────────────────────────────────────────────────────────────────────────
QUIZ_DATA = {
    "what_is_rl": [
        {
            "q": "Which best describes Reinforcement Learning?",
            "opts": [
                "Learning from labeled examples provided by a teacher",
                "Learning from trial-and-error using reward signals",
                "Finding hidden patterns in unlabeled data",
                "Optimizing a mathematical function using gradients",
            ],
            "ans": 1,
        },
        {
            "q": "Which application famously used RL to defeat world champions?",
            "opts": [
                "Google Translate, which translates 100+ languages",
                "ImageNet, which classifies millions of images",
                "AlphaGo, which defeated world champions at Go",
                "GPT-3, which generates human-like text",
            ],
            "ans": 2,
        },
        {
            "q": "What does an RL agent ultimately try to maximize?",
            "opts": [
                "The immediate reward at each single step",
                "The accuracy on a held-out test set",
                "Cumulative reward over time (not just immediate reward)",
                "The number of actions taken per episode",
            ],
            "ans": 2,
        },
    ],
    "rl_framework": [
        {
            "q": "What is the 'state' in Reinforcement Learning?",
            "opts": [
                "The action taken by the agent at each time step",
                "The total reward accumulated so far",
                "A snapshot of the current situation (e.g., the robot's position)",
                "The probability distribution over possible actions",
            ],
            "ans": 2,
        },
        {
            "q": "What does a discount factor γ = 0 mean for the agent?",
            "opts": [
                "The agent plans infinitely into the future",
                "The agent only cares about the immediate reward (completely shortsighted)",
                "The agent ignores all rewards",
                "The agent gives equal weight to all future rewards",
            ],
            "ans": 1,
        },
        {
            "q": "What is an 'episode' in Reinforcement Learning?",
            "opts": [
                "A single (state, action, reward) tuple",
                "One Q-table update step",
                "A complete run from the start state to a terminal state",
                "A batch of 100 training samples",
            ],
            "ans": 2,
        },
    ],
    "key_concepts": [
        {
            "q": "What does Q(s, a) represent?",
            "opts": [
                "The probability of choosing action a in state s",
                "Expected cumulative reward for taking action a in state s, then following policy π",
                "The immediate reward received after taking action a in state s",
                "The number of times action a was taken in state s",
            ],
            "ans": 1,
        },
        {
            "q": "In Q*(s,a) = r + γ max Q*(s',a'), what is the second term γ max Q*(s',a')?",
            "opts": [
                "The immediate reward for the current transition",
                "The discounted maximum future Q-value from the next state",
                "The learning rate controlling update speed",
                "The TD error measuring prediction accuracy",
            ],
            "ans": 1,
        },
        {
            "q": "What is a deterministic policy?",
            "opts": [
                "A policy that randomly picks actions from a distribution",
                "A policy that mixes exploration and exploitation",
                "A mapping that always selects the exact same action for a given state: π(s) = a",
                "A policy learned from human demonstrations",
            ],
            "ans": 2,
        },
    ],
    "q_learning": [
        {
            "q": "What is the TD Error in Q-learning?",
            "opts": [
                "The total reward accumulated in an episode",
                "The difference between the new target estimate and the current Q-value: [r + γ max Q(s',a') - Q(s,a)]",
                "The learning rate α multiplied by the reward r",
                "The number of exploratory steps taken",
            ],
            "ans": 1,
        },
        {
            "q": "In ε-greedy exploration with ε = 0.8, what happens?",
            "opts": [
                "The agent always picks the action with the highest Q-value",
                "The agent never explores randomly",
                "The agent explores randomly 80% of the time and exploits its Q-table 20% of the time",
                "The agent exploits 80% of the time and explores 20% of the time",
            ],
            "ans": 2,
        },
        {
            "q": "Why do we decay ε (epsilon) over time during training?",
            "opts": [
                "To make the Q-table grow faster",
                "To reduce the learning rate gradually",
                "The agent has learned enough to start exploiting its knowledge, so it needs less random exploration",
                "To increase the discount factor γ gradually",
            ],
            "ans": 2,
        },
    ],
    "final": [
        {
            "q": "In Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)], the symbol α is the...",
            "opts": [
                "Discount factor — controls how much we value future rewards",
                "Exploration rate — probability of taking a random action",
                "Learning rate — controls how fast we update our Q-value estimates",
                "Episode counter — number of training episodes completed",
            ],
            "ans": 2,
        },
        {
            "q": "Why doesn't tabular Q-learning scale to video games like Atari?",
            "opts": [
                "Video games run too fast for the algorithm to keep up",
                "The Q-table would need millions/billions of rows — one for each possible game state",
                "Q-learning only works with continuous action spaces",
                "Video games don't have clear reward signals",
            ],
            "ans": 1,
        },
        {
            "q": "DQN's key innovation over tabular Q-learning is...",
            "opts": [
                "Using a lookup table with billions of entries instead of a small table",
                "Removing the exploration-exploitation trade-off entirely",
                "Using a neural network to approximate Q(s,a) for all states, enabling generalization",
                "Training on a single episode instead of thousands",
            ],
            "ans": 2,
        },
        {
            "q": "In our grid world with γ=0.99, what happens if the agent reaches the goal in 8 steps?",
            "opts": [
                "All states receive exactly +10 reward",
                "Only the final state receives reward; all others get 0",
                "The rewards get discounted at each step, so early states get slightly less credit than the final +10",
                "The agent fails because 8 steps is too many",
            ],
            "ans": 2,
        },
        {
            "q": "In the grid world, what reward does the agent receive for hitting a boundary?",
            "opts": [
                "0 (no reward, just stays in place)",
                "-0.1 (same as a regular step penalty)",
                "-5.0 (same as a trap penalty)",
                "-0.5 (stays in place but gets penalized to discourage running into walls)",
            ],
            "ans": 3,
        },
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: quiz renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_quiz(section: str):
    questions = QUIZ_DATA[section]
    n = len(questions)
    submitted = st.session_state[f"quiz_submitted_{section}"]

    st.markdown("---")
    st.markdown("### 📝 Quick Quiz")

    for i, qdata in enumerate(questions):
        st.markdown(f"**Q{i+1}: {qdata['q']}**")
        key = f"quiz_answers_{section}_{i}"
        choice = st.radio(
            label="",
            options=qdata["opts"],
            index=qdata["opts"].index(st.session_state[key]) if st.session_state[key] in qdata["opts"] else 0,
            key=f"_radio_{section}_{i}",
            disabled=submitted,
            label_visibility="collapsed",
        )
        st.session_state[key] = choice

        if submitted:
            correct_opt = qdata["opts"][qdata["ans"]]
            if choice == correct_opt:
                st.markdown(f"<span class='quiz-correct'>✅ Correct!</span>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<span class='quiz-wrong'>❌ Incorrect.</span> "
                    f"Correct answer: **{correct_opt}**",
                    unsafe_allow_html=True,
                )
        st.markdown("---")

    col_btn, col_score = st.columns([2, 3])
    with col_btn:
        if not submitted:
            if st.button(f"Submit Quiz", key=f"submit_{section}"):
                score = sum(
                    1 for i, qdata in enumerate(questions)
                    if st.session_state[f"quiz_answers_{section}_{i}"] == qdata["opts"][qdata["ans"]]
                )
                st.session_state[f"quiz_score_{section}"] = score
                st.session_state[f"quiz_submitted_{section}"] = True
                st.rerun()
        else:
            if st.button(f"Retake Quiz", key=f"retake_{section}"):
                st.session_state[f"quiz_submitted_{section}"] = False
                for i in range(n):
                    st.session_state[f"quiz_answers_{section}_{i}"] = None
                st.session_state[f"quiz_score_{section}"] = 0
                st.rerun()

    with col_score:
        if submitted:
            score = st.session_state[f"quiz_score_{section}"]
            pct = int(score / n * 100)
            colour = "#4caf50" if pct >= 67 else "#ff9800"
            st.markdown(
                f"<div style='background:{colour}22;border:1px solid {colour};border-radius:8px;padding:10px;text-align:center'>"
                f"<b style='color:{colour}'>Score: {score}/{n} ({pct}%)</b></div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Helper: LLM chat component
# ─────────────────────────────────────────────────────────────────────────────
def render_llm_chat(context: str, placeholder_question: str, chat_key: str):
    with st.expander("💬 Ask AI Tutor"):
        question = st.text_input(
            "Your question:",
            placeholder=placeholder_question,
            key=f"llm_input_{chat_key}",
        )
        if st.button("Ask", key=f"llm_ask_{chat_key}"):
            provider = st.session_state.llm_provider
            api_key  = st.session_state.llm_api_key
            model    = st.session_state.llm_model
            if provider and api_key and model and question.strip():
                with st.spinner("Thinking..."):
                    response = get_llm_response(provider, api_key, model, question.strip(), context)
                st.markdown(
                    f"<div style='background:#0d2a5e;border-left:4px solid #4fc3f7;border-radius:6px;padding:12px 16px;margin:10px 0;color:#e0e0e0;'><b>AI Tutor:</b><br>{response}</div>",
                    unsafe_allow_html=True,
                )
            elif not question.strip():
                st.warning("Please type a question first.")
            else:
                st.info(
                    "Configure an LLM provider in the sidebar (select provider, enter API key, "
                    "pick a model) to enable the AI Tutor."
                )

# ─────────────────────────────────────────────────────────────────────────────
# Grid visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────
CELL_COLORS = {
    "empty": "#1e3a5f",
    "start": "#2ecc71",
    "trap":  "#e74c3c",
    "goal":  "#f39c12",
}

def _cell_color(env, row, col):
    pos = (row, col)
    if pos == env.start:
        return CELL_COLORS["start"]
    if pos == env.goal:
        return CELL_COLORS["goal"]
    if pos in env.traps:
        return CELL_COLORS["trap"]
    return CELL_COLORS["empty"]

def _grid_shapes_and_annotations(env):
    """Return Plotly shapes+annotations for the static grid background."""
    shapes = []
    annotations = []
    size = env.size
    # Cell state number (small, top-left)
    for r in range(size):
        for c in range(size):
            color = _cell_color(env, r, c)
            shapes.append(dict(
                type="rect",
                x0=c, x1=c+1, y0=size-r-1, y1=size-r,
                fillcolor=color, line=dict(color="#0f3460", width=2),
            ))
            # State index label
            state_idx = env.pos_to_state(r, c)
            annotations.append(dict(
                x=c + 0.08, y=size - r - 0.1,
                text=f"<b>{state_idx}</b>",
                xanchor="left", yanchor="top",
                font=dict(size=9, color="#aaaaaa"),
                showarrow=False,
            ))
            # Icon label
            pos = (r, c)
            if pos == env.goal:
                icon = "🏆"
            elif pos in env.traps:
                icon = "💀"
            elif pos == env.start:
                icon = "S"
            else:
                icon = ""
            if icon:
                annotations.append(dict(
                    x=c + 0.5, y=size - r - 0.5,
                    text=f"<b>{icon}</b>",
                    xanchor="center", yanchor="middle",
                    font=dict(size=22, color="white"),
                    showarrow=False,
                ))
    return shapes, annotations

ACTION_DELTAS = {0: (0, 0.35), 1: (0, -0.35), 2: (-0.35, 0), 3: (0.35, 0)}  # dx, dy in plot coords

def create_grid_figure(env, q_table=None, path=None, agent_pos=None, show_arrows=True, title="Grid World"):
    """Create a Plotly figure of the grid world."""
    size = env.size
    shapes, annotations = _grid_shapes_and_annotations(env)

    # Policy arrows
    if q_table is not None and show_arrows:
        for r in range(size):
            for c in range(size):
                pos = (r, c)
                if pos in env.traps or pos == env.goal:
                    continue
                state_idx = env.pos_to_state(r, c)
                best_action = int(np.argmax(q_table[state_idx]))
                dx, dy = ACTION_DELTAS[best_action]
                cx, cy = c + 0.5, size - r - 0.5
                annotations.append(dict(
                    x=cx + dx, y=cy + dy,
                    ax=cx - dx * 0.4, ay=cy - dy * 0.4,
                    xref="x", yref="y", axref="x", ayref="y",
                    text="",
                    arrowhead=2, arrowsize=1.2, arrowwidth=2,
                    arrowcolor="#4fc3f7",
                    showarrow=True,
                ))

    data = []

    # Path line
    if path:
        path_x = [c + 0.5 for (r, c) in path]
        path_y = [size - r - 0.5 for (r, c) in path]
        data.append(go.Scatter(
            x=path_x, y=path_y,
            mode="lines",
            line=dict(color="gold", width=3, dash="dot"),
            name="Path", showlegend=False,
        ))

    # Agent marker
    if agent_pos is not None:
        ar, ac = agent_pos
        data.append(go.Scatter(
            x=[ac + 0.5], y=[size - ar - 0.5],
            mode="markers+text",
            marker=dict(size=28, color="#9b59b6", symbol="circle"),
            text=["🤖"], textposition="middle center",
            name="Agent", showlegend=False,
        ))

    # Dummy scatter so we always have a trace
    if not data:
        data.append(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=dict(text=title, font=dict(color="#4fc3f7", size=16)),
        shapes=shapes, annotations=annotations,
        xaxis=dict(range=[0, size], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, size], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
    )
    return fig


def _grid_image(env, path, step):
    """Render the grid at the given step as a PNG (bytes) using Matplotlib."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    size = env.size
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.axis("off")

    # Color map for cell types
    cell_colors = {}
    for r in range(size):
        for c in range(size):
            pos = (r, c)
            if pos == env.goal:
                cell_colors[pos] = "#1b5e20"   # dark green
            elif pos in env.traps:
                cell_colors[pos] = "#b71c1c"   # dark red
            elif pos == env.start:
                cell_colors[pos] = "#1a237e"   # dark blue
            else:
                cell_colors[pos] = "#0d1b2a"   # near-black

    # Draw cells — note: matplotlib y=0 is bottom, but grid row 0 = top
    # We flip: draw row r at y = (size-1-r)
    for r in range(size):
        for c in range(size):
            pos = (r, c)
            y = size - 1 - r
            rect = patches.FancyBboxPatch(
                (c + 0.05, y + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                linewidth=1.5,
                edgecolor="#4fc3f7",
                facecolor=cell_colors[pos],
            )
            ax.add_patch(rect)
            # Cell labels
            label = ""
            if pos == env.start:
                label = "S"
            elif pos == env.goal:
                label = "G"
            elif pos in env.traps:
                label = "T"
            if label:
                ax.text(c + 0.5, y + 0.5, label,
                        ha="center", va="center",
                        fontsize=14, fontweight="bold", color="#ffffff")

    # Draw trail (visited cells up to current step)
    visited = path[:step]
    n_visited = len(visited)
    for idx, pos in enumerate(visited):
        r, c = pos
        y = size - 1 - r
        alpha = 0.25 + 0.55 * (idx / max(n_visited - 1, 1))
        trail = plt.Circle((c + 0.5, y + 0.5), 0.22, color="#fdd835", alpha=alpha, zorder=3)
        ax.add_patch(trail)

    # Draw agent (large bright ball) at current position
    if path:
        ar, ac = path[step]
        ay = size - 1 - ar
        # White halo
        halo = plt.Circle((ac + 0.5, ay + 0.5), 0.40, color="#ffffff", alpha=0.35, zorder=4)
        ax.add_patch(halo)
        # Agent circle
        agent = plt.Circle((ac + 0.5, ay + 0.5), 0.35, color="#ab47bc", zorder=5)
        ax.add_patch(agent)
        # Step label inside agent
        ax.text(ac + 0.5, ay + 0.5, str(step),
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="#ffffff", zorder=6)

    # Title
    pos = path[step] if path else (0, 0)
    suffix = ""
    if step == len(path) - 1 and path:
        if path[step] == env.goal:
            suffix = " - GOAL!"
        elif path[step] in env.traps:
            suffix = " - TRAP!"
    ax.set_title(f"Step {step}/{len(path)-1}  pos={pos}{suffix}",
                 color="#4fc3f7", fontsize=13, pad=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_step_animation(env, path, unique_key, title="Agent Path"):
    """Streamlit-native step-through animation using st.empty() + Matplotlib PNG frames."""
    import time
    if not path:
        st.warning("No path to display.")
        return
    n = len(path)
    c_speed, c_reset = st.columns([3, 1])
    with c_speed:
        delay = st.select_slider(
            "Animation speed",
            options=[0.2, 0.4, 0.7, 1.2],
            value=0.7,
            format_func=lambda x: {0.2: "Very fast", 0.4: "Fast", 0.7: "Medium", 1.2: "Slow"}[x],
            key=f"speed_{unique_key}",
        )
    with c_reset:
        if st.button("Reset", key=f"reset_{unique_key}"):
            st.session_state[f"astep_{unique_key}"] = 0

    if f"astep_{unique_key}" not in st.session_state:
        st.session_state[f"astep_{unique_key}"] = 0

    step = st.slider("Drag to scrub:", 0, n - 1, st.session_state[f"astep_{unique_key}"],
                     key=f"slider_{unique_key}")
    st.session_state[f"astep_{unique_key}"] = step

    chart_ph = st.empty()
    play_clicked = st.button("▶  Play animation", key=f"play_{unique_key}", type="primary")

    if play_clicked:
        for i in range(n):
            chart_ph.image(_grid_image(env, path, i), use_container_width=True)
            time.sleep(delay)
        st.session_state[f"astep_{unique_key}"] = n - 1
    else:
        chart_ph.image(_grid_image(env, path, step), use_container_width=True)


def create_training_charts(history):
    """Plotly subplot: reward, success rate, epsilon decay."""
    rewards  = history["episode_rewards"]
    success  = history["success_flags"]
    epsilon  = history["epsilon_history"] if "epsilon_history" in history else []

    # Use agent epsilon history from session state if not in history dict
    episodes = list(range(1, len(rewards) + 1))

    # Rolling averages
    w50  = 50
    w100 = 100
    def rolling(arr, w):
        return [np.mean(arr[max(0, i-w):i+1]) for i in range(len(arr))]

    roll_rew = rolling(rewards, w50)
    roll_suc = [x*100 for x in rolling(success, w100)]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Episode Reward", "Success Rate (100-ep avg)", "Epsilon (ε) Decay"],
    )

    # Panel 1: rewards
    fig.add_trace(go.Scatter(x=episodes, y=rewards, mode="lines",
                             line=dict(color="#4fc3f7", width=0.8), opacity=0.3,
                             name="Raw Reward", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=episodes, y=roll_rew, mode="lines",
                             line=dict(color="#ff9800", width=2),
                             name="50-ep avg", showlegend=False), row=1, col=1)

    # Panel 2: success rate
    fig.add_trace(go.Scatter(x=episodes, y=roll_suc, mode="lines",
                             fill="tozeroy", fillcolor="rgba(76,175,80,0.15)",
                             line=dict(color="#4caf50", width=2),
                             name="Success %", showlegend=False), row=1, col=2)

    # Panel 3: epsilon
    if epsilon:
        eps_episodes = list(range(1, len(epsilon)+1))
        fig.add_trace(go.Scatter(x=eps_episodes, y=epsilon, mode="lines",
                                 fill="tozeroy", fillcolor="rgba(244,67,54,0.15)",
                                 line=dict(color="#f44336", width=2),
                                 name="Epsilon", showlegend=False), row=1, col=3)

    fig.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#0f3460", row=1, col=i)
        fig.update_yaxes(gridcolor="#0f3460", row=1, col=i)
    return fig


def create_qtable_heatmap(env, q_table):
    """Plotly heatmap of max Q-value per state."""
    size = env.size
    max_q = np.max(q_table, axis=1).reshape(size, size)
    best_a = np.argmax(q_table, axis=1).reshape(size, size)
    symbols = GridWorld.ACTION_SYMBOLS

    text_matrix = []
    for r in range(size):
        row_text = []
        for c in range(size):
            pos = (r, c)
            if pos in env.traps:
                row_text.append("💀<br>TRAP")
            elif pos == env.goal:
                row_text.append("🏆<br>GOAL")
            else:
                row_text.append(f"{max_q[r,c]:.2f}<br>{symbols[best_a[r,c]]}")
        text_matrix.append(row_text)

    colorscale = [
        [0.0, "#c0392b"],
        [0.4, "#1a1a2e"],
        [1.0, "#27ae60"],
    ]

    fig = go.Figure(go.Heatmap(
        z=max_q,
        text=text_matrix,
        texttemplate="%{text}",
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title="Max Q", tickfont=dict(color="#e0e0e0"), title_font=dict(color="#e0e0e0")),
    ))
    fig.update_layout(
        title=dict(text="Q-Table Heatmap (Max Q per State)", font=dict(color="#4fc3f7")),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
PROVIDER_MODELS = {
    "Claude (Anthropic)": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
    "OpenAI":             ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "Groq":               ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
    "Gemini (Google)":    ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
}

with st.sidebar:
    st.markdown("## 🤖 RL Tutor")
    st.markdown("---")
    st.markdown("### LLM Configuration")

    provider = st.selectbox(
        "Provider",
        options=["(None)"] + list(PROVIDER_MODELS.keys()),
        key="_sb_provider",
    )
    if provider != "(None)":
        model_choice = st.selectbox("Model", options=PROVIDER_MODELS[provider], key="_sb_model")
        api_key_val  = st.text_input("API Key", type="password", key="_sb_api_key",
                                      placeholder="sk-...")
        st.session_state.llm_provider = provider
        st.session_state.llm_model    = model_choice
        st.session_state.llm_api_key  = api_key_val if api_key_val else None
    else:
        st.session_state.llm_provider = None
        st.session_state.llm_model    = None
        st.session_state.llm_api_key  = None

    st.markdown("---")
    st.markdown("### Your Progress")

    total_score = 0
    total_max   = 0
    for sec in SECTIONS:
        score = st.session_state[f"quiz_score_{sec}"]
        done  = st.session_state[f"quiz_submitted_{sec}"]
        mx    = SECTION_MAX[sec]
        total_score += score
        total_max   += mx
        label = SECTION_LABELS[sec]
        icon  = "✅" if done else "⬜"
        pct   = int(score/mx*100) if done else 0
        colour = "#4caf50" if pct >= 67 else ("#ff9800" if done else "#888")
        st.markdown(
            f"<div class='sidebar-score'>{icon} <b>{label}</b>: "
            f"<span style='color:{colour}'>{score}/{mx}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    pct_total = int(total_score / total_max * 100) if total_max else 0
    st.markdown(
        f"<div style='text-align:center;font-size:1.1rem'>"
        f"<b>Total: {total_score}/{total_max} ({pct_total}%)</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#888;text-align:center;line-height:1.6'>"
        "Developed by <b style='color:#b0bec5'>Sridhar Nerur</b><br>"
        "<a href='mailto:snerur@uta.edu' style='color:#4fc3f7;text-decoration:none'>snerur@uta.edu</a><br>"
        "Built with <b style='color:#b0bec5'>Claude</b> (Anthropic)<br><br>"
        "<span style='color:#ef9a9a'>For educational purposes only.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Default agent cache (train once)
# ─────────────────────────────────────────────────────────────────────────────
def ensure_default_agent():
    if st.session_state.default_agent is None:
        env   = GridWorld()
        agent = QLearningAgent(env.n_states, env.n_actions,
                               alpha=0.1, gamma=0.99,
                               epsilon=1.0, epsilon_decay=0.995,
                               epsilon_min=0.01, seed=42)
        with st.spinner("Training default agent (2000 episodes)…"):
            history = train_agent(env, agent, n_episodes=2000)
        history["epsilon_history"] = agent.epsilon_history
        st.session_state.default_env     = env
        st.session_state.default_agent   = agent
        st.session_state.default_history = history

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
TAB_NAMES = [
    "Home",
    "What is RL?",
    "RL Framework",
    "Key Concepts",
    "Q-Learning",
    "Grid World Demo",
    "Train Your Agent",
    "Summary & Challenges",
]

tabs = st.tabs(TAB_NAMES)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — HOME
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(
        "<h1 style='text-align:center;color:#4fc3f7;font-size:2.6rem'>"
        "🤖 Reinforcement Learning — Interactive Tutor</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;font-size:1.15rem;color:#aaa'>"
        "Learn RL from first principles — with interactive demos, quizzes, and an AI tutor.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Learning objectives
    col_obj, col_prog = st.columns([3, 2])
    with col_obj:
        st.markdown("### 🎯 Learning Objectives")
        objectives = [
            "Understand what Reinforcement Learning is and how it differs from supervised/unsupervised learning",
            "Master the RL framework: agent, environment, state, action, reward, episode",
            "Understand key concepts: policies, value functions, and the Bellman equation",
            "Learn the Q-Learning algorithm and epsilon-greedy exploration",
            "See Q-Learning in action on a 5x5 Grid World",
            "Train your own agent and tune hyperparameters interactively",
        ]
        for obj in objectives:
            st.markdown(f"- {obj}")

    with col_prog:
        st.markdown("### 📈 Your Progress")
        total_score = sum(st.session_state[f"quiz_score_{s}"] for s in SECTIONS)
        total_max   = sum(SECTION_MAX[s] for s in SECTIONS)
        pct = int(total_score / total_max * 100) if total_max else 0
        st.progress(pct / 100)
        st.markdown(f"**{total_score}/{total_max} correct ({pct}%)**")
        st.markdown("")
        for sec in SECTIONS:
            done  = st.session_state[f"quiz_submitted_{sec}"]
            score = st.session_state[f"quiz_score_{sec}"]
            mx    = SECTION_MAX[sec]
            icon  = "✅" if done else "⬜"
            st.markdown(f"{icon} {SECTION_LABELS[sec]}: **{score}/{mx}**")

    st.markdown("---")
    st.markdown("### 🌟 Features")
    f1, f2, f3, f4 = st.columns(4)
    feature_data = [
        ("📝", "Interactive Quizzes", "Test your understanding after each section. Immediate feedback helps reinforce concepts."),
        ("🎮", "Animated Demo", "Watch a trained Q-learning agent navigate the Grid World step by step."),
        ("🏋️", "Train Your Agent", "Tune hyperparameters and train your own agent — see how α, γ, ε affect learning."),
        ("💬", "Ask AI Tutor", "Connect Claude, GPT, Groq, or Gemini for instant answers to your questions."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], feature_data):
        with col:
            st.markdown(
                f"<div style='background:linear-gradient(135deg,#16213e,#0f3460);border:1px solid #4fc3f7;border-radius:10px;padding:18px;text-align:center;color:#e0e0e0;'><h3>{icon} {title}</h3><p style='color:#ccc;font-size:0.9rem'>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.info(
        "**How to use this app:** Navigate the tabs left-to-right. Each tab introduces new concepts, "
        "ends with a quiz, and has an AI Tutor expander. Configure your preferred LLM in the sidebar "
        "to unlock the AI Tutor. The Grid World Demo and Train Your Agent tabs let you interact with "
        "a live Q-learning simulation."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WHAT IS RL?
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## What is Reinforcement Learning?")

    # Dog trick analogy
    col_dog, col_def = st.columns([1, 2])
    with col_dog:
        st.markdown(
            "<div style='background:#16213e;border:1px solid #0f3460;border-radius:12px;padding:16px 20px;margin-bottom:12px;color:#e0e0e0;text-align:center;font-size:3rem'>🐕</div>",
            unsafe_allow_html=True,
        )
    with col_def:
        st.markdown("### The Dog Trick Analogy")
        st.markdown(
            "Imagine teaching a dog to sit. You don't give it a manual — you simply **reward** it "
            "with a treat when it does the right thing, and ignore (or gently correct) it when it doesn't. "
            "Over thousands of interactions, the dog **learns** which actions lead to treats.\n\n"
            "Reinforcement Learning works the same way: an **agent** learns by trying actions in an "
            "**environment** and receiving **rewards** or **penalties** as feedback — no labelled dataset required."
        )

    # Comparison table
    st.markdown("### Comparison with Other ML Paradigms")
    comparison_df = pd.DataFrame({
        "Paradigm":    ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"],
        "Input":       ["Labeled (X, y) pairs", "Unlabeled data X", "Environment interactions"],
        "Signal":      ["Ground-truth label", "Internal structure", "Reward/Penalty signal"],
        "Goal":        ["Predict labels on new data", "Find structure/clusters", "Maximize cumulative reward"],
        "Example":     ["Image classification", "Customer segmentation", "Game-playing AI, robotics"],
    })
    st.dataframe(
        comparison_df.set_index("Paradigm"),
        use_container_width=True,
    )

    # Real-world examples
    st.markdown("### 🌍 Real-World Examples")
    examples = [
        ("♟️ AlphaGo (DeepMind)", "AlphaGo mastered the ancient game of Go — a game with more possible positions than atoms in the universe — by playing millions of games against itself. Using RL with policy and value networks, it defeated world champion Lee Sedol in 2016, a milestone once thought decades away."),
        ("💬 ChatGPT / RLHF", "ChatGPT and other modern LLMs are fine-tuned with **Reinforcement Learning from Human Feedback (RLHF)**. Human raters compare model responses, and the model is rewarded for generating responses that humans prefer — making it more helpful, harmless, and honest."),
        ("🚗 Self-Driving Cars", "Autonomous vehicles use RL (among other techniques) to learn complex driving policies: when to brake, merge, or navigate intersections. The reward signal can include safety metrics, comfort, and arrival time."),
        ("❄️ Data Center Cooling", "Google used RL to reduce data center cooling energy by 40%. The agent learns to control thousands of sensors and actuators to minimize power while keeping servers within safe temperature ranges."),
    ]
    for title_ex, desc_ex in examples:
        with st.expander(title_ex):
            st.markdown(desc_ex)

    render_quiz("what_is_rl")
    render_llm_chat(
        context="We are discussing what Reinforcement Learning is, the dog-training analogy, and real-world applications like AlphaGo and ChatGPT RLHF.",
        placeholder_question="Why is RL different from supervised learning?",
        chat_key="what_is_rl",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RL FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## The RL Framework")

    # SVG diagram of the Agent-Environment loop
    st.markdown("### The Agent-Environment Loop")
    st.markdown("""
<div style="background:#16213e;border:1px solid #0f3460;border-radius:12px;padding:28px 20px 16px 20px;text-align:center;">
  <svg viewBox="0 0 720 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:720px;font-family:Arial,sans-serif;">
    <defs>
      <marker id="arr-orange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="#f39c12"/>
      </marker>
      <marker id="arr-red" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="#e74c3c"/>
      </marker>
      <marker id="arr-purple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="#9b59b6"/>
      </marker>
    </defs>
    <!-- Environment Box -->
    <rect x="20" y="40" width="200" height="110" rx="12" fill="#0d2a5e" stroke="#4fc3f7" stroke-width="2.5"/>
    <text x="120" y="80" text-anchor="middle" fill="#4fc3f7" font-size="15" font-weight="bold">ENVIRONMENT</text>
    <text x="120" y="100" text-anchor="middle" fill="#90caf9" font-size="11">Rules of the world</text>
    <text x="120" y="118" text-anchor="middle" fill="#90caf9" font-size="11">Game / Robot / Market</text>
    <text x="120" y="136" text-anchor="middle" fill="#90caf9" font-size="11">Returns state + reward</text>
    <!-- Agent Box -->
    <rect x="500" y="40" width="200" height="110" rx="12" fill="#0d3320" stroke="#4caf50" stroke-width="2.5"/>
    <text x="600" y="80" text-anchor="middle" fill="#4caf50" font-size="15" font-weight="bold">AGENT</text>
    <text x="600" y="100" text-anchor="middle" fill="#a5d6a7" font-size="11">Observes state + reward</text>
    <text x="600" y="118" text-anchor="middle" fill="#a5d6a7" font-size="11">Runs policy π</text>
    <text x="600" y="136" text-anchor="middle" fill="#a5d6a7" font-size="11">Chooses next action</text>
    <!-- State arrow: Environment to Agent (top) -->
    <line x1="220" y1="72" x2="500" y2="72" stroke="#f39c12" stroke-width="2.5" marker-end="url(#arr-orange)"/>
    <rect x="298" y="54" width="124" height="24" rx="6" fill="#3d2700"/>
    <text x="360" y="71" text-anchor="middle" fill="#f39c12" font-size="13" font-weight="bold">State  s_t</text>
    <!-- Reward arrow: Environment to Agent (middle) -->
    <line x1="220" y1="118" x2="500" y2="118" stroke="#e74c3c" stroke-width="2.5" marker-end="url(#arr-red)"/>
    <rect x="290" y="100" width="140" height="24" rx="6" fill="#3b0f0f"/>
    <text x="360" y="117" text-anchor="middle" fill="#e74c3c" font-size="13" font-weight="bold">Reward  r_t</text>
    <!-- Action arrow: Agent to Environment (curves under) -->
    <path d="M600,150 L600,195 L120,195 L120,150" stroke="#9b59b6" stroke-width="2.5" fill="none" marker-end="url(#arr-purple)"/>
    <rect x="296" y="177" width="128" height="24" rx="6" fill="#1e0b30"/>
    <text x="360" y="194" text-anchor="middle" fill="#9b59b6" font-size="13" font-weight="bold">Action  a_t</text>
    <!-- Caption -->
    <text x="360" y="225" text-anchor="middle" fill="#666" font-size="11">This loop repeats every time step t = 0, 1, 2, ...</text>
  </svg>
</div>
""", unsafe_allow_html=True)

    # Component cards
    st.markdown("### Core Components")
    comp_data = [
        ("🤖 Agent", "The learner and decision-maker. It observes the environment and chooses actions.", "#4fc3f7"),
        ("🌍 Environment", "Everything the agent interacts with. It receives actions and returns states and rewards.", "#2ecc71"),
        ("📍 State (s)", "A snapshot of the current situation. E.g., the robot's position, the game board, sensor readings.", "#f39c12"),
        ("⚡ Action (a)", "A choice the agent can make. E.g., move up/down/left/right, apply force, generate a word.", "#e74c3c"),
        ("🎁 Reward (r)", "A scalar feedback signal. Positive = good, negative = bad. The agent wants to maximize total reward.", "#9b59b6"),
        ("🔁 Episode", "A complete sequence from start to a terminal state (win/lose/timeout). Training uses many episodes.", "#1abc9c"),
    ]
    c1, c2, c3 = st.columns(3)
    for idx, (comp_name, comp_desc, comp_col) in enumerate(comp_data):
        col = [c1, c2, c3][idx % 3]
        with col:
            st.markdown(
                f"<div style='background:#16213e;border-left:4px solid {comp_col};border-radius:12px;padding:16px 20px;margin-bottom:12px;color:#e0e0e0;'>"
                f"<b style='color:{comp_col}'>{comp_name}</b><br><small>{comp_desc}</small></div>",
                unsafe_allow_html=True,
            )

    # Discount factor visualization
    st.markdown("---")
    st.markdown("### 📉 Discount Factor γ (Gamma)")
    st.markdown(
        "The discount factor γ ∈ [0, 1] controls how much the agent values future rewards vs. immediate rewards. "
        "The **cumulative return** G_t is defined as:"
    )
    st.latex(r"G_t = r_{t} + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}")

    gamma_slider = st.slider("γ (gamma)", 0.0, 1.0, 0.99, 0.01, key="gamma_demo")

    st.markdown(f"**With γ = {gamma_slider:.2f}, here's how a +10 goal reward is discounted back in time:**")
    horizon = 10
    rewards_demo = [0.0] * horizon + [10.0]
    discounted = [10.0 * (gamma_slider ** (horizon - t)) for t in range(horizon + 1)]

    disc_df = pd.DataFrame({
        "Step":          list(range(horizon + 1)),
        "Raw Reward":    rewards_demo,
        "Discounted Value": [round(d, 4) for d in discounted],
    })
    st.dataframe(disc_df, use_container_width=True, hide_index=True)

    if gamma_slider < 0.5:
        st.warning("With very low γ, the agent is extremely shortsighted and ignores future rewards.")
    elif gamma_slider > 0.95:
        st.success("High γ means the agent plans far into the future — good for long-horizon tasks.")
    else:
        st.info("Medium γ balances short-term and long-term rewards.")

    render_quiz("rl_framework")
    render_llm_chat(
        context="We are discussing the RL framework components: agent, environment, state, action, reward, episode, and the discount factor gamma.",
        placeholder_question="How does the discount factor affect the agent's behaviour?",
        chat_key="rl_framework",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — KEY CONCEPTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## Key Concepts")

    # Policy
    st.markdown("### 📋 Policy (π)")
    col_det, col_sto = st.columns(2)
    with col_det:
        st.markdown("**Deterministic Policy**")
        st.latex(r"\pi(s) = a")
        st.markdown(
            "Maps each state directly to a specific action. "
            "Given the same state, the agent always takes the same action. "
            "Simple and fast — great when the optimal action is clear."
        )
    with col_sto:
        st.markdown("**Stochastic Policy**")
        st.latex(r"\pi(a \mid s) = P(A_t = a \mid S_t = s)")
        st.markdown(
            "Maps each state to a probability distribution over actions. "
            "The agent samples an action from this distribution. "
            "Useful in games of chance or when multiple actions are equally good."
        )

    # Value functions
    st.markdown("---")
    st.markdown("### 💰 Value Functions")
    col_v, col_q = st.columns(2)
    with col_v:
        st.markdown("**State Value Function V(s)**")
        st.latex(r"V^{\pi}(s) = \mathbb{E}_{\pi}\left[G_t \mid S_t = s\right]")
        st.markdown(
            "The expected cumulative return starting from state *s* and following policy π. "
            "A high V(s) means this is a good state to be in."
        )
    with col_q:
        st.markdown("**Action-Value Function Q(s, a)**")
        st.latex(r"Q^{\pi}(s,a) = \mathbb{E}_{\pi}\left[G_t \mid S_t = s,\, A_t = a\right]")
        st.markdown(
            "The expected cumulative return from state *s*, taking action *a*, then following policy π. "
            "Q-learning directly estimates this function."
        )

    # Bellman equation
    st.markdown("---")
    st.markdown("### The Bellman Equation — The Core Insight of RL")

    st.markdown("""<div style='background:#0d2a5e;border-left:4px solid #4fc3f7;border-radius:6px;padding:12px 16px;margin:10px 0;color:#e0e0e0;'>
<b>The intuition:</b> The value of taking action <i>a</i> in state <i>s</i> equals
<b>what you earn right now</b> plus <b>the best you can earn from where you land</b>.
<br><br>
Think of it like planning a road trip: "The total time via this route = time on this road segment
+ the best possible total time from the next city." You don't need to know the whole map —
you just need to know the next hop and trust that the values ahead are correct.
</div>""", unsafe_allow_html=True)

    st.latex(r"Q^*(s,a) \;=\; \underbrace{r(s,a)}_{\substack{\text{immediate}\\\text{reward}}} \;+\; \gamma \cdot \underbrace{\max_{a'} Q^*(s',\, a')}_{\substack{\text{best future value}\\\text{from next state }s'}}")

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown("""<div style='background:#16213e;border:1px solid #0f3460;border-radius:12px;padding:16px 20px;margin-bottom:12px;color:#e0e0e0;border-top:4px solid #4fc3f7;text-align:center;'>
        <b style='color:#4fc3f7;font-size:1.1rem'>Q*(s, a)</b><br><br>
        <small>The <b>optimal Q-value</b>: the best possible cumulative reward when taking action a from state s</small>
        </div>""", unsafe_allow_html=True)
    with b2:
        st.markdown("""<div style='background:#16213e;border:1px solid #0f3460;border-radius:12px;padding:16px 20px;margin-bottom:12px;color:#e0e0e0;border-top:4px solid #2ecc71;text-align:center;'>
        <b style='color:#2ecc71;font-size:1.1rem'>r(s, a)</b><br><br>
        <small>The <b>immediate reward</b> you collect right now — e.g. −0.1 for a step, +10 for reaching the goal</small>
        </div>""", unsafe_allow_html=True)
    with b3:
        st.markdown("""<div style='background:#16213e;border:1px solid #0f3460;border-radius:12px;padding:16px 20px;margin-bottom:12px;color:#e0e0e0;border-top:4px solid #f39c12;text-align:center;'>
        <b style='color:#f39c12;font-size:1.1rem'>γ (gamma)</b><br><br>
        <small>The <b>discount factor</b>: rewards earned in the future are worth slightly less than today's rewards</small>
        </div>""", unsafe_allow_html=True)
    with b4:
        st.markdown("""<div style='background:#16213e;border:1px solid #0f3460;border-radius:12px;padding:16px 20px;margin-bottom:12px;color:#e0e0e0;border-top:4px solid #9b59b6;text-align:center;'>
        <b style='color:#9b59b6;font-size:1.1rem'>max Q*(s', a')</b><br><br>
        <small>The <b>best Q-value in the next state s'</b> — whatever the optimal action there will be</small>
        </div>""", unsafe_allow_html=True)

    with st.expander("📖 Concrete Grid World Example — click to expand"):
        st.markdown("""
**Scenario:** The agent is at grid position (3, 3) and takes action RIGHT, landing on (3, 4).

- **Immediate reward** r = −0.1  (regular step penalty — no trap or goal)
- **Next state** s' = (3, 4)  — one step below the goal at (4, 4)
- At (3, 4) the best action is DOWN toward the goal, with Q-value ≈ **+8.5**
- Discount factor γ = 0.99
        """)
        st.latex(r"Q^*\bigl((3,3),\;\text{RIGHT}\bigr) \;=\; -0.1 \;+\; 0.99 \times 8.5 \;=\; -0.1 + 8.415 \;=\; \mathbf{8.315}")
        st.markdown("""
**Why does this matter?** We begin training with all Q-values at zero.
After the agent accidentally reaches the goal and gets +10, the Bellman equation lets that reward
**propagate backwards** — the state just before the goal gets a high Q-value, then the state before
*that*, and so on. Within a few hundred episodes every state knows how close it is to the goal.
        """)

    # Interactive Bellman demo
    st.markdown("---")
    st.markdown("### Interactive Bellman Calculator")
    st.markdown("""<div style='background:#0d2a5e;border-left:4px solid #4fc3f7;border-radius:6px;padding:12px 16px;margin:10px 0;color:#e0e0e0;'>
Adjust the sliders to see how each ingredient of the Bellman equation affects the result.
The <b>TD Error</b> is the correction signal — it drives every Q-learning update.
</div>""", unsafe_allow_html=True)

    bl, br = st.columns(2)
    with bl:
        st.markdown("**Set the scenario:**")
        r_demo = st.select_slider(
            "Immediate reward r — what did the agent just receive?",
            options=[-5.0, -0.5, -0.1, 0.0, 10.0],
            value=-0.1,
            format_func=lambda x: {
                -5.0: "−5.0  (hit a trap  💀)",
                -0.5: "−0.5  (hit boundary 🧱)",
                -0.1: "−0.1  (regular step ➡)",
                 0.0: " 0.0  (neutral)",
                10.0: "+10.0 (reached goal 🏆)",
            }[x],
            key="bellman_r",
        )
        max_q_ns = st.slider(
            "Best Q-value at the next state  max Q(s', a')\n"
            "(How good is where the agent just landed?)",
            min_value=-5.0, max_value=10.0, value=7.5, step=0.1,
            key="bellman_maxq",
        )
        gamma_key = st.slider("Discount factor γ", 0.0, 1.0, 0.99, 0.01, key="bellman_gamma")
        current_q = st.slider(
            "Current Q(s, a) — the agent's existing estimate before this update",
            min_value=-6.0, max_value=10.0, value=3.0, step=0.1,
            key="bellman_cq",
        )
        alpha_demo = st.slider("Learning rate α", 0.01, 1.0, 0.1, 0.01, key="bellman_alpha")

    with br:
        target    = r_demo + gamma_key * max_q_ns
        td_error  = target - current_q
        new_q     = current_q + alpha_demo * td_error

        st.markdown("**Step-by-step result:**")
        calc_rows = [
            ("Immediate reward  r",                           f"{r_demo:.2f}"),
            ("Discount factor  γ",                            f"{gamma_key:.2f}"),
            ("Best future Q-value  max Q(s', a')",            f"{max_q_ns:.2f}"),
            ("γ × max Q(s', a')",                            f"{gamma_key * max_q_ns:.3f}"),
            ("TD Target  =  r + γ × max Q(s', a')",          f"{target:.3f}"),
            ("Current Q(s, a)  (before update)",              f"{current_q:.3f}"),
            ("TD Error  =  Target − Q(s, a)",                 f"{td_error:+.3f}"),
            (f"New Q  =  Q + α × TD Error  (α={alpha_demo})", f"{new_q:.3f}"),
        ]
        calc_df = pd.DataFrame(calc_rows, columns=["Component", "Value"])
        st.dataframe(calc_df, hide_index=True, use_container_width=True)

        if abs(td_error) < 0.05:
            st.success("TD Error ≈ 0 — the Q-value is already accurate. Very little update happens.")
        elif td_error > 0:
            st.info(
                f"TD Error = +{td_error:.3f}  \n"
                f"We **underestimated** this state — Q increases by **{alpha_demo * td_error:.3f}** "
                f"(α × TD Error)."
            )
        else:
            st.warning(
                f"TD Error = {td_error:.3f}  \n"
                f"We **overestimated** this state — Q decreases by **{abs(alpha_demo * td_error):.3f}**."
            )

    render_quiz("key_concepts")
    render_llm_chat(
        context="We are discussing policies (deterministic vs stochastic), value functions V(s) and Q(s,a), and the Bellman equation.",
        placeholder_question="What is the difference between V(s) and Q(s,a)?",
        chat_key="key_concepts",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Q-LEARNING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## Q-Learning Algorithm")

    # Q-table explanation
    st.markdown("### 🗂️ The Q-Table")
    st.markdown(
        "A Q-table is a 2-D lookup table: rows = states, columns = actions. "
        "Each cell Q(s, a) stores the estimated expected return. "
        "The agent always picks the action with the highest Q-value in its current state."
    )
    sample_q = pd.DataFrame(
        np.round(np.random.default_rng(7).uniform(-1, 5, (5, 4)), 2),
        columns=["↑ UP", "↓ DOWN", "← LEFT", "→ RIGHT"],
        index=[f"State {i}" for i in range(5)],
    )
    sample_q.index.name = "State"
    st.dataframe(sample_q.style.highlight_max(axis=1, color="#1b5e20"), use_container_width=True)
    st.caption("Highlighted cells show the best action (highest Q) per state.")

    # Update rule
    st.markdown("---")
    st.markdown("### ⚙️ Q-Learning Update Rule")
    st.latex(
        r"Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]"
    )

    # Term breakdown
    st.markdown("### Term Breakdown")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("α (alpha)", "Learning Rate", "Controls update speed")
    t2.metric("r", "Reward", "Immediate feedback")
    t3.metric("γ (gamma)", "Discount", "Values future rewards")
    t4.metric("TD Target", "r + γ max Q(s',a')", "What we aim for")
    t5.metric("TD Error", "Target − Q(s,a)", "Correction signal")

    st.markdown("---")
    st.markdown("### 🔍 Epsilon-Greedy Exploration")
    st.markdown(
        "ε-greedy balances **exploration** (trying random actions to discover better strategies) "
        "with **exploitation** (using the current best Q-values). With probability ε, the agent "
        "acts randomly; with probability (1 − ε), it picks the greedy best action."
    )

    eps_slider = st.slider("ε (epsilon)", 0.0, 1.0, 0.5, 0.01, key="eps_demo")

    ec1, ec2 = st.columns(2)
    with ec1:
        fig_eps = go.Figure(go.Pie(
            labels=["🎲 Explore (random)", "🎯 Exploit (greedy)"],
            values=[eps_slider, 1 - eps_slider],
            hole=0.4,
            marker=dict(colors=["#9b59b6", "#2ecc71"]),
            textfont=dict(size=14),
        ))
        fig_eps.update_layout(
            paper_bgcolor="#1a1a2e", font=dict(color="#e0e0e0"),
            margin=dict(l=10, r=10, t=10, b=10), height=280,
            showlegend=True,
        )
        st.plotly_chart(fig_eps, use_container_width=True, key="ql_eps_chart")
    with ec2:
        st.markdown(f"**ε = {eps_slider:.2f}**")
        st.markdown(f"- Explore probability: **{eps_slider*100:.0f}%**")
        st.markdown(f"- Exploit probability: **{(1-eps_slider)*100:.0f}%**")
        if eps_slider > 0.9:
            st.warning("Very high ε → mostly random. Agent hasn't learned yet (early training).")
        elif eps_slider < 0.1:
            st.success("Very low ε → mostly greedy. Agent is confident in its Q-table (late training).")
        else:
            st.info("Balanced exploration and exploitation (mid-training).")

    # Epsilon decay visualization
    st.markdown("---")
    st.markdown("### 📉 Epsilon Decay Over Episodes")
    col_de1, col_de2 = st.columns([2, 3])
    with col_de1:
        start_eps  = st.number_input("Start ε", 0.1, 1.0, 1.0, 0.05, key="dec_start")
        decay_rate = st.number_input("Decay rate", 0.990, 0.999, 0.995, 0.001, format="%.3f", key="dec_rate")
        min_eps    = st.number_input("Min ε", 0.001, 0.1, 0.01, 0.001, format="%.3f", key="dec_min")
        n_ep_dec   = st.number_input("Episodes", 100, 5000, 2000, 100, key="dec_ep")
    with col_de2:
        eps_vals = []
        e = float(start_eps)
        for _ in range(int(n_ep_dec)):
            e = max(float(min_eps), e * float(decay_rate))
            eps_vals.append(e)
        fig_decay = go.Figure(go.Scatter(
            y=eps_vals, mode="lines", line=dict(color="#f44336", width=2),
            fill="tozeroy", fillcolor="rgba(244,67,54,0.15)",
        ))
        fig_decay.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
            font=dict(color="#e0e0e0"),
            xaxis=dict(title="Episode", gridcolor="#0f3460"),
            yaxis=dict(title="ε", gridcolor="#0f3460"),
            margin=dict(l=40, r=20, t=20, b=40), height=250,
        )
        st.plotly_chart(fig_decay, use_container_width=True, key="ql_decay_chart")

    render_quiz("q_learning")
    render_llm_chat(
        context="We are discussing the Q-Learning algorithm, Q-table, update rule Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)], TD error, and epsilon-greedy exploration.",
        placeholder_question="Why do we need epsilon-greedy instead of always picking the best action?",
        chat_key="q_learning",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GRID WORLD DEMO
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## Grid World Demo")

    # Auto-train default agent on first visit
    ensure_default_agent()

    env_d     = st.session_state.default_env
    agent_d   = st.session_state.default_agent
    history_d = st.session_state.default_history

    # Layout info
    st.markdown("### The 5×5 Grid World Environment")
    info_c1, info_c2 = st.columns([2, 1])
    with info_c2:
        st.markdown("**Rewards**")
        st.markdown("- 🏆 Goal: **+10**")
        st.markdown("- 💀 Trap: **−5**")
        st.markdown("- ➡️ Step: **−0.1**")
        st.markdown("- 🧱 Boundary: **−0.5**")
        st.markdown("")
        st.markdown("**Layout**")
        st.markdown("- Start: (0, 0) top-left")
        st.markdown("- Goal: (4, 4) bottom-right")
        st.markdown("- Traps: (0,3), (1,1), (2,2), (3,3)")
    with info_c1:
        fig_static = create_grid_figure(env_d, q_table=agent_d.Q, show_arrows=True,
                                        title="Grid World — Optimal Policy Arrows")
        st.plotly_chart(fig_static, use_container_width=True, key="demo_static_grid")

    # Episode checkpoint selector
    st.markdown("---")
    st.markdown("### How the Agent Improves During Training")
    st.info(
        "The agent trains for **2,000 episodes**. An **episode** is one complete attempt: "
        "the agent starts at S, takes steps one at a time, and either reaches the goal, hits a trap, "
        "or runs out of moves. We captured a snapshot of the agent's knowledge (Q-table) and the path "
        "it took at five checkpoints. Select a checkpoint below to see how its strategy evolved."
    )
    CHECKPOINT_DESC = {
        50:   ("2.5% through training — mostly random",
               "Epsilon (exploration rate) ≈ 0.78. The agent still tries almost random actions. "
               "Paths are chaotic; it usually falls into a trap or wanders aimlessly."),
        200:  ("10% through training — starting to learn",
               "Epsilon ≈ 0.36. The agent has learned to avoid some traps but its policy still has big gaps."),
        500:  ("25% through training — much improved",
               "Epsilon ≈ 0.08. The agent reaches the goal most of the time and the arrows show a mostly-correct policy."),
        1000: ("50% through training — nearly optimal",
               "Epsilon is near its minimum (0.01). The agent reliably reaches the goal and Q-values are nearly converged."),
        2000: ("100% — fully trained",
               "Training complete. Compare these policy arrows to Episode 50 — completely different! "
               "The agent now follows a near-optimal path every time."),
    }
    available_eps = sorted(history_d["snapshots"].keys())
    n_total = len(history_d["episode_rewards"])
    ep_select = st.selectbox(
        "Select a training checkpoint to inspect:",
        options=available_eps,
        format_func=lambda x: f"Episode {x}  —  {CHECKPOINT_DESC[x][0]}",
        key="ep_select",
    )
    st.progress(ep_select / n_total, text=f"Training progress: episode {ep_select} of {n_total}")
    st.caption(CHECKPOINT_DESC[ep_select][1])
    path_snap = history_d["paths"].get(ep_select, [])
    q_snap    = history_d["snapshots"].get(ep_select)
    col_snap1, col_snap2 = st.columns(2)
    with col_snap1:
        fig_snap = create_grid_figure(env_d, q_table=q_snap, path=path_snap, show_arrows=True,
                                      title=f"Policy arrows + path — Episode {ep_select}")
        st.plotly_chart(fig_snap, use_container_width=True, key="demo_snap_path")
    with col_snap2:
        if path_snap:
            final_pos = path_snap[-1]
            if final_pos == env_d.goal:
                st.success("Outcome: Goal reached!")
            elif final_pos in env_d.traps:
                st.error("Outcome: Hit a trap!")
            else:
                st.warning("Outcome: Timed out (too many steps)")
            st.metric("Steps taken", len(path_snap) - 1)
            ep_idx = ep_select - 1
            window = history_d["success_flags"][max(0, ep_idx-99): ep_idx+1]
            sr = int(np.mean(window)*100) if window else 0
            st.metric("Success rate at this checkpoint", f"{sr}%")
            st.markdown(
                "**What the arrows show:** each arrow points to the action with the highest "
                "Q-value in that cell — the agent's current best guess about which direction "
                "leads to the most future reward."
            )

    # Animated path of fully-trained agent
    st.markdown("---")
    st.markdown("### Animated — Fully Trained Agent")
    st.info("Press **Play** to watch the agent navigate the grid. Drag the slider to scrub through steps manually.")
    final_path, final_reward = run_greedy_episode(env_d, agent_d)
    render_step_animation(env_d, final_path, unique_key="demo_main", title="Trained Agent (Greedy Policy)")
    st.success(f"Total reward: **{final_reward:.2f}** | Steps: **{len(final_path)-1}**")

    # Q-table heatmap + explanation
    st.markdown("---")
    st.markdown("### What the Agent Learned — Q-Table Heatmap")
    st.info(
        "Each cell shows the **maximum Q-value** across all four actions for that grid position — "
        "how much total reward the agent expects to collect from here if it plays optimally. "
        "**Green = high value** (near goal, easy to reach it). **Red = low/negative** (trap cells or very far from goal)."
    )
    fig_heat = create_qtable_heatmap(env_d, agent_d.Q)
    st.plotly_chart(fig_heat, use_container_width=True, key="demo_qtable_heat")
    with st.expander("Understanding these Q-values with γ = 0.99 — click to expand"):
        st.markdown("""
**Why are the values what they are?**

With γ = 0.99 (very close to 1) the agent values future rewards almost as much as immediate ones,
so it is willing to plan many steps ahead.

| Cell type | What the value means |
|-----------|----------------------|
| **Bright green (~9–10)** | Very close to the goal — reward ≈ +10 discounted by only 1–2 steps |
| **Mid green (~6–8)** | A few steps from the goal — reward is discounted by γ per step |
| **Near zero** | Far from the goal or the agent rarely visits here |
| **0 (trap/goal cells)** | Terminal states — the episode ends there, no future reward possible |

**The math behind it:** Goal = +10 reward. From the cell one step away, the agent earns
−0.1 (step penalty) + 0.99 × 10 = **9.9**. From two steps away:
−0.1 + 0.99 × 9.9 = **9.701**. Each additional step reduces the value by ~1%.
That is why the bottom row and right column are the greenest — they are closest to the goal.

**Try γ = 0.5 in the Train Your Agent tab.** With a short-sighted agent, states more than
3–4 steps from the goal will have Q-values near zero because future rewards are discounted
so aggressively (0.5^4 = 0.06). The agent will struggle to navigate longer paths.
        """)
        symbols = GridWorld.ACTION_SYMBOLS
        rows = []
        for r in range(env_d.size):
            row = {}
            for c in range(env_d.size):
                state_idx = env_d.pos_to_state(r, c)
                pos = (r, c)
                if pos == env_d.goal:
                    row[f"Col {c}"] = "GOAL"
                elif pos in env_d.traps:
                    row[f"Col {c}"] = "TRAP"
                else:
                    best_a = int(np.argmax(agent_d.Q[state_idx]))
                    best_q = float(np.max(agent_d.Q[state_idx]))
                    row[f"Col {c}"] = f"{symbols[best_a]} {best_q:.2f}"
            rows.append(row)
        qtdf = pd.DataFrame(rows, index=[f"Row {r}" for r in range(env_d.size)])
        st.dataframe(qtdf, use_container_width=True)

    # Training metrics
    st.markdown("---")
    st.markdown("### 📈 Training Metrics (Default Agent)")
    n_eps_d = len(history_d["episode_rewards"])
    success_rate = int(np.mean(history_d["success_flags"][-100:]) * 100)
    avg_reward   = float(np.mean(history_d["episode_rewards"][-100:]))
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Final Success Rate", f"{success_rate}%", "Last 100 episodes")
    mc2.metric("Final Avg Reward",   f"{avg_reward:.2f}", "Last 100 episodes")
    mc3.metric("Training Episodes",  f"{n_eps_d}", "Default: 2000")
    fig_metrics = create_training_charts(history_d)
    st.plotly_chart(fig_metrics, use_container_width=True, key="demo_train_metrics")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — TRAIN YOUR AGENT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("## Train Your Own Agent")
    st.markdown(
        "Tune the hyperparameters below and hit **Train Agent** to see how they affect learning. "
        "Compare multiple runs using the comparison table at the bottom."
    )

    # Hyperparameter sliders
    st.markdown("### ⚙️ Hyperparameters")
    hc1, hc2 = st.columns(2)
    with hc1:
        hp_alpha   = st.slider("Learning Rate α", 0.01, 1.0, 0.1, 0.01, key="hp_alpha")
        hp_gamma   = st.slider("Discount Factor γ", 0.5, 1.0, 0.99, 0.01, key="hp_gamma")
    with hc2:
        hp_eps_dec = st.slider("Epsilon Decay", 0.990, 0.999, 0.995, 0.001, format="%.3f", key="hp_eps_dec")
        hp_n_ep    = st.select_slider("Number of Episodes",
                                      options=[500, 1000, 2000, 5000], value=2000, key="hp_n_ep")

    st.markdown(
        f"<div style='background:#0d2a5e;border-left:4px solid #4fc3f7;border-radius:6px;padding:12px 16px;margin:10px 0;color:#e0e0e0;'>"
        f"<b>Config:</b> α={hp_alpha}, γ={hp_gamma}, ε_decay={hp_eps_dec:.3f}, episodes={hp_n_ep}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button("🚀 Train Agent", key="train_btn", type="primary"):
        env_c   = GridWorld()
        agent_c = QLearningAgent(
            env_c.n_states, env_c.n_actions,
            alpha=hp_alpha, gamma=hp_gamma,
            epsilon=1.0, epsilon_decay=hp_eps_dec, epsilon_min=0.01, seed=42,
        )
        with st.spinner(f"Training for {hp_n_ep} episodes…"):
            hist_c = train_agent(env_c, agent_c, n_episodes=hp_n_ep)
        hist_c["epsilon_history"] = agent_c.epsilon_history
        st.session_state.custom_agent   = agent_c
        st.session_state.custom_history = hist_c
        st.session_state.custom_env     = env_c

        # Store run for comparison
        sr_final = float(np.mean(hist_c["success_flags"][-100:])) * 100
        ar_final = float(np.mean(hist_c["episode_rewards"][-100:]))
        # Episodes to 80% success
        rolling_s = [np.mean(hist_c["success_flags"][max(0,i-100):i+1]) for i in range(len(hist_c["success_flags"]))]
        ep_80 = next((i+1 for i, v in enumerate(rolling_s) if v >= 0.8), hp_n_ep)
        st.session_state.training_runs.append({
            "α": hp_alpha, "γ": hp_gamma, "ε_decay": hp_eps_dec, "episodes": hp_n_ep,
            "Success %": round(sr_final, 1), "Avg Reward": round(ar_final, 2),
            "Ep to 80%": ep_80,
        })
        st.success(f"Training complete! Final success rate: {sr_final:.1f}%")
        st.rerun()

    # Show results if available
    if st.session_state.custom_agent is not None:
        hist_c   = st.session_state.custom_history
        agent_c  = st.session_state.custom_agent
        env_c    = st.session_state.get("custom_env", GridWorld())

        st.markdown("---")
        st.markdown("### 📊 Training Results")

        sr_final = float(np.mean(hist_c["success_flags"][-100:])) * 100
        ar_final = float(np.mean(hist_c["episode_rewards"][-100:]))
        rolling_s = [np.mean(hist_c["success_flags"][max(0,i-100):i+1]) for i in range(len(hist_c["success_flags"]))]
        ep_80 = next((i+1 for i, v in enumerate(rolling_s) if v >= 0.8), len(hist_c["success_flags"]))

        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("Final Success Rate", f"{sr_final:.1f}%", "Last 100 ep")
        rm2.metric("Final Avg Reward",   f"{ar_final:.2f}",  "Last 100 ep")
        rm3.metric("Episodes to 80%",    str(ep_80))

        fig_train_c = create_training_charts(hist_c)
        st.plotly_chart(fig_train_c, use_container_width=True, key="custom_train_metrics")

        # Policy & Q-table
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Policy Visualization (Arrows)**")
            fig_pol = create_grid_figure(env_c, q_table=agent_c.Q, show_arrows=True,
                                         title="Learned Policy")
            st.plotly_chart(fig_pol, use_container_width=True, key="custom_policy_grid")
        with tc2:
            st.markdown("**Q-Table Heatmap**")
            fig_hc = create_qtable_heatmap(env_c, agent_c.Q)
            st.plotly_chart(fig_hc, use_container_width=True, key="custom_qtable_heat")

        # Watch agent run
        st.markdown("**Watch Your Trained Agent**")
        cpath, creward = run_greedy_episode(env_c, agent_c)
        render_step_animation(env_c, cpath, unique_key="custom_anim", title="Your Custom Agent — Greedy Episode")
        st.success(f"Total reward: **{creward:.2f}** | Steps: **{len(cpath)-1}**")

    # Comparison table
    if len(st.session_state.training_runs) > 1:
        st.markdown("---")
        st.markdown("### 🔍 Run Comparison")
        runs_df = pd.DataFrame(st.session_state.training_runs)
        runs_df.index = [f"Run {i+1}" for i in range(len(runs_df))]
        st.dataframe(
            runs_df.style.highlight_max(subset=["Success %", "Avg Reward"], color="#1b5e20")
                         .highlight_min(subset=["Ep to 80%"], color="#1b5e20"),
            use_container_width=True,
        )

        if st.button("Clear comparison history", key="clear_runs"):
            st.session_state.training_runs = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SUMMARY & CHALLENGES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("## Summary & Challenges")

    # Summary table
    st.markdown("### 📋 Concepts Covered")
    summary_df = pd.DataFrame({
        "Concept": [
            "Reinforcement Learning",
            "Agent",
            "Environment",
            "State (s)",
            "Action (a)",
            "Reward (r)",
            "Episode",
            "Discount Factor (γ)",
            "Policy (π)",
            "Value Function V(s)",
            "Action-Value Q(s,a)",
            "Bellman Equation",
            "Q-Learning",
            "TD Error",
            "ε-greedy",
        ],
        "Definition": [
            "Learning by trial-and-error through reward signals",
            "The learner/decision-maker",
            "Everything the agent interacts with",
            "Current situation snapshot",
            "Agent's choice at each step",
            "Scalar feedback signal",
            "Complete sequence from start to terminal state",
            "Weights future vs immediate rewards",
            "Mapping from states to actions",
            "Expected return starting from state s",
            "Expected return taking action a in state s",
            "Q*(s,a) = r + γ max Q*(s',a')",
            "Model-free algorithm to learn Q-values",
            "Difference between target and current Q estimate",
            "Balance exploration vs exploitation",
        ],
        "Key Formula / Symbol": [
            "G_t = Σ γᵏ rₜ₊ₖ",
            "—",
            "—",
            "s ∈ S",
            "a ∈ A",
            "r ∈ ℝ",
            "s₀, a₀, r₁, …, sT",
            "γ ∈ [0, 1]",
            "π(s) = a  or  π(a|s)",
            "V(s) = 𝔼[G_t | S_t=s]",
            "Q(s,a) = 𝔼[G_t | S_t=s, A_t=a]",
            "Q*(s,a) = r + γ max Q*(s',a')",
            "Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]",
            "δ = r + γ max Q(s',a') - Q(s,a)",
            "Act randomly w.p. ε, greedy w.p. (1-ε)",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # RL Roadmap
    st.markdown("---")
    st.markdown("### 🗺️ RL Roadmap — What's Next?")
    roadmap = [
        ("✅ Tabular Q-Learning", "What you've mastered! Works for small, discrete state spaces.", "#2ecc71"),
        ("⬆️ Deep Q-Network (DQN)", "Replace the Q-table with a neural network. Enables RL on Atari games with raw pixels. Key innovations: experience replay, target networks.", "#4fc3f7"),
        ("⬆️ Policy Gradient Methods", "Instead of learning Q-values, directly optimise the policy π with gradient ascent. REINFORCE, PPO, TRPO.", "#f39c12"),
        ("⬆️ Actor-Critic (A2C/A3C)", "Combine policy gradients (actor) with a value function (critic) for stable, efficient training. Foundation of modern RLHF.", "#9b59b6"),
        ("⬆️ Advanced Methods", "SAC, TD3, Rainbow DQN, MuZero, AlphaZero, RLHF — state of the art for robotics, games, and LLM alignment.", "#e74c3c"),
    ]
    for step_name, step_desc, step_col in roadmap:
        st.markdown(
            f"<div style='background:#16213e;border-left:4px solid {step_col};border-radius:12px;padding:16px 20px;margin-bottom:12px;color:#e0e0e0;'>"
            f"<b style='color:{step_col}'>{step_name}</b><br>"
            f"<small>{step_desc}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Final challenge quiz
    st.markdown("---")
    st.markdown("### 🏆 Final Challenge")
    st.markdown(
        "<div style='background:#3d1f00;border-left:4px solid #ff9800;border-radius:6px;padding:12px 16px;margin:10px 0;color:#e0e0e0;'>These 5 questions test deeper understanding — think carefully!</div>",
        unsafe_allow_html=True,
    )
    render_quiz("final")

    # Resources
    st.markdown("---")
    st.markdown("### 📚 Resources to Go Deeper")
    res_c1, res_c2, res_c3 = st.columns(3)
    with res_c1:
        st.markdown("""
<div style='background:#16213e;border:1px solid #0f3460;border-radius:12px;padding:16px 20px;color:#e0e0e0;'>
<b style='color:#4fc3f7'>Books</b><br><br>
<span style='color:#e0e0e0'>• Sutton &amp; Barto — <i>Reinforcement Learning: An Introduction</i> (free online)</span><br>
<span style='color:#e0e0e0'>• Agarwal et al. — <i>Reinforcement Learning: Theory and Algorithms</i></span>
</div>""", unsafe_allow_html=True)
    with res_c2:
        st.markdown("""
<div style='background:#16213e;border:1px solid #0f3460;border-radius:12px;padding:16px 20px;color:#e0e0e0;'>
<b style='color:#4fc3f7'>Tools &amp; Libraries</b><br><br>
<span style='color:#e0e0e0'>• <code>gymnasium</code> — standard RL environments</span><br>
<span style='color:#e0e0e0'>• <code>stable-baselines3</code> — pre-built RL algorithms</span><br>
<span style='color:#e0e0e0'>• <code>ray[rllib]</code> — scalable RL training</span>
</div>""", unsafe_allow_html=True)
    with res_c3:
        st.markdown("""
<div style='background:#16213e;border:1px solid #0f3460;border-radius:12px;padding:16px 20px;color:#e0e0e0;'>
<b style='color:#4fc3f7'>Courses</b><br><br>
<span style='color:#e0e0e0'>• David Silver's RL course (UCL/DeepMind) — YouTube</span><br>
<span style='color:#e0e0e0'>• CS285 Deep RL (Berkeley) — YouTube</span><br>
<span style='color:#e0e0e0'>• Hugging Face Deep RL Course — free, hands-on</span>
</div>""", unsafe_allow_html=True)

    # Completion celebration
    st.markdown("---")
    all_submitted = all(st.session_state[f"quiz_submitted_{sec}"] for sec in SECTIONS)
    total_score_f = sum(st.session_state[f"quiz_score_{sec}"] for sec in SECTIONS)
    total_max_f   = sum(SECTION_MAX[sec] for sec in SECTIONS)

    if all_submitted:
        pct_f = int(total_score_f / total_max_f * 100)
        if pct_f >= 80:
            st.balloons()
            st.markdown(
                f"<div style='background:#0d3320;border-left:4px solid #4caf50;border-radius:6px;padding:12px 16px;margin:10px 0;color:#e0e0e0;'>"
                f"<h3>🎉 Congratulations! Outstanding performance!</h3>"
                f"You scored <b>{total_score_f}/{total_max_f} ({pct_f}%)</b>. "
                f"You have a strong grasp of Reinforcement Learning fundamentals. "
                f"You're ready to tackle DQN and Policy Gradient methods!"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif pct_f >= 60:
            st.markdown(
                f"<div style='background:#0d2a5e;border-left:4px solid #4fc3f7;border-radius:6px;padding:12px 16px;margin:10px 0;color:#e0e0e0;'>"
                f"<h3>Good work! Keep it up.</h3>"
                f"You scored <b>{total_score_f}/{total_max_f} ({pct_f}%)</b>. "
                f"Review the sections where you lost points, retake those quizzes, and try again!"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#3d1f00;border-left:4px solid #ff9800;border-radius:6px;padding:12px 16px;margin:10px 0;color:#e0e0e0;'>"
                f"<h3>Keep practising!</h3>"
                f"You scored <b>{total_score_f}/{total_max_f} ({pct_f}%)</b>. "
                f"Re-read the earlier sections, use the AI Tutor to ask questions, and retake the quizzes."
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='background:#0d2a5e;border-left:4px solid #4fc3f7;border-radius:6px;padding:12px 16px;margin:10px 0;color:#e0e0e0;'>Complete all quizzes in the earlier tabs to unlock your final score and celebration!</div>",
            unsafe_allow_html=True,
        )

    render_llm_chat(
        context="We have finished a comprehensive RL tutorial covering: RL basics, RL framework, policies, value functions, Bellman equation, Q-learning, Grid World, and the RL roadmap (DQN, Policy Gradients, Actor-Critic).",
        placeholder_question="What should I learn next after tabular Q-learning?",
        chat_key="summary",
    )
