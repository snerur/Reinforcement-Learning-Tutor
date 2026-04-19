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
}
.feature-card h3 { color: #4fc3f7; margin-bottom: 6px; }
/* ── info boxes ── */
.info-box {
    background: #0d47a1;
    border-left: 4px solid #4fc3f7;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
}
.success-box {
    background: #1b5e20;
    border-left: 4px solid #4caf50;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
}
.warning-box {
    background: #4a2c00;
    border-left: 4px solid #ff9800;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
}
/* ── quiz ── */
.quiz-card {
    background: #1e2a45;
    border: 1px solid #0f3460;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
}
.quiz-correct { color: #4caf50; font-weight: bold; }
.quiz-wrong   { color: #f44336; font-weight: bold; }
/* ── sidebar ── */
.sidebar-score {
    font-size: 0.9rem;
    padding: 4px 0;
}
/* ── formula ── */
.formula-box {
    background: #0a0a1a;
    border: 1px solid #4fc3f7;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    margin: 10px 0;
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
        st.markdown(f"<div class='quiz-card'>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)

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
                    f"<div class='info-box'><b>AI Tutor:</b><br>{response}</div>",
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


def create_animated_path(env, path, title="Agent Path Animation"):
    """Create an animated Plotly figure showing the agent traversing a path."""
    size = env.size
    shapes, base_annotations = _grid_shapes_and_annotations(env)

    path_x_all = [c + 0.5 for (r, c) in path]
    path_y_all = [size - r - 0.5 for (r, c) in path]

    frames = []
    for i in range(len(path)):
        frame_data = [
            go.Scatter(
                x=path_x_all[:i+1], y=path_y_all[:i+1],
                mode="lines",
                line=dict(color="gold", width=3, dash="dot"),
                showlegend=False,
            ),
            go.Scatter(
                x=[path_x_all[i]], y=[path_y_all[i]],
                mode="markers+text",
                marker=dict(size=28, color="#9b59b6", symbol="circle"),
                text=["🤖"], textposition="middle center",
                showlegend=False,
            ),
        ]
        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig = go.Figure(
        data=[
            go.Scatter(x=[path_x_all[0]], y=[path_y_all[0]],
                       mode="lines", line=dict(color="gold", width=3, dash="dot"), showlegend=False),
            go.Scatter(x=[path_x_all[0]], y=[path_y_all[0]],
                       mode="markers+text",
                       marker=dict(size=28, color="#9b59b6"), text=["🤖"],
                       textposition="middle center", showlegend=False),
        ],
        frames=frames,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color="#4fc3f7", size=16)),
        shapes=shapes, annotations=base_annotations,
        xaxis=dict(range=[0, size], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, size], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        margin=dict(l=10, r=10, t=60, b=10),
        height=440,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.1, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=400, redraw=True), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0))],
                        label=str(i)) for i in range(len(path))],
            currentvalue=dict(prefix="Step: "),
            x=0.05, len=0.9, y=0,
        )],
    )
    return fig


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
        colorbar=dict(title="Max Q", tickfont=dict(color="#e0e0e0"), titlefont=dict(color="#e0e0e0")),
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
    "🏠 Home",
    "❓ What is RL?",
    "🔄 RL Framework",
    "🧠 Key Concepts",
    "📊 Q-Learning",
    "🎮 Grid World Demo",
    "🏋️ Train Your Agent",
    "🎓 Summary & Challenges",
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
        st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)

    with col_prog:
        st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)

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
                f"<div class='feature-card'><h3>{icon} {title}</h3><p style='color:#ccc;font-size:0.9rem'>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown(
        "**How to use this app:** Navigate the tabs left-to-right. Each tab introduces new concepts, "
        "ends with a quiz, and has an AI Tutor expander. Configure your preferred LLM in the sidebar "
        "to unlock the AI Tutor. The Grid World Demo and Train Your Agent tabs let you interact with "
        "a live Q-learning simulation."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WHAT IS RL?
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-title'>❓ What is Reinforcement Learning?</div>", unsafe_allow_html=True)

    # Dog trick analogy
    col_dog, col_def = st.columns([1, 2])
    with col_dog:
        st.markdown(
            "<div class='rl-card' style='text-align:center;font-size:3rem'>🐕</div>",
            unsafe_allow_html=True,
        )
    with col_def:
        st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
        st.markdown("### The Dog Trick Analogy")
        st.markdown(
            "Imagine teaching a dog to sit. You don't give it a manual — you simply **reward** it "
            "with a treat when it does the right thing, and ignore (or gently correct) it when it doesn't. "
            "Over thousands of interactions, the dog **learns** which actions lead to treats.\n\n"
            "Reinforcement Learning works the same way: an **agent** learns by trying actions in an "
            "**environment** and receiving **rewards** or **penalties** as feedback — no labelled dataset required."
        )
        st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown("<div class='section-title'>🔄 The RL Framework</div>", unsafe_allow_html=True)

    # ASCII art loop
    st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
    st.markdown("### The Agent-Environment Loop")
    st.code(
        """
  ┌─────────────────────────────────────────────────────────────┐
  │                     ENVIRONMENT                             │
  │                                                             │
  │     State s_t ──────────────────────────────────────►      │
  │                                                      │      │
  │     Reward r_t ─────────────────────────────────►   │      │
  │                                                  │   │      │
  └──────────────────────────────────────────────────┼───┼──────┘
                                                     │   │
                                                     ▼   │
  ┌──────────────────────────────────────────────────┐   │
  │                      AGENT                       │   │
  │                                                  │   │
  │   Observes s_t, r_t  →  Policy π  →  Action a_t │   │
  │                                                  │   │
  └──────────────────────────────────────────────────┘   │
                            │                             │
                            └────────── a_t ──────────────┘
        """,
        language="",
    )
    st.markdown("</div>", unsafe_allow_html=True)

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
                f"<div class='rl-card' style='border-left:4px solid {comp_col}'>"
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
    st.markdown("<div class='section-title'>🧠 Key Concepts</div>", unsafe_allow_html=True)

    # Policy
    st.markdown("### 📋 Policy (π)")
    col_det, col_sto = st.columns(2)
    with col_det:
        st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
        st.markdown("**Deterministic Policy**")
        st.latex(r"\pi(s) = a")
        st.markdown(
            "Maps each state directly to a specific action. "
            "Given the same state, the agent always takes the same action. "
            "Simple and fast — great when the optimal action is clear."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_sto:
        st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
        st.markdown("**Stochastic Policy**")
        st.latex(r"\pi(a \mid s) = P(A_t = a \mid S_t = s)")
        st.markdown(
            "Maps each state to a probability distribution over actions. "
            "The agent samples an action from this distribution. "
            "Useful in games of chance or when multiple actions are equally good."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Value functions
    st.markdown("---")
    st.markdown("### 💰 Value Functions")
    col_v, col_q = st.columns(2)
    with col_v:
        st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
        st.markdown("**State Value Function V(s)**")
        st.latex(r"V^{\pi}(s) = \mathbb{E}_{\pi}\left[G_t \mid S_t = s\right]")
        st.markdown(
            "The expected cumulative return starting from state *s* and following policy π. "
            "A high V(s) means this is a good state to be in."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_q:
        st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
        st.markdown("**Action-Value Function Q(s, a)**")
        st.latex(r"Q^{\pi}(s,a) = \mathbb{E}_{\pi}\left[G_t \mid S_t = s,\, A_t = a\right]")
        st.markdown(
            "The expected cumulative return from state *s*, taking action *a*, then following policy π. "
            "Q-learning directly estimates this function."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Bellman equation
    st.markdown("---")
    st.markdown("### 🔔 The Bellman Equation")
    st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
    st.latex(r"Q^*(s,a) = r + \gamma \max_{a'} Q^*(s',a')")
    st.markdown(
        "The **Bellman Optimality Equation** says: the optimal Q-value of being in state *s* and taking action *a* "
        "equals the immediate reward *r* plus the discounted maximum future Q-value. "
        "This is the backbone of Q-learning — we keep updating Q-values until they satisfy this equation."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Bellman demo
    st.markdown("---")
    st.markdown("### 🔬 Interactive Bellman Calculation")

    np.random.seed(0)
    demo_q = np.random.uniform(-1, 5, (5, 4))

    state_pick = st.selectbox("Pick a state (0–4):", list(range(5)), key="bellman_state")
    action_pick = st.selectbox("Pick an action (0=↑, 1=↓, 2=←, 3=→):", [0, 1, 2, 3],
                                format_func=lambda x: GridWorld.ACTION_NAMES[x],
                                key="bellman_action")
    gamma_key = st.slider("γ for Bellman demo", 0.0, 1.0, 0.99, 0.01, key="bellman_gamma")
    r_demo    = st.number_input("Immediate reward r:", value=float(-0.1), step=0.1, key="bellman_r")

    next_s    = (state_pick + 1) % 5  # dummy next state
    max_q_ns  = float(np.max(demo_q[next_s]))
    current_q = float(demo_q[state_pick, action_pick])
    target    = r_demo + gamma_key * max_q_ns

    c_q1, c_q2, c_q3 = st.columns(3)
    c_q1.metric("Current Q(s,a)", f"{current_q:.3f}")
    c_q2.metric(f"r + γ·max Q(s',·)", f"{target:.3f}")
    c_q3.metric("TD Error (Target - Q)", f"{target - current_q:.3f}")

    st.markdown(
        f"<div class='info-box'>"
        f"Q({state_pick}, {GridWorld.ACTION_NAMES[action_pick]}) = {current_q:.3f} → "
        f"Target = {r_demo} + {gamma_key} × {max_q_ns:.3f} = {target:.3f} | "
        f"TD Error = {target - current_q:.3f}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Show demo Q-table
    with st.expander("Show demo Q-table"):
        q_df = pd.DataFrame(demo_q, columns=GridWorld.ACTION_NAMES)
        q_df.index.name = "State"
        st.dataframe(q_df.round(3), use_container_width=True)

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
    st.markdown("<div class='section-title'>📊 Q-Learning Algorithm</div>", unsafe_allow_html=True)

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
    st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
    st.latex(
        r"Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]"
    )
    st.markdown("</div>", unsafe_allow_html=True)

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
        st.plotly_chart(fig_eps, use_container_width=True)
    with ec2:
        st.markdown("<div class='rl-card' style='margin-top:40px'>", unsafe_allow_html=True)
        st.markdown(f"**ε = {eps_slider:.2f}**")
        st.markdown(f"- Explore probability: **{eps_slider*100:.0f}%**")
        st.markdown(f"- Exploit probability: **{(1-eps_slider)*100:.0f}%**")
        if eps_slider > 0.9:
            st.warning("Very high ε → mostly random. Agent hasn't learned yet (early training).")
        elif eps_slider < 0.1:
            st.success("Very low ε → mostly greedy. Agent is confident in its Q-table (late training).")
        else:
            st.info("Balanced exploration and exploitation (mid-training).")
        st.markdown("</div>", unsafe_allow_html=True)

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
        st.plotly_chart(fig_decay, use_container_width=True)

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
    st.markdown("<div class='section-title'>🎮 Grid World Demo</div>", unsafe_allow_html=True)

    # Auto-train default agent on first visit
    ensure_default_agent()

    env_d     = st.session_state.default_env
    agent_d   = st.session_state.default_agent
    history_d = st.session_state.default_history

    # Layout info
    st.markdown("### The 5×5 Grid World Environment")
    info_c1, info_c2 = st.columns([2, 1])
    with info_c2:
        st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)
    with info_c1:
        fig_static = create_grid_figure(env_d, q_table=agent_d.Q, show_arrows=True,
                                        title="Grid World — Optimal Policy Arrows")
        st.plotly_chart(fig_static, use_container_width=True)

    # Episode selector — show path from snapshot
    st.markdown("---")
    st.markdown("### 📽️ Agent Path at Different Training Stages")
    available_eps = sorted(history_d["snapshots"].keys())
    ep_select = st.selectbox(
        "Select training checkpoint episode:",
        options=available_eps,
        format_func=lambda x: f"Episode {x}",
        key="ep_select",
    )
    path_snap = history_d["paths"].get(ep_select, [])
    q_snap    = history_d["snapshots"].get(ep_select)

    col_snap1, col_snap2 = st.columns(2)
    with col_snap1:
        fig_snap = create_grid_figure(env_d, q_table=q_snap,
                                      path=path_snap, show_arrows=True,
                                      title=f"Policy & Path — Episode {ep_select}")
        st.plotly_chart(fig_snap, use_container_width=True)
    with col_snap2:
        if path_snap:
            final_pos = path_snap[-1]
            outcome = "🏆 Goal!" if final_pos == env_d.goal else ("💀 Trap" if final_pos in env_d.traps else "⏱️ Timeout")
            st.markdown(f"**Episode outcome:** {outcome}")
            st.markdown(f"**Path length:** {len(path_snap)} steps")
            if ep_select <= 50:
                st.warning("Early training: agent is mostly exploring randomly — path is chaotic.")
            elif ep_select <= 200:
                st.info("Mid training: agent is starting to learn useful paths but still makes mistakes.")
            else:
                st.success("Late training: agent has learned a near-optimal policy.")

    # Animated path of fully-trained agent
    st.markdown("---")
    st.markdown("### 🤖 Animated — Fully Trained Agent")
    final_path, final_reward = run_greedy_episode(env_d, agent_d)
    fig_anim = create_animated_path(env_d, final_path, "Trained Agent (Greedy Policy)")
    st.plotly_chart(fig_anim, use_container_width=True)
    st.markdown(f"Episode total reward: **{final_reward:.2f}** | Steps: **{len(final_path)}**")

    # Q-table heatmap
    st.markdown("---")
    st.markdown("### 🌡️ Q-Table Heatmap")
    fig_heat = create_qtable_heatmap(env_d, agent_d.Q)
    st.plotly_chart(fig_heat, use_container_width=True)

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
    st.plotly_chart(fig_metrics, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — TRAIN YOUR AGENT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("<div class='section-title'>🏋️ Train Your Own Agent</div>", unsafe_allow_html=True)
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
        f"<div class='info-box'>"
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
        st.plotly_chart(fig_train_c, use_container_width=True)

        # Policy & Q-table
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Policy Visualization (Arrows)**")
            fig_pol = create_grid_figure(env_c, q_table=agent_c.Q, show_arrows=True,
                                         title="Learned Policy")
            st.plotly_chart(fig_pol, use_container_width=True)
        with tc2:
            st.markdown("**Q-Table Heatmap**")
            fig_hc = create_qtable_heatmap(env_c, agent_c.Q)
            st.plotly_chart(fig_hc, use_container_width=True)

        # Watch agent run
        if st.button("👀 Watch Agent Run (Greedy)", key="watch_custom"):
            cpath, creward = run_greedy_episode(env_c, agent_c)
            fig_ca = create_animated_path(env_c, cpath, "Custom Agent — Greedy Episode")
            st.plotly_chart(fig_ca, use_container_width=True)
            st.markdown(f"Total reward: **{creward:.2f}** | Steps: **{len(cpath)}**")

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
    st.markdown("<div class='section-title'>🎓 Summary & Challenges</div>", unsafe_allow_html=True)

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
            f"<div class='rl-card' style='border-left:4px solid {step_col}'>"
            f"<b style='color:{step_col}'>{step_name}</b><br>"
            f"<small>{step_desc}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Final challenge quiz
    st.markdown("---")
    st.markdown("### 🏆 Final Challenge")
    st.markdown(
        "<div class='warning-box'>These 5 questions test deeper understanding — think carefully!</div>",
        unsafe_allow_html=True,
    )
    render_quiz("final")

    # Resources
    st.markdown("---")
    st.markdown("### 📚 Resources to Go Deeper")
    res_c1, res_c2, res_c3 = st.columns(3)
    with res_c1:
        st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
        st.markdown("**📖 Books**")
        st.markdown("- Sutton & Barto — *Reinforcement Learning: An Introduction* (free online)")
        st.markdown("- Agarwal et al. — *Reinforcement Learning: Theory and Algorithms*")
        st.markdown("</div>", unsafe_allow_html=True)
    with res_c2:
        st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
        st.markdown("**🛠️ Tools & Libraries**")
        st.markdown("- `gymnasium` — RL environments (formerly OpenAI Gym)")
        st.markdown("- `stable-baselines3` — Pre-built RL algorithms")
        st.markdown("- `ray[rllib]` — Scalable RL training")
        st.markdown("</div>", unsafe_allow_html=True)
    with res_c3:
        st.markdown("<div class='rl-card'>", unsafe_allow_html=True)
        st.markdown("**🎓 Courses**")
        st.markdown("- David Silver's RL course (UCL/DeepMind) — YouTube")
        st.markdown("- CS285 Deep RL (Berkeley) — YouTube")
        st.markdown("- Hugging Face Deep RL Course — free, hands-on")
        st.markdown("</div>", unsafe_allow_html=True)

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
                f"<div class='success-box'>"
                f"<h3>🎉 Congratulations! Outstanding performance!</h3>"
                f"You scored <b>{total_score_f}/{total_max_f} ({pct_f}%)</b>. "
                f"You have a strong grasp of Reinforcement Learning fundamentals. "
                f"You're ready to tackle DQN and Policy Gradient methods!"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif pct_f >= 60:
            st.markdown(
                f"<div class='info-box'>"
                f"<h3>Good work! Keep it up.</h3>"
                f"You scored <b>{total_score_f}/{total_max_f} ({pct_f}%)</b>. "
                f"Review the sections where you lost points, retake those quizzes, and try again!"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='warning-box'>"
                f"<h3>Keep practising!</h3>"
                f"You scored <b>{total_score_f}/{total_max_f} ({pct_f}%)</b>. "
                f"Re-read the earlier sections, use the AI Tutor to ask questions, and retake the quizzes."
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div class='info-box'>Complete all quizzes in the earlier tabs to unlock your final score and celebration!</div>",
            unsafe_allow_html=True,
        )

    render_llm_chat(
        context="We have finished a comprehensive RL tutorial covering: RL basics, RL framework, policies, value functions, Bellman equation, Q-learning, Grid World, and the RL roadmap (DQN, Policy Gradients, Actor-Critic).",
        placeholder_question="What should I learn next after tabular Q-learning?",
        chat_key="summary",
    )
