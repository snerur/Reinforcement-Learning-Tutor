import numpy as np
from typing import Dict, List, Tuple, Optional


class GridWorld:
    """5x5 Grid World RL environment."""
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTION_NAMES = ['↑ UP', '↓ DOWN', '← LEFT', '→ RIGHT']
    ACTION_SYMBOLS = ['↑', '↓', '←', '→']

    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.traps = {(0, 3), (1, 1), (2, 2), (3, 3)}
        self.reward_goal = 10.0
        self.reward_trap = -5.0
        self.reward_step = -0.1
        self.reward_boundary = -0.5
        self.state = self.start

    def pos_to_state(self, row, col):
        return row * self.size + col

    def state_to_pos(self, state):
        return divmod(state, self.size)

    def get_state_index(self):
        return self.pos_to_state(*self.state)

    def reset(self):
        self.state = self.start
        return self.get_state_index()

    def step(self, action):
        row, col = self.state
        if action == self.UP:
            new_row, new_col = row - 1, col
        elif action == self.DOWN:
            new_row, new_col = row + 1, col
        elif action == self.LEFT:
            new_row, new_col = row, col - 1
        else:
            new_row, new_col = row, col + 1

        hit_boundary = not (0 <= new_row < self.size and 0 <= new_col < self.size)
        if hit_boundary:
            new_row, new_col = row, col
            reward = self.reward_boundary
            done = False
        elif (new_row, new_col) == self.goal:
            reward = self.reward_goal
            done = True
        elif (new_row, new_col) in self.traps:
            reward = self.reward_trap
            done = True
        else:
            reward = self.reward_step
            done = False

        self.state = (new_row, new_col)
        return self.get_state_index(), reward, done


class QLearningAgent:
    """Tabular Q-Learning Agent with epsilon-greedy exploration."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seed=42):
        np.random.seed(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = np.zeros((n_states, n_actions))
        self.epsilon_history = []

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        current_q = self.Q[state, action]
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        td_error = target - current_q
        self.Q[state, action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


def train_agent(env, agent, n_episodes=2000, max_steps=100):
    """Train agent and return history with snapshots and paths."""
    snapshot_episodes = {50, 200, 500, 1000, n_episodes}
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_flags': [],
        'td_errors': [],
        'snapshots': {},
        'paths': {},
    }

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        ep_td_errors = []
        path = [env.state]

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            td_error = agent.update(state, action, reward, next_state, done)
            path.append(env.state)
            ep_td_errors.append(abs(td_error))
            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break

        agent.decay_epsilon()
        goal_idx = env.pos_to_state(*env.goal)
        succeeded = (state == goal_idx)

        history['episode_rewards'].append(total_reward)
        history['episode_lengths'].append(steps)
        history['success_flags'].append(succeeded)
        history['td_errors'].append(np.mean(ep_td_errors) if ep_td_errors else 0)

        if ep in snapshot_episodes:
            history['snapshots'][ep] = agent.Q.copy()
            history['paths'][ep] = path.copy()

    return history


def run_greedy_episode(env, agent, max_steps=50):
    """Run one greedy episode. Return path and total reward."""
    state = env.reset()
    path = [env.state]
    total_reward = 0

    for _ in range(max_steps):
        action = int(np.argmax(agent.Q[state]))
        next_state, reward, done = env.step(action)
        path.append(env.state)
        total_reward += reward
        state = next_state
        if done:
            break

    return path, total_reward
