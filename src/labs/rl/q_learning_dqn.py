"""
Lab: Reinforcement Learning - Q-Learning and DQN
==================================================
"""
import numpy as np


# =============================================================================
# TABULAR Q-LEARNING
# =============================================================================
def train_q_learning(env_name="FrozenLake-v1", episodes=5000):
    """Train a tabular Q-learning agent."""
    import gymnasium as gym

    env = gym.make(env_name, is_slippery=False)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    alpha, gamma, epsilon = 0.8, 0.95, 0.1

    rewards_history = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

    env.close()
    return Q, rewards_history


# =============================================================================
# SIMPLE DQN (PyTorch)
# =============================================================================
class DQNAgent:
    """Minimal Deep Q-Network agent for demonstration."""

    def __init__(self, state_dim, action_dim, hidden=128, lr=1e-3, gamma=0.99):
        import torch
        import torch.nn as nn

        self.gamma = gamma
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        ).to(self.device)

        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = []
        self.max_buffer = 10000

    def select_action(self, state, epsilon=0.1):
        import torch

        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(1).item()

    def store_transition(self, s, a, r, s_next, done):
        if len(self.replay_buffer) >= self.max_buffer:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((s, a, r, s_next, done))

    def train_step(self, batch_size=64):
        import torch

        if len(self.replay_buffer) < batch_size:
            return 0.0

        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = torch.nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    print("=" * 60)
    print("TABULAR Q-LEARNING ON FROZENLAKE")
    print("=" * 60)
    try:
        Q, rewards = train_q_learning(episodes=2000)
        action_labels = ["Left", "Down", "Right", "Up"]
        policy = np.argmax(Q, axis=1).reshape(4, 4)
        print("Learned Policy (0=L, 1=D, 2=R, 3=U):")
        print(policy)
        print(f"Success rate (last 100): {np.mean(rewards[-100:]):.2f}")
    except ImportError:
        print("Install gymnasium: pip install gymnasium")
