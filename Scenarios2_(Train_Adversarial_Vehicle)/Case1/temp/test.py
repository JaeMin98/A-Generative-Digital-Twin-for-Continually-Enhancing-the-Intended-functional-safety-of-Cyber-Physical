import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Placeholder rewards data for demonstration
# In a real scenario, this would come from the training process
rewards = np.random.binomial(1, 0.5, 1000).cumsum()

# 학습 과정에서의 보상을 시각화
def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards over Episodes in Bandit Problem')
    plt.legend()
    plt.grid(True)
    plt.show()


# 밴딧 문제 환경 설정
class BanditEnv:
    def __init__(self, k):
        self.k = k
        self.probs = np.random.rand(k)

    def step(self, action):
        reward = 1 if np.random.rand() < self.probs[action] else 0
        return reward

# Q-네트워크 정의
class QNetwork(nn.Module):
    def __init__(self, k):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(k, k)

    def forward(self, x):
        return self.fc(x)

# SAC 알고리즘 적용
class SACAgent:
    def __init__(self, k, lr=0.01, gamma=0.99, alpha=0.1):
        self.k = k
        self.q_net = QNetwork(k)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.alpha = alpha

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state)
        action = torch.argmax(q_values).item()
        return action

    def update(self, state, action, reward):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state)
        q_value = q_values[0, action]

        target = reward + self.alpha * torch.logsumexp(q_values / self.alpha, dim=1)
        loss = F.mse_loss(q_value, target)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

# 학습 과정
def train_bandit(env, agent, episodes=10000):
    rewards = []
    for episode in range(episodes):
        state = np.ones(env.k)  # 밴딧 문제에서는 상태가 고정
        action = agent.select_action(state)
        reward = env.step(action)
        agent.update(state, action, reward)
        rewards.append(reward)

    return rewards

# 주요 파라미터 설정
k = 10  # 밴딧의 개수
env = BanditEnv(k)
agent = SACAgent(k)

# 학습 실행
rewards = train_bandit(env, agent)

# 결과 출력
print(f"총 보상: {sum(rewards)}")
