""" Implementation of a Deep SARSA Agent to solve the CartPole-v1 environment. """

import gym
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Generator, List, Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

        # Initialize weights using orthogonal initialization
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)

    def forward(self, x) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Memory:
    def __init__(self, size):
        self.size = size
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, next_action, done) -> None:
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.position] = (
            state,
            action,
            reward,
            next_state,
            next_action,
            done,
        )
        self.position = (self.position + 1) % self.size

    def sample(
        self, batch_size
    ) -> List[Tuple[torch.Tensor, int, int, torch.Tensor, int, bool]]:
        return random.sample(self.memory, batch_size)


class Agent:
    """Deep SARSA (Replay Buffer Off-Policy Variation) Agent"""

    def __init__(
        self,
        env,
        n_inputs,
        n_outputs,
        lr,
        gamma,
        epsilon,
        epsilon_decay,
        epsilon_min,
        batch_size,
        memory_size,
    ):
        self.env = env

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.model = Network(n_inputs, n_outputs).to("cuda")
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.001
        )
        self.memory = Memory(self.memory_size)

    def get_action(self, state) -> int:
        """Get an action from the agent
        using an epsilon-greedy policy.
        """

        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()

        state = torch.tensor(state, dtype=torch.float, device="cuda")
        action = self.model(state)
        return torch.argmax(action).item()

    def update(self) -> None:
        """Process a batch of experiences, calculate the q-values,
        and propagate the loss backwards through the network.
        """

        if len(self.memory.memory) < self.memory_size:
            return

        batch = self.memory.sample(self.batch_size)

        # Convert batch to tensors, this speeds up computation a lot
        batched_state = torch.tensor(
            np.array([b[0] for b in batch]), dtype=torch.float, device="cuda"
        )

        batched_action = torch.tensor(
            np.array([b[1] for b in batch]), dtype=torch.long, device="cuda"
        )
        batched_reward = torch.tensor(
            np.array([b[2] for b in batch]), dtype=torch.float, device="cuda"
        )
        batched_next_state = torch.tensor(
            np.array([b[3] for b in batch]), dtype=torch.float, device="cuda"
        )
        batched_next_action = torch.tensor(
            np.array([b[4] for b in batch]), dtype=torch.long, device="cuda"
        )

        batched_dones = torch.tensor(
            np.array([b[5] for b in batch]), dtype=torch.float, device="cuda"
        )

        q_value = (
            self.model(batched_state).gather(1, batched_action.unsqueeze(1)).squeeze(1)
        )
        next_q_value = (
            self.model(batched_next_state)
            .gather(1, batched_next_action.unsqueeze(1))
            .squeeze(1)
            .detach()
        )
        expected_q_value = (
            batched_reward + next_q_value * (1 - batched_dones) * self.gamma
        )

        loss = self.loss_fn(q_value, expected_q_value)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train(self, episodes) -> Generator[int, None, None]:
        """Train the agent for a given number of episodes."""

        for episode in range(episodes):
            done = False
            truncated = False

            state = self.env.reset()
            action = self.get_action(state)

            episode_reward = 0
            while not done or not truncated:
                if episode > 40:
                    self.env.render()

                next_state, reward, done, truncated = self.env.step(action)

                next_action = self.get_action(next_state)

                self.memory.push(state, action, reward, next_state, next_action, done)

                self.update()
                state = next_state
                action = next_action

                episode_reward += reward

            print(
                "Episode: {}, Reward: {}, Epsilon: {}".format(
                    episode, episode_reward, self.epsilon
                )
            )
            yield episode_reward


if __name__ == "__main__":
    agent = Agent(
        env=gym.make("CartPole-v1"),
        n_inputs=4,
        n_outputs=2,
        lr=0.01,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        batch_size=32,
        memory_size=10000,
    )

    rewards = []
    plt.ion()
    for duration in agent.train(100):
        rewards.append(duration)

        plt.plot(rewards)

        # Calculate the moving average
        if len(rewards) > 10:
            moving_avg = np.convolve(rewards, np.ones((10,)) / 10, mode="valid")
            # Offset the moving average on the x-axis
            plt.plot(range(9, len(rewards)), moving_avg, color="orange")

        # Annotations
        plt.xlabel("Episode")
        plt.ylabel("Duration")

        plt.title("Deep SARSA on CartPole-v1")

        # Add legend for moving average
        plt.legend(["Duration", "Moving Average (10 Episodes)"])

        plt.draw()
        plt.pause(0.00001)
        plt.savefig("deep_sarsa.png")
        plt.clf()