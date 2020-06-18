#!/usr/bin/env python

import numpy as np

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter

ENV = 'CartPole-v0'
LEARNING_RATE = 0.01


class A2CNetwork(nn.Module):

    def __init__(self, observation_dims, actions_dims):

        super(A2CNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(observation_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Policy
        self.actor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, actions_dims)
        )

        # Value
        self.critic = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.actor(self.net(x)), self.critic(self.net(x))


class A2CAgent():

    def __init__(self, net, env):
        self.net = net
        self.env = env
        self.total_episode_reward = 0.

        self.state = np.float32(self.env.reset())

        self.action_list = list(range(self.env.action_space.n))

    def step(self, render=False):
        current_state = self.state

        logits_actions_v = self.net(torch.FloatTensor(current_state))[0]
        probs_actions_v = F.softmax(logits_actions_v, dim=0)

        action = np.random.choice(self.action_list, p=probs_actions_v.data.cpu().numpy())

        next_state, reward, done, _ = self.env.step(action)
        next_state = np.float32(next_state)

        self.total_episode_reward += reward

        if not done:
            self.state = next_state
            episode_reward = None  # not done
        else:
            episode_reward = self.total_episode_reward

            self.total_episode_reward = 0.
            self.state = np.float32(self.env.reset())

        return current_state, action, reward, episode_reward


def learn(env):
    net = A2CNetwork(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    agent = A2CAgent(net, env)

    writer = SummaryWriter(comment=f"-{ENV}-a2c")

    states = []
    actions = []
    rewards = []

    step_count = 0

    while True:
        step_count += 1

        state, action, reward, episode_reward = agent.step()

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        if step_count > 10: break


if __name__ == "__main__":
    import gym
    env = gym.make(ENV)

    learn(env)
