#!/usr/bin/env python
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from copy import copy

ENV = 'LunarLander-v2'
GAMMA = 0.99
LEARNING_RATE = 0.01
BATCH_EPISODES = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PolicyGradientNet(nn.Module):

    def __init__(self, num_observations, num_actions):

        super(PolicyGradientNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_observations, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_actions))

    def forward(self, x):
        return self.net(x)


def calc_qvalues(rewards, gamma=GAMMA):
    q_values = []
    R = 0.
    for r in reversed(rewards):
        R = r + gamma*R  # Bellmann's formula
        q_values.append(R)

    q_values = np.array(list(reversed(q_values)))
    return q_values - q_values.mean()


class PolicyGradientAgent():

    def __init__(self, net, env):
        self.net = net
        self.env = env

        self.state = np.float32(self.env.reset())
        self.tot_reward = 0

    def play_step(self, render=False):
        episode_reward = None

        current_state = copy(self.state)

        state_t = torch.tensor(current_state).to(device)
        prob_action_t = self.net(state_t)
        prob_action_t.detach()
        prob_action_t = F.softmax(prob_action_t, dim=0)
        action = np.random.choice(list(range(self.env.action_space.n)), p=prob_action_t.data.cpu().numpy())

        next_state, reward, done, _ = self.env.step(action)
        next_state = np.float32(next_state)

        self.tot_reward += reward
        # Hack
        #reward -= np.absolute(next_state[0])*10

        if render:
            env.render()

        if not done:
            self.state = next_state
        else:
            episode_reward = self.tot_reward
            self.tot_reward = 0
            self.state = np.float32(self.env.reset())
        
        return current_state, action, reward, episode_reward


def learn(env):
    writer = SummaryWriter(comment='-reinforce-'+ENV)
    net = PolicyGradientNet(env.observation_space.shape[0], env.action_space.n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    agent = PolicyGradientAgent(net, env)

    batch_states = []
    batch_actions = []
    rewards = []  # rewards are collected during episode and then transformed to qvalues
    batch_qvalues = []

    episode_count = 0
    episode_rewards = []
    max_mean_episode_rewards = 0.

    step_idx = 0

    while True:
        step_idx += 1

        state, action, reward, episode_reward = agent.play_step()

        batch_states.append(state)
        batch_actions.append(action)
        rewards.append(reward)

        if episode_reward is not None:
            # episode finished
            episode_count += 1

            batch_qvalues.extend(calc_qvalues(rewards, gamma=GAMMA))
            rewards.clear()

            writer.add_scalar('episode_reward', episode_reward, episode_count)

            episode_rewards.append(episode_reward)
            mean_episode_rewards = float(np.mean(episode_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, episode_reward, mean_episode_rewards, episode_count))

            if episode_count % BATCH_EPISODES == 0:
                # learn
                states_t = torch.FloatTensor(batch_states).to(device)
                actions_t = torch.LongTensor(batch_actions).to(device)
                qvalues_t = torch.FloatTensor(batch_qvalues).to(device)

                optimizer.zero_grad()
                logits_t = net(states_t)
                log_prob_t = F.log_softmax(logits_t, dim=1)
                scaled_log_prob_t = qvalues_t * log_prob_t[range(len(batch_states)), actions_t]
                loss_t = -scaled_log_prob_t.mean()

                loss_t.backward()
                optimizer.step()

                batch_states.clear()
                batch_actions.clear()
                batch_qvalues.clear()

                writer.add_scalar('loss', loss_t.data.cpu().numpy(), episode_count)

            if mean_episode_rewards > 195. and mean_episode_rewards > max_mean_episode_rewards:
                torch.save(net.state_dict(), f"{ENV}-reinforce.dat")
                max_mean_episode_rewards = mean_episode_rewards
    writer.close()


if __name__ == '__main__':
    import gym
    env = gym.make(ENV)
    env.seed(0)
    np.random.seed(0)
    learn(env)
