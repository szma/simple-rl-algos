#!/usr/bin/env python

import numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from collections import namedtuple, deque

ENV = 'LunarLander-v2'
#ENV = 'CartPole-v0'
SYNC_NETS = 500
WARMUP = 50000
REPLAY_BUFFER_SIZE = 1000000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_DECAY = .998  # 1/500.
LEARNING_RATE = 1e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DQNNet(nn.Module):
    def __init__(self, observation_dims, action_dims):
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dims))

    def forward(self, x):
        return self.net(x)


class Memory():
    Experience = namedtuple('Experience',
                            field_names=['state',
                                         'action',
                                         'reward',
                                         'done',
                                         'next_state'])

    def __init__(self, max_size):
        self.memory = deque([], max_size)

    def append(self, exp: Experience):
        self.memory.append(exp)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.memory[idx]
                                                             for idx in idxs])

        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.long),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool),
                np.array(next_states, dtype=np.float32))

    def is_fully_populated(self):
        return self.memory.maxlen == len(self.memory)


class Agent():

    def __init__(self, net, env, memory, epsilon=1.0):
        self.net = net
        self.env = env
        self.memory = memory

        self.state = np.float32(self.env.reset())
        self.tot_reward = 0

        self.epsilon = epsilon

    def play_step(self, render=False):
        episode_reward = None

        if np.random.rand() > self.epsilon or render:
            state_t = torch.tensor(self.state).to(device)
            q_action_values_t = self.net(state_t).detach()
            action = int(torch.argmax(q_action_values_t).data)
            #state_a = np.array([self.state], copy=False)
            #state_v = torch.tensor(state_a).to(device)
            #q_vals_v = net(state_v)
            #_, act_v = torch.max(q_vals_v, dim=1)
            #action = int(act_v.item())
        else:
            action = self.env.action_space.sample()

        next_state, reward, done, _ = self.env.step(action)
        next_state = np.float32(next_state)

        # Hack
        self.tot_reward += reward
        reward -= np.absolute(next_state[0])*10

        if render:
            env.render()

        self.memory.append(Memory.Experience(
            state=self.state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state))

        if not done:
            self.state = next_state
        else:
            episode_reward = self.tot_reward
            self.tot_reward = 0
            self.state = np.float32(self.env.reset())

        return episode_reward


def calc_loss(batch, net, target_net):

    # Q(s, a) = r + gamma * Q'(s', a')

    states, actions, rewards, dones, next_states = batch

    states_t = torch.tensor(states).to(device)
    actions_t = torch.tensor(actions).to(device)
    rewards_t = torch.tensor(rewards).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    next_states_t = torch.tensor(next_states).to(device)

    # Get calculated Q(s, a) from net. gather gathers the indices (taken actions) along the matrix
    # [[1,2,3],
    #  [4,5,6],
    #  [7,8,9]].gather(dim=1, [2,1,0]) => [3,5,7]
    q_vals_net = net(states_t).gather(dim=1, index=actions_t.unsqueeze(-1)).squeeze(-1)
    # Basic DQN (take actions and q value from target_net):
    # q_vals_next_target_net = target_net(next_states_t).max(dim=1)[0]  # .values
    # Double DQN (take actions from net and q value in target net)
    net_next_state_actions_t = net(next_states_t).max(dim=1)[1]  # .indices
    q_vals_next_target_net = target_net(next_states_t).gather(dim=1, index=net_next_state_actions_t.unsqueeze(-1)).squeeze(-1)

    # max returns indices and values
    q_vals_next_target_net[dones_t] = 0.0
    q_vals_next_target_net.detach()

    q_vals_expected = rewards_t + GAMMA * q_vals_next_target_net

    return nn.MSELoss()(q_vals_net, q_vals_expected)


def learn(env):
    # target_net.load_state_dict(net.state_dict())  # Copy weights
    net = DQNNet(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net = DQNNet(env.observation_space.shape[0], env.action_space.n).to(device)

    memory = Memory(REPLAY_BUFFER_SIZE)
    agent = Agent(net, env, memory)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    summary_writer = SummaryWriter(comment="-"+ENV)

    for i in range(WARMUP):
        agent.play_step()
    print(
        f"""Replay memory initialized. Fully populated: {agent.memory.is_fully_populated()} \
            with size {len(agent.memory.memory)} of {agent.memory.memory.maxlen}""")

    frame = 0
    episode = 1
    episode_rewards = []
    agent.epsilon = 1.0
    max_solve_ratio = 0.0
    while True:
        frame += 1

        #agent.epsilon = np.maximum(1.0 - episode * EPS_DECAY, 0.01)
        episode_reward = agent.play_step()

        if episode_reward is not None:
            agent.epsilon *= EPS_DECAY
            summary_writer.add_scalar('episode_reward', episode_reward, episode)
            summary_writer.add_scalar('epsilon', agent.epsilon, episode)
            episode_rewards.append(episode_reward)
            solve_ratio = np.mean(episode_rewards[-100:])
            print(f'Episode {episode}: {episode_reward}, Solve ratio: {solve_ratio}')

            solved = solve_ratio >= 200.

            print()
            if solved:
                if solve_ratio > max_solve_ratio:
                    max_solve_ratio = solve_ratio
                    torch.save(net.state_dict(), f"{ENV}-best.dat")
                    print("Solved better.")
            episode += 1

        optimizer.zero_grad()
        loss = calc_loss(memory.sample(BATCH_SIZE), net, target_net)
        loss.backward()
        optimizer.step()

        if frame % SYNC_NETS == 0:
            target_net.load_state_dict(net.state_dict())

        #if frame > 5e6:
            #break


def playback(env):
    net = DQNNet(env.observation_space.shape[0], env.action_space.n).to(device)
    net.load_state_dict(torch.load(f"{ENV}-best.dat"))

    memory = Memory(1)
    agent = Agent(net, env, memory)
    agent.epsilon = 0.0

    while(True):
        episode_reward = agent.play_step(render=True)
        if episode_reward is not None:
            print(f'Episode finished: {episode_reward}')


if __name__ == '__main__':
    import gym
    env = gym.make(ENV)

    #learn(env)
    playback(env)
