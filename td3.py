#!/usr/bin/env python

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from collections import namedtuple, deque

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
#device = 'cpu'

GAMMA = 0.99
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = int(1e6)
INIT_REPLAY_BUFFER_SIZE = int(1e5)
DELAYED_UPDATE_STEPS = 2


class DeterminisicPolicyNet(nn.Module):
    def __init__(self, observation_dims, action_dims):

        super(DeterminisicPolicyNet, self).__init__()

        self.mu_net = nn.Sequential(
            nn.Linear(observation_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dims),
            nn.Tanh()
        )

    def forward(self, x):
        return self.mu_net(x)


class QNet(nn.Module):
    def __init__(self, observation_dims, action_dims):

        super(QNet, self).__init__()

        self.q_net_obs_only = nn.Sequential(
            nn.Linear(observation_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.q_net = nn.Sequential(
            nn.Linear(512 + action_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, s, a):
        obs_only_layer = self.q_net_obs_only(s)
        return self.q_net(torch.cat((obs_only_layer, a), 1))


Experience = namedtuple('Experience',
                        field_names=['state',
                                     'action',
                                     'reward',
                                     'done',
                                     'next_state'])


class Memory():

    def __init__(self, max_size):
        self.memory = deque([], max_size)

    def append(self, exp: Experience):
        self.memory.append(exp)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.memory[idx]
                                                             for idx in idxs])

        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool),
                np.array(next_states, dtype=np.float32))

    def is_fully_populated(self):
        return self.memory.maxlen == len(self.memory)


class Agent():

    def __init__(self, net, env):
        self.net = net
        self.env = env

        self.state = np.float32(self.env.reset())
        self.tot_reward = 0

        self.noise = 0.03

    def play_step(self, random=False):
        episode_reward = None

        if not random:
            state_v = torch.tensor(self.state).to(device)
            action = self.net(state_v).cpu().detach().numpy() + self.noise * np.random.randn()
            action = np.clip(action, -1, 1)
        else:
            action = self.env.action_space.sample()

        next_state, reward, done, _ = self.env.step(action)
        next_state = np.float32(next_state)

        # Hack
        self.tot_reward += reward
        reward -= np.absolute(next_state[0])*10

        exp = Experience(
            state=self.state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state)

        if not done:
            self.state = next_state
        else:
            episode_reward = self.tot_reward
            self.tot_reward = 0
            self.state = np.float32(self.env.reset())

        return exp, episode_reward


def polyak_sync(target_net, net, polyak_param=.995):

    for target_p, net_p in zip(target_net.parameters(), net.parameters()):
        target_p.data.copy_(polyak_param * target_p.data + (1-polyak_param)*net_p)


def learn(env):

    mu_net = DeterminisicPolicyNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    mu_target_net = DeterminisicPolicyNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    mu_target_net.load_state_dict(mu_net.state_dict())

    q_net = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    q_target_net = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    q_target_net.load_state_dict(q_net.state_dict())

    mu_optim = torch.optim.Adam(mu_net.parameters(), lr=LEARNING_RATE)
    q_optim = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

    memory = Memory(REPLAY_BUFFER_SIZE)
    agent = Agent(mu_net, env)

    writer = SummaryWriter(comment='-tp3-pendulum')

    # Fill memory with random data for warmup
    for _ in range(INIT_REPLAY_BUFFER_SIZE):
        exp, episode_reward = agent.play_step(random=True)
        memory.append(exp)

    print(f'Memory filled with {len(memory.memory)} experiences.')

    episode_rewards = []
    best_mean_history = 0.0
    step_idx = 0
    episode_idx = 0
    while True:
        step_idx += 1

        exp, episode_reward = agent.play_step()
        memory.append(exp)

        if episode_reward is not None:
            # Episode done
            episode_idx += 1
            episode_rewards.append(episode_reward)
            mean_history = np.array(episode_rewards)[-20:].mean()
            if mean_history > best_mean_history:
                print('solved. new best')
                best_mean_history = mean_history
                torch.save(mu_net.state_dict(), "tp3-lunar-best.dat")

            print(f"{mean_history} ({episode_reward}), with noise {agent.noise}")
            writer.add_scalar("reward", episode_reward, step_idx)
        agent.noise *= 0.999999

        batch_exp = memory.sample(BATCH_SIZE)
        states, actions, rewards, dones, next_states = batch_exp

        states_t = torch.tensor(states).to(device)
        actions_t = torch.tensor(actions).to(device)
        rewards_t = torch.tensor(rewards).to(device)
        dones_t = torch.BoolTensor(dones).to(device)
        next_states_t = torch.tensor(next_states).to(device)

        q_optim.zero_grad()
        q_vals = q_net(states_t, actions_t)

        q_target = GAMMA * q_target_net(next_states_t, mu_target_net(next_states_t))
        q_target[dones_t] = 0.0
        q_target += rewards_t.view((BATCH_SIZE, -1))
        q_target = q_target.detach()

        loss_q = F.mse_loss(q_vals, q_target)
        loss_q.backward()
        q_optim.step()

        if step_idx % DELAYED_UPDATE_STEPS == 0:

            mu_optim.zero_grad()
            loss_mu = -q_net(states_t, mu_net(states_t)).mean()
            loss_mu.backward()
            mu_optim.step()

            polyak_sync(mu_target_net, mu_net)
            polyak_sync(q_target_net, q_net)


def best_play(env):
    import time
    mu_net = DeterminisicPolicyNet(env.observation_space.shape[0], env.action_space.shape[0])
    mu_net.load_state_dict(torch.load("tp3-lunar-best.dat"))

    state = np.float32(env.reset())

    episode_reward = 0.

    while True:
        #time.sleep(0.02)
        state_v = torch.tensor(state)
        action = mu_net(state_v).cpu().detach().numpy()
        action = np.clip(action, -1, 1)

        next_state, reward, done, _ = env.step(action)
        env.render()
        state = np.float32(next_state)
        episode_reward += reward
        if done:
            state = np.float32(env.reset())
            print(episode_reward)
            episode_reward = 0.


def random_play(env):
    import time

    env.reset()

    while True:
        time.sleep(0.02)
        s, r, d, _ = env.step(env.action_space.sample())
        if d:
            env.reset()


if __name__ == "__main__":
    import gym
    #import pybulletgym
    # RoboSchool Envs
    # env = gym.make("InvertedPendulumPyBulletEnv-v0")
    # env=gym.make("InvertedDoublePendulumPyBulletEnv-v0")
    #env=gym.make("InvertedPendulumSwingupPyBulletEnv-v0")
    # env=gym.make("ReacherPyBulletEnv-v0")
    # env=gym.make("Walker2DPyBulletEnv-v0")
    # env=gym.make("HalfCheetahPyBulletEnv-v0")
    # env=gym.make("AntPyBulletEnv-v0")
    # env=gym.make("HopperPyBulletEnv-v0")
    # env=gym.make("HumanoidPyBulletEnv-v0")
    # env=gym.make("HumanoidFlagrunPyBulletEnv-v0")
    # env=gym.make("HumanoidFlagrunHarderPyBulletEnv-v0")
    # env=gym.make("AtlasPyBulletEnv-v0")
    # env=gym.make("PusherPyBulletEnv-v0")
    # env=gym.make("ThrowerPyBulletEnv-v0")
    # env=gym.make("StrikerPyBulletEnv-v0")
    # MuJoCo Envs
    # env=gym.make("InvertedPendulumMuJoCoEnv-v0")
    # env = gym.make("InvertedDoublePendulumMuJoCoEnv-v0")
    # env=gym.make("ReacherMuJoCoEnv-v0"),not implemented
    # env=gym.make("Walker2DMuJoCoEnv-v0")
    # env=gym.make("HalfCheetahMuJoCoEnv-v0")
    # env=gym.make("AntMuJoCoEnv-v0")
    # env=gym.make("HopperMuJoCoEnv-v0")
    # env=gym.make("HumanoidMuJoCoEnv-v0")
    # env=gym.make("PusherMuJoCoEnv-v0"),not implemented
    # env=gym.make("ThrowerMuJoCoEnv-v0"),not implemented
    # env=gym.make("StrikerMuJoCoEnv-v0"),not implemented

    # In those envs rendering works different: call just once, before reset:
    # env.render()

    env = gym.make("LunarLanderContinuous-v2")

    print('Obs:', env.observation_space)
    print('Actions: ', env.action_space)

    #random_play(env)
    #learn(env)
    best_play(env)
