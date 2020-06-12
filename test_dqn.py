#!/usr/bin/env python
import unittest
import torch

from dqn import DQNNet, Memory


class TestDQN(unittest.TestCase):
    """Simple tests for the DQN net"""

    def test_net(self):
        """Simply tries to initialize the net and passes some observation through it."""
        net = DQNNet(2, 8)
        t_res = net(torch.tensor([1., 2.]))
        self.assertEqual(len(t_res), 8)


class TestMemory(unittest.TestCase):
    """Simple tests for the replay buffer called Memory"""

    def test_memory(self):
        mem = Memory(3)
        exp = Memory.Experience(
            state=0., action=1, reward=5., done=False, next_state=1.)

        mem.append(exp)
        self.assertFalse(mem.is_fully_populated())
        mem.append(exp)
        mem.append(exp)
        self.assertTrue(mem.is_fully_populated())

        mem.append(exp)
        self.assertTrue(mem.is_fully_populated())
        print(mem.memory.popleft())

        states, actions, rewards, dones, next_states = mem.sample(2)
        self.assertEqual(len(states), 2)


if __name__ == '__main__':
    unittest.main()
