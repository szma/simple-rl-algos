#!/usr/bin/env python


import numpy as np
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    import gym
    gym.make("InvertedPendulum-v2")