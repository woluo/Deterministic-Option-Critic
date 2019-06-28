#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
An implementation of random replay buffer
"""

import numpy as np


class Buffer:

    def __init__(self, state_dim, action_dim, capacity):

        self.sd = state_dim
        self.ad = action_dim
        self.capacity = capacity
        self.buffer = np.zeros(shape=(capacity, state_dim * 2 + action_dim + 1))
        self.isFilled = False
        self.pointer = 0

    def store(self, state, action, reward, next_state):
        """Store the transition"""

        # flatten the transition tuple
        transition = np.hstack((state, action, reward, next_state))
        self.buffer[self.pointer, :] = transition
        self.pointer += 1

        # prevent overflow
        if self.pointer == self.capacity:
            self.isFilled = True
            self.pointer = 0

    def sample(self, num):
        """Sample batch"""

        # sample batch memory from all memory
        index = np.random.choice(self.capacity, size=num)
        batch = self.buffer[index, :]

        # separate the batch memory to different dimensions
        sa = self.sd + self.ad
        return batch[:, :self.sd], batch[:, self.sd: sa], batch[:, sa: sa + 1], batch[:, sa + 1:]
