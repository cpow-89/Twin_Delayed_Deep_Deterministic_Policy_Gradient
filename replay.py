import numpy as np
import random
from collections import deque


class ReplayBuffer(object):

    """Fixed-size experience buffer"""
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.device = device

    def add(self, transition):
        """Add a new experience to memory."""
        self.memory.append(transition)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences_samples = random.sample(self.memory, k=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for sample in experiences_samples:
            state, next_state, action, reward, done = sample
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return [np.array(batch_states), np.array(batch_next_states), np.array(batch_actions),
                np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)]
