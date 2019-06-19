import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.network = self._create_network(state_dim, action_dim)

    @staticmethod
    def _create_network(state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh())

    def forward(self, state):
        return self.max_action * self.network(state)
