import torch
from models import Actor
from models import Critic


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwinDelayedDDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic_twin_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_twin_1_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_twin_1_target.load_state_dict(self.critic_twin_1.state_dict())
        self.critic_twin_1_optimizer = torch.optim.Adam(self.critic_twin_1.parameters())

        self.critic_twin_2 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_twin_2_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_twin_2_target.load_state_dict(self.critic_twin_2.state_dict())
        self.critic_twin_2_optimizer = torch.optim.Adam(self.critic_twin_2.parameters())

        self.max_action = max_action
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5

    def act(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action
