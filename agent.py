import torch
import torch.nn.functional as func
import utils
import os
from models import Actor
from models import Critic
from replay import ReplayBuffer


class TwinDelayedDDPG:
    def __init__(self, config, state_dim, action_dim, max_action):

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = max_action

        self.memory = ReplayBuffer(config["buffer_size"], self.device)

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic_twin_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_twin_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_twin_1_target.load_state_dict(self.critic_twin_1.state_dict())
        self.critic_twin_1_optimizer = torch.optim.Adam(self.critic_twin_1.parameters())

        self.critic_twin_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_twin_2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_twin_2_target.load_state_dict(self.critic_twin_2.state_dict())
        self.critic_twin_2_optimizer = torch.optim.Adam(self.critic_twin_2.parameters())

    def act(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def _add_noise_to_action(self, next_actions):
        noises = torch.randn_like(next_actions).normal_(0, self.config["policy_noise"]).to(self.device)
        noises = noises.clamp(-self.config["noise_clip"], self.config["noise_clip"])
        return (next_actions + noises).clamp(-self.max_action, self.max_action)

    def _update_weights(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _update_target_network_based_on_polyak_averaging(self, target, source):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.config["tau"] * param.data + (1 - self.config["tau"]) * target_param.data)

    def learn(self, iterations):
        for it in range(iterations):
            transitions = self.memory.sample(self.config["batch_size"])
            states, next_states, actions, rewards, dones = [torch.Tensor(elem).to(self.device) for elem in transitions]
            next_actions = self.actor_target(next_states)
            next_actions = self._add_noise_to_action(next_actions)

            target_q1 = self.critic_twin_1_target(next_states, next_actions)
            target_q2 = self.critic_twin_2_target(next_states, next_actions)
            min_target_q = torch.min(target_q1, target_q2)
            target_q = rewards + ((1 - dones) * self.config["discount_factor"] * min_target_q).detach()

            curr_q1 = self.critic_twin_1(states, actions)
            curr_q2 = self.critic_twin_2(states, actions)

            critic_twin_1_optimizer_loss = func.mse_loss(curr_q1, target_q)
            self._update_weights(self.critic_twin_1_optimizer, critic_twin_1_optimizer_loss)
            critic_twin_2_optimizer_loss = func.mse_loss(curr_q2, target_q)
            self._update_weights(self.critic_twin_2_optimizer, critic_twin_2_optimizer_loss)

            if it % self.config["policy_update_delay"] == 0:
                actor_loss = -self.critic_twin_1(states, self.actor(states)).mean()
                self._update_weights(self.actor_optimizer, actor_loss)
                self._update_target_network_based_on_polyak_averaging(self.actor_target, self.actor)
                self._update_target_network_based_on_polyak_averaging(self.critic_twin_1_target, self.critic_twin_1)
                self._update_target_network_based_on_polyak_averaging(self.critic_twin_2_target, self.critic_twin_2)

    def add_transition_to_memory(self, transition):
        self.memory.add(transition)

    def save(self):
        checkpoint_dir = os.path.join(".", *self.config["checkpoint_dir"], self.config["env_name"])
        utils.save_state_dict(os.path.join(checkpoint_dir, "actor"), self.actor.state_dict())
        utils.save_state_dict(os.path.join(checkpoint_dir, "actor_target"), self.actor_target.state_dict())
        utils.save_state_dict(os.path.join(checkpoint_dir, "critic_twin_1"), self.critic_twin_1.state_dict())
        utils.save_state_dict(os.path.join(checkpoint_dir, "critic_twin_1_target"), self.critic_twin_1.state_dict())
        utils.save_state_dict(os.path.join(checkpoint_dir, "critic_twin_2"), self.critic_twin_1.state_dict())
        utils.save_state_dict(os.path.join(checkpoint_dir, "critic_twin_2_target"), self.critic_twin_1.state_dict())

    def load(self):
        checkpoint_dir = os.path.join(".", *self.config["checkpoint_dir"], self.config["env_name"])
        path = os.path.join(checkpoint_dir, "actor", "*")
        self.actor.load_state_dict(utils.load_latest_available_state_dict(path))
        path = os.path.join(checkpoint_dir, "actor_target", "*")
        self.actor_target.load_state_dict(utils.load_latest_available_state_dict(path))
        path = os.path.join(checkpoint_dir, "critic_twin_1", "*")
        self.critic_twin_1.load_state_dict(utils.load_latest_available_state_dict(path))
        path = os.path.join(checkpoint_dir, "critic_twin_1_target", "*")
        self.critic_twin_1_target.load_state_dict(utils.load_latest_available_state_dict(path))
        path = os.path.join(checkpoint_dir, "critic_twin_2", "*")
        self.critic_twin_2.load_state_dict(utils.load_latest_available_state_dict(path))
        path = os.path.join(checkpoint_dir, "critic_twin_2_target", "*")
        self.critic_twin_2_target.load_state_dict(utils.load_latest_available_state_dict(path))
