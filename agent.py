import torch
import torch.nn.functional as func
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

    def _add_noise_to_action(self, next_actions):
        noises = torch.randn_like(next_actions).normal_(0, self.policy_noise).to(DEVICE)
        noises = noises.clamp(-self.noise_clip, self.noise_clip)
        return (next_actions + noises).clamp(-self.max_action, self.max_action)

    def _update_weights(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _update_target_network_based_on_polyak_averaging(self, target, source):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self, replay_buffer, iterations, batch_size=100, discount=0.99, policy_update_delay=2):
        for it in range(iterations):
            transitions = replay_buffer.sample(batch_size)
            states, next_states, actions, rewards, dones = [torch.Tensor(elem).to(DEVICE) for elem in transitions]
            next_actions = self.actor_target(next_states)
            next_actions = self._add_noise_to_action(next_actions)

            target_q1 = self.critic_twin_1_target(next_states, next_actions)
            target_q2 = self.critic_twin_2_target(next_states, next_actions)
            min_target_q = torch.min(target_q1, target_q2)
            target_q = rewards + ((1 - dones) * discount * min_target_q).detach()

            curr_q1 = self.critic_twin_1(states, actions)
            curr_q2 = self.critic_twin_2(states, actions)

            critic_twin_1_optimizer_loss = func.mse_loss(curr_q1, target_q)
            self._update_weights(self.critic_twin_1_optimizer, critic_twin_1_optimizer_loss)
            critic_twin_2_optimizer_loss = func.mse_loss(curr_q2, target_q)
            self._update_weights(self.critic_twin_2_optimizer, critic_twin_2_optimizer_loss)

            if it % policy_update_delay == 0:
                actor_loss = -self.critic_twin_1(states, self.actor(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self._update_target_network_based_on_polyak_averaging(self.actor_target, self.actor)
                self._update_target_network_based_on_polyak_averaging(self.critic_twin_1_target, self.critic_twin_1)
                self._update_target_network_based_on_polyak_averaging(self.critic_twin_2_target, self.critic_twin_2)