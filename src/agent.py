import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.network import ActorNetwork, CriticNetwork
from src.replay_buffer import ReplayBuffer
from src.noise import OUNoise, GaussianNoise

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 1e-2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size):
        """
        state_size(int) : dimension of each state
        action_size(int) : dimension of each action
        random_seed(int) : random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        # Actor Network
        self.actor = ActorNetwork(state_size, action_size).to(device)
        self.actor_target = ActorNetwork(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic = CriticNetwork(state_size, action_size).to(device)
        self.critic_target = CriticNetwork(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise
        # self.noise = OUNoise(action_size)
        self.noise = GaussianNoise(action_size,scale=0.25)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device)

    def step(self, state, action, reward, next_state, done):
        """save experience in replay memory, and use random sample from buffer to learn"""
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        # set to evaluation mode
        self.actor.eval()   

        # 勾配計算を行わない
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()

        # reset to training mode
        self.actor.train()
        
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1,1)  # -1 ~ action ~ 1 の範囲に収める
    
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        #-----------------update critic------------------#
        # compute TD target
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))  # TD targets

        # compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)  # element-wise mean squared loss

        # backpropagation
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #-----------------update actor------------------#
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()

        # backpropagation
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #-----------------update target network------------------#
        self.soft_update(self.critic, self.critic_target, TAU)
        self.soft_update(self.actor, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

    def save_model(self,model_name):
        torch.save(self.actor.state_dict(),f"./data/{model_name}_actor.pth")
        torch.save(self.critic.state_dict(),f"./data/{model_name}_critic.pth")

        
            
    