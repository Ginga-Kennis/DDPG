import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.network import ActorNetwork, CriticNetwork
from src.replay_buffer import ReplayBuffer
from src.noise import OUNoise, GaussianNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, buffer_size=int(1e6), batch_size=64, gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, weight_decay=1e-2, sigma=0.1):
        """
        state_size(int) : dimension of each state
        action_size(int) : dimension of each action
        buffer_size(int) : size of replpay buffer
        batch_size(int) : batch size
        gamma(float) : discount factor
        tau(float) : parameter for soft update
        lr_actor(float) : learning rate for actor network
        lr_critic(float) : learning rate for critic network
        weight_decay(float) : L2 weight decay
        sigma(float) : standard deviation of noise
        """
        self.state_size = state_size
        self.action_size = action_size

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma 
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay

        self.sigma = sigma

        # Actor Network
        self.actor = ActorNetwork(state_size, action_size).to(device)
        self.actor_target = ActorNetwork(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Critic Network
        self.critic = CriticNetwork(state_size, action_size).to(device)
        self.critic_target = CriticNetwork(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Noise
        # self.noise = OUNoise(action_size)
        self.noise = GaussianNoise(action_size, self.sigma)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, device)

    
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
    


    def step(self, state, action, reward, next_state, done):
        """save experience in replay memory, and use random sample from buffer to learn"""
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    

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
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

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

        
            
    