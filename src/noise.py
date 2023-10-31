import random
import copy
import numpy as np

class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    """

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)  # global mean
        self.theta = theta            #strength to pull towards the mean
        self.sigma = sigma            # global mean
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state
    
class GaussianNoise:
    """
    Gaussian noise
    """
    def __init__(self,action_size,scale):
        self.action_size = action_size
        self.scale = scale

    def reset(self):
        pass
    
    def sample(self):
        return np.random.randn(self.action_size) * self.scale