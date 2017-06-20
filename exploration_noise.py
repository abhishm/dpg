# https://gist.github.com/Anjum48/886a825aeb877909856407aa7e4c82f9

import numpy as np

class OUNoise:
    """An Ornstein - Uhlenbeck process is used to generate temporarially correlated noise"""
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.3, seed=123):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
        np.random.seed(seed)

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
