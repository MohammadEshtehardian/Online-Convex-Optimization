import numpy as np

class Gaussian:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def loss(self):
        x = self.std * np.random.randn(1) + self.mean
        return np.clip(x, 0, 1)

class Uniform:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def loss(self):
        x = np.random.uniform(self.a, self.b)
        return np.clip(x, 0, 1)

class Rayleigh:
    def __init__(self, sigma):
        self.sigma = sigma
    def loss(self):
        x = np.random.rayleigh(self.sigma)
        return np.clip(x, 0, 1)

class Exponential:
    def __init__(self, mean):
        self.mean = mean
    def loss(self):
        x = np.random.exponential(self.mean)
        return np.clip(x, 0, 1)
