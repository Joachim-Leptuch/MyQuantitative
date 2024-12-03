import numpy as np
import math


class GBM:

    def __init__(self, s, r, vol, T, N):
        self.s = s
        self.r = r
        self.v = vol
        self.T = T
        self.N = N
        self.dt = T / N
        self.path = None

    def generate_path(self):
        """
        Generates a path using geometric brownian motion.
        """
        steps = int(self.T * self.N)
        random_noise = np.random.normal(0, np.sqrt(self.dt), size=steps)
        log_returns = (self.r - 0.5 * self.v ** 2) * self.dt + self.v * random_noise  # Discrete GBM equation
        stock_prices = self.s * np.exp(np.cumsum(log_returns))

        return stock_prices

    def get_terminal_value(self):
        """
        Calculates the terminal stock value using the Geometric Brownian Motion (GBM) model.
        """
        z = np.random.normal(0, 1)  # Generate standard normal random variable
        st = self.s * math.exp((self.r - 0.5 * self.v ** 2) * self.T + self.v * math.sqrt(self.T) * z)

        return st
