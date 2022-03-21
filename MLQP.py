import numpy as np


class MLQP:
    """Multi-Layer Quadratic Perceptron"""

    def __init__(self, layers: list, lrl, lrq, decay, decay_period):
        self._depth = len(layers) - 1
        """Depth of network"""
        self._wl = [np.random.rand(layers[i] + 1, layers[i + 1]) for i in range(self._depth)]
        """Weights of linear elements"""
        self._wq = [np.random.rand(layers[i] + 1, layers[i + 1]) for i in range(self._depth)]
        """Weights of quadratic elements"""
        self._h = [np.ones([layers[i] + 1, 1]) for i in range(len(layers))]
        """Value of hidden layers"""
        self._lrl = lrl
        self._lrq = lrq
        self._step = 0
        self._decay = decay
        self._decay_period = decay_period

    @staticmethod
    def activation(x: np.ndarray):
        """Activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def de_activation(x: np.ndarray):
        """The derivative function of activation function"""
        return x * (1 - x)

    def forward(self, x):
        self._h[0] = np.array([x + [1]]).T
        for i in range(self._depth):
            self._h[i + 1][:-1] = self.activation(self._h[i].T @ self._wl[i] + np.square(self._h[i]).T @ self._wq[i]).T
        return self._h[-1]

    def update(self, error):
        if (self._step + 1) % self._decay_period == 0:
            self._lrl *= self._decay
            self._lrq *= self._decay
        grad = np.array(+ self.de_activation(self._h[-1]))
        delta_w, delta_v = [], []
        i = -2
        # Calculate gradients
        while self._depth + i > -2:
            delta_w.append((self._h[i] @ grad.T)[:, :-1])
            delta_v.append((np.square(self._h[i]) @ grad.T)[:, :-1])
            grad = (self._wl[i + 1] + self._wq[i + 1] * 2 * self._h[i]) @ grad[:-1]
            grad = grad * self.de_activation(self._h[i])
            i -= 1
        # Update weights
        delta_w.reverse()
        delta_v.reverse()
        for i in range(self._depth):
            self._wl[i] += self._lrl * error * delta_w[i]
            self._wq[i] += self._lrq * error * delta_v[i]
        self._step += 1