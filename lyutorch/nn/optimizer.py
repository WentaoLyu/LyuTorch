class Optimizer:
    def __init__(self, parameters: dict, lr=1e-3, decay=1e-4) -> None:
        self._parameters = parameters
        self._lr = lr
        self._iter = 0
        self._decay = decay

    def step(self):
        for key, value in self._parameters.items():
            value -= value.grad * self._lr
        self._iter += 1
        self._lr *= 1 - self._decay / (1 + self._decay * self._iter)

    def zero_grad(self):
        for key, value in self._parameters.items():
            value.grad = 0
