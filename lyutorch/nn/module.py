import numpy as np


class Module:
    def __init__(self):
        self._parameters = {}

    def parameters(self):
        return self._parameters

    def add_parameter(self, name, parameter):
        self._parameters[name] = parameter

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def save(self, filename):
        np.savez(filename, **self._parameters)

    def load(self, filename):
        data = np.load(filename)
        for key in data.keys():
            self._parameters[key] = data[key]

    def __call__(self, *input):
        return self.forward(*input)
