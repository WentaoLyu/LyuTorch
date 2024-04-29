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

    def __call__(self, *input):
        return self.forward(*input)
