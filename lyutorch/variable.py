from .Variables import *


def tensor(input_data, requires_grad=False, dtype=None, order=None, *args, **kwargs):
    return Tensor(input_data, requires_grad, dtype, order, *args, **kwargs)
