import numpy as np

from .Variables import Tensor


def exp(tensor: Tensor):
    ret = Tensor(np.exp(tensor), requires_grad=tensor.requires_grad)
