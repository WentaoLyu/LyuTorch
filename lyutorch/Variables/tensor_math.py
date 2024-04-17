from .gradient import *
from .tensor import Tensor

__all__ = ["__add__"]

from .utils import attach_backward_fn


def __add__(self, other):
    if isinstance(other, Tensor):
        requires_grad = self.requires_grad | other.requires_grad
        result = Tensor(super(Tensor, self).__add__(other), requires_grad=requires_grad)
        attach_backward_fn(result, requires_grad, add_grad, self, other)
        return result
    else:
        raise TypeError(
            f"unsupported operand type(s) for +: 'Tensor' and '{type(other)}'"
        )


def __mul__(self, other):
    pass


def __matmul__(self, other):
    if isinstance(other, Tensor):
        requires_grad = self.requires_grad | other.requires_grad
        result = Tensor(
            super(Tensor, self).__matmul__(other), requires_grad=requires_grad
        )
        attach_backward_fn(result, requires_grad, matmul_grad, self, other)
        return result
    else:
        raise TypeError(
            f"unsupported operand type(s) for @: 'Tensor' and '{type(other)}'"
        )
