from .gradient import *
from .tensor import Tensor

__all__ = ["__add__"]


def __add__(self, other):
    if isinstance(other, Tensor):
        requires_grad = self.requires_grad | other.requires_grad
        result = Tensor(super(Tensor, self).__add__(other), requires_grad=requires_grad)
        result.backward_fn = lambda pass_in_grad, pass_in: add_grad(
            result, self, other, pass_in_grad, pass_in
        )
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
        result.backward_fn = lambda pass_in_grad, pass_in: matmul_grad(
            result, self, other, pass_in_grad, pass_in
        )
        return result
    else:
        raise TypeError(
            f"unsupported operand type(s) for @: 'Tensor' and '{type(other)}'"
        )
