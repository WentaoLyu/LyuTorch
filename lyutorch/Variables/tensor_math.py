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
    if isinstance(other, Tensor):
        requires_grad = self.requires_grad | other.requires_grad
        result = Tensor(super(Tensor, self).__mul__(other), requires_grad=requires_grad)
        attach_backward_fn(result, requires_grad, mul_grad, self, other)
        return result
    else:
        raise TypeError(
            f"unsupported operand type(s) for *: 'Tensor' and '{type(other)}'"
        )


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


def t(self):
    if isinstance(self, Tensor):
        if self.ndim == 2:
            requires_grad = self.requires_grad
            result = Tensor(super(Tensor, self).T, requires_grad=requires_grad)
            attach_backward_fn(result, requires_grad, t_grad, self)
            return result
        else:
            raise ValueError("t() only supports 2D tensor")
    else:
        raise TypeError(f"unsupported operand type(s) for t(): '{type(self)}'")
