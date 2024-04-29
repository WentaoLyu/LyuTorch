import numpy as np

from .variable import Tensor
from .Variables.utils import make_grad, attach_backward_fn


def exp(tensor: Tensor):
    def exp_grad(self: Tensor, prev_self: Tensor, pass_in_grad, pass_in):
        if prev_self.requires_grad:
            grad = np.eye(self.size).reshape(self.shape * 2) * np.asarray(self)
            make_grad(
                self, pass_in_grad, pass_in, grad, 0, prev_self, broadcast_check=False
            )

    ret = Tensor(np.exp(tensor), requires_grad=tensor.requires_grad)
    attach_backward_fn(ret, tensor.requires_grad, exp_grad, tensor)
    return ret


def sum(tensor, axes=None, keepdims=False):
    def sum_grad(self, prev_self, pass_in_grad, pass_in, axis):
        if prev_self.requires_grad:
            if axis is None:
                grad = np.ones_like(prev_self)
            else:
                if self.ndim == 0:
                    grad = np.ones_like(prev_self)
                else:
                    shapes = np.array(prev_self.shape)
                    if isinstance(axis, tuple):
                        shapes[list(axis)] = 1
                    else:
                        shapes[axis] = 1
                    grad = np.eye(self.size).reshape(
                        self.shape + tuple(shapes)
                    ) * np.ones_like(prev_self)
            make_grad(
                self, pass_in_grad, pass_in, grad, 0, prev_self, broadcast_check=False
            )

    ret = np.sum(tensor, axis=axes, keepdims=keepdims)
    ret = Tensor(ret, requires_grad=tensor.requires_grad)
    attach_backward_fn(ret, tensor.requires_grad, sum_grad, tensor, axis=axes)
    return ret


def log(tensor: Tensor):
    def log_grad(self, prev_self, pass_in_grad, pass_in):
        if prev_self.requires_grad:
            grad = np.eye(self.size).reshape(self.shape * 2) * (
                1 / np.asarray(prev_self)
            )
            make_grad(self, pass_in_grad, pass_in, grad, 0, prev_self)

    ret = Tensor(np.log(tensor), requires_grad=tensor.requires_grad)
    attach_backward_fn(ret, tensor.requires_grad, log_grad, tensor)
    return ret
