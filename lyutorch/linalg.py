import numpy as np

from .Variables.tensor import Tensor
from .Variables.utils import make_grad, attach_backward_fn


def norm(tensor: Tensor, order: int = None):
    def norm_grad(self, prev_self, pass_in_grad, pass_in, _order):
        if _order == 1:
            grad = np.sign(prev_self)
        elif _order == 2:
            grad = prev_self / self
        else:
            grad = 0
        make_grad(self, pass_in_grad, pass_in, grad, 0, prev_self)

    if (order == 1) | (order == 2):
        result = Tensor(
            np.linalg.norm(tensor, order, keepdims=False),
            requires_grad=tensor.requires_grad,
        )
        attach_backward_fn(
            result, tensor.requires_grad, norm_grad, tensor, _order=order
        )
        return result
    else:
        raise NotImplementedError("Only support 1 and 2 order")


def norm2(tensor: Tensor, order=None):
    def norm2_grad(self, prev_self, pass_in_grad, pass_in, _order):
        if _order == 1:
            grad = np.sign(prev_self)
        elif _order == 2:
            grad = 2 * prev_self
        make_grad(self, pass_in_grad, pass_in, grad, 0, prev_self)

    if (order == 1) | (order == 2):
        result = Tensor(
            np.linalg.norm(tensor, order, keepdims=False) ** order,
            requires_grad=tensor.requires_grad,
        )
        attach_backward_fn(
            result, tensor.requires_grad, norm2_grad, tensor, _order=order
        )
        return result
    else:
        raise NotImplementedError("Only support 1 and 2 order")
