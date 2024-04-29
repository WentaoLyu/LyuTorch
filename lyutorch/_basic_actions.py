import numpy as np

from .Variables.utils import make_grad, attach_backward_fn
from .variable import Tensor

__all__ = ["squeeze"]


def squeeze(tensor: Tensor, axis=None):
    def squeeze_grad(
        self: Tensor,
        prev_self: Tensor,
        pass_in_grad: np.ndarray | float,
        pass_in: bool = True,
    ):
        sq_grad = np.eye(self.size)
        sq_grad = sq_grad.reshape(self.shape + prev_self.shape)
        make_grad(
            self, pass_in_grad, pass_in, sq_grad, 0, prev_self, broadcast_check=False
        )

    result = Tensor(np.squeeze(tensor, axis), requires_grad=tensor.requires_grad)
    attach_backward_fn(result, tensor.requires_grad, squeeze_grad, tensor)
    return result
