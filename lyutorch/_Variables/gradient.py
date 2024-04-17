import numpy as np

from .tensor import Tensor

__all__ = ["add_grad", "matmul_grad"]

from .utils import make_grad, _get_identity_tensor_like


def add_grad(
    self: Tensor,
    prev_self: Tensor,
    other: Tensor,
    pass_in_grad: np.ndarray,
    pass_in: bool = True,
):
    if prev_self.requires_grad | other.requires_grad:
        added_grad = _get_identity_tensor_like(self)
        make_grad(self, pass_in, pass_in_grad, added_grad, 0, prev_self, other)


def matmul_grad(
    self: Tensor,
    prev_self: Tensor,
    other: Tensor,
    pass_in_grad: np.ndarray,
    pass_in: bool = True,
):
    if prev_self.requires_grad:
        left_grad = np.broadcast_to(
            np.asarray(other), self.shape[:-2] + prev_self.shape[-1:] + self.shape[-1:]
        )
        left_grad = left_grad[..., None, None] * np.eye(self.shape[-2])
        if self.shape[:-2]:
            left_grad = (
                np.eye(np.prod(self.shape[:-2])).reshape(self.shape[:-2] * 2)[
                    ..., None, None, None, None
                ]
                * left_grad
            )
        else:
            left_grad *= np.asarray(1)[..., None, None, None, None]
        transpose_dims = (
            tuple(np.arange(self.ndim - 2))
            + tuple([2 * self.ndim - 2, 2 * self.ndim - 3])
            + tuple(np.arange(self.ndim - 2, 2 * self.ndim - 4))
            + tuple([2 * self.ndim - 1, 2 * self.ndim - 4])
        )
        left_grad = np.transpose(left_grad, transpose_dims)
        make_grad(self, pass_in, pass_in_grad, left_grad, 2, prev_self)
    if other.requires_grad:
        pass
