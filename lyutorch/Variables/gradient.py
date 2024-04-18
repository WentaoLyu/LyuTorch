import numpy as np

from .tensor import Tensor

__all__ = ["add_grad", "mul_grad", "matmul_grad"]

from .utils import make_grad, _get_idarray_like, append_none_matmul_dims


def add_grad(
    self: Tensor,
    prev_self: Tensor,
    other: Tensor,
    pass_in_grad: np.ndarray,
    pass_in: bool = True,
):
    if prev_self.requires_grad | other.requires_grad:
        added_grad = _get_idarray_like(self)
        make_grad(self, pass_in_grad, pass_in, added_grad, 0, prev_self, other)


def mul_grad(
    self: Tensor,
    prev_self: Tensor,
    other: Tensor,
    pass_in_grad: np.ndarray,
    pass_in: bool = True,
):
    if prev_self.requires_grad:
        added_grad = _get_idarray_like(self) * other
        make_grad(self, pass_in_grad, pass_in, added_grad, 0, prev_self)
    if other.requires_grad:
        added_grad = _get_idarray_like(self) * prev_self
        make_grad(self, pass_in_grad, pass_in, added_grad, 0, other)


def matmul_grad(
    self: Tensor,
    prev_self: Tensor,
    other: Tensor,
    pass_in_grad: np.ndarray,
    pass_in: bool = True,
):
    if prev_self.requires_grad:
        left_grad = np.broadcast_to(
            np.asarray(other), self.shape[:-2] + other.shape[-2:]
        )
        left_grad = left_grad[..., None, None] * np.eye(self.shape[-2])
        left_grad = append_none_matmul_dims(self, left_grad)
        transpose_dims = (
            tuple(np.arange(self.ndim - 2))
            + tuple([2 * self.ndim - 2, 2 * self.ndim - 3])
            + tuple(np.arange(self.ndim - 2, 2 * self.ndim - 4))
            + tuple([2 * self.ndim - 1, 2 * self.ndim - 4])
        )
        left_grad = np.transpose(left_grad, transpose_dims)
        make_grad(self, pass_in_grad, pass_in, left_grad, 2, prev_self)
    if other.requires_grad:
        right_grad = np.broadcast_to(
            np.asarray(prev_self), self.shape[:-2] + prev_self.shape[-2:]
        )
        right_grad = right_grad[..., None, None] * np.eye(self.shape[-1])
        right_grad = append_none_matmul_dims(self, right_grad)
        transpose_dims = (
            tuple(np.arange(self.ndim - 2))
            + tuple([2 * self.ndim - 4, 2 * self.ndim - 1])
            + tuple(np.arange(self.ndim - 2, 2 * self.ndim - 4))
            + tuple([2 * self.ndim - 3, 2 * self.ndim - 2])
        )
        right_grad = np.transpose(right_grad, transpose_dims)
        make_grad(self, pass_in_grad, pass_in, right_grad, 2, other)
